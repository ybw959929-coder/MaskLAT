"""Stage-3 proposal groups used by the Group-in-VLM refinement path.

This module is intentionally independent from ``topology_group_decoder.py``.
The old topology V1/V2 experiments regroup and feed back at every prediction
stage, while this path establishes one hard Query-to-Group identity at st3,
inserts one token per valid Group into the VLM, and keeps that identity fixed
for st4--st9.

Tensor conventions:

* Query tensors are ``[B, Q, D]``;
* physical Group tables are also ``[B, Q, ...]`` and are valid only where
  ``group_valid_mask`` is true (the physical row is the component root);
* packed Group-token tables are ``[B, Gmax, ...]``;
* ``query_to_group`` stores physical root rows, never packed token slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .spatial_validity import (
    masked_normalized_resize,
    normalize_sam_valid_regions,
    valid_mask_from_normalized_boxes,
)
from .topology_group_decoder import MaskTopologyGrouper, SpatialMaskAligner


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _require_finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _masked_mean(features: Tensor, weights: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Pool ``[B,C,H,W]`` features with ``[B,N,H,W]`` soft masks."""

    if features.ndim != 4 or weights.ndim != 4:
        raise ValueError("features and weights must both be four-dimensional")
    if features.shape[0] != weights.shape[0] or features.shape[-2:] != weights.shape[-2:]:
        raise ValueError(
            "feature/mask grids must agree, got "
            f"{tuple(features.shape)} and {tuple(weights.shape)}"
        )
    numer = torch.einsum(
        "bnhw,bchw->bnc",
        weights.to(torch.float32),
        features.to(torch.float32),
    )
    denom = weights.to(torch.float32).sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
    pooled = numer / denom.clamp_min(eps)
    pooled = torch.where(denom.gt(eps), pooled, torch.zeros_like(pooled))
    return pooled.to(features.dtype)


def pack_physical_groups(
    physical: Tensor,
    group_valid_mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pack root rows of a physical ``[B,Q,...]`` table to ``[B,Gmax,...]``."""

    if physical.ndim < 2 or group_valid_mask.shape != physical.shape[:2]:
        raise ValueError("physical and group_valid_mask must share [B,Q]")
    if group_valid_mask.dtype != torch.bool:
        raise TypeError("group_valid_mask must be boolean")
    counts = group_valid_mask.sum(dim=1)
    if not bool(counts.ge(1).all()):
        raise ValueError("every sample must contain at least one valid Group")
    batch_size, num_queries = group_valid_mask.shape
    max_groups = int(counts.max().item())
    packed = physical.new_zeros((batch_size, max_groups) + physical.shape[2:])
    packed_valid = torch.zeros(
        batch_size,
        max_groups,
        dtype=torch.bool,
        device=physical.device,
    )
    packed_roots = torch.full(
        (batch_size, max_groups),
        -1,
        dtype=torch.long,
        device=physical.device,
    )
    query_ids = torch.arange(num_queries, device=physical.device)
    for batch_index in range(batch_size):
        roots = query_ids[group_valid_mask[batch_index]]
        group_count = int(roots.numel())
        packed[batch_index, :group_count] = physical[batch_index, roots]
        packed_valid[batch_index, :group_count] = True
        packed_roots[batch_index, :group_count] = roots
    return packed, packed_valid, packed_roots


def unpack_packed_groups(
    packed: Tensor,
    packed_valid_mask: Tensor,
    packed_root_indices: Tensor,
    *,
    num_queries: int,
) -> Tensor:
    """Scatter packed Group rows back to a zero-filled physical table."""

    if packed.ndim < 2:
        raise ValueError("packed must be [B,G,...]")
    expected = packed.shape[:2]
    if packed_valid_mask.shape != expected or packed_root_indices.shape != expected:
        raise ValueError("packed masks/root indices must match [B,G]")
    physical = packed.new_zeros((packed.shape[0], int(num_queries)) + packed.shape[2:])
    for batch_index in range(packed.shape[0]):
        valid = packed_valid_mask[batch_index]
        roots = packed_root_indices[batch_index, valid]
        if not bool(roots.ge(0).all() and roots.lt(num_queries).all()):
            raise ValueError("packed Group root index is out of range")
        physical[batch_index, roots] = packed[batch_index, valid]
    return physical


@dataclass
class St3GroupProposal:
    """All fixed proposal identity and visual evidence established at st3."""

    query_to_group: Tensor
    group_valid_mask: Tensor
    group_sizes: Tensor
    pairwise_iou: Tensor
    topology_centrality: Tensor
    member_attention_weights: Tensor
    group_features: Tensor
    group_soft_masks: Tensor
    group_boxes_cxcywh: Tensor
    group_geometry_valid_mask: Tensor
    sam_region_evidence: Tensor
    siglip_region_evidence: Tensor
    sam_region_evidence_valid_mask: Tensor
    siglip_region_evidence_valid_mask: Tensor
    query_mask_on_siglip: Tensor
    siglip_valid_mask: Tensor
    sam_valid_boxes_normalized: Tensor
    packed_group_tokens: Tensor
    packed_group_valid_mask: Tensor
    packed_group_root_indices: Tensor


class GaussianFourierBoxEncoder(nn.Module):
    """Falcon-style separate Gaussian Fourier encoders for center and size.

    Geometry is the original-image-normalized ``(cx, cy, w, h)``.  Area is not
    encoded.  The random matrices are deterministic persistent buffers, so a
    checkpoint has an unambiguous geometry coordinate system.
    """

    def __init__(
        self,
        hidden_dim: int,
        fourier_features: int = 64,
        sigma: float = 10.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if int(fourier_features) <= 0:
            raise ValueError("fourier_features must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        center_matrix = torch.randn(2, int(fourier_features), generator=generator) * float(sigma)
        size_matrix = torch.randn(2, int(fourier_features), generator=generator) * float(sigma)
        self.register_buffer("center_matrix", center_matrix, persistent=True)
        self.register_buffer("size_matrix", size_matrix, persistent=True)
        encoded_dim = 4 * int(fourier_features)
        self.projection = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection.apply(_init_linear)

    @staticmethod
    def _encode(values: Tensor, matrix: Tensor) -> Tensor:
        phase = 2.0 * math.pi * torch.matmul(values.to(torch.float32), matrix.to(torch.float32))
        return torch.cat((phase.cos(), phase.sin()), dim=-1)

    def forward(self, boxes_cxcywh: Tensor, valid_mask: Tensor) -> Tensor:
        if boxes_cxcywh.shape[:-1] != valid_mask.shape or boxes_cxcywh.shape[-1] != 4:
            raise ValueError("boxes must be [B,Q,4] and valid_mask must be [B,Q]")
        center = self._encode(boxes_cxcywh[..., :2], self.center_matrix)
        size = self._encode(boxes_cxcywh[..., 2:], self.size_matrix)
        projection_dtype = self.projection[0].weight.dtype
        encoded = self.projection(
            torch.cat((center, size), dim=-1).to(dtype=projection_dtype)
        )
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)


class GroupVisualTokenizer(nn.Module):
    """Create one proposal-aware VLM token for every valid st3 Group.

    The seed combines the st3 Group hypothesis with Group-mask-pooled SAM and
    SigLIP evidence.  The standard image-token path already supplies the
    dense visual grid to the VLM, so this proposal token deliberately avoids a
    second pre-VLM dense visual cross-attention.  Geometry is added after
    content encoding using the separate center/size Fourier representation
    above.
    """

    def __init__(
        self,
        hidden_dim: int,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        llm_hidden_dim: int,
        fourier_features: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.sam_projector = nn.Linear(sam_feature_dim, hidden_dim)
        self.siglip_projector = nn.Linear(siglip_feature_dim, hidden_dim)
        self.seed_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.content_norm = nn.LayerNorm(hidden_dim)
        self.geometry_encoder = GaussianFourierBoxEncoder(
            hidden_dim,
            fourier_features=fourier_features,
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden_dim),
        )
        self.apply(_init_linear)

    def forward(
        self,
        *,
        group_features: Tensor,
        group_valid_mask: Tensor,
        group_boxes_cxcywh: Tensor,
        geometry_valid_mask: Tensor,
        sam_region_evidence: Tensor,
        siglip_region_evidence: Tensor,
        sam_region_evidence_valid_mask: Tensor,
        siglip_region_evidence_valid_mask: Tensor,
    ) -> Tensor:
        if group_features.ndim != 3:
            raise ValueError("group_features must be [B,Q,D]")
        if group_valid_mask.shape != group_features.shape[:2]:
            raise ValueError("group_valid_mask must be [B,Q]")
        expected_region_valid_shape = group_features.shape[:2] + (1,)
        for name, valid_mask in (
            ("sam_region_evidence_valid_mask", sam_region_evidence_valid_mask),
            ("siglip_region_evidence_valid_mask", siglip_region_evidence_valid_mask),
        ):
            if valid_mask.shape != expected_region_valid_shape:
                raise ValueError(
                    f"{name} must be {expected_region_valid_shape}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            if valid_mask.dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
        sam_region = self.sam_projector(sam_region_evidence)
        siglip_region = self.siglip_projector(siglip_region_evidence)
        # Gate after the learnable projections as well as before them: their
        # biases must never turn an empty region into visual evidence.
        sam_region = torch.where(
            sam_region_evidence_valid_mask,
            sam_region,
            torch.zeros_like(sam_region),
        )
        siglip_region = torch.where(
            siglip_region_evidence_valid_mask,
            siglip_region,
            torch.zeros_like(siglip_region),
        )
        seed = self.seed_fusion(torch.cat((group_features, sam_region, siglip_region), dim=-1))
        seed = seed * group_valid_mask.unsqueeze(-1).to(seed.dtype)
        content = self.content_norm(seed)
        geometry = self.geometry_encoder(group_boxes_cxcywh, geometry_valid_mask)
        tokens = self.token_mlp(content + geometry)
        tokens = tokens * group_valid_mask.unsqueeze(-1).to(tokens.dtype)
        _require_finite("proposal_group_tokens", tokens)
        return tokens


class St3ProposalBuilder(nn.Module):
    """Build fixed st3 Groups and their proposal-aware VLM tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        llm_hidden_dim: int,
        num_heads: int = 8,
        mask_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        fourier_features: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.mask_threshold = float(mask_threshold)
        self.grouper = MaskTopologyGrouper(mask_threshold, iou_threshold)
        self.spatial_aligner = SpatialMaskAligner()
        self.group_query = nn.Linear(hidden_dim, hidden_dim)
        self.member_key = nn.Linear(hidden_dim, hidden_dim)
        self.member_value = nn.Linear(hidden_dim, hidden_dim)
        self.group_output = nn.Linear(hidden_dim, hidden_dim)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.tokenizer = GroupVisualTokenizer(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            fourier_features=fourier_features,
        )
        self.apply(_init_linear)

    @property
    def siglip_region_projector(self) -> nn.Linear:
        """Compatibility view of the Group-level SigLIP projector.

        ``MaskLATSegmentor`` uses this width to make repeated configuration calls
        idempotent.  The projector belongs to the Group tokenizer; exposing
        this read-only view does not reintroduce Query-level MaskPool state.
        """

        return self.tokenizer.siglip_projector

    @staticmethod
    def _group_boxes(
        group_soft_masks: Tensor,
        group_valid_mask: Tensor,
        spatial_metadata: Dict[str, Tensor],
        mask_threshold: float,
    ) -> Tuple[Tensor, Tensor]:
        """Return original-image-normalized ``cx,cy,w,h`` for Groups.

        The masks live on the padded SAM canvas.  Their support box is first
        measured in continuous SAM input coordinates, then mapped through the
        inverse recorded ``original_to_sam`` transform.  This keeps the
        geometry token invariant to SAM resize/padding and also handles any
        future recoverable affine crop/translation recorded by the dataset.
        """

        batch_size, num_groups, height, width = group_soft_masks.shape
        required = ("original_size", "original_to_sam", "sam_input_size", "sam_valid_region")
        missing = [key for key in required if key not in spatial_metadata]
        if missing:
            raise ValueError(
                f"spatial_metadata is missing Group-geometry fields: {missing}"
            )
        metadata = {
            key: spatial_metadata[key].to(
                device=group_soft_masks.device,
                dtype=torch.float32,
            )
            for key in required
        }
        expected_shapes = {
            "original_size": (batch_size, 2),
            "original_to_sam": (batch_size, 3, 3),
            "sam_input_size": (batch_size, 2),
            "sam_valid_region": (batch_size, 4),
        }
        for key, value in metadata.items():
            if tuple(value.shape) != expected_shapes[key]:
                raise ValueError(
                    f"spatial_metadata[{key!r}] must be "
                    f"{expected_shapes[key]}, got {tuple(value.shape)}"
                )
            _require_finite(f"spatial_metadata[{key}]", value)
        if not bool(metadata["original_size"].gt(0).all()):
            raise ValueError("original_size must be positive")
        determinant = torch.linalg.det(metadata["original_to_sam"])
        if not bool(determinant.abs().gt(1.0e-8).all()):
            raise ValueError("original_to_sam must be invertible")

        sam_valid_boxes = normalize_sam_valid_regions(
            metadata["sam_input_size"],
            metadata["sam_valid_region"],
        )
        valid_pixels = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            (height, width),
        )[:, 0]
        binary = group_soft_masks.detach().ge(float(mask_threshold)) & valid_pixels[:, None]
        y0_grid = (
            torch.arange(height, device=binary.device, dtype=torch.float32) / float(height)
        ).view(1, 1, height, 1)
        y1_grid = (
            (torch.arange(height, device=binary.device, dtype=torch.float32) + 1.0)
            / float(height)
        ).view(1, 1, height, 1)
        x0_grid = (
            torch.arange(width, device=binary.device, dtype=torch.float32) / float(width)
        ).view(1, 1, 1, width)
        x1_grid = (
            (torch.arange(width, device=binary.device, dtype=torch.float32) + 1.0)
            / float(width)
        ).view(1, 1, 1, width)
        large = torch.tensor(2.0, device=binary.device, dtype=torch.float32)
        x_min = torch.where(binary, x0_grid, large).amin(dim=(-2, -1))
        y_min = torch.where(binary, y0_grid, large).amin(dim=(-2, -1))
        x_max = torch.where(binary, x1_grid, -large).amax(dim=(-2, -1))
        y_max = torch.where(binary, y1_grid, -large).amax(dim=(-2, -1))
        geometry_valid = binary.flatten(2).any(dim=-1) & group_valid_mask

        sam_height = metadata["sam_input_size"][:, 0, None]
        sam_width = metadata["sam_input_size"][:, 1, None]
        x0_sam = x_min * sam_width
        x1_sam = x_max * sam_width
        y0_sam = y_min * sam_height
        y1_sam = y_max * sam_height
        ones = torch.ones_like(x0_sam)
        sam_corners = torch.stack(
            (
                torch.stack((x0_sam, y0_sam, ones), dim=-1),
                torch.stack((x1_sam, y0_sam, ones), dim=-1),
                torch.stack((x0_sam, y1_sam, ones), dim=-1),
                torch.stack((x1_sam, y1_sam, ones), dim=-1),
            ),
            dim=2,
        )
        sam_to_original = torch.linalg.inv(metadata["original_to_sam"])
        original_corners = torch.einsum(
            "bij,bgkj->bgki",
            sam_to_original,
            sam_corners,
        )
        original_corners = original_corners / original_corners[..., 2:].clamp_min(
            1.0e-8
        )
        original_height = metadata["original_size"][:, 0, None]
        original_width = metadata["original_size"][:, 1, None]
        x0 = (
            original_corners[..., 0].amin(dim=2) / original_width
        ).clamp(0.0, 1.0)
        x1 = (
            original_corners[..., 0].amax(dim=2) / original_width
        ).clamp(0.0, 1.0)
        y0 = (
            original_corners[..., 1].amin(dim=2) / original_height
        ).clamp(0.0, 1.0)
        y1 = (
            original_corners[..., 1].amax(dim=2) / original_height
        ).clamp(0.0, 1.0)
        boxes = torch.stack(
            (
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                (x1 - x0).clamp_min(0.0),
                (y1 - y0).clamp_min(0.0),
            ),
            dim=-1,
        )
        boxes = torch.where(geometry_valid.unsqueeze(-1), boxes, torch.zeros_like(boxes))
        return boxes, geometry_valid

    def forward(
        self,
        *,
        query_states: Tensor,
        mask_logits: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> St3GroupProposal:
        if query_states.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError("query_states/mask_logits must be [B,Q,D] and [B,Q,H,W]")
        if query_states.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("query and mask B/Q dimensions differ")
        sam_valid_boxes = normalize_sam_valid_regions(
            spatial_metadata["sam_input_size"],
            spatial_metadata["sam_valid_region"],
        ).to(device=mask_logits.device)
        sam_mask_valid = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            mask_logits.shape[-2:],
        )
        sam_feature_valid = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            sam_mask_features.shape[-2:],
        )
        grouping = self.grouper(mask_logits, sam_valid_mask=sam_mask_valid)
        mask_prob = mask_logits.sigmoid() * sam_mask_valid.to(mask_logits.dtype)
        # Normalize through the valid source coverage so SAM padding cannot
        # attenuate masks along a resize boundary.  This is the same spatial
        # contract used by the original topology evidence path.  This tensor
        # is retained only to form the Group mask; no Query-level MaskPool is
        # performed in this proposal path.
        mask_on_sam_features, _ = masked_normalized_resize(
            mask_prob,
            sam_mask_valid,
            sam_mask_features.shape[-2:],
            target_valid_mask=sam_feature_valid,
        )

        query_mask_on_siglip, resolved_siglip_valid = self.spatial_aligner(
            mask_prob,
            spatial_metadata,
            siglip_spatial_features.shape[-2:],
            supplied_valid_mask=siglip_valid_mask,
            sam_source_valid_mask=sam_mask_valid,
        )
        # The grouping attention is intentionally based directly on the st3
        # Query states.  SAM/SigLIP enter only after member masks have been
        # aggregated into one Group mask below.
        member = query_states

        batch_size, num_queries, _ = member.shape
        membership = grouping.query_to_group[:, None, :].eq(
            torch.arange(num_queries, device=member.device).view(1, num_queries, 1)
        )
        group_sizes = grouping.group_sizes.clamp_min(1).to(member.dtype)
        mean_group = torch.einsum("bgq,bqd->bgd", membership.to(member.dtype), member)
        mean_group = mean_group / group_sizes.unsqueeze(-1)
        logits = torch.einsum(
            "bgd,bqd->bgq",
            self.group_query(mean_group).to(torch.float32),
            self.member_key(member).to(torch.float32),
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits.masked_fill(~membership, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(membership, weights, torch.zeros_like(weights))
        weights = weights * grouping.group_valid_mask.unsqueeze(-1).to(weights.dtype)
        attended = torch.einsum(
            "bgq,bqd->bgd",
            weights.to(member.dtype),
            self.member_value(member),
        )
        group_features = self.group_norm(mean_group + self.group_output(attended))
        group_features = group_features * grouping.group_valid_mask.unsqueeze(-1).to(group_features.dtype)
        group_soft_masks = torch.einsum(
            "bgq,bqhw->bghw",
            weights.to(mask_prob.dtype),
            mask_prob,
        )
        group_soft_masks = group_soft_masks * grouping.group_valid_mask[:, :, None, None].to(
            group_soft_masks.dtype
        )
        boxes, geometry_valid = self._group_boxes(
            group_soft_masks,
            grouping.group_valid_mask,
            spatial_metadata,
            self.mask_threshold,
        )
        # Region evidence for a Group must be pooled from the same aggregated
        # mask that defines that Group.  Reading ``sam_raw[root]`` or
        # ``siglip_raw[root]`` would make the VLM token depend on the arbitrary
        # minimum Query id chosen as the physical component root.  The linear
        # mixtures below are exactly the member-attention counterpart of
        # aligning/pooling ``group_soft_masks`` on each visual grid.
        group_mask_on_sam_features = torch.einsum(
            "bgq,bqhw->bghw",
            weights.to(mask_on_sam_features.dtype),
            mask_on_sam_features,
        )
        group_mask_on_siglip = torch.einsum(
            "bgq,bqhw->bghw",
            weights.to(query_mask_on_siglip.dtype),
            query_mask_on_siglip,
        )
        group_sam_raw = _masked_mean(
            sam_mask_features,
            group_mask_on_sam_features,
        )
        group_siglip_raw = _masked_mean(
            siglip_spatial_features,
            group_mask_on_siglip,
        )
        # A topology-empty Group still has tiny finite sigmoid mass.  That
        # mass must not be normalized into an artificial full-image region
        # descriptor.  Geometry validity uses the exact same detached binary
        # threshold/valid-region contract as Group construction.
        group_has_region = geometry_valid.unsqueeze(-1)
        group_sam_has_evidence = (
            group_mask_on_sam_features.to(torch.float32)
            .sum(dim=(-2, -1), keepdim=False)
            .unsqueeze(-1)
            .gt(1.0e-6)
            & group_has_region
        )
        group_siglip_has_evidence = (
            group_mask_on_siglip.to(torch.float32)
            .sum(dim=(-2, -1), keepdim=False)
            .unsqueeze(-1)
            .gt(1.0e-6)
            & group_has_region
        )
        group_sam_raw = torch.where(
            group_sam_has_evidence,
            group_sam_raw,
            torch.zeros_like(group_sam_raw),
        )
        group_siglip_raw = torch.where(
            group_siglip_has_evidence,
            group_siglip_raw,
            torch.zeros_like(group_siglip_raw),
        )
        member_weights = weights.gather(
            1,
            grouping.query_to_group.unsqueeze(1),
        ).squeeze(1)
        physical_tokens = self.tokenizer(
            group_features=group_features,
            group_valid_mask=grouping.group_valid_mask,
            group_boxes_cxcywh=boxes,
            geometry_valid_mask=geometry_valid,
            sam_region_evidence=group_sam_raw,
            siglip_region_evidence=group_siglip_raw,
            sam_region_evidence_valid_mask=group_sam_has_evidence,
            siglip_region_evidence_valid_mask=group_siglip_has_evidence,
        )
        packed_tokens, packed_valid, packed_roots = pack_physical_groups(
            physical_tokens,
            grouping.group_valid_mask,
        )
        for name, value in (
            ("st3_group_features", group_features),
            ("st3_group_soft_masks", group_soft_masks),
            ("st3_group_boxes", boxes),
            ("st3_packed_group_tokens", packed_tokens),
        ):
            _require_finite(name, value)
        return St3GroupProposal(
            query_to_group=grouping.query_to_group,
            group_valid_mask=grouping.group_valid_mask,
            group_sizes=grouping.group_sizes,
            pairwise_iou=grouping.pairwise_iou,
            topology_centrality=grouping.topology_centrality,
            member_attention_weights=member_weights,
            group_features=group_features,
            group_soft_masks=group_soft_masks,
            group_boxes_cxcywh=boxes,
            group_geometry_valid_mask=geometry_valid,
            sam_region_evidence=group_sam_raw,
            siglip_region_evidence=group_siglip_raw,
            sam_region_evidence_valid_mask=group_sam_has_evidence,
            siglip_region_evidence_valid_mask=group_siglip_has_evidence,
            query_mask_on_siglip=query_mask_on_siglip,
            siglip_valid_mask=resolved_siglip_valid,
            sam_valid_boxes_normalized=sam_valid_boxes,
            packed_group_tokens=packed_tokens,
            packed_group_valid_mask=packed_valid,
            packed_group_root_indices=packed_roots,
        )


class GatedResidualFusion(nn.Module):
    """Query-dependent residual fusion without an additional scalar alpha/beta."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.left_norm = nn.LayerNorm(hidden_dim)
        self.right_norm = nn.LayerNorm(hidden_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.apply(_init_linear)
        # Exact residual initialization protects the pretrained MaskLAT path;
        # the first backward pass still gives a non-zero gradient to the last
        # delta projection.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(self, left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
        if left.shape != right.shape:
            raise ValueError(f"fusion tensors must match: {tuple(left.shape)} != {tuple(right.shape)}")
        x = self.left_norm(left)
        y = self.right_norm(right)
        interaction = x * y
        delta_value = self.delta_mlp(torch.cat((x, y), dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat((x, y, interaction), dim=-1)))
        delta = gate * delta_value
        updated = left + delta.to(left.dtype)
        _require_finite("gated_residual_delta", delta)
        _require_finite("gated_residual_output", updated)
        return updated, delta


class FixedGroupVLMRefiner(nn.Module):
    """Post-VLM Group/Query/Cond interactions for the fixed st3 partition."""

    def __init__(self, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        # Hugging Face's recursive gradient-checkpointing switch discovers
        # modules through this attribute and injects
        # ``_gradient_checkpointing_func``.  The refiner runs after each
        # checkpointed decoder layer, so it needs its own pure-compute
        # checkpoint boundary to avoid retaining all st3--st9 activations.
        self.gradient_checkpointing = False
        self.group_input_norm = nn.LayerNorm(hidden_dim)
        self.group_cond_attention = nn.MultiheadAttention(
            hidden_dim,
            int(num_heads),
            batch_first=True,
        )
        self.group_member_query = nn.Linear(hidden_dim, hidden_dim)
        self.member_key = nn.Linear(hidden_dim, hidden_dim)
        self.member_value = nn.Linear(hidden_dim, hidden_dim)
        self.group_from_cond = GatedResidualFusion(hidden_dim)
        self.group_from_seg = GatedResidualFusion(hidden_dim)
        self.group_from_members = GatedResidualFusion(hidden_dim)
        self.query_from_group_seg = GatedResidualFusion(hidden_dim)
        self.query_from_group = GatedResidualFusion(hidden_dim)
        self.member_quality = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_linear)
        # ``self.apply`` also visits every nested GatedResidualFusion and would
        # otherwise overwrite the exact-zero residual deltas established by
        # their constructors.  Restore those projections after the outer
        # initialization so the new feedback path is an identity map at step
        # zero while remaining trainable from the first backward pass.
        for fusion in (
            self.group_from_cond,
            self.group_from_seg,
            self.group_from_members,
            self.query_from_group_seg,
            self.query_from_group,
        ):
            nn.init.zeros_(fusion.delta_mlp[-1].weight)
            nn.init.zeros_(fusion.delta_mlp[-1].bias)

    @staticmethod
    def _gather_groups(group_features: Tensor, query_to_group: Tensor) -> Tensor:
        return group_features.gather(
            1,
            query_to_group.unsqueeze(-1).expand(-1, -1, group_features.shape[-1]),
        )

    def _read_conditions(
        self,
        group_features: Tensor,
        group_valid_mask: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tensor:
        if local_conditions.ndim != 3:
            raise ValueError("local_conditions must be [Nseg,C,D]")
        if local_condition_valid_mask.shape != local_conditions.shape[:2]:
            raise ValueError("local condition validity must be [Nseg,C]")
        if not bool(local_condition_valid_mask.any(dim=1).all()):
            raise ValueError("every SEG row must own at least one local condition")
        context, _ = self.group_cond_attention(
            query=group_features,
            key=local_conditions,
            value=local_conditions,
            key_padding_mask=~local_condition_valid_mask,
            need_weights=False,
        )
        context = context * group_valid_mask.unsqueeze(-1).to(context.dtype)
        updated, _ = self.group_from_cond(group_features, context)
        return updated * group_valid_mask.unsqueeze(-1).to(updated.dtype)

    def initialize_after_vlm(
        self,
        *,
        query_states: Tensor,
        group_states: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        seg_states: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
        proposal_group_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Reverse Group read, then one explicit Group+SEG Query update."""

        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._gradient_checkpointing_func(
                self._initialize_after_vlm_impl,
                query_states,
                group_states,
                query_to_group,
                group_valid_mask,
                seg_states,
                local_conditions,
                local_condition_valid_mask,
                proposal_group_states,
            )
        return self._initialize_after_vlm_impl(
            query_states,
            group_states,
            query_to_group,
            group_valid_mask,
            seg_states,
            local_conditions,
            local_condition_valid_mask,
            proposal_group_states,
        )

    def _initialize_after_vlm_impl(
        self,
        query_states: Tensor,
        group_states: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        seg_states: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
        proposal_group_states: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Side-effect-free implementation used by activation checkpointing."""

        if proposal_group_states is not None:
            if proposal_group_states.shape != group_states.shape:
                raise ValueError(
                    "proposal_group_states and VLM group_states must match"
                )
            group_states = self.group_input_norm(
                group_states + proposal_group_states
            )
            group_states = group_states * group_valid_mask.unsqueeze(-1).to(
                group_states.dtype
            )
        groups = self._read_conditions(
            group_states,
            group_valid_mask,
            local_conditions,
            local_condition_valid_mask,
        )
        seg_for_group = seg_states[:, None, :].expand_as(groups)
        groups, _ = self.group_from_seg(groups, seg_for_group)
        groups = groups * group_valid_mask.unsqueeze(-1).to(groups.dtype)
        member_group = self._gather_groups(groups, query_to_group)
        seg_for_query = seg_states[:, None, :].expand_as(query_states)
        group_seg = member_group + seg_for_query
        queries, _ = self.query_from_group_seg(query_states, group_seg)
        return queries, groups

    def update_after_stage(
        self,
        *,
        query_states: Tensor,
        group_states: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
        apply_query_feedback: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Member→Group, Group→local Cond, and optional Group→member Query."""

        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._gradient_checkpointing_func(
                self._update_after_stage_impl,
                query_states,
                group_states,
                query_to_group,
                group_valid_mask,
                local_conditions,
                local_condition_valid_mask,
                apply_query_feedback,
            )
        return self._update_after_stage_impl(
            query_states,
            group_states,
            query_to_group,
            group_valid_mask,
            local_conditions,
            local_condition_valid_mask,
            apply_query_feedback,
        )

    def _update_after_stage_impl(
        self,
        query_states: Tensor,
        group_states: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
        apply_query_feedback: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Side-effect-free implementation used by activation checkpointing."""

        num_queries = query_states.shape[1]
        membership = query_to_group[:, None, :].eq(
            torch.arange(num_queries, device=query_states.device).view(1, num_queries, 1)
        )
        logits = torch.einsum(
            "bgd,bqd->bgq",
            self.group_member_query(group_states).to(torch.float32),
            self.member_key(query_states).to(torch.float32),
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits.masked_fill(~membership, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(membership, weights, torch.zeros_like(weights))
        weights = weights * group_valid_mask.unsqueeze(-1).to(weights.dtype)
        member_context = torch.einsum(
            "bgq,bqd->bgd",
            weights.to(query_states.dtype),
            self.member_value(query_states),
        )
        groups, _ = self.group_from_members(group_states, member_context)
        groups = self._read_conditions(
            groups,
            group_valid_mask,
            local_conditions,
            local_condition_valid_mask,
        )
        if apply_query_feedback:
            member_group = self._gather_groups(groups, query_to_group)
            queries, _ = self.query_from_group(query_states, member_group)
        else:
            queries = query_states
        # Keep the public stage-output contract identical to the existing
        # topology aggregator: one attention weight per Query member, not the
        # internal sparse [B, physical_group, Q] matrix.
        member_weights = weights.gather(
            1,
            query_to_group.unsqueeze(1),
        ).squeeze(1)
        return queries, groups, member_weights

    def quality_logits(
        self,
        query_states: Tensor,
        group_states: Tensor,
        query_to_group: Tensor,
    ) -> Tensor:
        member_group = self._gather_groups(group_states, query_to_group)
        quality_input = torch.cat(
            (
                query_states,
                member_group,
                query_states * member_group,
                query_states - member_group,
            ),
            dim=-1,
        )
        logits = self.member_quality(quality_input).squeeze(-1)
        _require_finite("fixed_group_member_quality_logits", logits)
        return logits


def build_group_local_self_attention_mask(
    query_to_group: Tensor,
    *,
    dtype: torch.dtype,
    num_heads: int,
) -> Tensor:
    """Build additive self-attention masks that isolate fixed Group members."""

    if query_to_group.ndim != 2:
        raise ValueError("query_to_group must be [B,Q]")
    allowed = query_to_group[:, :, None].eq(query_to_group[:, None, :])
    floor = torch.finfo(dtype).min
    additive = torch.zeros(
        allowed.shape,
        dtype=dtype,
        device=query_to_group.device,
    ).masked_fill(~allowed, floor)
    return additive[:, None].expand(-1, int(num_heads), -1, -1).flatten(0, 1)


__all__ = [
    "FixedGroupVLMRefiner",
    "GaussianFourierBoxEncoder",
    "GatedResidualFusion",
    "GroupVisualTokenizer",
    "St3GroupProposal",
    "St3ProposalBuilder",
    "build_group_local_self_attention_mask",
    "pack_physical_groups",
    "unpack_packed_groups",
]
