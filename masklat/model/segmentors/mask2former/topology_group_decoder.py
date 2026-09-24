"""Mask-topology grouped hierarchical decoder building blocks.

The modules in this file never construct a group mask.  Discrete group
membership is computed from detached, binarized query masks; every formal
prediction produced by :class:`GroupPredictionAdapter` is gathered from one
of the original query masks.

Tensor conventions used throughout this file are:

* query tensors: ``[B, Q, D]``;
* query masks: ``[B, Q, H, W]``;
* sparse group tables: ``[B, Q, ...]`` with valid rows selected by
  ``group_valid_mask``;
* ``query_to_group[b, q]`` is the minimum query id (the physical root) of
  query ``q``'s connected component.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mask_topology_grouping import connected_component_roots_from_adjacency
from .spatial_validity import (
    masked_normalized_grid_sample,
    masked_normalized_resize,
    normalize_sam_valid_regions,
    valid_mask_from_normalized_boxes,
)


def _init_linear(module: nn.Module) -> None:
    """Initialize newly introduced linear layers with the requested scheme."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _require_finite(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _finite_logit_bound(tensor: Tensor) -> float:
    """Return a large finite masking value that is safe for fp16/bf16."""

    return float(min(1.0e4, torch.finfo(tensor.dtype).max / 2.0))


@dataclass
class TopologyGroupingOutput:
    """Detached result of grouping ``Q`` masks for every batch element.

    ``group_sizes`` is a sparse physical table: sizes are written at root
    rows and all non-root rows are zero.  ``topology_centrality`` is defined
    per member query.  No returned tensor participates in autograd.
    """

    query_to_group: Tensor
    group_valid_mask: Tensor
    group_sizes: Tensor
    pairwise_iou: Tensor
    topology_centrality: Tensor


@dataclass
class TopologyGroupStageOutput:
    """Complete output of one prediction stage.

    The structural fields are populated by the decoder core.  Classification
    logits are attached later by ``MaskLATSegmentor`` so that matching and loss
    remain separate from the decoder implementation.
    """

    stage_index: int
    query_states: Tensor
    query_mask_logits: Tensor
    query_to_group: Tensor
    group_valid_mask: Tensor
    group_sizes: Tensor
    pairwise_iou: Tensor
    topology_centrality: Tensor
    sam_region_evidence: Tensor
    siglip_region_evidence: Tensor
    member_evidence: Tensor
    group_features: Tensor
    member_attention_weights: Tensor
    query_mask_on_siglip: Tensor
    siglip_valid_mask: Tensor
    sam_valid_boxes_normalized: Tensor
    group_transport: Optional[Tensor] = None
    transported_group_prior: Optional[Tensor] = None
    group_innovation: Optional[Tensor] = None
    new_group_mask: Optional[Tensor] = None
    group_class_logits: Optional[Tensor] = None
    member_quality_logits: Optional[Tensor] = None
    feedback_delta_norm: Optional[Tensor] = None
    feedback_applied: bool = False


class MaskTopologyGrouper(nn.Module):
    """Build deterministic mask-IoU connected components.

    Input:
        ``mask_logits`` float tensor ``[B,Q,Hm,Wm]``.
    Output:
        :class:`TopologyGroupingOutput` with root-based physical indices.

    The sigmoid, threshold, IoU, and label-propagation path is executed under
    ``no_grad``.  Empty/empty pairs have off-diagonal IoU zero and every
    diagonal is forced to one.  Label propagation is device-native and never
    transfers grouping state to CPU.
    """

    def __init__(self, mask_threshold: float = 0.5, iou_threshold: float = 0.7):
        super().__init__()
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError(f"mask_threshold must be in [0,1], got {mask_threshold}")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold}")
        self.mask_threshold = float(mask_threshold)
        self.iou_threshold = float(iou_threshold)

    @torch.no_grad()
    def forward(
        self,
        mask_logits: Tensor,
        sam_valid_mask: Optional[Tensor] = None,
    ) -> TopologyGroupingOutput:
        if mask_logits.ndim != 4:
            raise ValueError(f"mask_logits must be [B,Q,H,W], got {tuple(mask_logits.shape)}")
        if min(mask_logits.shape) <= 0:
            raise ValueError(f"mask_logits dimensions must be non-empty, got {tuple(mask_logits.shape)}")
        _require_finite("mask_logits", mask_logits)

        batch_size, num_queries, height, width = mask_logits.shape
        if sam_valid_mask is None:
            sam_valid_mask = torch.ones(
                batch_size,
                1,
                height,
                width,
                dtype=torch.bool,
                device=mask_logits.device,
            )
        expected_valid_shape = (batch_size, 1, height, width)
        if sam_valid_mask.shape != expected_valid_shape:
            raise ValueError(
                "sam_valid_mask must match the Query-mask grid: "
                f"{tuple(sam_valid_mask.shape)} != {expected_valid_shape}"
            )
        if sam_valid_mask.device != mask_logits.device:
            raise ValueError("sam_valid_mask and mask_logits must share one device")
        if sam_valid_mask.dtype != torch.bool:
            raise TypeError("sam_valid_mask must have dtype torch.bool")

        binary_masks = (
            mask_logits.sigmoid().detach().ge(self.mask_threshold)
            & sam_valid_mask
        )
        flat_masks = binary_masks.flatten(2).to(torch.float32)
        intersections = torch.matmul(flat_masks, flat_masks.transpose(1, 2))
        areas = flat_masks.sum(dim=-1)
        unions = areas[:, :, None] + areas[:, None, :] - intersections
        pairwise_iou = torch.where(
            unions > 0,
            intersections / unions.clamp_min(1.0),
            torch.zeros_like(intersections),
        )
        diagonal = torch.arange(num_queries, device=mask_logits.device)
        pairwise_iou[:, diagonal, diagonal] = 1.0

        adjacency = pairwise_iou.ge(self.iou_threshold)
        adjacency = adjacency | adjacency.transpose(1, 2)
        adjacency[:, diagonal, diagonal] = True
        query_to_group = connected_component_roots_from_adjacency(adjacency)

        query_ids = diagonal.unsqueeze(0).expand(batch_size, -1)
        group_valid_mask = query_to_group.eq(query_ids)
        if not bool(group_valid_mask.any(dim=1).all()):
            raise RuntimeError("topology grouping produced a sample without a valid group")

        group_sizes = torch.zeros(
            batch_size,
            num_queries,
            dtype=torch.long,
            device=mask_logits.device,
        )
        group_sizes.scatter_add_(
            1,
            query_to_group,
            torch.ones_like(query_to_group),
        )
        same_group = query_to_group[:, :, None].eq(query_to_group[:, None, :])
        member_iou_sum = (pairwise_iou * same_group.to(pairwise_iou.dtype)).sum(dim=-1)
        member_sizes = group_sizes.gather(1, query_to_group).clamp_min(1).to(torch.float32)
        topology_centrality = member_iou_sum / member_sizes
        topology_centrality = torch.where(
            member_sizes.eq(1),
            torch.ones_like(topology_centrality),
            topology_centrality,
        )

        if not bool(query_to_group.ge(0).all() and query_to_group.lt(num_queries).all()):
            raise RuntimeError("every query must map to exactly one in-range physical root")
        if not bool(group_valid_mask.sum(dim=1).ge(1).all()):
            raise RuntimeError("group count must be in [1,Q]")
        _require_finite("pairwise_iou", pairwise_iou)
        _require_finite("topology_centrality", topology_centrality)
        return TopologyGroupingOutput(
            query_to_group=query_to_group.detach(),
            group_valid_mask=group_valid_mask.detach(),
            group_sizes=group_sizes.detach(),
            pairwise_iou=pairwise_iou.detach(),
            topology_centrality=topology_centrality.detach(),
        )


class SpatialMaskAligner(nn.Module):
    """Map SAM-coordinate query masks to SigLIP patch centers.

    ``spatial_metadata`` must contain batched tensors:

    * ``original_size [B,2]`` (height, width);
    * ``original_to_sam [B,3,3]``;
    * ``sam_input_size [B,2]``;
    * ``sam_valid_region [B,4]`` (top, left, bottom, right);
    * ``original_to_siglip [B,3,3]``;
    * ``siglip_input_size [B,2]``;
    * ``siglip_valid_region [B,4]``;
    * ``siglip_patch_size [B,2]``.

    The returned mask ``[B,Q,Hs,Ws]`` is differentiable with respect to the
    input mask probabilities.  Metadata and validity decisions are not
    differentiable.  Sampling uses bilinear ``grid_sample``, zero padding,
    and ``align_corners=False`` exactly.
    """

    REQUIRED_KEYS = (
        "original_size",
        "original_to_sam",
        "sam_input_size",
        "sam_valid_region",
        "original_to_siglip",
        "siglip_input_size",
        "siglip_valid_region",
        "siglip_patch_size",
    )

    def forward(
        self,
        mask_prob: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_spatial_size: Tuple[int, int],
        supplied_valid_mask: Optional[Tensor] = None,
        sam_source_valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if mask_prob.ndim != 4:
            raise ValueError(f"mask_prob must be [B,Q,H,W], got {tuple(mask_prob.shape)}")
        if spatial_metadata is None:
            raise ValueError("spatial_metadata is required for SAM-to-SigLIP alignment")
        missing = [key for key in self.REQUIRED_KEYS if key not in spatial_metadata]
        if missing:
            raise ValueError(f"spatial_metadata is missing required fields: {missing}")

        batch_size, _, _, _ = mask_prob.shape
        hs, ws = int(siglip_spatial_size[0]), int(siglip_spatial_size[1])
        if hs <= 0 or ws <= 0:
            raise ValueError(f"invalid SigLIP spatial size {(hs, ws)}")

        metadata = {
            key: spatial_metadata[key].to(device=mask_prob.device, dtype=torch.float32)
            for key in self.REQUIRED_KEYS
        }
        expected_shapes = {
            "original_size": (batch_size, 2),
            "original_to_sam": (batch_size, 3, 3),
            "sam_input_size": (batch_size, 2),
            "sam_valid_region": (batch_size, 4),
            "original_to_siglip": (batch_size, 3, 3),
            "siglip_input_size": (batch_size, 2),
            "siglip_valid_region": (batch_size, 4),
            "siglip_patch_size": (batch_size, 2),
        }
        for key, value in metadata.items():
            if tuple(value.shape) != expected_shapes[key]:
                raise ValueError(
                    f"spatial_metadata[{key!r}] must be "
                    f"{expected_shapes[key]}, got {tuple(value.shape)}"
                )
            _require_finite(f"spatial_metadata[{key}]", value)
        for key in (
            "original_size",
            "sam_input_size",
            "siglip_input_size",
            "siglip_patch_size",
        ):
            if not bool(metadata[key].gt(0).all()):
                raise ValueError(f"spatial_metadata[{key!r}] must be positive")
        for key in ("original_to_sam", "original_to_siglip"):
            determinant = torch.linalg.det(metadata[key])
            if not bool(determinant.abs().gt(1e-8).all()):
                invalid = torch.nonzero(
                    determinant.abs().le(1e-8),
                    as_tuple=False,
                ).flatten().tolist()
                raise ValueError(
                    f"spatial_metadata[{key!r}] is singular at batch "
                    f"indices {invalid}"
                )

        sam_valid_boxes = normalize_sam_valid_regions(
            metadata["sam_input_size"],
            metadata["sam_valid_region"],
        )
        siglip_size = metadata["siglip_input_size"]
        siglip_region = metadata["siglip_valid_region"]
        siglip_top, siglip_left, siglip_bottom, siglip_right = (
            siglip_region.unbind(dim=1)
        )
        siglip_region_valid = (
            siglip_top.ge(0)
            & siglip_left.ge(0)
            & siglip_bottom.gt(siglip_top)
            & siglip_right.gt(siglip_left)
            & siglip_bottom.le(siglip_size[:, 0])
            & siglip_right.le(siglip_size[:, 1])
        )
        if not bool(siglip_region_valid.all()):
            invalid = torch.nonzero(
                ~siglip_region_valid,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "siglip_valid_region must be a non-empty half-open rectangle "
                f"inside siglip_input_size; invalid batch indices {invalid}"
            )
        covered_siglip_size = metadata["siglip_patch_size"] * torch.tensor(
            [hs, ws],
            dtype=torch.float32,
            device=mask_prob.device,
        )
        uncovered = siglip_size - covered_siglip_size
        patch_grid_valid = (
            uncovered.ge(0).all(dim=1)
            & uncovered.lt(metadata["siglip_patch_size"]).all(dim=1)
        )
        if not bool(patch_grid_valid.all()):
            invalid = torch.nonzero(
                ~patch_grid_valid,
                as_tuple=False,
            ).flatten().tolist()
            raise ValueError(
                "SigLIP patch grid/patch size does not tile the declared "
                f"input with only a sub-patch remainder; invalid batch "
                f"indices {invalid}"
            )
        derived_source_valid_mask = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            mask_prob.shape[-2:],
        )
        if sam_source_valid_mask is None:
            sam_source_valid_mask = derived_source_valid_mask
        else:
            if sam_source_valid_mask.shape != derived_source_valid_mask.shape:
                raise ValueError(
                    "sam_source_valid_mask must match the Query-mask grid: "
                    f"{tuple(sam_source_valid_mask.shape)} != "
                    f"{tuple(derived_source_valid_mask.shape)}"
                )
            if sam_source_valid_mask.device != mask_prob.device:
                raise ValueError(
                    "sam_source_valid_mask and mask_prob must share one device"
                )
            if sam_source_valid_mask.dtype != torch.bool:
                raise TypeError("sam_source_valid_mask must have dtype torch.bool")
            if not bool(torch.equal(
                sam_source_valid_mask,
                derived_source_valid_mask,
            )):
                raise ValueError(
                    "sam_source_valid_mask disagrees with continuous SAM "
                    "spatial metadata"
                )

        patch_size = metadata["siglip_patch_size"]
        y_index = torch.arange(hs, device=mask_prob.device, dtype=torch.float32) + 0.5
        x_index = torch.arange(ws, device=mask_prob.device, dtype=torch.float32) + 0.5
        patch_y, patch_x = torch.meshgrid(y_index, x_index, indexing="ij")
        siglip_x = patch_x.unsqueeze(0) * patch_size[:, 1, None, None]
        siglip_y = patch_y.unsqueeze(0) * patch_size[:, 0, None, None]
        ones = torch.ones_like(siglip_x)
        siglip_points = torch.stack((siglip_x, siglip_y, ones), dim=-1)

        siglip_to_original = torch.linalg.inv(metadata["original_to_siglip"])
        original_points = torch.einsum(
            "bij,bhwj->bhwi",
            siglip_to_original,
            siglip_points,
        )
        original_points = original_points / original_points[..., 2:].clamp_min(1e-8)
        sam_points = torch.einsum(
            "bij,bhwj->bhwi",
            metadata["original_to_sam"],
            original_points,
        )
        sam_points = sam_points / sam_points[..., 2:].clamp_min(1e-8)

        sam_height = metadata["sam_input_size"][:, 0, None, None].clamp_min(1.0)
        sam_width = metadata["sam_input_size"][:, 1, None, None].clamp_min(1.0)
        grid_x = 2.0 * sam_points[..., 0] / sam_width - 1.0
        grid_y = 2.0 * sam_points[..., 1] / sam_height - 1.0
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)

        siglip_region = metadata["siglip_valid_region"]
        in_siglip = (
            siglip_x.ge(siglip_region[:, 1, None, None])
            & siglip_x.lt(siglip_region[:, 3, None, None])
            & siglip_y.ge(siglip_region[:, 0, None, None])
            & siglip_y.lt(siglip_region[:, 2, None, None])
        )
        original_size = metadata["original_size"]
        in_original = (
            original_points[..., 0].ge(0)
            & original_points[..., 0].lt(original_size[:, 1, None, None])
            & original_points[..., 1].ge(0)
            & original_points[..., 1].lt(original_size[:, 0, None, None])
        )
        sam_region = metadata["sam_valid_region"]
        in_sam = (
            sam_points[..., 0].ge(sam_region[:, 1, None, None])
            & sam_points[..., 0].lt(sam_region[:, 3, None, None])
            & sam_points[..., 1].ge(sam_region[:, 0, None, None])
            & sam_points[..., 1].lt(sam_region[:, 2, None, None])
        )
        valid_mask = (in_siglip & in_original & in_sam).unsqueeze(1)
        if supplied_valid_mask is not None:
            if supplied_valid_mask.shape != (batch_size, 1, hs, ws):
                raise ValueError(
                    "supplied SigLIP valid mask must be "
                    f"{(batch_size, 1, hs, ws)}, got {tuple(supplied_valid_mask.shape)}"
                )
            if supplied_valid_mask.device != mask_prob.device:
                raise ValueError(
                    "supplied SigLIP valid mask and mask_prob must share one "
                    "device"
                )
            if supplied_valid_mask.dtype != torch.bool:
                raise TypeError(
                    "supplied SigLIP valid mask must have dtype torch.bool"
                )
            valid_mask = valid_mask & supplied_valid_mask
        aligned, source_coverage = masked_normalized_grid_sample(
            mask_prob,
            sam_source_valid_mask,
            sampling_grid,
            target_valid_mask=valid_mask,
        )
        valid_mask = valid_mask & source_coverage.gt(1e-6)
        aligned = aligned * valid_mask.to(aligned.dtype)
        _require_finite("query_mask_on_siglip", aligned)
        return aligned, valid_mask


class DualRegionEvidencePooler(nn.Module):
    """Pool SAM and spatially aligned SigLIP evidence for every query.

    All normalized mask pooling reductions use float32.  Projection and
    fusion are differentiable, as are the soft-mask weights.  Empty weighted
    regions produce zero evidence.  Missing SigLIP features or alignment
    metadata is an error when SigLIP evidence is enabled.
    """

    def __init__(
        self,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        hidden_dim: int,
        num_stages: int,
        region_init_scale: float = 1e-2,
        use_sam_region_evidence: bool = True,
        use_siglip_region_evidence: bool = True,
        require_spatial_alignment: bool = True,
        mask_threshold: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_sam_region_evidence = bool(use_sam_region_evidence)
        self.use_siglip_region_evidence = bool(use_siglip_region_evidence)
        self.require_spatial_alignment = bool(require_spatial_alignment)
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError(
                f"mask_threshold must be in [0,1], got {mask_threshold}"
            )
        self.mask_threshold = float(mask_threshold)
        self.eps = float(eps)
        self.sam_region_projector = nn.Linear(sam_feature_dim, hidden_dim)
        self.siglip_region_projector = nn.Linear(siglip_feature_dim, hidden_dim)
        self.sam_norm = nn.LayerNorm(hidden_dim)
        self.siglip_norm = nn.LayerNorm(hidden_dim)
        self.region_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_evidence_norm = nn.LayerNorm(hidden_dim)
        self.alpha_region = nn.Parameter(
            torch.full((num_stages,), float(region_init_scale), dtype=torch.float32)
        )
        self.aligner = SpatialMaskAligner()
        self.apply(_init_linear)

    @staticmethod
    def _normalized_pool(mask_weights: Tensor, features: Tensor, eps: float) -> Tensor:
        weights = mask_weights.to(torch.float32)
        values = features.to(torch.float32)
        mass = weights.flatten(2).sum(dim=-1, keepdim=True)
        pooled = torch.einsum("bqhw,bdhw->bqd", weights, values)
        pooled = pooled / mass.clamp_min(eps)
        pooled = torch.where(mass.gt(eps), pooled, torch.zeros_like(pooled))
        return pooled

    def forward(
        self,
        query_states: Tensor,
        mask_logits: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Optional[Tensor],
        spatial_metadata: Optional[Dict[str, Tensor]],
        stage_index: int,
        siglip_valid_mask: Optional[Tensor] = None,
        sam_mask_valid_mask: Optional[Tensor] = None,
        sam_feature_valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if query_states.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError(
                "query_states and mask_logits must be [B,Q,D] and [B,Q,H,W], "
                f"got {tuple(query_states.shape)} and {tuple(mask_logits.shape)}"
            )
        if query_states.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("query state and mask batch/query dimensions do not match")
        if sam_mask_features.ndim != 4 or sam_mask_features.shape[0] != query_states.shape[0]:
            raise ValueError("sam_mask_features must be [B,Dm,Hf,Wf] with matching batch")
        if (
            query_states.device != mask_logits.device
            or sam_mask_features.device != mask_logits.device
        ):
            raise ValueError(
                "query_states, mask_logits, and sam_mask_features must share "
                "one device"
            )
        if not 0 <= stage_index < self.alpha_region.numel():
            raise IndexError(f"stage_index {stage_index} is out of range")

        batch_size, _, mask_height, mask_width = mask_logits.shape
        if sam_mask_valid_mask is None:
            sam_mask_valid_mask = torch.ones(
                batch_size,
                1,
                mask_height,
                mask_width,
                dtype=torch.bool,
                device=mask_logits.device,
            )
        expected_mask_valid_shape = (
            batch_size,
            1,
            mask_height,
            mask_width,
        )
        if sam_mask_valid_mask.shape != expected_mask_valid_shape:
            raise ValueError(
                "sam_mask_valid_mask must match the Query-mask grid: "
                f"{tuple(sam_mask_valid_mask.shape)} != "
                f"{expected_mask_valid_shape}"
            )
        expected_feature_valid_shape = (
            batch_size,
            1,
            sam_mask_features.shape[-2],
            sam_mask_features.shape[-1],
        )
        if sam_feature_valid_mask is None:
            sam_feature_valid_mask = torch.ones(
                expected_feature_valid_shape,
                dtype=torch.bool,
                device=mask_logits.device,
            )
        if sam_feature_valid_mask.shape != expected_feature_valid_shape:
            raise ValueError(
                "sam_feature_valid_mask must match the SAM feature grid: "
                f"{tuple(sam_feature_valid_mask.shape)} != "
                f"{expected_feature_valid_shape}"
            )
        for name, valid_mask in (
            ("sam_mask_valid_mask", sam_mask_valid_mask),
            ("sam_feature_valid_mask", sam_feature_valid_mask),
        ):
            if valid_mask.device != mask_logits.device:
                raise ValueError(f"{name} and mask_logits must share one device")
            if valid_mask.dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")

        mask_prob = mask_logits.sigmoid()
        # "Empty Query mask" follows the same detached binary-mask contract
        # as topology grouping.  A finite negative logit still has a tiny
        # positive sigmoid mass, so mass-only gating would otherwise pool an
        # almost uniform full-image feature for a topology-empty Query.
        binary_nonempty = (
            (
                mask_prob.detach().ge(self.mask_threshold)
                & sam_mask_valid_mask
            )
            .flatten(2)
            .any(dim=-1, keepdim=True)
        )
        sam_weights, _ = masked_normalized_resize(
            mask_prob,
            sam_mask_valid_mask,
            sam_mask_features.shape[-2:],
            target_valid_mask=sam_feature_valid_mask,
            eps=self.eps,
        )
        sam_raw = self._normalized_pool(sam_weights, sam_mask_features, self.eps)
        sam_has_evidence = (
            sam_weights.flatten(2).sum(dim=-1, keepdim=True).gt(self.eps)
            & binary_nonempty
        )
        if not self.use_sam_region_evidence:
            sam_raw = torch.zeros_like(sam_raw)
            sam_has_evidence = torch.zeros_like(sam_has_evidence)
        sam_dtype = self.sam_region_projector.weight.dtype
        vsam = self.sam_region_projector(sam_raw.to(sam_dtype))
        vsam = torch.where(
            sam_has_evidence,
            vsam,
            torch.zeros_like(vsam),
        )

        if siglip_spatial_features is None:
            if self.use_siglip_region_evidence:
                raise ValueError(
                    "SigLIP spatial features are required when topology grouping is enabled"
                )
            batch_size, num_queries = query_states.shape[:2]
            query_mask_on_siglip = mask_prob.new_zeros(batch_size, num_queries, 1, 1)
            resolved_valid_mask = torch.zeros(
                batch_size,
                1,
                1,
                1,
                dtype=torch.bool,
                device=mask_prob.device,
            )
            siglip_raw = sam_raw.new_zeros(batch_size, num_queries, self.siglip_region_projector.in_features)
            siglip_has_evidence = torch.zeros(
                batch_size,
                num_queries,
                1,
                dtype=torch.bool,
                device=query_states.device,
            )
        else:
            if siglip_spatial_features.ndim != 4:
                raise ValueError(
                    "siglip_spatial_features must be [B,Ds,Hs,Ws], "
                    f"got {tuple(siglip_spatial_features.shape)}"
                )
            if siglip_spatial_features.shape[0] != query_states.shape[0]:
                raise ValueError("SigLIP and query batches do not match after cond_lens expansion")
            if siglip_spatial_features.device != query_states.device:
                raise ValueError(
                    "SigLIP spatial features and Query tensors must share one "
                    "device"
                )
            if self.require_spatial_alignment:
                query_mask_on_siglip, resolved_valid_mask = self.aligner(
                    mask_prob,
                    spatial_metadata,
                    siglip_spatial_features.shape[-2:],
                    siglip_valid_mask,
                    sam_mask_valid_mask,
                )
            else:
                if siglip_valid_mask is None:
                    resolved_valid_mask = torch.ones(
                        query_states.shape[0],
                        1,
                        *siglip_spatial_features.shape[-2:],
                        dtype=torch.bool,
                        device=query_states.device,
                    )
                else:
                    expected_siglip_valid_shape = (
                        query_states.shape[0],
                        1,
                        *siglip_spatial_features.shape[-2:],
                    )
                    if siglip_valid_mask.shape != expected_siglip_valid_shape:
                        raise ValueError(
                            "siglip_valid_mask must match the SigLIP feature "
                            f"grid: {tuple(siglip_valid_mask.shape)} != "
                            f"{expected_siglip_valid_shape}"
                        )
                    if siglip_valid_mask.device != query_states.device:
                        raise ValueError(
                            "siglip_valid_mask and Query tensors must share "
                            "one device"
                        )
                    if siglip_valid_mask.dtype != torch.bool:
                        raise TypeError(
                            "siglip_valid_mask must have dtype torch.bool"
                        )
                    resolved_valid_mask = siglip_valid_mask
                query_mask_on_siglip, siglip_source_coverage = (
                    masked_normalized_resize(
                        mask_prob,
                        sam_mask_valid_mask,
                        siglip_spatial_features.shape[-2:],
                        target_valid_mask=resolved_valid_mask,
                        eps=self.eps,
                    )
                )
                resolved_valid_mask = (
                    resolved_valid_mask
                    & siglip_source_coverage.gt(self.eps)
                )
            siglip_weights = query_mask_on_siglip * resolved_valid_mask.to(
                query_mask_on_siglip.dtype
            )
            siglip_raw = self._normalized_pool(
                siglip_weights,
                siglip_spatial_features,
                self.eps,
            )
            siglip_has_evidence = (
                siglip_weights.flatten(2)
                .sum(dim=-1, keepdim=True)
                .gt(self.eps)
                & binary_nonempty
            )
        if not self.use_siglip_region_evidence:
            siglip_raw = torch.zeros_like(siglip_raw)
            siglip_has_evidence = torch.zeros_like(siglip_has_evidence)
        siglip_dtype = self.siglip_region_projector.weight.dtype
        vsig = self.siglip_region_projector(siglip_raw.to(siglip_dtype))
        vsig = torch.where(
            siglip_has_evidence,
            vsig,
            torch.zeros_like(vsig),
        )

        fusion_dtype = self.region_fusion[0].weight.dtype
        region = self.region_fusion(
            torch.cat((self.sam_norm(vsam), self.siglip_norm(vsig)), dim=-1).to(
                fusion_dtype
            )
        )
        region_has_evidence = sam_has_evidence | siglip_has_evidence
        region = torch.where(
            region_has_evidence,
            region,
            torch.zeros_like(region),
        )
        query_dtype = query_states.dtype
        scale = self.alpha_region[stage_index].to(device=query_states.device, dtype=query_dtype)
        member_evidence = self.query_evidence_norm(
            query_states + scale * region.to(query_dtype)
        )
        _require_finite("sam_region_evidence", vsam)
        _require_finite("siglip_region_evidence", vsig)
        _require_finite("member_evidence", member_evidence)
        return (
            vsam,
            vsig,
            member_evidence,
            query_mask_on_siglip,
            resolved_valid_mask,
        )


class TopologyGroupAggregator(nn.Module):
    """Aggregate member evidence within each dynamic connected component."""

    def __init__(self, hidden_dim: int, topology_bias_init: float = 1.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.member_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.lambda_topology = nn.Parameter(
            torch.tensor(float(topology_bias_init), dtype=torch.float32)
        )
        self.apply(_init_linear)

    def forward(
        self,
        member_evidence: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        topology_centrality: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if member_evidence.ndim != 3:
            raise ValueError("member_evidence must be [B,Q,D]")
        batch_size, num_queries, hidden_dim = member_evidence.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden_dim}")
        expected_shape = (batch_size, num_queries)
        for name, tensor in (
            ("query_to_group", query_to_group),
            ("group_valid_mask", group_valid_mask),
            ("topology_centrality", topology_centrality),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must be {expected_shape}, got {tuple(tensor.shape)}")

        membership = F.one_hot(query_to_group, num_classes=num_queries)
        membership = membership.transpose(1, 2).to(torch.bool)  # [B,root,member]
        membership_float = membership.to(member_evidence.dtype)
        group_sizes = membership_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        initial_group = torch.matmul(membership_float, member_evidence) / group_sizes

        group_query = self.query_projection(self.group_norm(initial_group))
        member_key = self.key_projection(self.member_norm(member_evidence))
        logits = torch.einsum("bgd,bqd->bgq", group_query, member_key)
        logits = logits / float(self.hidden_dim) ** 0.5
        topology_bias = self.lambda_topology.to(logits.dtype) * topology_centrality[:, None, :].to(
            logits.dtype
        )
        logits = logits + topology_bias
        logits = logits.masked_fill(~membership, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * membership_float
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        values = self.value_projection(member_evidence)
        context = torch.matmul(weights, values)
        group_features = self.output_norm(
            initial_group + self.output_projection(context)
        )
        group_features = group_features * group_valid_mask.unsqueeze(-1).to(
            group_features.dtype
        )

        gather_index = query_to_group[:, None, :]
        member_attention_weights = weights.gather(1, gather_index).squeeze(1)
        _require_finite("group_features", group_features)
        _require_finite("member_attention_weights", member_attention_weights)
        return group_features, member_attention_weights


class DynamicTopologyGroupAggregator(nn.Module):
    """V2 group bank update with membership-masked cross-attention.

    The query of each valid physical group row is either its state transported
    from the preceding stage or a shared learned seed.  Keys and values are
    the current stage's member evidence.  Unlike the legacy aggregator this
    module never initializes a group by averaging its members and does not
    inject a hand-crafted centrality bias into attention logits.
    """

    def __init__(self, hidden_dim: int, num_attention_heads: int = 8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        num_attention_heads = min(int(num_attention_heads), self.hidden_dim)
        while self.hidden_dim % num_attention_heads != 0:
            num_attention_heads -= 1
        self.num_attention_heads = num_attention_heads
        self.learned_group_seed = nn.Parameter(torch.empty(1, self.hidden_dim))
        self.prior_norm = nn.LayerNorm(self.hidden_dim)
        self.member_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_attention_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.learned_group_seed, mean=0.0, std=0.02)

    def forward(
        self,
        member_evidence: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        transported_prior: Optional[Tensor] = None,
        transport_has_predecessor: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if member_evidence.ndim != 3:
            raise ValueError("member_evidence must be [B,Q,D]")
        batch_size, num_queries, hidden_dim = member_evidence.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"expected hidden dim {self.hidden_dim}, got {hidden_dim}"
            )
        expected_shape = (batch_size, num_queries)
        if query_to_group.shape != expected_shape:
            raise ValueError("query_to_group must be [B,Q]")
        if group_valid_mask.shape != expected_shape:
            raise ValueError("group_valid_mask must be [B,Q]")
        if transported_prior is not None and transported_prior.shape != (
            batch_size,
            num_queries,
            hidden_dim,
        ):
            raise ValueError("transported_prior must be [B,Q,D]")
        if transport_has_predecessor is not None and (
            transport_has_predecessor.shape != expected_shape
            or transport_has_predecessor.dtype != torch.bool
        ):
            raise ValueError(
                "transport_has_predecessor must be a boolean [B,Q] tensor"
            )

        membership = F.one_hot(
            query_to_group,
            num_classes=num_queries,
        ).transpose(1, 2).to(torch.bool)
        seed = self.learned_group_seed.to(
            device=member_evidence.device,
            dtype=member_evidence.dtype,
        )
        group_rows = []
        member_weight_rows = []
        prior_rows = []
        query_ids = torch.arange(num_queries, device=member_evidence.device)
        for batch_index in range(batch_size):
            roots = group_valid_mask[batch_index].nonzero(
                as_tuple=False
            ).flatten()
            if roots.numel() == 0:
                raise RuntimeError("every sample must contain at least one group")

            prior = seed.expand(num_queries, -1)
            if transported_prior is not None:
                if transport_has_predecessor is None:
                    has_predecessor = group_valid_mask[batch_index]
                else:
                    has_predecessor = transport_has_predecessor[batch_index]
                prior = torch.where(
                    has_predecessor[:, None],
                    transported_prior[batch_index],
                    prior,
                )
            prior = prior * group_valid_mask[batch_index, :, None].to(
                prior.dtype
            )

            group_queries = self.prior_norm(prior[roots]).unsqueeze(0)
            member_keys = self.member_norm(
                member_evidence[batch_index]
            ).unsqueeze(0)
            disallowed = ~membership[batch_index, roots]
            context, attention_weights = self.cross_attention(
                group_queries,
                member_keys,
                member_keys,
                attn_mask=disallowed,
                need_weights=True,
                average_attn_weights=True,
            )
            updated = self.output_norm(
                prior[roots].unsqueeze(0) + context
            ).squeeze(0)
            physical_groups = member_evidence.new_zeros(
                num_queries,
                hidden_dim,
            ).index_copy(0, roots, updated)
            physical_attention = member_evidence.new_zeros(
                num_queries,
                num_queries,
            ).index_copy(0, roots, attention_weights.squeeze(0))
            per_member_attention = physical_attention[
                query_to_group[batch_index],
                query_ids,
            ]
            group_rows.append(physical_groups)
            member_weight_rows.append(per_member_attention)
            prior_rows.append(prior)

        group_features = torch.stack(group_rows, dim=0)
        member_attention_weights = torch.stack(member_weight_rows, dim=0)
        resolved_prior = torch.stack(prior_rows, dim=0)
        _require_finite("v2_group_features", group_features)
        _require_finite(
            "v2_member_attention_weights",
            member_attention_weights,
        )
        _require_finite("v2_transported_group_prior", resolved_prior)
        return group_features, member_attention_weights, resolved_prior


class GroupToQueryFeedback(nn.Module):
    """Apply member-specific, condition-free group feedback to raw queries."""

    def __init__(
        self,
        hidden_dim: int,
        num_feedback_stages: int,
        feedback_init_scale: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.feedback_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feedback_gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.alpha_feedback = nn.Parameter(
            torch.full(
                (num_feedback_stages,),
                float(feedback_init_scale),
                dtype=torch.float32,
            )
        )
        self.apply(_init_linear)

    def forward(
        self,
        raw_query_states: Tensor,
        group_features: Tensor,
        query_to_group: Tensor,
        stage_index: int,
    ) -> Tuple[Tensor, Tensor]:
        if raw_query_states.ndim != 3:
            raise ValueError("raw_query_states must be [B,Q,D]")
        if group_features.shape != raw_query_states.shape:
            raise ValueError("group_features and raw_query_states must have identical shape")
        if query_to_group.shape != raw_query_states.shape[:2]:
            raise ValueError("query_to_group must be [B,Q]")
        if not 0 <= stage_index < self.alpha_feedback.numel():
            raise IndexError(
                f"feedback stage {stage_index} outside [0,{self.alpha_feedback.numel()})"
            )

        gather_index = query_to_group.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        gathered_group = group_features.gather(1, gather_index)
        query_norm = self.query_norm(raw_query_states)
        group_norm = self.group_norm(gathered_group)
        interaction = query_norm * group_norm
        difference = query_norm - group_norm
        delta = self.feedback_mlp(
            torch.cat((query_norm, group_norm, interaction, difference), dim=-1)
        )
        gate = torch.sigmoid(
            self.feedback_gate(torch.cat((query_norm, group_norm, interaction), dim=-1))
        )
        scale = self.alpha_feedback[stage_index].to(
            device=raw_query_states.device,
            dtype=raw_query_states.dtype,
        )
        scaled_delta = scale * gate * delta.to(raw_query_states.dtype)
        updated = raw_query_states + scaled_delta
        _require_finite("feedback_delta", scaled_delta)
        _require_finite("feedback_query_states", updated)
        return updated, scaled_delta


class InnovationGroupToQueryFeedback(nn.Module):
    """V2 member-gated feedback driven only by new group information."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.innovation_norm = nn.LayerNorm(hidden_dim)
        self.candidate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.feedback_gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_init_linear)
        # V2 starts as an exact no-op.  There is deliberately no independent
        # alpha_feedback parameter in addition to the member-specific gate.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        raw_query_states: Tensor,
        group_innovation: Tensor,
        query_to_group: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if raw_query_states.ndim != 3:
            raise ValueError("raw_query_states must be [B,Q,D]")
        if group_innovation.shape != raw_query_states.shape:
            raise ValueError(
                "group_innovation and raw_query_states must have identical "
                "physical [B,Q,D] shapes"
            )
        if query_to_group.shape != raw_query_states.shape[:2]:
            raise ValueError("query_to_group must be [B,Q]")
        gather_index = query_to_group.unsqueeze(-1).expand(
            -1,
            -1,
            self.hidden_dim,
        )
        member_innovation = group_innovation.gather(1, gather_index)
        query_norm = self.query_norm(raw_query_states)
        innovation_norm = self.innovation_norm(member_innovation)
        interaction = query_norm * innovation_norm
        joint = torch.cat(
            (query_norm, innovation_norm, interaction),
            dim=-1,
        )
        gate = torch.sigmoid(self.feedback_gate(joint))
        candidate = self.candidate(joint)
        delta = gate * self.output_projection(candidate).to(
            raw_query_states.dtype
        )
        updated = raw_query_states + delta
        _require_finite("v2_feedback_delta", delta)
        _require_finite("v2_feedback_query_states", updated)
        return updated, delta


class WithinGroupQualityHead(nn.Module):
    """Predict final-stage member quality without using GT at inference."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sam_norm = nn.LayerNorm(hidden_dim)
        self.siglip_norm = nn.LayerNorm(hidden_dim)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.quality_head = nn.Sequential(
            nn.Linear(5 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_linear)

    def forward(
        self,
        query_states: Tensor,
        sam_region_evidence: Tensor,
        siglip_region_evidence: Tensor,
        group_features: Tensor,
        query_to_group: Tensor,
        topology_centrality: Tensor,
    ) -> Tensor:
        if query_states.ndim != 3:
            raise ValueError("query_states must be [B,Q,D]")
        gather_index = query_to_group.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        gathered_group = group_features.gather(1, gather_index)
        q_norm = self.query_norm(query_states)
        sam_norm = self.sam_norm(sam_region_evidence)
        siglip_norm = self.siglip_norm(siglip_region_evidence)
        group_norm = self.group_norm(gathered_group)
        interaction = q_norm * group_norm
        quality_input = torch.cat(
            (
                q_norm,
                sam_norm,
                siglip_norm,
                group_norm,
                interaction,
                topology_centrality.unsqueeze(-1).to(q_norm.dtype),
            ),
            dim=-1,
        )
        logits = self.quality_head(quality_input).squeeze(-1)
        _require_finite("member_quality_logits", logits)
        return logits


class DisagreementAwareIntraGroupTournament(nn.Module):
    """Rank final-stage members from their pairwise exclusive mask regions.

    For members ``i`` and ``j`` of one topology group, the two soft exclusive
    regions are ``p_i * (1 - p_j)`` and ``p_j * (1 - p_i)``.  A shared support
    network scores both directions from spatially aligned SAM and SigLIP
    evidence plus the group prototype.  Subtraction enforces
    ``a_ij = -a_ji`` exactly; a Borda mean then produces the compatible
    per-query ``member_quality_logits`` used by the formal adapter.

    Exhaustive comparisons are always used in evaluation.  During training,
    a large group uses a deterministic stratified circulant comparison graph:
    every member has at most ``train_max_opponents_per_member`` opponents and
    every undirected edge is evaluated once.  This preserves antisymmetry and
    the Borda mean while bounding saved autograd tensors by ``O(mK)`` instead
    of ``O(m^2)`` for group size ``m`` and configured degree ``K``.  A value
    of zero explicitly restores exhaustive training comparisons.
    """

    def __init__(
        self,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        hidden_dim: int,
        pair_chunk_size: int = 64,
        train_max_opponents_per_member: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        if int(pair_chunk_size) <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if int(train_max_opponents_per_member) < 0:
            raise ValueError(
                "train_max_opponents_per_member must be non-negative"
            )
        if (
            int(train_max_opponents_per_member) > 0
            and int(train_max_opponents_per_member) % 2 != 0
        ):
            raise ValueError(
                "train_max_opponents_per_member must be even; use 0 only "
                "for exhaustive training"
            )
        self.hidden_dim = int(hidden_dim)
        self.pair_chunk_size = int(pair_chunk_size)
        self.train_max_opponents_per_member = int(
            train_max_opponents_per_member
        )
        self.eps = float(eps)
        self.sam_projection = nn.Linear(sam_feature_dim, hidden_dim)
        self.siglip_projection = nn.Linear(siglip_feature_dim, hidden_dim)
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.support_network = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(_init_linear)

    def _local_pair_indices(
        self,
        member_count: int,
        device: torch.device,
    ) -> Tensor:
        """Return deterministic undirected local-member comparison edges.

        Evaluation, an explicit zero training cap, and groups no larger than
        the cap all use the complete upper triangle.  Otherwise, K/2 offsets
        are deterministically stratified over the non-antipodal half-ring.
        Each offset contributes one circulant edge per member, so every
        member has exactly K distinct opponents and the graph has ``mK/2``
        undirected edges.
        """

        member_count = int(member_count)
        if member_count < 2:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        max_opponents = self.train_max_opponents_per_member
        exhaustive = (
            not self.training
            or max_opponents == 0
            or member_count - 1 <= max_opponents
        )
        if exhaustive:
            return torch.triu_indices(
                member_count,
                member_count,
                offset=1,
                device=device,
            )

        half_degree = max_opponents // 2
        max_offset = (member_count - 1) // 2
        offset_slots = torch.arange(
            half_degree,
            dtype=torch.long,
            device=device,
        )
        offsets = torch.div(
            (2 * offset_slots + 1) * max_offset,
            2 * half_degree,
            rounding_mode="floor",
        ) + 1
        local_i = torch.arange(
            member_count,
            dtype=torch.long,
            device=device,
        ).repeat_interleave(half_degree)
        local_j = (
            local_i + offsets.repeat(member_count)
        ).remainder(member_count)
        return torch.stack((local_i, local_j), dim=0)

    @staticmethod
    def _weighted_pool(weights: Tensor, features: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
        weights_fp32 = weights.to(torch.float32)
        features_fp32 = features.to(torch.float32)
        mass = weights_fp32.flatten(1).sum(dim=-1, keepdim=True)
        pooled = torch.einsum(
            "nhw,dhw->nd",
            weights_fp32,
            features_fp32,
        )
        pooled = pooled / mass.clamp_min(eps)
        pooled = torch.where(
            mass.gt(eps),
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled, mass

    def _direction_support(
        self,
        sam_exclusive: Tensor,
        siglip_exclusive: Tensor,
        sam_features: Tensor,
        siglip_features: Tensor,
        group_feature: Tensor,
        sam_valid_pixels: Tensor,
        siglip_valid_pixels: Tensor,
    ) -> Tensor:
        sam_pooled, sam_mass = self._weighted_pool(
            sam_exclusive,
            sam_features,
            self.eps,
        )
        siglip_pooled, siglip_mass = self._weighted_pool(
            siglip_exclusive,
            siglip_features,
            self.eps,
        )
        module_dtype = self.sam_projection.weight.dtype
        sam_evidence = self.sam_projection(sam_pooled.to(module_dtype))
        siglip_evidence = self.siglip_projection(
            siglip_pooled.to(module_dtype)
        )
        count = sam_exclusive.shape[0]
        group_evidence = self.group_norm(
            group_feature.to(module_dtype)
        ).unsqueeze(0).expand(count, -1)
        sam_fraction = (
            sam_mass / sam_valid_pixels.clamp_min(1.0)
        ).to(module_dtype)
        siglip_fraction = (
            siglip_mass / siglip_valid_pixels.clamp_min(1.0)
        ).to(module_dtype)
        support_input = torch.cat(
            (
                sam_evidence,
                siglip_evidence,
                group_evidence,
                sam_fraction,
                siglip_fraction,
            ),
            dim=-1,
        )
        return self.support_network(support_input).squeeze(-1)

    def _score_sample(
        self,
        sam_mask_prob: Tensor,
        siglip_mask_prob: Tensor,
        sam_features: Tensor,
        siglip_features: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        group_features: Tensor,
        sam_valid_mask: Tensor,
        siglip_valid_mask: Tensor,
        return_pairwise: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        num_queries = sam_mask_prob.shape[0]
        sample_scores = sam_mask_prob.new_zeros(num_queries)
        pairwise = (
            sam_mask_prob.new_zeros(num_queries, num_queries)
            if return_pairwise
            else None
        )
        sam_valid = sam_valid_mask.to(sam_mask_prob.dtype)
        siglip_valid = siglip_valid_mask.to(siglip_mask_prob.dtype)
        sam_valid_pixels = sam_valid.sum().reshape(1, 1).to(torch.float32)
        siglip_valid_pixels = (
            siglip_valid.sum().reshape(1, 1).to(torch.float32)
        )
        # Convert once per sample.  Converting inside every pair direction
        # would retain redundant cast nodes in the training autograd graph.
        sam_features = sam_features.to(torch.float32)
        siglip_features = siglip_features.to(torch.float32)
        for root in group_valid_mask.nonzero(as_tuple=False).flatten():
            members = query_to_group.eq(root).nonzero(
                as_tuple=False
            ).flatten()
            member_count = int(members.numel())
            if member_count == 1:
                continue
            local_pairs = self._local_pair_indices(
                member_count,
                members.device,
            )
            group_score = sam_mask_prob.new_zeros(member_count)
            comparison_count = sam_mask_prob.new_zeros(member_count)
            for start in range(0, local_pairs.shape[1], self.pair_chunk_size):
                pair_slice = local_pairs[
                    :,
                    start : start + self.pair_chunk_size,
                ]
                local_i, local_j = pair_slice[0], pair_slice[1]
                query_i = members[local_i]
                query_j = members[local_j]
                sam_i = sam_mask_prob[query_i]
                sam_j = sam_mask_prob[query_j]
                siglip_i = siglip_mask_prob[query_i]
                siglip_j = siglip_mask_prob[query_j]
                sam_i_only = sam_i * (1.0 - sam_j) * sam_valid
                sam_j_only = sam_j * (1.0 - sam_i) * sam_valid
                siglip_i_only = (
                    siglip_i * (1.0 - siglip_j) * siglip_valid
                )
                siglip_j_only = (
                    siglip_j * (1.0 - siglip_i) * siglip_valid
                )
                support_i = self._direction_support(
                    sam_i_only,
                    siglip_i_only,
                    sam_features,
                    siglip_features,
                    group_features[root],
                    sam_valid_pixels,
                    siglip_valid_pixels,
                )
                support_j = self._direction_support(
                    sam_j_only,
                    siglip_j_only,
                    sam_features,
                    siglip_features,
                    group_features[root],
                    sam_valid_pixels,
                    siglip_valid_pixels,
                )
                preference = (support_i - support_j).to(
                    group_score.dtype
                )
                group_score = group_score.index_add(
                    0,
                    local_i,
                    preference,
                )
                group_score = group_score.index_add(
                    0,
                    local_j,
                    -preference,
                )
                comparison_count = comparison_count.index_add(
                    0,
                    local_i,
                    torch.ones_like(preference),
                )
                comparison_count = comparison_count.index_add(
                    0,
                    local_j,
                    torch.ones_like(preference),
                )
                if pairwise is not None:
                    pairwise = pairwise.index_put(
                        (query_i, query_j),
                        preference,
                    )
                    pairwise = pairwise.index_put(
                        (query_j, query_i),
                        -preference,
                    )
            if not bool(comparison_count.gt(0).all()):
                raise RuntimeError(
                    "bounded tournament left a non-singleton member without "
                    "an opponent"
                )
            group_score = group_score / comparison_count
            sample_scores = sample_scores.scatter(
                0,
                members,
                group_score,
            )
        return sample_scores, pairwise

    def _forward_impl(
        self,
        mask_logits: Tensor,
        query_mask_on_siglip: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        group_features: Tensor,
        sam_mask_valid_mask: Tensor,
        sam_feature_valid_mask: Tensor,
        siglip_valid_mask: Tensor,
        return_pairwise: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask_logits.ndim != 4:
            raise ValueError("mask_logits must be [B,Q,H,W]")
        batch_size, num_queries = mask_logits.shape[:2]
        if query_mask_on_siglip.shape[:2] != (batch_size, num_queries):
            raise ValueError(
                "query_mask_on_siglip must share mask batch/query axes"
            )
        if siglip_spatial_features.shape[-2:] != query_mask_on_siglip.shape[-2:]:
            raise ValueError(
                "SigLIP masks and spatial features must already share a grid"
            )
        sam_mask_prob, _ = masked_normalized_resize(
            mask_logits.sigmoid(),
            sam_mask_valid_mask,
            sam_mask_features.shape[-2:],
            target_valid_mask=sam_feature_valid_mask,
            eps=self.eps,
        )
        all_scores = []
        all_pairwise = []
        for batch_index in range(batch_size):
            scores, pairwise = self._score_sample(
                sam_mask_prob=sam_mask_prob[batch_index],
                siglip_mask_prob=query_mask_on_siglip[batch_index],
                sam_features=sam_mask_features[batch_index],
                siglip_features=siglip_spatial_features[batch_index],
                query_to_group=query_to_group[batch_index],
                group_valid_mask=group_valid_mask[batch_index],
                group_features=group_features[batch_index],
                sam_valid_mask=sam_feature_valid_mask[batch_index, 0],
                siglip_valid_mask=siglip_valid_mask[batch_index, 0],
                return_pairwise=return_pairwise,
            )
            all_scores.append(scores)
            if pairwise is not None:
                all_pairwise.append(pairwise)
        member_quality_logits = torch.stack(all_scores, dim=0)
        pairwise_preferences = (
            torch.stack(all_pairwise, dim=0)
            if return_pairwise
            else None
        )
        _require_finite(
            "v2_member_quality_logits",
            member_quality_logits,
        )
        if pairwise_preferences is not None:
            _require_finite(
                "v2_pairwise_preferences",
                pairwise_preferences,
            )
        return member_quality_logits, pairwise_preferences

    def forward(
        self,
        mask_logits: Tensor,
        query_mask_on_siglip: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        group_features: Tensor,
        sam_mask_valid_mask: Tensor,
        sam_feature_valid_mask: Tensor,
        siglip_valid_mask: Tensor,
    ) -> Tensor:
        scores, _ = self._forward_impl(
            mask_logits=mask_logits,
            query_mask_on_siglip=query_mask_on_siglip,
            sam_mask_features=sam_mask_features,
            siglip_spatial_features=siglip_spatial_features,
            query_to_group=query_to_group,
            group_valid_mask=group_valid_mask,
            group_features=group_features,
            sam_mask_valid_mask=sam_mask_valid_mask,
            sam_feature_valid_mask=sam_feature_valid_mask,
            siglip_valid_mask=siglip_valid_mask,
            return_pairwise=False,
        )
        return scores

    def forward_with_pairwise(self, **kwargs) -> Tuple[Tensor, Tensor]:
        """Diagnostic/test path returning the antisymmetric preference matrix."""

        scores, pairwise = self._forward_impl(
            return_pairwise=True,
            **kwargs,
        )
        if pairwise is None:
            raise RuntimeError("pairwise preferences were not produced")
        return scores, pairwise


class GroupConditionClassifier(nn.Module):
    """Classify every valid group independently along the condition axis.

    Open-vocabulary classification reuses the caller-provided ``logit_scale``
    and condition embeddings.  Closed-set classification reuses the original
    ``class_predictor`` module.  Invalid condition columns are filled with a
    large negative finite value; invalid physical group rows are ignored by
    the criterion and adapter.
    """

    def __init__(self, *, float32_open_vocab: bool = False) -> None:
        super().__init__()
        self.float32_open_vocab = bool(float32_open_vocab)

    def forward(
        self,
        group_features: Tensor,
        group_valid_mask: Tensor,
        cond_embeddings: Optional[Tensor] = None,
        embed_masks: Optional[Tensor] = None,
        logit_scale: Optional[Tensor] = None,
        class_predictor: Optional[nn.Module] = None,
    ) -> Tensor:
        if cond_embeddings is None:
            if class_predictor is None:
                raise ValueError("closed-set group classification requires class_predictor")
            logits = class_predictor(group_features)
        else:
            if logit_scale is None:
                raise ValueError("open-vocabulary group classification requires logit_scale")
            if cond_embeddings.ndim != 3 or cond_embeddings.shape[0] != group_features.shape[0]:
                raise ValueError("cond_embeddings must be [B,Nc,D] with matching batch")
            _require_finite(
                "group_classifier_group_features",
                group_features,
            )
            _require_finite(
                "group_classifier_cond_embeddings",
                cond_embeddings,
            )
            _require_finite("group_classifier_logit_scale", logit_scale)
            if self.float32_open_vocab:
                # V2 keeps this precision-sensitive reduction in FP32.  In
                # particular, FP16 cannot represent F.normalize's default
                # epsilon for a zero-padded Condition vector.  V1 deliberately
                # retains its historical same-dtype computation.
                group_norm = F.normalize(
                    group_features.to(torch.float32),
                    dim=-1,
                )
                cond_norm = F.normalize(
                    cond_embeddings.to(torch.float32),
                    dim=-1,
                )
                scale = logit_scale.to(torch.float32).exp()
            else:
                group_norm = F.normalize(group_features, dim=-1)
                cond_norm = F.normalize(cond_embeddings, dim=-1)
                scale = logit_scale.exp()
            _require_finite("group_classifier_exp_logit_scale", scale)
            logits = scale * torch.einsum(
                "bqd,bcd->bqc",
                group_norm,
                cond_norm,
            )
            logits = logits.clamp(min=-500.0, max=500.0)
            if self.float32_open_vocab:
                logits = logits.to(group_features.dtype)
            if embed_masks is not None:
                if embed_masks.shape != logits.shape[:1] + logits.shape[2:]:
                    raise ValueError(
                        f"embed_masks must be [B,Nc], got {tuple(embed_masks.shape)}"
                    )
                invalid_logit = -_finite_logit_bound(logits)
                logits = logits.masked_fill(
                    ~embed_masks[:, None, :].to(torch.bool),
                    invalid_logit,
                )
        if group_valid_mask.shape != group_features.shape[:2]:
            raise ValueError("group_valid_mask must be [B,Q]")
        _require_finite("group_class_logits", logits)
        return logits


class GroupPredictionAdapter(nn.Module):
    """Select one original query mask per final-stage group.

    Formal logits and masks retain the physical length ``Q`` for compatibility
    with existing post-processors.  Root rows contain the group logits and the
    exact gathered original query mask.  Non-root rows are made background-only
    and receive a strongly negative mask logit.
    """

    def __init__(self, invalid_mask_logit: float = -20.0):
        super().__init__()
        self.invalid_mask_logit = float(invalid_mask_logit)

    @torch.no_grad()
    def select_representatives(
        self,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        member_quality_logits: Tensor,
    ) -> Tensor:
        if query_to_group.shape != group_valid_mask.shape:
            raise ValueError("query_to_group and group_valid_mask shapes differ")
        if member_quality_logits.shape != query_to_group.shape:
            raise ValueError("member_quality_logits must be [B,Q]")
        batch_size, num_queries = query_to_group.shape
        max_quality = torch.full(
            (batch_size, num_queries),
            torch.finfo(member_quality_logits.dtype).min,
            dtype=member_quality_logits.dtype,
            device=member_quality_logits.device,
        )
        max_quality.scatter_reduce_(
            1,
            query_to_group,
            member_quality_logits,
            reduce="amax",
            include_self=True,
        )
        member_max = max_quality.gather(1, query_to_group)
        query_ids = torch.arange(num_queries, device=query_to_group.device)
        query_ids = query_ids.unsqueeze(0).expand(batch_size, -1)
        candidates = torch.where(
            member_quality_logits.eq(member_max),
            query_ids,
            torch.full_like(query_ids, num_queries),
        )
        representatives = torch.full(
            (batch_size, num_queries),
            num_queries,
            dtype=torch.long,
            device=query_to_group.device,
        )
        representatives.scatter_reduce_(
            1,
            query_to_group,
            candidates,
            reduce="amin",
            include_self=True,
        )
        representatives = torch.where(
            group_valid_mask,
            representatives,
            torch.full_like(representatives, -1),
        )
        if not bool(
            representatives[group_valid_mask].ge(0).all()
            and representatives[group_valid_mask].lt(num_queries).all()
        ):
            raise RuntimeError("failed to select one representative for every valid group")
        return representatives

    def forward(
        self,
        group_class_logits: Tensor,
        final_query_masks: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
        member_quality_logits: Tensor,
        embed_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        representatives = self.select_representatives(
            query_to_group,
            group_valid_mask,
            member_quality_logits,
        )
        batch_size, num_queries, height, width = final_query_masks.shape
        gather_ids = representatives.clamp_min(0)
        gathered_masks = final_query_masks.gather(
            1,
            gather_ids[:, :, None, None].expand(-1, -1, height, width),
        )
        formal_masks = torch.where(
            group_valid_mask[:, :, None, None],
            gathered_masks,
            torch.full_like(gathered_masks, self.invalid_mask_logit),
        )

        formal_logits = group_class_logits.clone()
        if embed_masks is None:
            valid_conditions = torch.ones(
                formal_logits.shape[0],
                formal_logits.shape[-1],
                dtype=torch.bool,
                device=formal_logits.device,
            )
        else:
            if embed_masks.shape != (
                formal_logits.shape[0],
                formal_logits.shape[-1],
            ):
                raise ValueError(
                    "embed_masks must be [B,Nc] and match group logits, got "
                    f"{tuple(embed_masks.shape)} for "
                    f"{tuple(formal_logits.shape)}"
                )
            valid_conditions = embed_masks.to(torch.bool)
        if formal_logits.shape[-1] <= 0:
            raise ValueError("group logits must contain a background condition")
        if not bool(valid_conditions[:, -1].all()):
            raise ValueError(
                "the final condition column must be valid background for every sample"
            )
        background_ids = torch.full(
            (formal_logits.shape[0],),
            formal_logits.shape[-1] - 1,
            dtype=torch.long,
            device=formal_logits.device,
        )
        invalid_rows = ~group_valid_mask
        logit_bound = _finite_logit_bound(formal_logits)
        formal_logits = formal_logits.masked_fill(
            invalid_rows.unsqueeze(-1),
            -logit_bound,
        )
        formal_logits.scatter_(
            2,
            background_ids[:, None, None].expand(-1, num_queries, 1),
            torch.where(
                invalid_rows[:, :, None],
                torch.full(
                    (batch_size, num_queries, 1),
                    logit_bound,
                    dtype=formal_logits.dtype,
                    device=formal_logits.device,
                ),
                formal_logits.gather(
                    2,
                    background_ids[:, None, None].expand(-1, num_queries, 1),
                ),
            ),
        )
        _require_finite("formal_group_class_logits", formal_logits)
        _require_finite("formal_group_masks", formal_masks)
        return formal_logits, formal_masks, representatives


class TopologyGroupDecoderCore(nn.Module):
    """Shared structural modules used by all ten prediction stages."""

    def __init__(
        self,
        hidden_dim: int,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        num_stages: int,
        num_queries: int,
        group_mask_threshold: float = 0.5,
        group_iou_threshold: float = 0.7,
        use_sam_region_evidence: bool = True,
        use_siglip_region_evidence: bool = True,
        require_spatial_alignment: bool = True,
        region_init_scale: float = 1e-2,
        feedback_init_scale: float = 1e-3,
        topology_bias_init: float = 1.0,
        require_expected_shapes: bool = True,
        topology_group_version: int = 1,
        tournament_pair_chunk_size: int = 64,
        tournament_train_max_opponents_per_member: int = 4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_stages = int(num_stages)
        self.require_expected_shapes = bool(require_expected_shapes)
        self.topology_group_version = int(topology_group_version)
        if self.topology_group_version not in (1, 2):
            raise ValueError(
                "topology_group_version must be either 1 or 2, got "
                f"{topology_group_version}"
            )
        if self.num_stages < 2:
            raise ValueError("topology decoder needs at least two prediction stages")
        self.grouper = MaskTopologyGrouper(
            mask_threshold=group_mask_threshold,
            iou_threshold=group_iou_threshold,
        )
        self.region_pooler = DualRegionEvidencePooler(
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_feature_dim,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            region_init_scale=region_init_scale,
            use_sam_region_evidence=use_sam_region_evidence,
            use_siglip_region_evidence=use_siglip_region_evidence,
            require_spatial_alignment=require_spatial_alignment,
            mask_threshold=group_mask_threshold,
        )
        if self.topology_group_version == 1:
            self.group_aggregator = TopologyGroupAggregator(
                hidden_dim=hidden_dim,
                topology_bias_init=topology_bias_init,
            )
            self.feedback = GroupToQueryFeedback(
                hidden_dim=hidden_dim,
                num_feedback_stages=num_stages - 1,
                feedback_init_scale=feedback_init_scale,
            )
            self.quality_head = WithinGroupQualityHead(
                hidden_dim=hidden_dim
            )
        else:
            self.group_aggregator = DynamicTopologyGroupAggregator(
                hidden_dim=hidden_dim,
            )
            self.feedback = InnovationGroupToQueryFeedback(
                hidden_dim=hidden_dim,
            )
            self.quality_head = DisagreementAwareIntraGroupTournament(
                sam_feature_dim=sam_feature_dim,
                siglip_feature_dim=siglip_feature_dim,
                hidden_dim=hidden_dim,
                pair_chunk_size=tournament_pair_chunk_size,
                train_max_opponents_per_member=(
                    tournament_train_max_opponents_per_member
                ),
            )

    @staticmethod
    def build_group_transport(
        current_query_to_group: Tensor,
        previous_query_to_group: Tensor,
        current_group_valid_mask: Tensor,
        previous_group_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Construct ``row_norm(P_cur^T P_prev)`` from persistent Query IDs."""

        if current_query_to_group.shape != previous_query_to_group.shape:
            raise ValueError(
                "current and previous query_to_group tensors must match"
            )
        batch_size, num_queries = current_query_to_group.shape
        expected_shape = (batch_size, num_queries)
        for name, tensor in (
            ("current_group_valid_mask", current_group_valid_mask),
            ("previous_group_valid_mask", previous_group_valid_mask),
        ):
            if tensor.shape != expected_shape or tensor.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean [B,Q] tensor")
        current_membership = F.one_hot(
            current_query_to_group,
            num_classes=num_queries,
        ).to(torch.float32)
        previous_membership = F.one_hot(
            previous_query_to_group,
            num_classes=num_queries,
        ).to(torch.float32)
        overlap = torch.matmul(
            current_membership.transpose(1, 2),
            previous_membership,
        )
        valid_pairs = (
            current_group_valid_mask[:, :, None]
            & previous_group_valid_mask[:, None, :]
        )
        overlap = overlap * valid_pairs.to(overlap.dtype)
        row_mass = overlap.sum(dim=-1, keepdim=True)
        transport = torch.where(
            row_mass.gt(0),
            overlap / row_mass.clamp_min(1.0),
            torch.zeros_like(overlap),
        )
        has_predecessor = (
            row_mass.squeeze(-1).gt(0) & current_group_valid_mask
        )
        _require_finite("group_transport", transport)
        return transport.detach(), has_predecessor.detach()

    def _validate_runtime_shapes(self, query_states: Tensor, stage_index: int) -> None:
        if query_states.ndim != 3:
            raise ValueError("query_states must be [B,Q,D]")
        if query_states.shape[1] != self.num_queries:
            raise ValueError(
                f"runtime query count {query_states.shape[1]} != configured {self.num_queries}"
            )
        if query_states.shape[2] != self.hidden_dim:
            raise ValueError(
                f"runtime hidden dim {query_states.shape[2]} != configured {self.hidden_dim}"
            )
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(f"stage {stage_index} outside [0,{self.num_stages})")
        if self.require_expected_shapes:
            if self.num_queries != 200 or self.hidden_dim != 256 or self.num_stages != 10:
                raise AssertionError(
                    "MaskLAT topology mode expects runtime Q=200, D=256, T=10; "
                    f"got Q={self.num_queries}, D={self.hidden_dim}, T={self.num_stages}"
                )

    def build_stage(
        self,
        query_states: Tensor,
        mask_logits: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        stage_index: int,
        siglip_valid_mask: Optional[Tensor] = None,
        previous_stage_output: Optional[TopologyGroupStageOutput] = None,
    ) -> TopologyGroupStageOutput:
        self._validate_runtime_shapes(query_states, stage_index)
        if spatial_metadata is None:
            raise ValueError(
                "topology stage construction requires SAM spatial metadata"
            )
        missing_sam_metadata = [
            key
            for key in ("sam_input_size", "sam_valid_region")
            if key not in spatial_metadata
        ]
        if missing_sam_metadata:
            raise ValueError(
                "spatial_metadata is missing SAM validity fields: "
                f"{missing_sam_metadata}"
            )
        sam_valid_boxes = normalize_sam_valid_regions(
            spatial_metadata["sam_input_size"],
            spatial_metadata["sam_valid_region"],
        )
        sam_mask_valid_mask = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            mask_logits.shape[-2:],
        )
        sam_feature_valid_mask = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            sam_mask_features.shape[-2:],
        )
        grouping = self.grouper(
            mask_logits,
            sam_valid_mask=sam_mask_valid_mask,
        )
        (
            vsam,
            vsig,
            member_evidence,
            query_mask_on_siglip,
            resolved_siglip_valid_mask,
        ) = self.region_pooler(
            query_states=query_states,
            mask_logits=mask_logits,
            sam_mask_features=sam_mask_features,
            siglip_spatial_features=siglip_spatial_features,
            spatial_metadata=spatial_metadata,
            stage_index=stage_index,
            siglip_valid_mask=siglip_valid_mask,
            sam_mask_valid_mask=sam_mask_valid_mask,
            sam_feature_valid_mask=sam_feature_valid_mask,
        )
        group_transport = None
        transported_group_prior = None
        group_innovation = None
        new_group_mask = None
        if self.topology_group_version == 1:
            group_features, member_attention_weights = self.group_aggregator(
                member_evidence=member_evidence,
                query_to_group=grouping.query_to_group,
                group_valid_mask=grouping.group_valid_mask,
                topology_centrality=grouping.topology_centrality,
            )
        else:
            if stage_index == 0:
                if previous_stage_output is not None:
                    raise ValueError(
                        "V2 stage 0 must not receive a previous stage"
                    )
                group_transport = member_evidence.new_zeros(
                    member_evidence.shape[0],
                    self.num_queries,
                    self.num_queries,
                    dtype=torch.float32,
                )
                transport_has_predecessor = torch.zeros_like(
                    grouping.group_valid_mask
                )
                raw_transported_prior = member_evidence.new_zeros(
                    member_evidence.shape
                )
            else:
                if previous_stage_output is None:
                    raise ValueError(
                        "V2 stages after st0 require previous_stage_output"
                    )
                if previous_stage_output.stage_index != stage_index - 1:
                    raise ValueError(
                        "previous topology stage must be consecutive"
                    )
                group_transport, transport_has_predecessor = (
                    self.build_group_transport(
                        current_query_to_group=grouping.query_to_group,
                        previous_query_to_group=(
                            previous_stage_output.query_to_group
                        ),
                        current_group_valid_mask=grouping.group_valid_mask,
                        previous_group_valid_mask=(
                            previous_stage_output.group_valid_mask
                        ),
                    )
                )
                raw_transported_prior = torch.matmul(
                    group_transport.to(
                        previous_stage_output.group_features.dtype
                    ),
                    previous_stage_output.group_features,
                )
            (
                group_features,
                member_attention_weights,
                transported_group_prior,
            ) = self.group_aggregator(
                member_evidence=member_evidence,
                query_to_group=grouping.query_to_group,
                group_valid_mask=grouping.group_valid_mask,
                transported_prior=raw_transported_prior,
                transport_has_predecessor=transport_has_predecessor,
            )
            new_group_mask = (
                grouping.group_valid_mask & ~transport_has_predecessor
            )
            group_innovation = (
                group_features - transported_group_prior
            ) * grouping.group_valid_mask.unsqueeze(-1).to(
                group_features.dtype
            )
            _require_finite("group_innovation", group_innovation)
        quality_logits = None
        if stage_index == self.num_stages - 1:
            if self.topology_group_version == 1:
                quality_logits = self.quality_head(
                    query_states=query_states,
                    sam_region_evidence=vsam,
                    siglip_region_evidence=vsig,
                    group_features=group_features,
                    query_to_group=grouping.query_to_group,
                    topology_centrality=grouping.topology_centrality,
                )
            else:
                quality_logits = self.quality_head(
                    mask_logits=mask_logits,
                    query_mask_on_siglip=query_mask_on_siglip,
                    sam_mask_features=sam_mask_features,
                    siglip_spatial_features=siglip_spatial_features,
                    query_to_group=grouping.query_to_group,
                    group_valid_mask=grouping.group_valid_mask,
                    group_features=group_features,
                    sam_mask_valid_mask=sam_mask_valid_mask,
                    sam_feature_valid_mask=sam_feature_valid_mask,
                    siglip_valid_mask=resolved_siglip_valid_mask,
                )
        return TopologyGroupStageOutput(
            stage_index=stage_index,
            query_states=query_states,
            query_mask_logits=mask_logits,
            query_to_group=grouping.query_to_group,
            group_valid_mask=grouping.group_valid_mask,
            group_sizes=grouping.group_sizes,
            pairwise_iou=grouping.pairwise_iou,
            topology_centrality=grouping.topology_centrality,
            sam_region_evidence=vsam,
            siglip_region_evidence=vsig,
            member_evidence=member_evidence,
            group_features=group_features,
            member_attention_weights=member_attention_weights,
            query_mask_on_siglip=query_mask_on_siglip,
            siglip_valid_mask=resolved_siglip_valid_mask,
            sam_valid_boxes_normalized=sam_valid_boxes,
            group_transport=group_transport,
            transported_group_prior=transported_group_prior,
            group_innovation=group_innovation,
            new_group_mask=new_group_mask,
            member_quality_logits=quality_logits,
        )

    def apply_feedback(
        self,
        raw_query_states: Tensor,
        stage_output: TopologyGroupStageOutput,
    ) -> Tensor:
        if stage_output.stage_index >= self.num_stages - 1:
            raise ValueError("the final prediction stage must not apply feedback")
        if self.topology_group_version == 1:
            updated, delta = self.feedback(
                raw_query_states=raw_query_states,
                group_features=stage_output.group_features,
                query_to_group=stage_output.query_to_group,
                stage_index=stage_output.stage_index,
            )
        else:
            if stage_output.group_innovation is None:
                raise ValueError(
                    "V2 feedback requires the current group innovation"
                )
            updated, delta = self.feedback(
                raw_query_states=raw_query_states,
                group_innovation=stage_output.group_innovation,
                query_to_group=stage_output.query_to_group,
            )
        stage_output.feedback_delta_norm = delta.to(torch.float32).norm(dim=-1).mean()
        stage_output.feedback_applied = True
        return updated


__all__ = [
    "DisagreementAwareIntraGroupTournament",
    "DualRegionEvidencePooler",
    "DynamicTopologyGroupAggregator",
    "GroupConditionClassifier",
    "GroupPredictionAdapter",
    "GroupToQueryFeedback",
    "InnovationGroupToQueryFeedback",
    "MaskTopologyGrouper",
    "SpatialMaskAligner",
    "TopologyGroupAggregator",
    "TopologyGroupDecoderCore",
    "TopologyGroupStageOutput",
    "TopologyGroupingOutput",
    "WithinGroupQualityHead",
]
