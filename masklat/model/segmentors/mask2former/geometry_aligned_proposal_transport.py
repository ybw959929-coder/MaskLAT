"""Geometry-aligned visual fusion and proposal-mediated transport.

This module is an opt-in experiment.  It deliberately does not subclass or
modify the retained latent-only implementations:

* SigLIP patches are mapped into the continuous SAM coordinate system;
* SAM and aligned SigLIP features are fused on a 64x64 grid;
* pixel-unshuffle produces exactly 32x32 VLM visual tokens;
* st0 Query/mask proposals initialize 64 latents; M0/M1/M2 then pool fresh
  fused-region evidence before the Q/Z updates in st1/st2/st3;
* st4--st9 use gate-free, shared-relation Q/Z/(Cond+BG) transport.

The forward contracts are intentionally strict.  A malformed transform or a
shape change must fail instead of silently falling back to interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .spatial_validity import (
    masked_normalized_resize,
    normalize_sam_valid_regions,
    valid_mask_from_normalized_boxes,
)
from .st3_bipartite_latent_transport import (
    BipartiteLatentTransportStage,
    GateFreeLatentBlock,
)
from .st3_group_vlm_refiner import (
    GaussianFourierBoxEncoder,
    St3ProposalBuilder,
    _masked_mean,
    _require_finite,
)


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _init_small_output(module: nn.Linear, std: float) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=float(std))
    if module.bias is not None:
        nn.init.zeros_(module.bias)


@dataclass
class GeometryAlignedFusionOutput:
    """Dense fused SAM-grid features and their 32x32 VLM projection."""

    fused_features_64: Tensor
    fused_valid_mask_64: Tensor
    vlm_tokens: Tensor


class GeometryAwareSiglipToSamAligner(nn.Module):
    """Sample SigLIP features at SAM-grid centers using recorded transforms.

    Sampling is mask-normalized: both ``feature * validity`` and validity are
    sampled, then divided.  This prevents padded SigLIP pixels from attenuating
    valid boundary features under bilinear interpolation.
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

    def __init__(self, target_size: int = 64, eps: float = 1.0e-6) -> None:
        super().__init__()
        if int(target_size) <= 0:
            raise ValueError("target_size must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        self.target_size = int(target_size)
        self.eps = float(eps)

    def forward(
        self,
        siglip_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if siglip_features.ndim != 4:
            raise ValueError("siglip_features must be [B,C,H,W]")
        missing = [key for key in self.REQUIRED_KEYS if key not in spatial_metadata]
        if missing:
            raise ValueError(f"spatial_metadata is missing fields: {missing}")

        batch_size, _, source_height, source_width = siglip_features.shape
        metadata = {
            key: spatial_metadata[key].to(
                device=siglip_features.device,
                dtype=torch.float32,
            )
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
            if not bool(determinant.abs().gt(1.0e-8).all()):
                raise ValueError(f"spatial_metadata[{key!r}] contains a singular transform")
        for prefix in ("sam", "siglip"):
            size = metadata[f"{prefix}_input_size"]
            region = metadata[f"{prefix}_valid_region"]
            top, left, bottom, right = region.unbind(dim=1)
            region_valid = (
                top.ge(0)
                & left.ge(0)
                & bottom.gt(top)
                & right.gt(left)
                & bottom.le(size[:, 0])
                & right.le(size[:, 1])
            )
            if not bool(region_valid.all()):
                raise ValueError(
                    f"{prefix}_valid_region must be a non-empty rectangle "
                    f"inside {prefix}_input_size"
                )
        covered_siglip_size = metadata["siglip_patch_size"] * torch.tensor(
            [source_height, source_width],
            device=siglip_features.device,
            dtype=torch.float32,
        )
        uncovered = metadata["siglip_input_size"] - covered_siglip_size
        if not bool((
            uncovered.ge(0).all(dim=1)
            & uncovered.lt(metadata["siglip_patch_size"]).all(dim=1)
        ).all()):
            raise ValueError(
                "SigLIP feature grid and patch size do not tile its input"
            )

        if siglip_valid_mask is None:
            siglip_valid_mask = torch.ones(
                batch_size,
                1,
                source_height,
                source_width,
                dtype=torch.bool,
                device=siglip_features.device,
            )
        if siglip_valid_mask.shape != (
            batch_size,
            1,
            source_height,
            source_width,
        ):
            raise ValueError("siglip_valid_mask must be [B,1,Hsiglip,Wsiglip]")
        if siglip_valid_mask.dtype != torch.bool:
            raise TypeError("siglip_valid_mask must have dtype torch.bool")
        if siglip_valid_mask.device != siglip_features.device:
            raise ValueError("siglip_valid_mask and siglip_features must share a device")

        patch_size = metadata["siglip_patch_size"]
        source_y = (
            torch.arange(
                source_height,
                device=siglip_features.device,
                dtype=torch.float32,
            )
            + 0.5
        )
        source_x = (
            torch.arange(
                source_width,
                device=siglip_features.device,
                dtype=torch.float32,
            )
            + 0.5
        )
        patch_y, patch_x = torch.meshgrid(source_y, source_x, indexing="ij")
        source_siglip_x = patch_x.unsqueeze(0) * patch_size[:, 1, None, None]
        source_siglip_y = patch_y.unsqueeze(0) * patch_size[:, 0, None, None]
        siglip_region = metadata["siglip_valid_region"]
        metadata_source_valid = (
            source_siglip_x.ge(siglip_region[:, 1, None, None])
            & source_siglip_x.lt(siglip_region[:, 3, None, None])
            & source_siglip_y.ge(siglip_region[:, 0, None, None])
            & source_siglip_y.lt(siglip_region[:, 2, None, None])
        ).unsqueeze(1)
        siglip_valid_mask = siglip_valid_mask & metadata_source_valid

        target = self.target_size
        y_index = torch.arange(target, device=siglip_features.device, dtype=torch.float32) + 0.5
        x_index = torch.arange(target, device=siglip_features.device, dtype=torch.float32) + 0.5
        target_y, target_x = torch.meshgrid(y_index, x_index, indexing="ij")
        sam_height = metadata["sam_input_size"][:, 0, None, None]
        sam_width = metadata["sam_input_size"][:, 1, None, None]
        sam_x = target_x.unsqueeze(0) * sam_width / float(target)
        sam_y = target_y.unsqueeze(0) * sam_height / float(target)
        ones = torch.ones_like(sam_x)
        sam_points = torch.stack((sam_x, sam_y, ones), dim=-1)

        sam_to_original = torch.linalg.inv(metadata["original_to_sam"])
        original_points = torch.einsum("bij,bhwj->bhwi", sam_to_original, sam_points)
        original_points = original_points / original_points[..., 2:].clamp_min(self.eps)
        siglip_points = torch.einsum(
            "bij,bhwj->bhwi",
            metadata["original_to_siglip"],
            original_points,
        )
        siglip_points = siglip_points / siglip_points[..., 2:].clamp_min(self.eps)

        # ``align_corners=False`` maps feature-cell j to normalized coordinate
        # ``2 * (j + .5) / W - 1``.  A 27x27 patch-14 feature map covers only
        # 378x378 pixels of a 384x384 SigLIP canvas, so normalizing by the full
        # input size would shift every sample (and the rightmost one by about
        # 0.414 source cells).  Normalize by the patch grid's actual coverage.
        covered_siglip_height = (
            metadata["siglip_patch_size"][:, 0, None, None]
            * float(source_height)
        ).clamp_min(1.0)
        covered_siglip_width = (
            metadata["siglip_patch_size"][:, 1, None, None]
            * float(source_width)
        ).clamp_min(1.0)
        grid_x = 2.0 * siglip_points[..., 0] / covered_siglip_width - 1.0
        grid_y = 2.0 * siglip_points[..., 1] / covered_siglip_height - 1.0
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)

        sam_region = metadata["sam_valid_region"]
        in_sam = (
            sam_x.ge(sam_region[:, 1, None, None])
            & sam_x.lt(sam_region[:, 3, None, None])
            & sam_y.ge(sam_region[:, 0, None, None])
            & sam_y.lt(sam_region[:, 2, None, None])
        )
        original_size = metadata["original_size"]
        in_original = (
            original_points[..., 0].ge(0)
            & original_points[..., 0].lt(original_size[:, 1, None, None])
            & original_points[..., 1].ge(0)
            & original_points[..., 1].lt(original_size[:, 0, None, None])
        )
        in_siglip = (
            siglip_points[..., 0].ge(siglip_region[:, 1, None, None])
            & siglip_points[..., 0].lt(siglip_region[:, 3, None, None])
            & siglip_points[..., 1].ge(siglip_region[:, 0, None, None])
            & siglip_points[..., 1].lt(siglip_region[:, 2, None, None])
        )
        in_patch_coverage = (
            siglip_points[..., 0].ge(0)
            & siglip_points[..., 0].lt(covered_siglip_width)
            & siglip_points[..., 1].ge(0)
            & siglip_points[..., 1].lt(covered_siglip_height)
        )
        geometric_valid = (
            in_sam & in_original & in_siglip & in_patch_coverage
        ).unsqueeze(1)

        # grid_sample performs its coordinate arithmetic in float32.  Cast
        # only for this operation and restore the feature dtype afterwards.
        validity = siglip_valid_mask.to(torch.float32)
        numerator = F.grid_sample(
            siglip_features.float() * validity,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        coverage = F.grid_sample(
            validity,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        resolved_valid = geometric_valid & coverage.gt(self.eps)
        aligned = numerator / coverage.clamp_min(self.eps)
        aligned = aligned * resolved_valid.to(aligned.dtype)
        aligned = aligned.to(siglip_features.dtype)
        _require_finite("geometry_aligned_siglip_features", aligned)
        return aligned, resolved_valid


class AlignedSamSiglipTokenProjector(nn.Module):
    """Fuse on 64x64, then pixel-unshuffle to 1024 VLM tokens."""

    def __init__(
        self,
        *,
        siglip_hidden_dim: int,
        sam_hidden_dim: int,
        fusion_dim: int,
        llm_hidden_dim: int,
        target_size: int = 64,
        output_size: int = 32,
    ) -> None:
        super().__init__()
        if int(target_size) != 2 * int(output_size):
            raise ValueError("pixel-unshuffle requires target_size == 2 * output_size")
        self.siglip_hidden_dim = int(siglip_hidden_dim)
        self.sam_hidden_dim = int(sam_hidden_dim)
        self.fusion_dim = int(fusion_dim)
        self.llm_hidden_dim = int(llm_hidden_dim)
        self.target_size = int(target_size)
        self.output_size = int(output_size)
        self.aligner = GeometryAwareSiglipToSamAligner(target_size=target_size)
        self.sam_norm = nn.LayerNorm(self.sam_hidden_dim)
        self.siglip_norm = nn.LayerNorm(self.siglip_hidden_dim)
        self.local_fusion = nn.Sequential(
            nn.Linear(self.sam_hidden_dim + self.siglip_hidden_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        unshuffled_dim = 4 * self.fusion_dim
        self.output_projector = nn.Sequential(
            nn.LayerNorm(unshuffled_dim),
            nn.Linear(unshuffled_dim, self.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
        )
        self.apply(_init_linear)

    def forward(
        self,
        *,
        siglip_spatial_features: Tensor,
        sam_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> GeometryAlignedFusionOutput:
        if siglip_spatial_features.ndim != 4 or sam_spatial_features.ndim != 4:
            raise ValueError("SAM/SigLIP spatial features must be [B,C,H,W]")
        batch_size = siglip_spatial_features.shape[0]
        if siglip_spatial_features.shape[1] != self.siglip_hidden_dim:
            raise ValueError("unexpected SigLIP feature width")
        if sam_spatial_features.shape[:2] != (batch_size, self.sam_hidden_dim):
            raise ValueError("unexpected SAM feature batch/width")
        if tuple(sam_spatial_features.shape[-2:]) != (
            self.target_size,
            self.target_size,
        ):
            raise ValueError(
                "geometry fusion requires the raw SAM target grid to be "
                f"{self.target_size}x{self.target_size}"
            )

        aligned_siglip, valid = self.aligner(
            siglip_spatial_features,
            spatial_metadata,
            siglip_valid_mask,
        )
        sam_tokens = sam_spatial_features.permute(0, 2, 3, 1)
        siglip_tokens = aligned_siglip.permute(0, 2, 3, 1)
        fused = self.local_fusion(
            torch.cat(
                (self.sam_norm(sam_tokens), self.siglip_norm(siglip_tokens)),
                dim=-1,
            )
        ).permute(0, 3, 1, 2).contiguous()
        fused = fused * valid.to(fused.dtype)

        unshuffled = F.pixel_unshuffle(fused, downscale_factor=2)
        # A 2x2 output cell is valid when it contains any valid source point.
        output_valid = F.max_pool2d(valid.float(), kernel_size=2, stride=2).bool()
        tokens = unshuffled.permute(0, 2, 3, 1).reshape(
            batch_size,
            self.output_size * self.output_size,
            4 * self.fusion_dim,
        )
        tokens = self.output_projector(tokens)
        tokens = tokens * output_valid.flatten(1).unsqueeze(-1).to(tokens.dtype)
        _require_finite("geometry_aligned_fused_features_64", fused)
        _require_finite("geometry_aligned_vlm_tokens", tokens)
        return GeometryAlignedFusionOutput(
            fused_features_64=fused,
            fused_valid_mask_64=valid,
            vlm_tokens=tokens,
        )


@dataclass
class GeometryProposalLatents:
    """st0 proposal evidence and the recurrent latent state reaching st3."""

    proposal_features: Tensor
    latent_features: Tensor
    query_boxes_cxcywh: Tensor
    query_geometry_valid_mask: Tensor
    sam_valid_boxes_normalized: Tensor
    packed_group_tokens: Tensor
    packed_group_valid_mask: Tensor


class FusedProposalBuilder(nn.Module):
    """Build fused proposals and own the st0--st3 recurrent latent path.

    The formal st0 prediction initializes ``Z0``.  At st1, st2, and st3, the
    preceding masks ``M0``, ``M1``, and ``M2`` respectively pool fresh region
    evidence from the same geometry-aligned SAM+SigLIP ``F64`` map.  One
    independent latent block consumes that evidence before the corresponding
    Query--latent transport stage.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        fused_feature_dim: int,
        llm_hidden_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        mask_threshold: float = 0.5,
        fourier_features: int = 64,
        pre_vlm_depth: int = 2,
        prefix_transport_depth: int = 3,
        output_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(num_latents) <= 0:
            raise ValueError("hidden_dim and num_latents must be positive")
        if int(num_heads) <= 0 or int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if int(pre_vlm_depth) <= 0:
            raise ValueError("pre_vlm_depth must be positive")
        if int(prefix_transport_depth) != 3:
            raise ValueError(
                "geometry prefix transport must cover exactly st1--st3"
            )
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError("mask_threshold must be in [0,1]")
        self.hidden_dim = int(hidden_dim)
        self.fused_feature_dim = int(fused_feature_dim)
        self.num_latents = int(num_latents)
        self.mask_threshold = float(mask_threshold)
        self.query_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fused_projector = nn.Linear(self.fused_feature_dim, self.hidden_dim)
        self.geometry_encoder = GaussianFourierBoxEncoder(
            self.hidden_dim,
            fourier_features=int(fourier_features),
        )
        self.geometry_area_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.proposal_norm = nn.LayerNorm(self.hidden_dim)
        self.learned_latents = nn.Parameter(
            torch.empty(self.num_latents, self.hidden_dim)
        )
        self.latent_blocks = nn.ModuleList(
            GateFreeLatentBlock(self.hidden_dim, int(num_heads))
            for _ in range(int(pre_vlm_depth))
        )
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, int(llm_hidden_dim)),
        )
        self.apply(_init_linear)
        nn.init.normal_(self.learned_latents, mean=0.0, std=0.02)
        # Each prefix stage has its own fused-region refresh: M0 -> st1,
        # M1 -> st2, M2 -> st3.  Construct these after the builder-wide pass so
        # their class-local initialization remains self-contained.
        self.prefix_fused_latent_stages = nn.ModuleList(
            GateFreeLatentBlock(self.hidden_dim, int(num_heads))
            for _ in range(int(prefix_transport_depth))
        )
        # Also construct transport after the builder-wide Xavier initialization
        # so its deliberately small output projections stay at the requested
        # non-zero initialization instead of being overwritten.
        self.prefix_transport_stages = nn.ModuleList(
            BipartiteLatentTransportStage(
                self.hidden_dim,
                int(num_heads),
                output_init_std=float(output_init_std),
                enable_latent_writeback=True,
            )
            for _ in range(int(prefix_transport_depth))
        )

    @property
    def llm_hidden_dim(self) -> int:
        return int(self.vlm_projector[-1].out_features)

    def project_latents_for_vlm(self, latent_states: Tensor) -> Tensor:
        """Project the recurrent Z3 state, rather than its st0 initialization."""

        if latent_states.ndim != 3 or latent_states.shape[1:] != (
            self.num_latents,
            self.hidden_dim,
        ):
            raise ValueError(
                "latent_states must be [B,num_latents,hidden_dim]"
            )
        tokens = self.vlm_projector(latent_states)
        _require_finite("geometry_proposal_vlm_tokens", tokens)
        return tokens

    def build_proposal_features(
        self,
        *,
        query_states: Tensor,
        mask_logits: Tensor,
        fused_features_64: Tensor,
        fused_valid_mask_64: Tensor,
        spatial_metadata: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build one stage's Query/region/geometry proposal table.

        ``mask_logits`` always belongs to the prediction immediately preceding
        the stage being updated.  The returned proposal table is therefore
        suitable both for the st0 initialization and for the M0/M1/M2-driven
        latent refreshes before st1/st2/st3 Query--latent transport.
        """

        if query_states.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError("query_states/mask_logits must be [B,Q,D]/[B,Q,H,W]")
        if query_states.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("query and mask batch/query axes differ")
        if query_states.shape[-1] != self.hidden_dim:
            raise ValueError("unexpected query width")
        if fused_features_64.ndim != 4 or (
            fused_features_64.shape[:2]
            != (query_states.shape[0], self.fused_feature_dim)
        ):
            raise ValueError("fused_features_64 has an invalid batch/width")
        if fused_valid_mask_64.shape != (
            query_states.shape[0],
            1,
            fused_features_64.shape[-2],
            fused_features_64.shape[-1],
        ):
            raise ValueError("fused_valid_mask_64 has an invalid shape")
        if fused_valid_mask_64.dtype != torch.bool:
            raise TypeError("fused_valid_mask_64 must have dtype torch.bool")

        sam_valid_boxes = normalize_sam_valid_regions(
            spatial_metadata["sam_input_size"],
            spatial_metadata["sam_valid_region"],
        ).to(device=mask_logits.device)
        source_valid = valid_mask_from_normalized_boxes(
            sam_valid_boxes,
            mask_logits.shape[-2:],
        )
        mask_prob = mask_logits.sigmoid() * source_valid.to(mask_logits.dtype)
        mask_on_fused, _ = masked_normalized_resize(
            mask_prob,
            source_valid,
            fused_features_64.shape[-2:],
            target_valid_mask=fused_valid_mask_64,
        )
        query_valid = torch.ones(
            query_states.shape[:2],
            dtype=torch.bool,
            device=query_states.device,
        )
        boxes, geometry_valid = St3ProposalBuilder._group_boxes(
            mask_prob,
            query_valid,
            spatial_metadata,
            self.mask_threshold,
        )
        pooled = _masked_mean(fused_features_64, mask_on_fused)
        has_evidence = (
            mask_on_fused.float().sum((-2, -1)).unsqueeze(-1).gt(1.0e-6)
            & geometry_valid.unsqueeze(-1)
        )
        pooled = torch.where(has_evidence, pooled, torch.zeros_like(pooled))
        pooled_token = self.fused_projector(pooled)
        pooled_token = torch.where(
            has_evidence,
            pooled_token,
            torch.zeros_like(pooled_token),
        )
        area = (boxes[..., 2] * boxes[..., 3]).unsqueeze(-1)
        area_token = self.geometry_area_encoder(
            area.to(dtype=self.geometry_area_encoder[0].weight.dtype)
        )
        area_token = area_token * geometry_valid.unsqueeze(-1).to(
            area_token.dtype
        )
        proposal = self.proposal_norm(
            self.query_projector(query_states)
            + pooled_token
            + self.geometry_encoder(boxes, geometry_valid)
            + area_token
        )
        _require_finite("geometry_proposal_features", proposal)
        return proposal, boxes, geometry_valid, sam_valid_boxes

    def refine_prefix_latents(
        self,
        *,
        stage_index: int,
        query_states: Tensor,
        mask_logits: Tensor,
        latent_states: Tensor,
        fused_features_64: Tensor,
        fused_valid_mask_64: Tensor,
        spatial_metadata: Dict[str, Tensor],
    ) -> Tensor:
        """Refresh Z for exactly one of st1--st3 from its preceding mask."""

        stage_index = int(stage_index)
        if stage_index < 0 or stage_index >= len(
            self.prefix_fused_latent_stages
        ):
            raise ValueError("stage_index must identify exactly st1, st2, or st3")
        if latent_states.ndim != 3 or latent_states.shape[1:] != (
            self.num_latents,
            self.hidden_dim,
        ):
            raise ValueError("latent_states must be [B,num_latents,hidden_dim]")
        if latent_states.shape[0] != query_states.shape[0]:
            raise ValueError("latent and Query batches differ")
        proposal, _, _, _ = self.build_proposal_features(
            query_states=query_states,
            mask_logits=mask_logits,
            fused_features_64=fused_features_64,
            fused_valid_mask_64=fused_valid_mask_64,
            spatial_metadata=spatial_metadata,
        )
        latent_states = self.prefix_fused_latent_stages[stage_index](
            latent_states,
            proposal,
        )
        _require_finite("geometry_prefix_fused_latents", latent_states)
        return latent_states

    def forward(
        self,
        *,
        query_states: Tensor,
        mask_logits: Tensor,
        fused_features_64: Tensor,
        fused_valid_mask_64: Tensor,
        spatial_metadata: Dict[str, Tensor],
    ) -> GeometryProposalLatents:
        proposal, boxes, geometry_valid, sam_valid_boxes = (
            self.build_proposal_features(
                query_states=query_states,
                mask_logits=mask_logits,
                fused_features_64=fused_features_64,
                fused_valid_mask_64=fused_valid_mask_64,
                spatial_metadata=spatial_metadata,
            )
        )

        latent = self.learned_latents.unsqueeze(0).expand(
            query_states.shape[0], -1, -1
        )
        for block in self.latent_blocks:
            latent = block(latent, proposal)
        # This is the initialization-only projection.  The staged caller
        # replaces it with project_latents_for_vlm(Z3) after st1--st3 have
        # recurrently updated both Queries and latents.
        vlm_tokens = self.project_latents_for_vlm(latent)
        valid = torch.ones(
            query_states.shape[0],
            self.num_latents,
            dtype=torch.bool,
            device=query_states.device,
        )
        _require_finite("geometry_proposal_latents", latent)
        _require_finite("geometry_proposal_vlm_tokens", vlm_tokens)
        return GeometryProposalLatents(
            proposal_features=proposal,
            latent_features=latent,
            query_boxes_cxcywh=boxes,
            query_geometry_valid_mask=geometry_valid,
            sam_valid_boxes_normalized=sam_valid_boxes,
            packed_group_tokens=vlm_tokens,
            packed_group_valid_mask=valid,
        )


class TripartiteRelationTransport(nn.Module):
    """One gate-free Q/Z/(Cond+BG) update using shared relation logits.

    ``R_zq`` and ``R_zc`` are computed in parallel from the same old Z with
    separate Query/Condition projections.  Each relation's exact transpose is
    used for its reverse write, so its two directions cannot learn
    inconsistent routing matrices.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        output_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if int(num_heads) <= 0 or int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if float(output_init_std) <= 0.0:
            raise ValueError("output_init_std must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.latent_query_relation_norm = nn.LayerNorm(hidden_dim)
        self.latent_condition_relation_norm = nn.LayerNorm(hidden_dim)
        self.query_relation_norm = nn.LayerNorm(hidden_dim)
        self.condition_relation_norm = nn.LayerNorm(hidden_dim)
        self.latent_query_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.latent_condition_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.condition_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_value_norm = nn.LayerNorm(hidden_dim)
        self.condition_value_norm = nn.LayerNorm(hidden_dim)
        self.query_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.condition_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.latent_read_output = nn.Linear(2 * hidden_dim, hidden_dim)

        self.latent_self_norm = nn.LayerNorm(hidden_dim)
        self.latent_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.latent_ffn_norm = nn.LayerNorm(hidden_dim)
        self.latent_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.latent_value_norm = nn.LayerNorm(hidden_dim)
        self.latent_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_write_output = nn.Linear(hidden_dim, hidden_dim)
        self.condition_write_output = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_init_linear)
        for output in (
            self.latent_read_output,
            self.latent_self_attention.out_proj,
            self.latent_ffn[-1],
            self.query_write_output,
            self.condition_write_output,
        ):
            _init_small_output(output, output_init_std)

    def _heads(self, value: Tensor) -> Tensor:
        return value.reshape(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        return value.transpose(1, 2).contiguous().reshape(
            value.shape[0], value.shape[2], self.hidden_dim
        )

    def forward(
        self,
        *,
        query_states: Tensor,
        latent_states: Tensor,
        condition_states: Tensor,
        condition_anchor: Tensor,
        condition_valid_mask: Tensor,
        condition_update_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if any(value.ndim != 3 for value in (query_states, latent_states, condition_states)):
            raise ValueError("Q/Z/C states must be [B,T,D]")
        if condition_anchor.shape != condition_states.shape:
            raise ValueError("condition_anchor and condition_states must have equal shapes")
        if condition_valid_mask.shape != condition_states.shape[:2] or (
            condition_update_mask.shape != condition_states.shape[:2]
        ):
            raise ValueError("condition masks must be [B,C]")
        if condition_valid_mask.dtype != torch.bool or condition_update_mask.dtype != torch.bool:
            raise TypeError("condition masks must be boolean")
        if not bool(condition_valid_mask.any(dim=1).all()):
            raise ValueError("every sample must have at least one valid condition")
        if not bool((condition_update_mask & ~condition_valid_mask).sum().eq(0)):
            raise ValueError("condition_update_mask must be a subset of validity")
        expected_batch_width = (query_states.shape[0], self.hidden_dim)
        if any(
            (value.shape[0], value.shape[-1]) != expected_batch_width
            for value in (latent_states, condition_states)
        ):
            raise ValueError("Q/Z/C batch sizes or widths differ")

        old_latents = latent_states
        z_query_relation = self._heads(
            self.latent_query_relation(
                self.latent_query_relation_norm(old_latents)
            )
        )
        z_condition_relation = self._heads(
            self.latent_condition_relation(
                self.latent_condition_relation_norm(old_latents)
            )
        )
        q_relation = self._heads(
            self.query_relation(self.query_relation_norm(query_states))
        )
        c_relation = self._heads(
            self.condition_relation(
                self.condition_relation_norm(condition_states)
            )
        )
        relation_zq = torch.matmul(
            z_query_relation,
            q_relation.transpose(-1, -2),
        ) * self.scale
        relation_zc = torch.matmul(
            z_condition_relation,
            c_relation.transpose(-1, -2),
        ) * self.scale
        relation_zc = relation_zc.masked_fill(
            ~condition_valid_mask[:, None, None, :],
            torch.finfo(relation_zc.dtype).min,
        )

        q_values = self._heads(
            self.query_value(self.query_value_norm(query_states))
        )
        c_values = self._heads(
            self.condition_value(self.condition_value_norm(condition_states))
        )
        z_from_q = torch.matmul(relation_zq.softmax(dim=-1), q_values)
        z_from_c = torch.matmul(relation_zc.softmax(dim=-1), c_values)
        read_delta = self.latent_read_output(
            torch.cat((self._merge_heads(z_from_q), self._merge_heads(z_from_c)), dim=-1)
        )
        latent_states = old_latents + read_delta

        normalized = self.latent_self_norm(latent_states)
        self_delta, _ = self.latent_self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        latent_states = latent_states + self_delta
        latent_states = latent_states + self.latent_ffn(
            self.latent_ffn_norm(latent_states)
        )

        z_values = self._heads(
            self.latent_value(self.latent_value_norm(latent_states))
        )
        q_write_weights = relation_zq.transpose(-1, -2).softmax(dim=-1)
        c_write_weights = relation_zc.transpose(-1, -2).softmax(dim=-1)
        q_delta = self.query_write_output(
            self._merge_heads(torch.matmul(q_write_weights, z_values))
        )
        c_delta = self.condition_write_output(
            self._merge_heads(torch.matmul(c_write_weights, z_values))
        )
        next_queries = query_states + q_delta
        candidate_conditions = condition_states + c_delta
        next_conditions = torch.where(
            condition_update_mask.unsqueeze(-1),
            candidate_conditions,
            condition_states,
        )
        next_conditions = torch.where(
            condition_valid_mask.unsqueeze(-1),
            next_conditions,
            condition_anchor,
        )
        _require_finite("proposal_transport_queries", next_queries)
        _require_finite("proposal_transport_latents", latent_states)
        _require_finite("proposal_transport_conditions", next_conditions)
        return next_queries, latent_states, next_conditions


class ProposalMediatedTripartiteBridge(nn.Module):
    """Combine recurrent/VLM Z and own the six st4--st9 transports."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        transport_depth: int = 6,
        output_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if int(transport_depth) <= 0:
            raise ValueError("transport_depth must be positive")
        self.hidden_dim = int(hidden_dim)
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        self.transport_stages = nn.ModuleList(
            TripartiteRelationTransport(
                hidden_dim,
                num_heads,
                output_init_std=output_init_std,
            )
            for _ in range(int(transport_depth))
        )

    def initialize(
        self,
        *,
        query_states: Tensor,
        proposal_latent_states: Tensor,
        vlm_latent_states: Tensor,
        seg_states: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if proposal_latent_states.shape != vlm_latent_states.shape:
            raise ValueError("proposal/VLM latent shapes differ")
        if query_states.shape[0] != proposal_latent_states.shape[0] or (
            query_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("entry Query/latent batch or width mismatch")
        if seg_states.shape != (query_states.shape[0], self.hidden_dim):
            raise ValueError("SEG states must be [B,D]")
        if local_conditions.ndim != 3 or (
            local_conditions.shape[0] != query_states.shape[0]
            or local_conditions.shape[-1] != self.hidden_dim
        ):
            raise ValueError("local conditions must be [B,C,D]")
        if local_condition_valid_mask.shape != local_conditions.shape[:2]:
            raise ValueError("condition validity mask must be [B,C]")
        if not bool(local_condition_valid_mask[:, -1].all()):
            raise ValueError("the final condition column must be valid BG")

        latent_states = self.latent_input_norm(
            proposal_latent_states + vlm_latent_states
        )
        # SEG is already in decoder width.  A direct residual retains the
        # original conditioning role without introducing a gate or a second
        # one-time Cond bridge.
        query_states = query_states + seg_states.unsqueeze(1)
        condition_anchor = local_conditions
        condition_states = local_conditions
        # BG is deliberately treated exactly like every foreground Cond.
        condition_update_mask = local_condition_valid_mask.clone()
        return (
            query_states,
            latent_states,
            condition_states,
            condition_anchor,
            condition_update_mask,
        )


__all__ = [
    "AlignedSamSiglipTokenProjector",
    "FusedProposalBuilder",
    "GeometryAlignedFusionOutput",
    "GeometryAwareSiglipToSamAligner",
    "GeometryProposalLatents",
    "ProposalMediatedTripartiteBridge",
    "TripartiteRelationTransport",
]
