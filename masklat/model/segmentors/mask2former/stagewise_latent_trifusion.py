"""Stagewise visual latents and Query/Latent/Cond transport.

This module is an opt-in successor to the retained stage-3 bipartite path.  It
keeps four image-only Mask2Former stages aligned with four SigLIP/SAM feature
levels, carries one recurrent bank of 64 latents through those stages, and
uses a single gated residual at every st4--st9 transport endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .spatial_validity import (
    masked_normalized_resize,
    normalize_sam_valid_regions,
    valid_mask_from_normalized_boxes,
)
from .st3_group_vlm_refiner import (
    GaussianFourierBoxEncoder,
    St3ProposalBuilder,
    _masked_mean,
    _require_finite,
)
from .topology_group_decoder import SpatialMaskAligner


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SamSiglipFusionProjector(nn.Module):
    """Align SAM ViT features to SigLIP patches and fuse them into VLM tokens."""

    def __init__(
        self,
        *,
        siglip_hidden_dim: int,
        sam_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        super().__init__()
        fusion_dim = int(siglip_hidden_dim) + int(sam_hidden_dim)
        self.siglip_hidden_dim = int(siglip_hidden_dim)
        self.sam_hidden_dim = int(sam_hidden_dim)
        self.llm_hidden_dim = int(llm_hidden_dim)
        self.spatial_aligner = SpatialMaskAligner()
        self.siglip_norm = nn.LayerNorm(self.siglip_hidden_dim)
        self.sam_norm = nn.LayerNorm(self.sam_hidden_dim)
        self.fusion_merger = nn.Sequential(
            nn.Linear(fusion_dim, self.siglip_hidden_dim),
            nn.GELU(),
            nn.Linear(self.siglip_hidden_dim, self.llm_hidden_dim),
        )
        self.apply(_init_linear)

    def forward(
        self,
        *,
        siglip_spatial_features: Tensor,
        sam_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if siglip_spatial_features.ndim != 4 or sam_spatial_features.ndim != 4:
            raise ValueError("SAM/SigLIP fusion features must be [B,C,H,W]")
        batch_size, siglip_dim, height, width = siglip_spatial_features.shape
        if siglip_dim != self.siglip_hidden_dim:
            raise ValueError(f"SigLIP fusion width must be {self.siglip_hidden_dim}, got {siglip_dim}")
        if sam_spatial_features.shape[:2] != (batch_size, self.sam_hidden_dim):
            raise ValueError(
                "SAM fusion batch/width mismatch: "
                f"{tuple(sam_spatial_features.shape[:2])} != "
                f"{(batch_size, self.sam_hidden_dim)}"
            )
        aligned_sam, resolved_valid = self.spatial_aligner(
            sam_spatial_features,
            spatial_metadata,
            (height, width),
            supplied_valid_mask=siglip_valid_mask,
        )
        siglip_tokens = siglip_spatial_features.permute(0, 2, 3, 1)
        sam_tokens = aligned_sam.to(siglip_spatial_features.dtype).permute(
            0, 2, 3, 1
        )
        fusion_features = torch.cat(
            (self.siglip_norm(siglip_tokens), self.sam_norm(sam_tokens)),
            dim=-1,
        ).reshape(batch_size, height * width, -1)
        fused = self.fusion_merger(fusion_features)
        fusion_valid = resolved_valid.flatten(1).unsqueeze(-1)
        fused = fused * fusion_valid.to(fused.dtype)
        _require_finite("stagewise_fused_vlm_tokens", fused)
        return fused


class StagewiseLatentBlock(nn.Module):
    """Update one recurrent latent bank from one formal proposal stage."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.apply(_init_linear)

    def forward(self, latent_states: Tensor, proposal_states: Tensor) -> Tensor:
        if latent_states.ndim != 3 or proposal_states.ndim != 3:
            raise ValueError("stagewise latent/proposal states must be [B,T,D]")
        if latent_states.shape[0] != proposal_states.shape[0] or (
            latent_states.shape[-1] != self.hidden_dim or proposal_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("stagewise latent/proposal batch or width mismatch")

        cross_delta, _ = self.cross_attention(
            query=self.cross_query_norm(latent_states),
            key=self.cross_memory_norm(proposal_states),
            value=self.cross_memory_norm(proposal_states),
            need_weights=False,
        )
        latent_states = latent_states + cross_delta
        normalized = self.self_norm(latent_states)
        self_delta, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        latent_states = latent_states + self_delta
        latent_states = latent_states + self.ffn(self.ffn_norm(latent_states))
        _require_finite("stagewise_recurrent_latents", latent_states)
        return latent_states


@dataclass
class StagewiseLatentProposal:
    """Four proposal stages and their recurrent latent representations."""

    proposal_features: Tuple[Tensor, ...]
    stage_latent_features: Tuple[Tensor, ...]
    stage_vlm_tokens: Tuple[Tensor, ...]
    latent_features: Tensor
    query_boxes_cxcywh: Tuple[Tensor, ...]
    query_geometry_valid_mask: Tuple[Tensor, ...]
    query_mask_on_siglip: Tensor
    siglip_valid_mask: Tensor
    sam_valid_boxes_normalized: Tensor
    packed_group_tokens: Tensor
    packed_group_valid_mask: Tensor


class StagewiseProposalLatentBuilder(nn.Module):
    """Build proposal_i and recurrent z_i for formal stages st0--st3."""

    num_stages = 4

    def __init__(
        self,
        *,
        hidden_dim: int,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        llm_hidden_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        mask_threshold: float = 0.5,
        fourier_features: int = 64,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_latents = int(num_latents)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_latents <= 0:
            raise ValueError("hidden_dim and num_latents must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")

        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.mask_threshold = float(mask_threshold)
        self.sam_feature_dim = int(sam_feature_dim)
        self.siglip_feature_dim = int(siglip_feature_dim)
        self.spatial_aligner = SpatialMaskAligner()
        self.sam_projectors = nn.ModuleList(
            nn.Linear(self.sam_feature_dim, hidden_dim) for _ in range(self.num_stages)
        )
        self.siglip_projectors = nn.ModuleList(
            nn.Linear(self.siglip_feature_dim, hidden_dim) for _ in range(self.num_stages)
        )
        self.proposal_fusions = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(self.num_stages)
        )
        self.geometry_encoders = nn.ModuleList(
            GaussianFourierBoxEncoder(
                hidden_dim,
                fourier_features=int(fourier_features),
            )
            for _ in range(self.num_stages)
        )
        self.proposal_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(self.num_stages))
        self.learned_latents = nn.Parameter(torch.empty(num_latents, hidden_dim))
        self.latent_blocks = nn.ModuleList(StagewiseLatentBlock(hidden_dim, num_heads) for _ in range(self.num_stages))
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(llm_hidden_dim)),
        )
        self.apply(_init_linear)
        nn.init.normal_(self.learned_latents, mean=0.0, std=0.02)

    @property
    def llm_hidden_dim(self) -> int:
        return int(self.vlm_projector[-1].out_features)

    def _build_stage_proposal(
        self,
        stage_index: int,
        query_states: Tensor,
        mask_logits: Tensor,
        sam_features: Tensor,
        siglip_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            sam_features.shape[-2:],
        )
        mask_prob = mask_logits.sigmoid() * sam_mask_valid.to(mask_logits.dtype)
        mask_on_sam, _ = masked_normalized_resize(
            mask_prob,
            sam_mask_valid,
            sam_features.shape[-2:],
            target_valid_mask=sam_feature_valid,
        )
        mask_on_siglip, resolved_siglip_valid = self.spatial_aligner(
            mask_prob,
            spatial_metadata,
            siglip_features.shape[-2:],
            supplied_valid_mask=siglip_valid_mask,
            sam_source_valid_mask=sam_mask_valid,
        )
        query_valid = torch.ones(
            mask_prob.shape[:2],
            dtype=torch.bool,
            device=mask_prob.device,
        )
        boxes, geometry_valid = St3ProposalBuilder._group_boxes(
            mask_prob,
            query_valid,
            spatial_metadata,
            self.mask_threshold,
        )

        sam_mass = mask_on_sam.to(torch.float32).sum((-2, -1)).unsqueeze(-1)
        siglip_mass = mask_on_siglip.to(torch.float32).sum((-2, -1)).unsqueeze(-1)
        sam_has_evidence = sam_mass.gt(1.0e-6) & geometry_valid.unsqueeze(-1)
        siglip_has_evidence = siglip_mass.gt(1.0e-6) & geometry_valid.unsqueeze(-1)
        pooled_sam = _masked_mean(sam_features, mask_on_sam)
        pooled_siglip = _masked_mean(siglip_features, mask_on_siglip)
        sam_region = torch.where(
            sam_has_evidence,
            pooled_sam,
            torch.zeros_like(pooled_sam),
        )
        siglip_region = torch.where(
            siglip_has_evidence,
            pooled_siglip,
            torch.zeros_like(pooled_siglip),
        )
        sam_token = self.sam_projectors[stage_index](sam_region)
        siglip_token = self.siglip_projectors[stage_index](siglip_region)
        sam_token = torch.where(
            sam_has_evidence,
            sam_token,
            torch.zeros_like(sam_token),
        )
        siglip_token = torch.where(
            siglip_has_evidence,
            siglip_token,
            torch.zeros_like(siglip_token),
        )
        proposal = self.proposal_fusions[stage_index](torch.cat((query_states, sam_token, siglip_token), dim=-1))
        proposal = self.proposal_norms[stage_index](
            proposal + self.geometry_encoders[stage_index](boxes, geometry_valid)
        )
        _require_finite(f"stagewise_proposal_st{stage_index}", proposal)
        return (
            proposal,
            boxes,
            geometry_valid,
            mask_on_siglip,
            resolved_siglip_valid,
            sam_valid_boxes,
        )

    def forward(
        self,
        *,
        query_states: Sequence[Tensor],
        mask_logits: Sequence[Tensor],
        sam_spatial_features: Sequence[Tensor],
        siglip_spatial_features: Sequence[Tensor],
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> StagewiseLatentProposal:
        stage_inputs = (
            query_states,
            mask_logits,
            sam_spatial_features,
            siglip_spatial_features,
        )
        if any(len(values) != self.num_stages for values in stage_inputs):
            raise ValueError("stagewise proposal inputs must each contain st0--st3")

        proposals = []
        boxes = []
        geometry_valid_masks = []
        latent_stages = []
        vlm_stages = []
        final_mask_on_siglip = None
        resolved_siglip_valid = None
        sam_valid_boxes = None
        batch_size = int(query_states[0].shape[0])
        latent = self.learned_latents.unsqueeze(0).expand(batch_size, -1, -1)

        for stage_index in range(self.num_stages):
            query = query_states[stage_index]
            masks = mask_logits[stage_index]
            sam_features = sam_spatial_features[stage_index]
            siglip_features = siglip_spatial_features[stage_index]
            if query.ndim != 3 or masks.ndim != 4:
                raise ValueError("stage query/mask tensors must be [B,Q,D]/[B,Q,H,W]")
            if query.shape[:2] != masks.shape[:2] or query.shape[-1] != self.hidden_dim:
                raise ValueError(f"invalid query/mask contract at st{stage_index}")
            if sam_features.ndim != 4 or sam_features.shape[1] != self.sam_feature_dim:
                raise ValueError(f"invalid SAM feature contract at st{stage_index}")
            if siglip_features.ndim != 4 or (siglip_features.shape[1] != self.siglip_feature_dim):
                raise ValueError(f"invalid SigLIP feature contract at st{stage_index}")

            (
                proposal,
                stage_boxes,
                stage_geometry_valid,
                final_mask_on_siglip,
                resolved_siglip_valid,
                sam_valid_boxes,
            ) = self._build_stage_proposal(
                stage_index,
                query,
                masks,
                sam_features,
                siglip_features,
                spatial_metadata,
                siglip_valid_mask,
            )
            latent = self.latent_blocks[stage_index](latent, proposal)
            vlm_tokens = self.vlm_projector(latent)
            proposals.append(proposal)
            boxes.append(stage_boxes)
            geometry_valid_masks.append(stage_geometry_valid)
            latent_stages.append(latent)
            vlm_stages.append(vlm_tokens)
            _require_finite(f"stagewise_vlm_latents_st{stage_index}", vlm_tokens)

        valid = torch.ones(
            batch_size,
            self.num_latents,
            dtype=torch.bool,
            device=latent.device,
        )
        return StagewiseLatentProposal(
            proposal_features=tuple(proposals),
            stage_latent_features=tuple(latent_stages),
            stage_vlm_tokens=tuple(vlm_stages),
            latent_features=latent_stages[-1],
            query_boxes_cxcywh=tuple(boxes),
            query_geometry_valid_mask=tuple(geometry_valid_masks),
            query_mask_on_siglip=final_mask_on_siglip,
            siglip_valid_mask=resolved_siglip_valid,
            sam_valid_boxes_normalized=sam_valid_boxes,
            packed_group_tokens=vlm_stages[-1],
            packed_group_valid_mask=valid,
        )


class QueryLatentConditionFusion(nn.Module):
    """One st4--st9 tri-partite update with one gate per endpoint."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        gate_init_bias: float = -2.0,
        update_latents: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.update_latents = bool(update_latents)
        self.latent_query_norm = nn.LayerNorm(hidden_dim)
        self.query_key_value_norm = nn.LayerNorm(hidden_dim)
        self.condition_key_value_norm = nn.LayerNorm(hidden_dim)
        self.query_cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.condition_cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        initial_gate = float(gate_init_bias)
        self.query_gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        self.condition_gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        if self.update_latents:
            self.latent_gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        else:
            self.register_parameter("latent_gate", None)
        self.apply(_init_linear)

    @staticmethod
    def _scatter(weights: Tensor, values: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is not None:
            weights = weights * valid_mask[:, None, :].to(weights.dtype)
        column_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("blt,bld->btd", column_weights, values)

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
        if query_states.ndim != 3 or latent_states.ndim != 3 or condition_states.ndim != 3:
            raise ValueError("Q/Z/C states must be [B,T,D]")
        if condition_anchor.shape != condition_states.shape:
            raise ValueError("condition anchor/state shapes differ")
        if condition_valid_mask.shape != condition_states.shape[:2] or (
            condition_update_mask.shape != condition_states.shape[:2]
        ):
            raise ValueError("condition masks must be [B,C]")
        if condition_valid_mask.dtype != torch.bool or condition_update_mask.dtype != torch.bool:
            raise TypeError("condition masks must be boolean")
        batch_widths = {(value.shape[0], value.shape[-1]) for value in (query_states, latent_states, condition_states)}
        if batch_widths != {(query_states.shape[0], self.hidden_dim)}:
            raise ValueError("Q/Z/C batch or hidden widths differ")
        if not bool(condition_valid_mask.any(dim=1).all()):
            raise ValueError("every condition row needs one valid key")

        normalized_latents = self.latent_query_norm(latent_states)
        normalized_queries = self.query_key_value_norm(query_states)
        normalized_conditions = self.condition_key_value_norm(condition_states)
        from_queries, query_attention = self.query_cross_attention(
            query=normalized_latents,
            key=normalized_queries,
            value=normalized_queries,
            need_weights=True,
            average_attn_weights=True,
        )
        from_conditions, condition_attention = self.condition_cross_attention(
            query=normalized_latents,
            key=normalized_conditions,
            value=normalized_conditions,
            key_padding_mask=~condition_valid_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        candidate = self.candidate_fusion(torch.cat((latent_states, from_queries, from_conditions), dim=-1))

        query_delta = self._scatter(query_attention, candidate, None)
        condition_delta = self._scatter(
            condition_attention,
            candidate,
            condition_valid_mask,
        )
        query_gate = torch.sigmoid(self.query_gate).to(query_states.dtype)
        condition_gate = torch.sigmoid(self.condition_gate).to(condition_states.dtype)
        next_queries = query_states + query_gate * query_delta.to(query_states.dtype)
        next_conditions = condition_anchor + condition_gate * condition_delta.to(condition_anchor.dtype)
        next_conditions = torch.where(
            condition_update_mask.unsqueeze(-1),
            next_conditions,
            condition_anchor,
        )
        if self.update_latents:
            latent_gate = torch.sigmoid(self.latent_gate).to(latent_states.dtype)
            next_latents = latent_states + latent_gate * (candidate.to(latent_states.dtype) - latent_states)
        else:
            next_latents = latent_states

        _require_finite("trifusion_queries", next_queries)
        _require_finite("trifusion_conditions", next_conditions)
        _require_finite("trifusion_latents", next_latents)
        return next_queries, next_latents, next_conditions


class StagewiseLatentTriFusionBridge(nn.Module):
    """Initialize st4 once from SEG and own the six tri-fusion stages."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        transport_depth: int = 6,
        gate_init_bias: float = -2.0,
        keep_background_fixed: bool = True,
    ) -> None:
        super().__init__()
        if int(transport_depth) <= 0:
            raise ValueError("transport_depth must be positive")
        self.hidden_dim = int(hidden_dim)
        self.keep_background_fixed = bool(keep_background_fixed)
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        self.seg_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.seg_gate = nn.Parameter(torch.tensor(float(gate_init_bias), dtype=torch.float32))
        self.transport_stages = nn.ModuleList(
            QueryLatentConditionFusion(
                hidden_dim,
                num_heads,
                gate_init_bias=gate_init_bias,
                update_latents=stage_index < int(transport_depth) - 1,
            )
            for stage_index in range(int(transport_depth))
        )
        self.apply(_init_linear)

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
        if query_states.shape[0] != proposal_latent_states.shape[0] or (query_states.shape[-1] != self.hidden_dim):
            raise ValueError("entry Query/latent batch or width mismatch")
        if seg_states.shape != (query_states.shape[0], self.hidden_dim):
            raise ValueError("entry SEG states must be [B,D]")
        if local_conditions.ndim != 3 or local_conditions.shape[0] != query_states.shape[0]:
            raise ValueError("entry condition states must be [B,C,D]")
        if local_condition_valid_mask.shape != local_conditions.shape[:2]:
            raise ValueError("entry condition validity mask must be [B,C]")

        latent_states = self.latent_input_norm(proposal_latent_states + vlm_latent_states)
        seg_delta = self.seg_projector(seg_states).unsqueeze(1)
        seg_gate = torch.sigmoid(self.seg_gate).to(query_states.dtype)
        query_states = query_states + seg_gate * seg_delta.to(query_states.dtype)
        condition_anchor = local_conditions
        condition_states = condition_anchor
        condition_update_mask = local_condition_valid_mask.clone()
        if self.keep_background_fixed:
            # The packed condition contract places BG in the final column.
            condition_update_mask[:, -1] = False
        return (
            query_states,
            latent_states,
            condition_states,
            condition_anchor,
            condition_update_mask,
        )


__all__ = [
    "QueryLatentConditionFusion",
    "SamSiglipFusionProjector",
    "StagewiseLatentBlock",
    "StagewiseLatentProposal",
    "StagewiseLatentTriFusionBridge",
    "StagewiseProposalLatentBuilder",
]
