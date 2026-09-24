"""Stage-3 proposal latents with persistent decoder-side memory.

The bridge deliberately avoids a hard pre-language instance partition.  The
first three Mask2Former layers produce 200 image-only mask proposals.  A
fixed-capacity bank of learned latent queries softly reads mask-aware proposal
tokens, passes exactly ``num_latents`` visual tokens through the VLM, and then
uses the same latent/proposal affinity to initialize all 200 decoder queries.
The semantic latents remain alive through stages 4--9: after each ordinary
decoder layer a non-causal masked Mixer synchronizes the 200 mask Queries and
64 latent-memory tokens *before* the formal mask prediction.

Tensor conventions:

* proposal/query tensors are ``[B, Q, D]``;
* latent tensors are ``[B, L, D]``;
* affinity logits are ``[B, L, Q]``;
* mask logits are ``[B, Q, H, W]``.
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


def _content_gate(projection: nn.Linear, features: Tensor) -> Tensor:
    """Evaluate a data-dependent gate in fp32 and return the input dtype."""

    return torch.sigmoid(projection(features).to(torch.float32)).to(
        dtype=features.dtype
    )


class GatedLatentInteractionBlock(nn.Module):
    """Non-causal latent self-attention followed by gated memory reading."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_init_bias = float(gate_init_bias)
        self.latent_self_norm = nn.LayerNorm(hidden_dim)
        self.latent_self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.latent_self_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.latent_cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.latent_cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.latent_cross_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.latent_ffn_norm = nn.LayerNorm(hidden_dim)
        self.latent_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.latent_ffn_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.apply(_init_linear)
        self.reset_residual_initialization()

    def reset_residual_initialization(self) -> None:
        """Restore gates after a parent module performs recursive init."""

        for gate in (
            self.latent_self_gate,
            self.latent_cross_gate,
            self.latent_ffn_gate,
        ):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, self.gate_init_bias)

    @staticmethod
    def _gated_residual(
        source: Tensor,
        delta: Tensor,
        gate_projection: nn.Linear,
    ) -> Tensor:
        gate = _content_gate(
            gate_projection,
            torch.cat((source, delta), dim=-1),
        )
        return source + gate * delta.to(source.dtype)

    def forward(
        self,
        latent_states: Tensor,
        memory_states: Tensor,
        memory_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if latent_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("latent/memory states must be [B,T,D]")
        if latent_states.shape[0] != memory_states.shape[0]:
            raise ValueError("latent and memory batches differ")
        if latent_states.shape[-1] != self.hidden_dim or (
            memory_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("latent and memory widths must equal hidden_dim")
        if memory_valid_mask is not None:
            if memory_valid_mask.shape != memory_states.shape[:2]:
                raise ValueError("memory_valid_mask must be [B,Tmemory]")
            if memory_valid_mask.dtype != torch.bool:
                raise TypeError("memory_valid_mask must be boolean")
            if not bool(memory_valid_mask.any(dim=1).all()):
                raise ValueError("every row must contain valid memory")

        normalized = self.latent_self_norm(latent_states)
        self_context, _ = self.latent_self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        latent_states = self._gated_residual(
            latent_states,
            self_context,
            self.latent_self_gate,
        )

        cross_context, _ = self.latent_cross_attention(
            query=self.latent_cross_norm(latent_states),
            key=self.memory_norm(memory_states),
            value=self.memory_norm(memory_states),
            key_padding_mask=(
                None if memory_valid_mask is None else ~memory_valid_mask
            ),
            need_weights=False,
        )
        latent_states = self._gated_residual(
            latent_states,
            cross_context,
            self.latent_cross_gate,
        )

        normalized = self.latent_ffn_norm(latent_states)
        ffn_context = self.latent_ffn(normalized)
        latent_states = self._gated_residual(
            latent_states,
            ffn_context,
            self.latent_ffn_gate,
        )
        _require_finite("gated_latent_interaction", latent_states)
        return latent_states


class PersistentLatentMixer(nn.Module):
    """Joint non-causal Q/latent Mixer that is an exact no-op at step zero.

    Query rows may read their own Query token and every latent, but not another
    Query.  Latent rows may read all Queries and all latents.  This preserves
    the original decoder's responsibility for Query-to-Query interaction while
    providing bidirectional Query/latent transport in one attention operation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_init_bias = float(gate_init_bias)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.joint_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.query_attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.latent_attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.query_attention_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.latent_attention_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.latent_ffn_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.latent_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.query_ffn_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.latent_ffn_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.apply(_init_linear)

        self.reset_residual_initialization()

    def reset_residual_initialization(self) -> None:
        """Restore the exact identity start after recursive parent init."""

        for gate in (
            self.query_attention_gate,
            self.latent_attention_gate,
            self.query_ffn_gate,
            self.latent_ffn_gate,
        ):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, self.gate_init_bias)
        # Protect the pretrained decoder exactly at initialization.  Gradients
        # reach these final projections on the first backward pass; subsequent
        # steps then open the attention and FFN transports smoothly.
        for output in (
            self.query_attention_output,
            self.latent_attention_output,
            self.query_ffn[-1],
            self.latent_ffn[-1],
        ):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @staticmethod
    def _joint_attention_mask(
        query_count: int,
        latent_count: int,
        device: torch.device,
    ) -> Tensor:
        total = int(query_count) + int(latent_count)
        blocked = torch.zeros(total, total, dtype=torch.bool, device=device)
        blocked[:query_count, :query_count] = ~torch.eye(
            query_count,
            dtype=torch.bool,
            device=device,
        )
        return blocked

    @staticmethod
    def _update(
        source: Tensor,
        delta: Tensor,
        output_projection: nn.Linear,
        gate_projection: nn.Linear,
    ) -> Tensor:
        delta = output_projection(delta)
        gate = _content_gate(
            gate_projection,
            torch.cat((source, delta), dim=-1),
        )
        return source + gate * delta.to(source.dtype)

    def forward(
        self,
        query_states: Tensor,
        latent_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if query_states.ndim != 3 or latent_states.ndim != 3:
            raise ValueError("Query/latent states must be [B,T,D]")
        if query_states.shape[0] != latent_states.shape[0]:
            raise ValueError("Query and latent batches differ")
        if query_states.shape[-1] != self.hidden_dim or (
            latent_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("Query and latent widths must equal hidden_dim")

        query_count = int(query_states.shape[1])
        latent_count = int(latent_states.shape[1])
        joint = torch.cat(
            (self.query_norm(query_states), self.latent_norm(latent_states)),
            dim=1,
        )
        context, _ = self.joint_attention(
            joint,
            joint,
            joint,
            attn_mask=self._joint_attention_mask(
                query_count,
                latent_count,
                joint.device,
            ),
            need_weights=False,
        )
        query_context = context[:, :query_count]
        latent_context = context[:, query_count:]
        query_states = self._update(
            query_states,
            query_context,
            self.query_attention_output,
            self.query_attention_gate,
        )
        latent_states = self._update(
            latent_states,
            latent_context,
            self.latent_attention_output,
            self.latent_attention_gate,
        )

        query_ffn = self.query_ffn(self.query_ffn_norm(query_states))
        latent_ffn = self.latent_ffn(self.latent_ffn_norm(latent_states))
        query_states = self._update(
            query_states,
            query_ffn,
            nn.Identity(),
            self.query_ffn_gate,
        )
        latent_states = self._update(
            latent_states,
            latent_ffn,
            nn.Identity(),
            self.latent_ffn_gate,
        )
        _require_finite("persistent_mixer_queries", query_states)
        _require_finite("persistent_mixer_latents", latent_states)
        return query_states, latent_states


@dataclass
class St3LatentProposal:
    """Image-level proposal state retained while latent tokens cross the VLM."""

    proposal_features: Tensor
    latent_features: Tensor
    affinity_logits: Tensor
    query_boxes_cxcywh: Tensor
    query_geometry_valid_mask: Tensor
    query_mask_on_siglip: Tensor
    siglip_valid_mask: Tensor
    sam_valid_boxes_normalized: Tensor
    packed_group_tokens: Tensor
    packed_group_valid_mask: Tensor


class St3ProposalLatentBuilder(nn.Module):
    """Compress 200 mask-aware stage-3 proposals into learned visual latents."""

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
        use_sam_region: bool = True,
        use_siglip_region: bool = True,
        use_xywh: bool = True,
        pre_vlm_depth: int = 2,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_latents = int(num_latents)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_latents <= 0:
            raise ValueError("hidden_dim and num_latents must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        for name, value in (
            ("use_sam_region", use_sam_region),
            ("use_siglip_region", use_siglip_region),
            ("use_xywh", use_xywh),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")

        self.hidden_dim = hidden_dim
        self.num_latents = num_latents
        self.mask_threshold = float(mask_threshold)
        self.use_sam_region = use_sam_region
        self.use_siglip_region = use_siglip_region
        self.use_xywh = use_xywh
        self.spatial_aligner = SpatialMaskAligner()

        self.sam_projector = nn.Linear(int(sam_feature_dim), hidden_dim)
        self.siglip_projector = nn.Linear(int(siglip_feature_dim), hidden_dim)
        self.proposal_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geometry_encoder = GaussianFourierBoxEncoder(
            hidden_dim,
            fourier_features=int(fourier_features),
        )
        self.proposal_norm = nn.LayerNorm(hidden_dim)

        self.learned_latents = nn.Parameter(torch.empty(num_latents, hidden_dim))
        self.latent_query = nn.Linear(hidden_dim, hidden_dim)
        self.proposal_key = nn.Linear(hidden_dim, hidden_dim)
        if int(pre_vlm_depth) <= 0:
            raise ValueError("pre_vlm_depth must be positive")
        self.pre_vlm_blocks = nn.ModuleList(
            GatedLatentInteractionBlock(
                hidden_dim,
                num_heads,
                gate_init_bias=gate_init_bias,
            )
            for _ in range(int(pre_vlm_depth))
        )
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(llm_hidden_dim)),
        )

        self.apply(_init_linear)
        for block in self.pre_vlm_blocks:
            block.reset_residual_initialization()
        nn.init.normal_(self.learned_latents, mean=0.0, std=0.02)

    @property
    def llm_hidden_dim(self) -> int:
        return int(self.vlm_projector[-1].out_features)

    @staticmethod
    def _query_boxes(
        mask_probabilities: Tensor,
        spatial_metadata: Dict[str, Tensor],
        mask_threshold: float,
    ) -> Tuple[Tensor, Tensor]:
        # The established Group geometry implementation already contains the
        # complete SAM resize/padding/crop inverse.  Treat every Query as a
        # physical row and reuse that exact coordinate contract.
        query_valid = torch.ones(
            mask_probabilities.shape[:2],
            dtype=torch.bool,
            device=mask_probabilities.device,
        )
        return St3ProposalBuilder._group_boxes(
            mask_probabilities,
            query_valid,
            spatial_metadata,
            mask_threshold,
        )

    def forward(
        self,
        *,
        query_states: Tensor,
        mask_logits: Tensor,
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
        latent_seed: Optional[Tensor] = None,
        compute_affinity: bool = True,
    ) -> St3LatentProposal:
        if query_states.ndim != 3 or mask_logits.ndim != 4:
            raise ValueError(
                "query_states/mask_logits must be [B,Q,D] and [B,Q,H,W]"
            )
        if query_states.shape[:2] != mask_logits.shape[:2]:
            raise ValueError("query and mask B/Q dimensions differ")
        if query_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"query hidden width must be {self.hidden_dim}, got "
                f"{query_states.shape[-1]}"
            )
        if sam_mask_features.ndim != 4 or siglip_spatial_features.ndim != 4:
            raise ValueError("SAM and SigLIP spatial features must be [B,C,H,W]")
        if not (
            query_states.shape[0]
            == sam_mask_features.shape[0]
            == siglip_spatial_features.shape[0]
        ):
            raise ValueError("proposal and spatial-feature batches differ")

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
        mask_prob = mask_logits.sigmoid() * sam_mask_valid.to(mask_logits.dtype)
        mask_on_sam, _ = masked_normalized_resize(
            mask_prob,
            sam_mask_valid,
            sam_mask_features.shape[-2:],
            target_valid_mask=sam_feature_valid,
        )
        mask_on_siglip, resolved_siglip_valid = self.spatial_aligner(
            mask_prob,
            spatial_metadata,
            siglip_spatial_features.shape[-2:],
            supplied_valid_mask=siglip_valid_mask,
            sam_source_valid_mask=sam_mask_valid,
        )

        boxes, geometry_valid = self._query_boxes(
            mask_prob,
            spatial_metadata,
            self.mask_threshold,
        )
        sam_has_evidence = (
            mask_on_sam.to(torch.float32)
            .sum(dim=(-2, -1), keepdim=False)
            .unsqueeze(-1)
            .gt(1.0e-6)
            & geometry_valid.unsqueeze(-1)
        )
        siglip_has_evidence = (
            mask_on_siglip.to(torch.float32)
            .sum(dim=(-2, -1), keepdim=False)
            .unsqueeze(-1)
            .gt(1.0e-6)
            & geometry_valid.unsqueeze(-1)
        )
        sam_region = _masked_mean(sam_mask_features, mask_on_sam)
        siglip_region = _masked_mean(siglip_spatial_features, mask_on_siglip)
        sam_region = torch.where(
            sam_has_evidence,
            sam_region,
            torch.zeros_like(sam_region),
        )
        siglip_region = torch.where(
            siglip_has_evidence,
            siglip_region,
            torch.zeros_like(siglip_region),
        )

        sam_token = self.sam_projector(sam_region)
        siglip_token = self.siglip_projector(siglip_region)
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
        # Multiplication by zero deliberately keeps every retained S2 module
        # in the autograd graph while removing its information contribution.
        # This makes the source ablations parameter-count and DDP compatible.
        if not self.use_sam_region:
            sam_token = sam_token * 0.0
        if not self.use_siglip_region:
            siglip_token = siglip_token * 0.0
        proposal = self.proposal_fusion(
            torch.cat((query_states, sam_token, siglip_token), dim=-1)
        )
        geometry_token = self.geometry_encoder(boxes, geometry_valid)
        if not self.use_xywh:
            geometry_token = geometry_token * 0.0
        proposal = self.proposal_norm(
            proposal + geometry_token
        )

        batch_size = proposal.shape[0]
        if latent_seed is None:
            if self.learned_latents is None:
                raise ValueError("this recurrent stage requires a latent_seed")
            latent_seed = self.learned_latents.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        elif latent_seed.shape != (
            batch_size, self.num_latents, self.hidden_dim
        ):
            raise ValueError("latent_seed must be [B,num_latents,hidden_dim]")
        latent = latent_seed
        for block in self.pre_vlm_blocks:
            latent = block(latent, proposal)

        if compute_affinity:
            affinity_logits = torch.einsum(
                "bld,bqd->blq",
                self.latent_query(latent).to(torch.float32),
                self.proposal_key(proposal).to(torch.float32),
            ) / math.sqrt(float(self.hidden_dim))
        else:
            # The bipartite bridge builds its own post-VLM relations.  The
            # recurrent variant retains this field only for bundle compatibility.
            affinity_logits = torch.zeros(
                batch_size, self.num_latents, proposal.shape[1],
                device=latent.device, dtype=torch.float32,
            )
        vlm_tokens = self.vlm_projector(latent)
        latent_valid = torch.ones(
            batch_size,
            self.num_latents,
            dtype=torch.bool,
            device=latent.device,
        )

        for name, value in (
            ("st3_proposal_features", proposal),
            ("st3_latent_features", latent),
            ("st3_latent_affinity_logits", affinity_logits),
            ("st3_latent_vlm_tokens", vlm_tokens),
            ("st3_query_boxes", boxes),
        ):
            _require_finite(name, value)
        return St3LatentProposal(
            proposal_features=proposal,
            latent_features=latent,
            affinity_logits=affinity_logits,
            query_boxes_cxcywh=boxes,
            query_geometry_valid_mask=geometry_valid,
            query_mask_on_siglip=mask_on_siglip,
            siglip_valid_mask=resolved_siglip_valid,
            sam_valid_boxes_normalized=sam_valid_boxes,
            packed_group_tokens=vlm_tokens,
            packed_group_valid_mask=latent_valid,
        )


class PosteriorLatentQueryBridge(nn.Module):
    """Condition local latents, initialize Queries, and persist their memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        *,
        post_vlm_depth: int = 2,
        persistent_mixer_depth: int = 6,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        self.hidden_dim = hidden_dim
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        if int(post_vlm_depth) <= 0:
            raise ValueError("post_vlm_depth must be positive")
        if int(persistent_mixer_depth) <= 0:
            raise ValueError("persistent_mixer_depth must be positive")
        self.post_vlm_blocks = nn.ModuleList(
            GatedLatentInteractionBlock(
                hidden_dim,
                num_heads,
                gate_init_bias=gate_init_bias,
            )
            for _ in range(int(post_vlm_depth))
        )
        self.persistent_mixers = nn.ModuleList(
            PersistentLatentMixer(
                hidden_dim,
                num_heads,
                gate_init_bias=gate_init_bias,
            )
            for _ in range(int(persistent_mixer_depth))
        )
        self.feedback_value = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_output = nn.Linear(hidden_dim, hidden_dim)
        self.query_output_norm = nn.LayerNorm(hidden_dim)
        self.apply(_init_linear)
        for block in self.post_vlm_blocks:
            block.reset_residual_initialization()
        for mixer in self.persistent_mixers:
            mixer.reset_residual_initialization()
        # Preserve the pretrained split decoder at initialization.  The
        # feedback projection starts learning immediately, while upstream
        # latent gradients become non-zero after its first optimizer update.
        nn.init.zeros_(self.feedback_output.weight)
        nn.init.zeros_(self.feedback_output.bias)

    def forward(
        self,
        *,
        query_states: Tensor,
        proposal_latent_states: Tensor,
        vlm_latent_states: Tensor,
        affinity_logits: Tensor,
        seg_states: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if query_states.ndim != 3:
            raise ValueError("query_states must be [Nseg,Q,D]")
        if proposal_latent_states.shape != vlm_latent_states.shape:
            raise ValueError("proposal and VLM latent tables must have equal shapes")
        if proposal_latent_states.ndim != 3:
            raise ValueError("latent states must be [Nseg,L,D]")
        if affinity_logits.shape != (
            query_states.shape[0],
            proposal_latent_states.shape[1],
            query_states.shape[1],
        ):
            raise ValueError(
                "affinity_logits must be [Nseg,L,Q], got "
                f"{tuple(affinity_logits.shape)}"
            )
        if seg_states.shape != (query_states.shape[0], self.hidden_dim):
            raise ValueError("seg_states must be [Nseg,D]")
        if local_conditions.ndim != 3 or local_condition_valid_mask.shape != (
            local_conditions.shape[0],
            local_conditions.shape[1],
        ):
            raise ValueError("local conditions must be [Nseg,C,D] and [Nseg,C]")
        if local_conditions.shape[0] != query_states.shape[0]:
            raise ValueError("local condition rows must match SEG rows")
        if local_condition_valid_mask.dtype != torch.bool:
            raise TypeError("local_condition_valid_mask must be boolean")
        if not bool(local_condition_valid_mask.any(dim=1).all()):
            raise ValueError("every SEG row must contain a valid local condition")

        latent = self.latent_input_norm(
            proposal_latent_states + vlm_latent_states
        )
        # SEG is deliberately excluded here: it is shared by every Query in a
        # row and is injected directly below.  The semantic latent bank reads
        # only this SEG row's Cond set plus its explicit BG token.
        semantic_latents = latent
        for block in self.post_vlm_blocks:
            semantic_latents = block(
                semantic_latents,
                local_conditions,
                local_condition_valid_mask,
            )

        # Reuse the exact forward affinity with the dual normalization: each
        # proposal receives a convex combination over the 64 semantic
        # carriers, instead of introducing a second unrelated cross-attention.
        reverse_weights = torch.softmax(
            affinity_logits.to(torch.float32),
            dim=1,
        )
        feedback = torch.einsum(
            "nlq,nld->nqd",
            reverse_weights.to(semantic_latents.dtype),
            self.feedback_value(semantic_latents),
        )
        updated_queries = self.query_output_norm(
            query_states
            + seg_states.unsqueeze(1).to(query_states.dtype)
            + self.feedback_output(feedback)
        )
        for name, value in (
            ("st3_semantic_latents", semantic_latents),
            ("st3_reverse_affinity", reverse_weights),
            ("st3_updated_queries", updated_queries),
        ):
            _require_finite(name, value)
        return updated_queries, semantic_latents, reverse_weights

    def mix_stage(
        self,
        *,
        stage_index: int,
        query_states: Tensor,
        latent_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Synchronize Q/Z after a decoder layer and before its mask head."""

        mixer_index = int(stage_index) - 4
        if not 0 <= mixer_index < len(self.persistent_mixers):
            raise IndexError(
                f"stage_index must address st4--st{3 + len(self.persistent_mixers)}, "
                f"got st{stage_index}"
            )
        return self.persistent_mixers[mixer_index](
            query_states,
            latent_states,
        )


__all__ = [
    "GatedLatentInteractionBlock",
    "PosteriorLatentQueryBridge",
    "PersistentLatentMixer",
    "St3LatentProposal",
    "St3ProposalLatentBuilder",
]
