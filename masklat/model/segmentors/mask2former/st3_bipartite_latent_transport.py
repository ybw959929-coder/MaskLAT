"""Gate-free stage-3 proposal latents with shared bipartite transport.

This module is intentionally independent from the already trained
``st3_proposal_latent_bridge`` experiment.  The old gate-heavy modules remain
available without any state-dict or configuration changes.

The new path has three explicit responsibilities:

* two gate-free latent blocks compress the stage-3 mask-aware proposals;
* one gate-free post-VLM block grounds the persistent latent table in the
  current SEG row's local Cond/BG table;
* stages 4--9 use one Query--latent affinity matrix to read latent context into
  Queries; stages 4--8 also column-normalize that same matrix to write the
  subsequently self-attended Queries back to the latent slots.  Stage 9 is
  read-only because no later stage consumes a final latent writeback.

Tensor conventions are batch-first unless noted otherwise:

* Query/proposal states: ``[B, Q, D]``;
* latent states: ``[B, L, D]``;
* shared relation logits: ``[B, Q, L]``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from .st3_group_vlm_refiner import _require_finite
from .st3_proposal_latent_bridge import (
    St3ProposalLatentBuilder,
    _init_linear,
)


class GateFreeLatentBlock(nn.Module):
    """Standard pre-norm residual latent block without content gates.

    Cross-attention deliberately precedes latent self-attention: the learned
    slots first read their external memory, then exchange the newly collected
    evidence with one another.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads"
            )
        self.hidden_dim = hidden_dim
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
            if tuple(memory_valid_mask.shape) != tuple(memory_states.shape[:2]):
                raise ValueError("memory_valid_mask must be [B,Tmemory]")
            if memory_valid_mask.dtype != torch.bool:
                raise TypeError("memory_valid_mask must be boolean")
            if not bool(memory_valid_mask.any(dim=1).all()):
                raise ValueError("every row must contain valid memory")

        normalized_memory = self.cross_memory_norm(memory_states)
        cross_context, _ = self.cross_attention(
            query=self.cross_query_norm(latent_states),
            key=normalized_memory,
            value=normalized_memory,
            key_padding_mask=(
                None if memory_valid_mask is None else ~memory_valid_mask
            ),
            need_weights=False,
        )
        latent_states = latent_states + cross_context.to(latent_states.dtype)

        normalized_latents = self.self_norm(latent_states)
        self_context, _ = self.self_attention(
            normalized_latents,
            normalized_latents,
            normalized_latents,
            need_weights=False,
        )
        latent_states = latent_states + self_context.to(latent_states.dtype)
        latent_states = latent_states + self.ffn(
            self.ffn_norm(latent_states)
        ).to(latent_states.dtype)
        _require_finite("gate_free_latent_block", latent_states)
        return latent_states


class LateStageVLMConditionRefresh(nn.Module):
    """Reapply the trained st3 VLM/Cond refresh at one later stage.

    At the entrance to a selected formal decoder stage this module performs

    ``Z = LN(Z_persistent + Z_VLM)``
    ``Z = CondRefresh(Z, Cond + BG)``.

    There is deliberately no scalar or vector gate.  ``input_norm`` and
    ``condition_refresh`` have the same structure as the corresponding st3
    modules, which lets an old final latent checkpoint initialize them exactly.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.condition_refresh = GateFreeLatentBlock(
            self.hidden_dim,
            int(num_heads),
        )

    def forward(
        self,
        persistent_latent_states: Tensor,
        vlm_latent_states: Tensor,
        local_conditions: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tensor:
        if persistent_latent_states.shape != vlm_latent_states.shape:
            raise ValueError(
                "persistent and cached VLM latent tables must have equal shapes"
            )
        if persistent_latent_states.ndim != 3 or (
            persistent_latent_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("late-stage latent states must be [B,L,D]")
        if local_conditions.ndim != 3 or tuple(
            local_condition_valid_mask.shape
        ) != tuple(local_conditions.shape[:2]):
            raise ValueError("local conditions must be [B,C,D] and [B,C]")
        if local_conditions.shape[0] != persistent_latent_states.shape[0] or (
            local_conditions.shape[-1] != self.hidden_dim
        ):
            raise ValueError("local condition rows/width disagree with latents")
        if local_condition_valid_mask.dtype != torch.bool:
            raise TypeError("local_condition_valid_mask must be boolean")

        refreshed = self.input_norm(
            persistent_latent_states + vlm_latent_states
        )
        refreshed = self.condition_refresh(
            refreshed,
            local_conditions,
            local_condition_valid_mask,
        )
        _require_finite("late_stage_vlm_condition_refresh", refreshed)
        return refreshed


class St3BipartiteLatentBuilder(St3ProposalLatentBuilder):
    """Reuse the audited proposal geometry with a gate-free latent encoder."""

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
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            mask_threshold=mask_threshold,
            fourier_features=fourier_features,
            use_sam_region=use_sam_region,
            use_siglip_region=use_siglip_region,
            use_xywh=use_xywh,
            pre_vlm_depth=pre_vlm_depth,
            # The gated blocks constructed by ``super`` are replaced below;
            # this value can never affect the registered new architecture.
            gate_init_bias=-2.0,
        )
        self.pre_vlm_blocks = nn.ModuleList(
            GateFreeLatentBlock(hidden_dim, num_heads)
            for _ in range(int(pre_vlm_depth))
        )
        # These two heads belong to the retained gated bridge inherited from
        # St3ProposalLatentBuilder.  The bipartite bridge constructs its own
        # Query<->latent relations after the VLM and never consumes the
        # pre-VLM affinity_logits.  Keeping the legacy heads trainable gives
        # them no gradient and makes ordinary DDP fail on the next iteration.
        self.latent_query.requires_grad_(False)
        self.proposal_key.requires_grad_(False)


class QueryLatentRead(nn.Module):
    """Write latent memory to Queries and expose the shared relation logits."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        output_init_std: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        output_init_std = float(output_init_std)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(output_init_std) or output_init_std <= 0.0:
            raise ValueError("output_init_std must be finite and positive")
        self.hidden_dim = hidden_dim
        self.output_init_std = output_init_std
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.query_relation = nn.Linear(hidden_dim, hidden_dim)
        self.latent_relation = nn.Linear(hidden_dim, hidden_dim)
        self.latent_value = nn.Linear(hidden_dim, hidden_dim)
        self.query_output = nn.Linear(hidden_dim, hidden_dim)
        self.apply(_init_linear)
        self.reset_output_initialization()

    def reset_output_initialization(self) -> None:
        # Small *non-zero* initialization protects the pretrained decoder while
        # preserving a first-step gradient to relation/value projections.
        nn.init.normal_(
            self.query_output.weight,
            mean=0.0,
            std=self.output_init_std,
        )
        if self.query_output.bias is not None:
            nn.init.zeros_(self.query_output.bias)

    def relation_logits(
        self,
        query_states: Tensor,
        latent_states: Tensor,
    ) -> Tensor:
        if query_states.ndim != 3 or latent_states.ndim != 3:
            raise ValueError("Query/latent states must be [B,T,D]")
        if query_states.shape[0] != latent_states.shape[0]:
            raise ValueError("Query and latent batches differ")
        if query_states.shape[-1] != self.hidden_dim or (
            latent_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("Query and latent widths must equal hidden_dim")
        query = self.query_relation(self.query_norm(query_states)).to(
            torch.float32
        )
        latent = self.latent_relation(self.latent_norm(latent_states)).to(
            torch.float32
        )
        logits = torch.einsum("bqd,bld->bql", query, latent) / math.sqrt(
            float(self.hidden_dim)
        )
        logits = logits.clamp(min=-80.0, max=80.0)
        _require_finite("bipartite_relation_logits", logits)
        return logits

    @staticmethod
    def routing_weights(relation_logits: Tensor) -> Tuple[Tensor, Tensor]:
        if relation_logits.ndim != 3:
            raise ValueError("relation_logits must be [B,Q,L]")
        query_to_latent = torch.softmax(relation_logits, dim=-1)
        latent_to_query = torch.softmax(relation_logits, dim=1).transpose(1, 2)
        return query_to_latent, latent_to_query

    def forward(
        self,
        query_states: Tensor,
        latent_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        relation_logits = self.relation_logits(query_states, latent_states)
        query_to_latent, _ = self.routing_weights(relation_logits)
        latent_values = self.latent_value(
            self.latent_norm(latent_states)
        )
        context = torch.einsum(
            "bql,bld->bqd",
            query_to_latent.to(latent_values.dtype),
            latent_values,
        )
        updated = (
            query_states
            + self.query_output(context).to(query_states.dtype)
        )
        _require_finite("bipartite_query_update", updated)
        return updated, relation_logits


class BipartiteLatentTransportStage(nn.Module):
    """One gate-free stage sharing a relation for both transport directions.

    The final decoder stage still reads latent evidence into its Queries, but
    has no following stage that could consume a Query-to-latent writeback.
    ``enable_latent_writeback=False`` therefore makes that last stage
    explicitly read-only instead of registering permanently unused weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        output_init_std: float = 1.0e-3,
        enable_latent_writeback: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads"
            )
        self.hidden_dim = hidden_dim
        if not isinstance(enable_latent_writeback, bool):
            raise TypeError("enable_latent_writeback must be a bool")
        self.enable_latent_writeback = enable_latent_writeback
        self.query_read = QueryLatentRead(
            hidden_dim,
            output_init_std=output_init_std,
        )
        if enable_latent_writeback:
            self.query_value_norm = nn.LayerNorm(hidden_dim)
            self.query_value = nn.Linear(hidden_dim, hidden_dim)
            self.latent_output = nn.Linear(hidden_dim, hidden_dim)
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
        self.apply(_init_linear)
        self.query_read.reset_output_initialization()
        if enable_latent_writeback:
            nn.init.normal_(
                self.latent_output.weight,
                mean=0.0,
                std=float(output_init_std),
            )
            if self.latent_output.bias is not None:
                nn.init.zeros_(self.latent_output.bias)

    def update_queries(
        self,
        query_states: Tensor,
        latent_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.query_read(query_states, latent_states)

    def update_latents(
        self,
        query_states: Tensor,
        latent_states: Tensor,
        relation_logits: Tensor,
    ) -> Tensor:
        expected = (
            query_states.shape[0],
            query_states.shape[1],
            latent_states.shape[1],
        )
        if tuple(relation_logits.shape) != expected:
            raise ValueError(
                "shared relation must be [B,Q,L], got "
                f"{tuple(relation_logits.shape)}; expected {expected}"
            )
        if not self.enable_latent_writeback:
            _require_finite("bipartite_final_latent_state", latent_states)
            return latent_states
        _, latent_to_query = QueryLatentRead.routing_weights(
            relation_logits
        )
        query_values = self.query_value(
            self.query_value_norm(query_states)
        )
        context = torch.einsum(
            "blq,bqd->bld",
            latent_to_query.to(query_values.dtype),
            query_values,
        )
        latent_states = (
            latent_states
            + self.latent_output(context).to(latent_states.dtype)
        )
        normalized = self.latent_self_norm(latent_states)
        self_context, _ = self.latent_self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        latent_states = latent_states + self_context.to(latent_states.dtype)
        latent_states = latent_states + self.latent_ffn(
            self.latent_ffn_norm(latent_states)
        ).to(latent_states.dtype)
        _require_finite("bipartite_latent_update", latent_states)
        return latent_states


class BipartiteLatentTransportBridge(nn.Module):
    """Ground VLM latents, initialize stage 4, and own six transports.

    Disabling latent writeback keeps the grounded entry latent table unchanged
    across all following stages.  Query reads remain differentiable, so this
    ablates state updates without freezing or detaching the latent producer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        *,
        post_vlm_depth: int = 1,
        transport_depth: int = 6,
        output_init_std: float = 1.0e-3,
        late_condition_refresh_stages: Iterable[int] = (),
        enable_latent_writeback: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        post_vlm_depth = int(post_vlm_depth)
        transport_depth = int(transport_depth)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads"
            )
        if post_vlm_depth <= 0 or transport_depth <= 0:
            raise ValueError("post_vlm_depth/transport_depth must be positive")
        if not isinstance(enable_latent_writeback, bool):
            raise TypeError("enable_latent_writeback must be a bool")
        refresh_stages = tuple(int(stage) for stage in late_condition_refresh_stages)
        if not enable_latent_writeback and refresh_stages:
            raise ValueError("read-only latent transport cannot enable late refresh")
        self.enable_latent_writeback = enable_latent_writeback
        self.hidden_dim = hidden_dim
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        self.post_vlm_blocks = nn.ModuleList(
            GateFreeLatentBlock(hidden_dim, num_heads)
            for _ in range(post_vlm_depth)
        )
        self.entry_query_read = QueryLatentRead(
            hidden_dim,
            output_init_std=output_init_std,
        )
        self.transport_stages = nn.ModuleList(
            BipartiteLatentTransportStage(
                hidden_dim,
                num_heads,
                output_init_std=output_init_std,
                enable_latent_writeback=(
                    enable_latent_writeback and index < transport_depth - 1
                ),
            )
            for index in range(transport_depth)
        )
        if len(set(refresh_stages)) != len(refresh_stages):
            raise ValueError("late condition refresh stages must be unique")
        if any(stage < 4 or stage > 9 for stage in refresh_stages):
            raise ValueError("late condition refresh stages must be within st4--st9")
        self.late_condition_refresh_stages = refresh_stages
        self.late_condition_refreshers = nn.ModuleDict(
            {
                f"st{stage}": LateStageVLMConditionRefresh(
                    hidden_dim,
                    num_heads,
                )
                for stage in refresh_stages
            }
        )
        # A full S3 run starts from S1 plus latent S2.  That S2 checkpoint
        # contains the proposal builder but deliberately contains no trained
        # post-VLM bridge, so there is no checkpoint key from which the new
        # st6/st9 blocks can be populated.  Make all three anchor blocks begin
        # from exactly the same parameters at construction time.  A later
        # checkpoint load still takes precedence through _load_from_state_dict.
        for refresher in self.late_condition_refreshers.values():
            refresher.input_norm.load_state_dict(
                self.latent_input_norm.state_dict()
            )
            refresher.condition_refresh.load_state_dict(
                self.post_vlm_blocks[0].state_dict()
            )

    def condition_refresher(self, stage_index: int) -> Optional[nn.Module]:
        """Return the opt-in refresher for a formal stage, if configured."""

        key = f"st{int(stage_index)}"
        return (
            self.late_condition_refreshers[key]
            if key in self.late_condition_refreshers
            else None
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Initialize absent late refreshers from trained st3 refresh weights.

        A legacy final checkpoint has no ``late_condition_refreshers`` keys.
        Before recursive loading reaches the new children, copy st3's trained
        input LayerNorm and first CondRefresh block into every requested stage.
        New-format checkpoints already carrying those keys always take
        precedence, so resume never overwrites learned st6/st9 parameters.
        """

        for stage in self.late_condition_refresh_stages:
            target_root = prefix + f"late_condition_refreshers.st{stage}."
            norm_source = prefix + "latent_input_norm."
            norm_target = target_root + "input_norm."
            block_source = prefix + "post_vlm_blocks.0."
            block_target = target_root + "condition_refresh."
            for key in tuple(state_dict.keys()):
                if key.startswith(norm_source):
                    target_key = norm_target + key[len(norm_source):]
                elif key.startswith(block_source):
                    target_key = block_target + key[len(block_source):]
                else:
                    continue
                if target_key not in state_dict:
                    state_dict[target_key] = state_dict[key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        *,
        query_states: Tensor,
        proposal_latent_states: Tensor,
        vlm_latent_states: Tensor,
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
        if seg_states.shape != (query_states.shape[0], self.hidden_dim):
            raise ValueError("seg_states must be [Nseg,D]")
        if local_conditions.ndim != 3 or tuple(
            local_condition_valid_mask.shape
        ) != tuple(local_conditions.shape[:2]):
            raise ValueError("local conditions must be [Nseg,C,D] and [Nseg,C]")
        if local_conditions.shape[0] != query_states.shape[0] or (
            local_conditions.shape[-1] != self.hidden_dim
        ):
            raise ValueError("local condition rows/width disagree with Queries")
        if local_condition_valid_mask.dtype != torch.bool:
            raise TypeError("local_condition_valid_mask must be boolean")
        if not bool(local_condition_valid_mask.any(dim=1).all()):
            raise ValueError("every SEG row must contain a valid local condition")

        latent_states = self.latent_input_norm(
            proposal_latent_states + vlm_latent_states
        )
        for block in self.post_vlm_blocks:
            latent_states = block(
                latent_states,
                local_conditions,
                local_condition_valid_mask,
            )

        updated_queries, entry_relation = self.entry_query_read(
            query_states,
            latent_states,
        )
        updated_queries = (
            updated_queries
            + seg_states.unsqueeze(1).to(updated_queries.dtype)
        )
        _require_finite("transport_entry_queries", updated_queries)
        return updated_queries, latent_states, entry_relation


__all__ = [
    "BipartiteLatentTransportBridge",
    "BipartiteLatentTransportStage",
    "GateFreeLatentBlock",
    "LateStageVLMConditionRefresh",
    "QueryLatentRead",
    "St3BipartiteLatentBuilder",
]
