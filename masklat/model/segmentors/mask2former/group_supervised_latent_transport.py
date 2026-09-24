"""Opt-in grouped-latent experiment; legacy transport is not modified.

The decoder is responsible for this module's explicit delayed-writeback
protocol: retain the relation returned by ``update_queries``, finish the
query's self-attention and FFN, predict its mask, then call ``update_latents``
with that final query and the *same* relation.  No graph tensors are cached in
module attributes, so checkpoint recomputation does not change this contract.

Unlike the original transport, ST9 writes back too: the final condition reader
consumes its updated latent table for final classification.  The initial ST3
Cond grounding remains inherited; ST6/ST9 Cond refresh is deliberately absent.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from .st3_bipartite_latent_transport import (
    BipartiteLatentTransportBridge,
    BipartiteLatentTransportStage,
    QueryLatentRead,
)
from .st3_group_vlm_refiner import _require_finite
from .st3_proposal_latent_bridge import _init_linear


class GroupSupervisedLatentTransportStage(BipartiteLatentTransportStage):
    """Shared early R, but query-to-latent writeback after mask prediction.

    This marker opts into the decoder's new return protocol.  Both transport
    directions themselves are inherited, including the column-normalization
    of the original early relation for the late writeback.  There is no second
    relation projection and no latent state stored between forward calls.
    """

    writeback_after_prediction = True

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        output_init_std: float = 1.0e-3,
        enable_latent_writeback: bool = True,
    ) -> None:
        if enable_latent_writeback is not True:
            raise ValueError("group-supervised stages require latent writeback")
        super().__init__(
            hidden_dim,
            num_heads,
            output_init_std=output_init_std,
            enable_latent_writeback=True,
        )

    @staticmethod
    def group_supervision_weights(relation_logits: Tensor) -> Tensor:
        """Return the latent<-Query probabilities supervised by the mask loss."""

        _, latent_to_query = QueryLatentRead.routing_weights(relation_logits)
        return latent_to_query


class FinalConditionLatentRead(nn.Module):
    """Residual Cond <- final latent cross-attention, without Cond self-attn.

    Local condition tables reserve their final column for background.  Only
    valid foreground columns are updated.  Background and padding are returned
    bit-for-bit unchanged; invalid columns cannot influence foreground reads.
    Every foreground condition independently queries the same latent memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        output_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        output_init_std = float(output_init_std)
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not math.isfinite(output_init_std) or output_init_std <= 0:
            raise ValueError("output_init_std must be finite and positive")
        self.hidden_dim = hidden_dim
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0,
        )
        self.apply(_init_linear)
        # Non-zero initialization preserves gradients to the ST9 latent path
        # from the first S3 step without adding a learned residual gate.
        nn.init.normal_(
            self.cross_attention.out_proj.weight, mean=0.0,
            std=output_init_std,
        )
        nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(
        self,
        local_conditions: Tensor,
        final_latent_states: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tensor:
        if local_conditions.ndim != 3 or final_latent_states.ndim != 3:
            raise ValueError("conditions/latents must be [B,T,D]")
        if local_conditions.shape[0] != final_latent_states.shape[0]:
            raise ValueError("condition and latent batches differ")
        if local_conditions.shape[-1] != self.hidden_dim or (
            final_latent_states.shape[-1] != self.hidden_dim
        ):
            raise ValueError("condition and latent widths must equal hidden_dim")
        if not local_conditions.shape[1] or not final_latent_states.shape[1]:
            raise ValueError("condition and latent tables cannot be empty")
        if tuple(local_condition_valid_mask.shape) != tuple(local_conditions.shape[:2]):
            raise ValueError("local_condition_valid_mask must be [B,C]")
        if local_condition_valid_mask.dtype != torch.bool:
            raise TypeError("local_condition_valid_mask must be boolean")
        if not bool(local_condition_valid_mask[:, -1].all()):
            raise ValueError("the last local condition must be a valid BG column")

        foreground_valid = local_condition_valid_mask.clone()
        foreground_valid[:, -1] = False
        # Sanitizing inactive query columns avoids attention to arbitrary pad
        # values. Cross-attention has no mixing along the condition/query axis.
        # Keeping one batched call also leaves valid zero gradients (rather than
        # unused parameters) in a BG-only minibatch under distributed training.
        attention_queries = torch.where(
            foreground_valid.unsqueeze(-1), local_conditions,
            torch.zeros_like(local_conditions),
        )
        memory = self.latent_norm(final_latent_states)
        residual, _ = self.cross_attention(
            query=self.condition_norm(attention_queries),
            key=memory,
            value=memory,
            need_weights=False,
        )
        output = torch.where(
            foreground_valid.unsqueeze(-1),
            local_conditions + residual.to(local_conditions.dtype),
            local_conditions,
        )
        _require_finite("final_condition_latent_read", output)
        return output


class GroupSupervisedLatentTransportBridge(BipartiteLatentTransportBridge):
    """ST3 grounding, six delayed writebacks, and one final Cond readout.

    Existing ST3/ST4--ST8 state-dict names are retained.  The additional state
    is the previously absent ST9 writeback plus ``final_condition_reader``.
    Old experiment constructors and state dictionaries remain untouched.
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
        if tuple(late_condition_refresh_stages):
            raise ValueError("group-supervised transport does not use late Cond refresh")
        if enable_latent_writeback is not True:
            raise ValueError("group-supervised transport requires ST4--ST9 writeback")
        if int(transport_depth) != 6:
            raise ValueError("group-supervised transport requires six ST4--ST9 stages")
        if int(post_vlm_depth) != 1:
            raise ValueError("group-supervised transport uses exactly one ST3 Cond read")
        super().__init__(
            hidden_dim,
            num_heads,
            post_vlm_depth=post_vlm_depth,
            transport_depth=transport_depth,
            output_init_std=output_init_std,
            late_condition_refresh_stages=(),
            enable_latent_writeback=True,
        )
        stages = []
        for old_stage in self.transport_stages:
            new_stage = GroupSupervisedLatentTransportStage(
                hidden_dim, num_heads, output_init_std=output_init_std,
            )
            # Preserve the inherited initializations where modules already
            # exist. Only ST9 writeback has no original weights to inherit.
            new_stage.load_state_dict(old_stage.state_dict(), strict=False)
            stages.append(new_stage)
        self.transport_stages = nn.ModuleList(stages)
        self.final_condition_reader = FinalConditionLatentRead(
            hidden_dim, num_heads, output_init_std=output_init_std,
        )

    def readout_conditions(
        self,
        local_conditions: Tensor,
        final_latent_states: Tensor,
        local_condition_valid_mask: Tensor,
    ) -> Tensor:
        return self.final_condition_reader(
            local_conditions, final_latent_states, local_condition_valid_mask,
        )


__all__ = [
    "FinalConditionLatentRead",
    "GroupSupervisedLatentTransportBridge",
    "GroupSupervisedLatentTransportStage",
]
