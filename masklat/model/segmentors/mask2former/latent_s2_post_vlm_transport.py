"""Matched post-VLM decoder ablations initialized from a trained latent S3.

Both variants reuse the complete, trained S3 handoff after their single VLM
pass: latent residual/normalization, Cond+BG refresh, Query read, and SEG
injection.  They differ only in the recurrent st4--st9 relation update:

* ``direct`` updates Query and Cond+BG through one shared Q/C relation;
* ``tripartite`` keeps the returned latent as a mediator and reuses the exact
  gate-free Q/Z/(Cond+BG) transport used by the geometry experiment.

The direct stage contains shared endpoint FFNs solely to keep its trainable
parameter count close to the tripartite stage.  They do not add a latent
route: after the one-time VLM-to-Query handoff, Z remains unchanged.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from .geometry_aligned_proposal_transport import TripartiteRelationTransport
from .st3_bipartite_latent_transport import (
    GateFreeLatentBlock,
    QueryLatentRead,
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


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


class SharedEndpointFFN(nn.Module):
    """One residual FFN shared by the Query and Condition endpoints."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        output_init_std: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.apply(_init_linear)
        _init_small_output(self.ffn[-1], output_init_std)

    def forward(self, states: Tensor) -> Tensor:
        return states + self.ffn(self.norm(states)).to(states.dtype)


class DirectQueryConditionTransport(nn.Module):
    """One gate-free, shared-relation Query <-> (Cond+BG) update.

    The same relation logits route Condition values into Queries and, after
    transposition, Query values into Conditions.  BG is a normal valid
    condition and therefore updates; padded condition slots stay anchored.
    ``latent_states`` is accepted only to satisfy the common decoder contract
    and is returned byte-for-byte unchanged.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        output_init_std: float = 1.0e-3,
        endpoint_ffn_depth: int = 2,
    ) -> None:
        super().__init__()
        if int(num_heads) <= 0 or int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("num_heads must divide hidden_dim")
        if int(endpoint_ffn_depth) < 0:
            raise ValueError("endpoint_ffn_depth must be non-negative")
        if not math.isfinite(float(output_init_std)) or output_init_std <= 0:
            raise ValueError("output_init_std must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.query_relation_norm = nn.LayerNorm(hidden_dim)
        self.condition_relation_norm = nn.LayerNorm(hidden_dim)
        self.query_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.condition_relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_value_norm = nn.LayerNorm(hidden_dim)
        self.condition_value_norm = nn.LayerNorm(hidden_dim)
        self.query_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.condition_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_read_output = nn.Linear(hidden_dim, hidden_dim)
        self.condition_read_output = nn.Linear(hidden_dim, hidden_dim)
        self.endpoint_ffns = nn.ModuleList(
            SharedEndpointFFN(
                hidden_dim,
                output_init_std=output_init_std,
            )
            for _ in range(int(endpoint_ffn_depth))
        )
        self.apply(_init_linear)
        _init_small_output(self.query_read_output, output_init_std)
        _init_small_output(self.condition_read_output, output_init_std)
        for endpoint_ffn in self.endpoint_ffns:
            _init_small_output(endpoint_ffn.ffn[-1], output_init_std)

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
        if any(
            value.ndim != 3
            for value in (query_states, latent_states, condition_states)
        ):
            raise ValueError("Q/Z/C states must be [B,T,D]")
        if condition_anchor.shape != condition_states.shape:
            raise ValueError("condition anchor/state shapes differ")
        if condition_valid_mask.shape != condition_states.shape[:2] or (
            condition_update_mask.shape != condition_states.shape[:2]
        ):
            raise ValueError("condition masks must be [B,C]")
        if condition_valid_mask.dtype != torch.bool or (
            condition_update_mask.dtype != torch.bool
        ):
            raise TypeError("condition masks must be boolean")
        if not bool(condition_valid_mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one valid condition")
        if bool((condition_update_mask & ~condition_valid_mask).any()):
            raise ValueError("condition update mask must be a validity subset")
        expected_batch_width = (query_states.shape[0], self.hidden_dim)
        if any(
            (value.shape[0], value.shape[-1]) != expected_batch_width
            for value in (latent_states, condition_states)
        ):
            raise ValueError("Q/Z/C batch sizes or widths differ")

        query_relation = self._heads(
            self.query_relation(self.query_relation_norm(query_states))
        )
        condition_relation = self._heads(
            self.condition_relation(
                self.condition_relation_norm(condition_states)
            )
        )
        relation_qc = torch.matmul(
            query_relation,
            condition_relation.transpose(-1, -2),
        ) * self.scale
        relation_qc = relation_qc.masked_fill(
            ~condition_valid_mask[:, None, None, :],
            torch.finfo(relation_qc.dtype).min,
        )

        condition_values = self._heads(
            self.condition_value(self.condition_value_norm(condition_states))
        )
        query_values = self._heads(
            self.query_value(self.query_value_norm(query_states))
        )
        query_context = torch.matmul(
            relation_qc.softmax(dim=-1), condition_values
        )
        condition_context = torch.matmul(
            relation_qc.transpose(-1, -2).softmax(dim=-1), query_values
        )
        next_queries = query_states + self.query_read_output(
            self._merge_heads(query_context)
        ).to(query_states.dtype)
        candidate_conditions = condition_states + self.condition_read_output(
            self._merge_heads(condition_context)
        ).to(condition_states.dtype)

        for endpoint_ffn in self.endpoint_ffns:
            next_queries = endpoint_ffn(next_queries)
            candidate_conditions = endpoint_ffn(candidate_conditions)
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
        _require_finite("direct_transport_queries", next_queries)
        _require_finite("direct_transport_conditions", next_conditions)
        return next_queries, latent_states, next_conditions


class LatentS2PostVLMBridge(nn.Module):
    """Own the matched direct or tripartite st4--st9 experiment."""

    VALID_TRANSPORT_TYPES = ("direct", "tripartite")

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        transport_type: str,
        post_vlm_depth: int = 1,
        transport_depth: int = 6,
        output_init_std: float = 1.0e-3,
        direct_endpoint_ffn_depth: int = 2,
    ) -> None:
        super().__init__()
        if transport_type not in self.VALID_TRANSPORT_TYPES:
            raise ValueError(
                f"transport_type must be one of {self.VALID_TRANSPORT_TYPES}"
            )
        if int(transport_depth) <= 0:
            raise ValueError("transport_depth must be positive")
        if int(post_vlm_depth) <= 0:
            raise ValueError("post_vlm_depth must be positive")
        self.hidden_dim = int(hidden_dim)
        self.transport_type = str(transport_type)
        # These three subtrees deliberately mirror the trained S3 bridge key
        # layout. MaskLATModel remaps the final S3 checkpoint into them exactly;
        # they are the shared, frozen handoff for both comparison branches.
        self.latent_input_norm = nn.LayerNorm(hidden_dim)
        self.post_vlm_blocks = nn.ModuleList(
            GateFreeLatentBlock(hidden_dim, num_heads)
            for _ in range(int(post_vlm_depth))
        )
        self.entry_query_read = QueryLatentRead(
            hidden_dim,
            output_init_std=output_init_std,
        )
        stage_type = (
            DirectQueryConditionTransport
            if self.transport_type == "direct"
            else TripartiteRelationTransport
        )
        self.transport_stages = nn.ModuleList(
            (
                stage_type(
                    hidden_dim,
                    num_heads,
                    output_init_std=output_init_std,
                    endpoint_ffn_depth=direct_endpoint_ffn_depth,
                )
                if self.transport_type == "direct"
                else stage_type(
                    hidden_dim,
                    num_heads,
                    output_init_std=output_init_std,
                )
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
        if local_condition_valid_mask.dtype != torch.bool:
            raise TypeError("condition validity mask must be boolean")
        if not bool(local_condition_valid_mask[:, -1].all()):
            raise ValueError("the final condition column must be valid BG")

        latent_states = self.latent_input_norm(
            proposal_latent_states + vlm_latent_states
        )
        for block in self.post_vlm_blocks:
            latent_states = block(
                latent_states,
                local_conditions,
                local_condition_valid_mask,
            )
        query_states, _ = self.entry_query_read(query_states, latent_states)
        query_states = query_states + seg_states.unsqueeze(1).to(
            query_states.dtype
        )
        condition_anchor = local_conditions
        condition_states = local_conditions
        # Foreground Cond and BG update; padding remains fixed to its anchor.
        condition_update_mask = local_condition_valid_mask.clone()
        return (
            query_states,
            latent_states,
            condition_states,
            condition_anchor,
            condition_update_mask,
        )


__all__ = [
    "DirectQueryConditionTransport",
    "LatentS2PostVLMBridge",
    "SharedEndpointFFN",
]
