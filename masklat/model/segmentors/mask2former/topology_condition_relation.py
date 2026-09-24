"""Cross-stage Group--Condition relation hypotheses for topology V2.

The original condition embeddings are immutable semantic anchors.  This
module never writes visual evidence back to them (or to decoder Queries).
Instead, it carries one compact relation state for every valid
``(physical Group root, foreground Condition)`` pair.  States are transported
between stages by persistent Query membership, so Group split/merge events do
not require stable Group ids.

Tensor conventions:

* sparse Group tables: ``[B, Q, ...]``; only ``group_valid_mask`` rows exist;
* Conditions: ``[B, Nc, D]``; the final column is background;
* relation state: ``[B, Q, Nc - 1, Dr]``.

There is deliberately no normalization over the Group dimension.  Multiple
Groups may therefore receive a high score for the same Condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _require_finite(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _finite_negative(tensor: Tensor) -> float:
    return -float(min(1.0e4, torch.finfo(tensor.dtype).max / 2.0))


def _stable_nonnegative_sqrt(values: Tensor, eps: float) -> Tensor:
    """Exact non-negative square root with a finite backward derivative.

    ``sqrt(0)`` has an infinite derivative.  Masking a zero-distance entry
    later in the graph is not sufficient: backward can still evaluate
    ``0 * inf`` and return NaN.  The detached correction preserves the exact
    forward value ``sqrt(x)`` while autograd follows the epsilon-regularized
    branch ``sqrt(x + eps)``.  Consequently the zero derivative is finite
    without quantizing or suppressing small non-zero geometry values.
    """

    nonnegative = values.clamp_min(0.0)
    safe_root = (nonnegative + float(eps)).sqrt()
    exact_root = nonnegative.sqrt()
    return exact_root.detach() + (safe_root - safe_root.detach())


@dataclass
class TopologyConditionRelationOutput:
    """Per-stage V2 classification results and transported hypotheses."""

    stage_logits: Tuple[Tensor, ...]
    relation_states: Tuple[Tensor, ...]


class TopologyConditionRelationBank(nn.Module):
    """Maintain independent, bounded Group--Condition relation hypotheses.

    For every stage, the unchanged text Condition attends to exactly three
    visual evidence tokens:

    1. the current Group feature;
    2. deterministic member-context and mask-geometry evidence;
    3. Group innovation relative to the transported previous-stage feature.

    A shared recurrent update accumulates the evidence in a compact relation
    state.  A bounded residual is added only to foreground anchor logits.
    The background/no-object logit is copied bit-for-bit from the caller's
    original anchor classifier.
    """

    _TRANSPORT_FIELDS = (
        "group_transport",
        "transport_matrix",
        "group_transport_matrix",
        "previous_group_transport",
    )

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int = 64,
        num_heads: int = 4,
        relation_logit_bound: float = 4.0,
        condition_chunk_size: int = 16,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        relation_dim = int(relation_dim)
        num_heads = int(num_heads)
        condition_chunk_size = int(condition_chunk_size)
        if hidden_dim <= 0 or relation_dim <= 0:
            raise ValueError("hidden_dim and relation_dim must be positive")
        if num_heads <= 0 or relation_dim % num_heads != 0:
            raise ValueError(
                "relation_dim must be divisible by a positive num_heads"
            )
        if relation_logit_bound <= 0:
            raise ValueError("relation_logit_bound must be positive")
        if condition_chunk_size <= 0:
            raise ValueError("condition_chunk_size must be positive")

        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim
        self.num_heads = num_heads
        self.head_dim = relation_dim // num_heads
        self.relation_logit_bound = float(relation_logit_bound)
        self.condition_chunk_size = condition_chunk_size
        self.eps = float(eps)

        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.condition_projector = nn.Linear(hidden_dim, relation_dim)
        self.group_projector = nn.Linear(hidden_dim, relation_dim)
        self.innovation_projector = nn.Linear(hidden_dim, relation_dim)
        # The context token combines mean member evidence, a learned summary of
        # the other valid Groups, absolute mask geometry, and the corresponding
        # attention-weighted relative geometry.  Pairwise work is performed
        # only on compact Group/geometry tensors; no QxQ spatial tensor is
        # materialized.
        self.neighbor_query = nn.Linear(hidden_dim, relation_dim)
        self.neighbor_key = nn.Linear(hidden_dim, relation_dim)
        self.neighbor_value = nn.Linear(hidden_dim, hidden_dim)
        self.relative_geometry_score = nn.Linear(6, 1)
        self.context_projector = nn.Linear(
            2 * hidden_dim + 12,
            relation_dim,
        )

        self.query_projector = nn.Linear(relation_dim, relation_dim)
        self.key_projector = nn.Linear(relation_dim, relation_dim)
        self.value_projector = nn.Linear(relation_dim, relation_dim)
        self.attention_output = nn.Linear(relation_dim, relation_dim)

        recurrent_width = 3 * relation_dim
        self.update_candidate = nn.Linear(recurrent_width, relation_dim)
        self.update_gate = nn.Linear(recurrent_width, relation_dim)
        self.state_norm = nn.LayerNorm(relation_dim)
        self.relation_logit = nn.Linear(relation_dim, 1)

        self.apply(self._init_weights)
        # Begin very close to the anchor classifier while retaining a
        # non-zero first-step gradient for the attention and recurrent path.
        nn.init.normal_(self.relation_logit.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.relation_logit.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def build_group_transport(
        current_query_to_group: Tensor,
        previous_query_to_group: Tensor,
        current_group_valid_mask: Optional[Tensor] = None,
        previous_group_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build row-normalized ``current Group <- previous Group`` overlap.

        Persistent Query ids form the correspondence.  Consequently, a split
        copies the appropriate fraction of its parent state and a merge takes
        the member-count-weighted mean of all parents.
        """

        if current_query_to_group.shape != previous_query_to_group.shape:
            raise ValueError(
                "adjacent query_to_group tensors must have equal [B,Q] shape"
            )
        if current_query_to_group.ndim != 2:
            raise ValueError("query_to_group must be [B,Q]")
        batch_size, num_queries = current_query_to_group.shape
        for name, mapping in (
            ("current_query_to_group", current_query_to_group),
            ("previous_query_to_group", previous_query_to_group),
        ):
            if mapping.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must have integer dtype")
            if not bool(mapping.ge(0).all() and mapping.lt(num_queries).all()):
                raise ValueError(f"{name} contains an invalid physical root")

        current_membership = F.one_hot(
            current_query_to_group.to(torch.long),
            num_classes=num_queries,
        ).to(torch.float32)
        previous_membership = F.one_hot(
            previous_query_to_group.to(torch.long),
            num_classes=num_queries,
        ).to(torch.float32)
        overlap = torch.bmm(
            current_membership.transpose(1, 2),
            previous_membership,
        )
        if current_group_valid_mask is not None:
            if current_group_valid_mask.shape != (batch_size, num_queries):
                raise ValueError("current_group_valid_mask must be [B,Q]")
            overlap = overlap * current_group_valid_mask[:, :, None].to(
                overlap.dtype
            )
        if previous_group_valid_mask is not None:
            if previous_group_valid_mask.shape != (batch_size, num_queries):
                raise ValueError("previous_group_valid_mask must be [B,Q]")
            overlap = overlap * previous_group_valid_mask[:, None, :].to(
                overlap.dtype
            )
        denominator = overlap.sum(dim=-1, keepdim=True)
        transport = torch.where(
            denominator.gt(0),
            overlap / denominator.clamp_min(1.0),
            torch.zeros_like(overlap),
        )
        _require_finite("group_transport", transport)
        return transport

    @staticmethod
    def _root_mean(
        member_values: Tensor,
        query_to_group: Tensor,
        group_valid_mask: Tensor,
    ) -> Tensor:
        """Average member values into the sparse physical-root table."""

        if member_values.ndim < 3:
            raise ValueError("member_values must be [B,Q,...]")
        if member_values.shape[:2] != query_to_group.shape:
            raise ValueError("member_values and query_to_group shapes differ")
        batch_size, num_queries = query_to_group.shape
        flat_values = member_values.reshape(batch_size, num_queries, -1)
        root_values = flat_values.new_zeros(flat_values.shape)
        root_index = query_to_group.unsqueeze(-1).expand_as(flat_values)
        root_values.scatter_add_(1, root_index, flat_values)
        counts = flat_values.new_zeros(batch_size, num_queries, 1)
        counts.scatter_add_(
            1,
            query_to_group.unsqueeze(-1),
            torch.ones_like(query_to_group, dtype=flat_values.dtype).unsqueeze(-1),
        )
        root_values = root_values / counts.clamp_min(1.0)
        root_values = root_values * group_valid_mask.unsqueeze(-1).to(
            root_values.dtype
        )
        return root_values.reshape_as(member_values)

    @staticmethod
    def _transport_by_membership(
        previous_values: Tensor,
        previous_query_to_group: Tensor,
        current_query_to_group: Tensor,
        current_group_valid_mask: Tensor,
    ) -> Tensor:
        """Transport root values without materializing a dense Q-by-Q product."""

        batch_size, num_queries = current_query_to_group.shape
        if previous_values.shape[:2] != (batch_size, num_queries):
            raise ValueError("previous_values must use the same [B,Q] table")
        gather_shape = (batch_size, num_queries) + previous_values.shape[2:]
        gather_index = previous_query_to_group.reshape(
            batch_size,
            num_queries,
            *((1,) * (previous_values.ndim - 2)),
        ).expand(gather_shape)
        values_per_persistent_query = previous_values.gather(1, gather_index)

        transported = torch.zeros_like(previous_values)
        scatter_index = current_query_to_group.reshape(
            batch_size,
            num_queries,
            *((1,) * (previous_values.ndim - 2)),
        ).expand_as(values_per_persistent_query)
        transported.scatter_add_(1, scatter_index, values_per_persistent_query)

        counts = previous_values.new_zeros(batch_size, num_queries)
        counts.scatter_add_(
            1,
            current_query_to_group,
            torch.ones_like(current_query_to_group, dtype=previous_values.dtype),
        )
        denominator = counts.reshape(
            batch_size,
            num_queries,
            *((1,) * (previous_values.ndim - 2)),
        )
        transported = transported / denominator.clamp_min(1.0)
        valid_shape = (
            batch_size,
            num_queries,
            *((1,) * (previous_values.ndim - 2)),
        )
        return transported * current_group_valid_mask.reshape(
            valid_shape
        ).to(transported.dtype)

    def _resolve_explicit_transport(
        self,
        stage_output: object,
        current_group_valid_mask: Tensor,
        previous_group_valid_mask: Tensor,
    ) -> Optional[Tensor]:
        transport = None
        for field_name in self._TRANSPORT_FIELDS:
            candidate = getattr(stage_output, field_name, None)
            if candidate is not None:
                transport = candidate
                break
        if transport is None:
            return None
        batch_size, num_queries = current_group_valid_mask.shape
        if transport.shape != (batch_size, num_queries, num_queries):
            raise ValueError(
                f"{field_name} must be [B,Qcurrent,Qprevious], got "
                f"{tuple(transport.shape)}"
            )
        transport = transport.to(
            device=current_group_valid_mask.device,
            dtype=torch.float32,
        )
        _require_finite(field_name, transport)
        if bool(transport.lt(0).any()):
            raise ValueError(f"{field_name} must be non-negative")
        transport = (
            transport
            * current_group_valid_mask[:, :, None].to(transport.dtype)
            * previous_group_valid_mask[:, None, :].to(transport.dtype)
        )
        denominator = transport.sum(dim=-1, keepdim=True)
        return torch.where(
            denominator.gt(self.eps),
            transport / denominator.clamp_min(self.eps),
            torch.zeros_like(transport),
        )

    @staticmethod
    def _transport_dense(transport: Tensor, values: Tensor) -> Tensor:
        batch_size, current_groups, previous_groups = transport.shape
        if values.shape[:2] != (batch_size, previous_groups):
            raise ValueError("transport and previous values have incompatible shapes")
        # ``P_cur^T P_prev`` has at most one edge contribution per persistent
        # Query and is consequently very sparse.  Applying its non-zero edges
        # avoids the otherwise costly QxQ by (Nc*Dr) dense product.
        edges = transport.gt(0).nonzero(as_tuple=False)
        result = values.new_zeros(
            batch_size,
            current_groups,
            *values.shape[2:],
        )
        if edges.numel() == 0:
            return result
        source_flat = values.reshape(
            batch_size * previous_groups,
            -1,
        )
        result_flat = result.reshape(batch_size * current_groups, -1)
        source_index = edges[:, 0] * previous_groups + edges[:, 2]
        target_index = edges[:, 0] * current_groups + edges[:, 1]
        edge_weight = transport[
            edges[:, 0],
            edges[:, 1],
            edges[:, 2],
        ].to(values.dtype)
        contributions = (
            source_flat.index_select(0, source_index)
            * edge_weight.unsqueeze(-1)
        )
        result_flat.index_add_(0, target_index, contributions)
        return result

    def _query_geometry(self, stage_output: object) -> Tensor:
        """Return six finite normalized geometry scalars per Query."""

        mask_probability = getattr(stage_output, "query_mask_on_siglip", None)
        valid_mask = getattr(stage_output, "siglip_valid_mask", None)
        if mask_probability is None:
            mask_logits = getattr(stage_output, "query_mask_logits")
            mask_probability = mask_logits.sigmoid()
            valid_mask = None
        if mask_probability.ndim != 4:
            raise ValueError("stage mask probabilities must be [B,Q,H,W]")
        batch_size, num_queries, height, width = mask_probability.shape
        probability = mask_probability.to(torch.float32)
        if valid_mask is None:
            valid = torch.ones(
                batch_size,
                1,
                height,
                width,
                device=probability.device,
                dtype=torch.bool,
            )
        else:
            if valid_mask.shape != (batch_size, 1, height, width):
                raise ValueError("stage validity mask must be [B,1,H,W]")
            valid = valid_mask.to(torch.bool)
        valid_float = valid.to(probability.dtype)
        probability = probability * valid_float

        y_coordinates = torch.linspace(
            0.0,
            1.0,
            height,
            device=probability.device,
            dtype=probability.dtype,
        ).view(1, 1, height, 1)
        x_coordinates = torch.linspace(
            0.0,
            1.0,
            width,
            device=probability.device,
            dtype=probability.dtype,
        ).view(1, 1, 1, width)
        mass = probability.flatten(2).sum(dim=-1)
        valid_area = valid_float.flatten(2).sum(dim=-1).clamp_min(1.0)
        area = mass / valid_area
        safe_mass = mass.clamp_min(self.eps)
        centre_x = (probability * x_coordinates).flatten(2).sum(-1) / safe_mass
        centre_y = (probability * y_coordinates).flatten(2).sum(-1) / safe_mass
        centre_x = torch.where(mass.gt(self.eps), centre_x, area.new_full((), 0.5))
        centre_y = torch.where(mass.gt(self.eps), centre_y, area.new_full((), 0.5))
        variance_x = (
            probability
            * (x_coordinates - centre_x[:, :, None, None]).square()
        ).flatten(2).sum(-1) / safe_mass
        variance_y = (
            probability
            * (y_coordinates - centre_y[:, :, None, None]).square()
        ).flatten(2).sum(-1) / safe_mass
        spread_x = _stable_nonnegative_sqrt(variance_x, self.eps)
        spread_y = _stable_nonnegative_sqrt(variance_y, self.eps)
        confidence = (
            (probability - 0.5).abs()
            * 2.0
            * valid_float
        ).flatten(2).sum(-1) / valid_area
        geometry = torch.stack(
            (area, centre_x, centre_y, spread_x, spread_y, confidence),
            dim=-1,
        )
        _require_finite("query_geometry", geometry)
        return geometry

    def _inter_group_context(
        self,
        group_features: Tensor,
        group_geometry: Tensor,
        group_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Summarize other Groups and their relative mask geometry.

        For a directed pair ``g <- h`` the compact geometry descriptor is
        ``[dx, dy, distance, log(area_h/area_g), dspread_x, dspread_y]``.
        Semantic affinity and a learned relative-geometry bias jointly weight
        valid *other* Groups.  A singleton receives deterministic zero
        neighborhood evidence instead of attending to itself.
        """

        if group_features.ndim != 3:
            raise ValueError("group_features must be [B,Q,D]")
        if group_geometry.shape != (*group_features.shape[:2], 6):
            raise ValueError("group_geometry must be [B,Q,6]")
        if group_valid_mask.shape != group_features.shape[:2]:
            raise ValueError("group_valid_mask must be [B,Q]")

        module_dtype = self.neighbor_query.weight.dtype
        features = group_features.to(module_dtype)
        query = self.neighbor_query(features)
        key = self.neighbor_key(features)
        semantic_score = torch.einsum("bqd,bkd->bqk", query, key)
        semantic_score = semantic_score * (self.relation_dim ** -0.5)

        geometry = group_geometry.to(torch.float32)
        area = geometry[..., 0].clamp_min(self.eps)
        centre_x = geometry[..., 1]
        centre_y = geometry[..., 2]
        spread_x = geometry[..., 3]
        spread_y = geometry[..., 4]
        dx = centre_x[:, None, :] - centre_x[:, :, None]
        dy = centre_y[:, None, :] - centre_y[:, :, None]
        distance = _stable_nonnegative_sqrt(
            dx.square() + dy.square(),
            self.eps,
        )
        log_area_ratio = (
            area[:, None, :].log() - area[:, :, None].log()
        )
        delta_spread_x = spread_x[:, None, :] - spread_x[:, :, None]
        delta_spread_y = spread_y[:, None, :] - spread_y[:, :, None]
        relative_geometry = torch.stack(
            (
                dx,
                dy,
                distance,
                log_area_ratio,
                delta_spread_x,
                delta_spread_y,
            ),
            dim=-1,
        )
        relative_bias = self.relative_geometry_score(
            relative_geometry.to(module_dtype)
        ).squeeze(-1)

        num_groups = group_features.shape[1]
        not_self = ~torch.eye(
            num_groups,
            device=group_features.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        pair_valid = (
            group_valid_mask[:, :, None]
            & group_valid_mask[:, None, :]
            & not_self
        )
        score = (semantic_score + relative_bias).to(torch.float32)
        score = score.masked_fill(~pair_valid, -1.0e4)
        attention = torch.softmax(score, dim=-1)
        attention = attention * pair_valid.to(attention.dtype)
        attention = attention / attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        neighbor_features = torch.einsum(
            "bqk,bkd->bqd",
            attention.to(module_dtype),
            self.neighbor_value(features),
        )
        relative_summary = torch.einsum(
            "bqk,bqkr->bqr",
            attention,
            relative_geometry,
        )
        valid_float = group_valid_mask.unsqueeze(-1)
        neighbor_features = neighbor_features * valid_float.to(
            neighbor_features.dtype
        )
        relative_summary = relative_summary * valid_float.to(
            relative_summary.dtype
        )
        _require_finite("inter_group_neighbor_features", neighbor_features)
        _require_finite("inter_group_relative_geometry", relative_summary)
        return neighbor_features, relative_summary

    def _stage_evidence(
        self,
        stage_output: object,
        transported_previous_group: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        group_features = getattr(stage_output, "group_features")
        group_valid_mask = getattr(stage_output, "group_valid_mask")
        query_to_group = getattr(stage_output, "query_to_group")
        if group_features.ndim != 3 or group_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"group_features must be [B,Q,{self.hidden_dim}]"
            )
        if group_valid_mask.shape != group_features.shape[:2]:
            raise ValueError("group_valid_mask must be [B,Q]")
        if group_valid_mask.dtype != torch.bool:
            raise TypeError("group_valid_mask must have dtype torch.bool")
        if query_to_group.shape != group_features.shape[:2]:
            raise ValueError("query_to_group must be [B,Q]")

        member_evidence = getattr(stage_output, "member_evidence", None)
        if member_evidence is None:
            member_evidence = getattr(stage_output, "query_states")
        if member_evidence.shape != group_features.shape:
            raise ValueError("member_evidence must match group_features")
        mean_member_evidence = self._root_mean(
            member_evidence,
            query_to_group,
            group_valid_mask,
        )
        mean_geometry = self._root_mean(
            self._query_geometry(stage_output).to(member_evidence.dtype),
            query_to_group,
            group_valid_mask,
        )
        neighbor_features, relative_geometry = self._inter_group_context(
            group_features,
            mean_geometry,
            group_valid_mask,
        )

        stage_innovation = getattr(stage_output, "group_innovation", None)
        if stage_innovation is not None:
            if stage_innovation.shape != group_features.shape:
                raise ValueError("group_innovation must match group_features")
            innovation = stage_innovation
        elif transported_previous_group is None:
            innovation = group_features
        else:
            innovation = group_features - transported_previous_group
        context = torch.cat(
            (
                mean_member_evidence,
                neighbor_features.to(mean_member_evidence.dtype),
                mean_geometry,
                relative_geometry.to(mean_member_evidence.dtype),
            ),
            dim=-1,
        )
        token_dtype = self.group_projector.weight.dtype
        tokens = torch.stack(
            (
                self.group_projector(group_features.to(token_dtype)),
                self.context_projector(context.to(token_dtype)),
                self.innovation_projector(innovation.to(token_dtype)),
            ),
            dim=2,
        )
        tokens = tokens * group_valid_mask[:, :, None, None].to(tokens.dtype)
        _require_finite("relation_evidence_tokens", tokens)
        return tokens, group_features

    def _cross_attend(
        self,
        condition_state: Tensor,
        evidence_tokens: Tensor,
    ) -> Tensor:
        """Condition-query cross-attention over three evidence tokens."""

        batch_size, condition_count, relation_dim = condition_state.shape
        _, group_count, token_count, _ = evidence_tokens.shape
        if token_count != 3 or relation_dim != self.relation_dim:
            raise ValueError("relation cross-attention received invalid shapes")
        query = self.query_projector(condition_state).view(
            batch_size,
            condition_count,
            self.num_heads,
            self.head_dim,
        )
        keys = self.key_projector(evidence_tokens).view(
            batch_size,
            group_count,
            token_count,
            self.num_heads,
            self.head_dim,
        )
        values = self.value_projector(evidence_tokens).view_as(keys)
        scores = torch.einsum(
            "bchd,bqthd->bqcht",
            query,
            keys,
        ) * (self.head_dim ** -0.5)
        attention = torch.softmax(scores.to(torch.float32), dim=-1).to(
            values.dtype
        )
        attended = torch.einsum(
            "bqcht,bqthd->bqchd",
            attention,
            values,
        ).reshape(batch_size, group_count, condition_count, relation_dim)
        attended = self.attention_output(attended)
        _require_finite("relation_cross_attention", attended)
        return attended

    def _update_relation_state(
        self,
        transported_state: Tensor,
        condition_state: Tensor,
        observation: Tensor,
        pair_valid_mask: Tensor,
    ) -> Tensor:
        expanded_condition = condition_state[:, None, :, :].expand_as(
            transported_state
        )
        recurrent_input = torch.cat(
            (transported_state, observation, expanded_condition),
            dim=-1,
        )
        candidate = torch.tanh(self.update_candidate(recurrent_input))
        gate = torch.sigmoid(self.update_gate(recurrent_input))
        state = self.state_norm(
            transported_state + gate * candidate
        )
        state = state * pair_valid_mask.unsqueeze(-1).to(state.dtype)
        _require_finite("relation_state", state)
        return state

    def forward(
        self,
        stage_outputs: Sequence[object],
        cond_embeddings: Tensor,
        anchor_logits: Sequence[Tensor],
        embed_masks: Optional[Tensor] = None,
    ) -> TopologyConditionRelationOutput:
        """Classify all ordered stages while preserving the text anchors."""

        if not stage_outputs:
            raise ValueError("stage_outputs cannot be empty")
        if len(stage_outputs) != len(anchor_logits):
            raise ValueError("every stage requires one anchor-logit tensor")
        if cond_embeddings.ndim != 3:
            raise ValueError("cond_embeddings must be [B,Nc,D]")
        if cond_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"condition width must be {self.hidden_dim}, got "
                f"{cond_embeddings.shape[-1]}"
            )
        batch_size, condition_count, _ = cond_embeddings.shape
        if condition_count < 1:
            raise ValueError("the final background Condition column is required")
        if embed_masks is None:
            condition_valid = torch.ones(
                batch_size,
                condition_count,
                device=cond_embeddings.device,
                dtype=torch.bool,
            )
        else:
            if embed_masks.shape != (batch_size, condition_count):
                raise ValueError("embed_masks must be [B,Nc]")
            condition_valid = embed_masks.to(
                device=cond_embeddings.device,
                dtype=torch.bool,
            )
        if not bool(condition_valid[:, -1].all()):
            raise ValueError("the final background Condition must be valid")

        foreground_count = condition_count - 1
        condition_dtype = self.condition_projector.weight.dtype
        # No in-place operation is performed on C0.  This projected tensor is
        # a read-only query used by every Group hypothesis and every stage.
        foreground_condition_state = self.condition_projector(
            self.condition_norm(
                cond_embeddings[:, :foreground_count].to(condition_dtype)
            )
        )
        foreground_valid = condition_valid[:, :foreground_count]

        classified_stages = []
        relation_states = []
        previous_state = None
        previous_group_features = None
        previous_query_to_group = None
        previous_group_valid_mask = None

        for expected_stage, (stage_output, anchor) in enumerate(
            zip(stage_outputs, anchor_logits)
        ):
            stage_index = int(getattr(stage_output, "stage_index"))
            if stage_index != expected_stage:
                raise AssertionError(
                    "topology relation stages must be ordered; "
                    f"position {expected_stage} stores st{stage_index}"
                )
            group_features = getattr(stage_output, "group_features")
            group_valid_mask = getattr(stage_output, "group_valid_mask")
            query_to_group = getattr(stage_output, "query_to_group")
            num_queries = group_features.shape[1]
            if group_features.shape[:2] != (
                batch_size,
                num_queries,
            ):
                raise ValueError("stage and Condition batches do not match")
            if anchor.shape != (
                batch_size,
                num_queries,
                condition_count,
            ):
                raise ValueError(
                    "anchor logits must be [B,Q,Nc] at every stage"
                )
            _require_finite("anchor_logits", anchor)

            if previous_state is None:
                transported_state = group_features.new_zeros(
                    batch_size,
                    num_queries,
                    foreground_count,
                    self.relation_dim,
                    dtype=self.relation_logit.weight.dtype,
                )
                transported_previous_group = None
            else:
                explicit_transport = self._resolve_explicit_transport(
                    stage_output,
                    group_valid_mask,
                    previous_group_valid_mask,
                )
                if explicit_transport is not None:
                    transported_state = self._transport_dense(
                        explicit_transport,
                        previous_state,
                    )
                    transported_previous_group = self._transport_dense(
                        explicit_transport,
                        previous_group_features,
                    )
                else:
                    transported_state = self._transport_by_membership(
                        previous_state,
                        previous_query_to_group,
                        query_to_group,
                        group_valid_mask,
                    )
                    transported_previous_group = self._transport_by_membership(
                        previous_group_features,
                        previous_query_to_group,
                        query_to_group,
                        group_valid_mask,
                    )

            evidence_tokens, current_group_features = self._stage_evidence(
                stage_output,
                transported_previous_group,
            )
            pair_valid = (
                group_valid_mask[:, :, None]
                & foreground_valid[:, None, :]
            )

            state_chunks = []
            for start in range(0, foreground_count, self.condition_chunk_size):
                end = min(
                    foreground_count,
                    start + self.condition_chunk_size,
                )
                condition_chunk = foreground_condition_state[:, start:end]
                observation = self._cross_attend(
                    condition_chunk,
                    evidence_tokens,
                )
                state_chunks.append(
                    self._update_relation_state(
                        transported_state[:, :, start:end],
                        condition_chunk,
                        observation,
                        pair_valid[:, :, start:end],
                    )
                )
            if state_chunks:
                current_state = torch.cat(state_chunks, dim=2)
                relation_residual = self.relation_logit_bound * torch.tanh(
                    self.relation_logit(current_state).squeeze(-1)
                )
                foreground_logits = (
                    anchor[:, :, :foreground_count]
                    + relation_residual.to(anchor.dtype)
                )
                foreground_logits = foreground_logits.masked_fill(
                    ~pair_valid,
                    _finite_negative(anchor),
                )
            else:
                current_state = transported_state
                foreground_logits = anchor[:, :, :0]

            # The background column is never modified by relation evidence.
            stage_logits = torch.cat(
                (foreground_logits, anchor[:, :, -1:]),
                dim=-1,
            )
            _require_finite("topology_relation_logits", stage_logits)
            classified_stages.append(stage_logits)
            relation_states.append(current_state)

            previous_state = current_state
            previous_group_features = current_group_features
            previous_query_to_group = query_to_group
            previous_group_valid_mask = group_valid_mask

        return TopologyConditionRelationOutput(
            stage_logits=tuple(classified_stages),
            relation_states=tuple(relation_states),
        )


__all__ = [
    "TopologyConditionRelationBank",
    "TopologyConditionRelationOutput",
]
