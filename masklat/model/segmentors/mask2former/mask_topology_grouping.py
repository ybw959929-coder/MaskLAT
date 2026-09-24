"""Mask-only topology grouping helpers for offline MaskLAT diagnostics.

The grouping functions in this module deliberately accept only predicted
binary masks (or their pairwise IoU matrix). Ground truth and classification
scores are not inputs to grouping or medoid selection.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .spatial_validity import masked_normalized_resize
except (ImportError, ValueError):  # Standalone path-loading in unit tests.
    from spatial_validity import masked_normalized_resize  # type: ignore


MaskSize = Union[int, Tuple[int, int], List[int]]


class GroupingResult(NamedTuple):
    """A deterministic, complete partition of the query IDs."""

    query_to_group: Tensor
    groups: List[List[int]]
    group_sizes: List[int]


class GroupMedoidResult(NamedTuple):
    """Medoid IDs and optional tensors gathered at those IDs."""

    group_medoid_query_ids: List[int]
    group_medoid_masks: Optional[Tensor]
    group_medoid_class_logits: Optional[Tensor]


class BridgeDiagnostics(NamedTuple):
    """Per-group connected-component bridge diagnostics.

    ``medoid_min_ious`` and ``group_min_pair_ious`` are aligned with
    ``non_singleton_group_ids``. ``bridge_group_flags`` is aligned with all
    input groups.
    """

    bridge_group_flags: List[bool]
    bridge_group_ids: List[int]
    non_singleton_group_ids: List[int]
    medoid_min_ious: List[float]
    group_min_pair_ious: List[float]

    @property
    def bridge_group_ratio(self) -> float:
        """Return the bridge ratio over non-singleton groups without NaNs."""

        denominator = len(self.non_singleton_group_ids)
        return len(self.bridge_group_ids) / denominator if denominator else 0.0


class TopScoreSelection(NamedTuple):
    """Result of selecting the top-score group and its top-score query."""

    selected_group_id: int
    selected_query_id: int
    group_scores: Tensor
    group_top_query_ids: List[int]
    selected_query_mask: Optional[Tensor]


def _normalize_mask_size(mask_size: MaskSize) -> Tuple[int, int]:
    if isinstance(mask_size, int):
        size = (mask_size, mask_size)
    elif isinstance(mask_size, (tuple, list)) and len(mask_size) == 2:
        size = (int(mask_size[0]), int(mask_size[1]))
    else:
        raise TypeError("mask_size must be a positive integer or a two-element sequence")

    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"mask_size must be positive, got {size}")
    return size


def _as_finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def resize_and_binarize_masks(
    query_mask_logits: Tensor,
    mask_size: MaskSize = 64,
    mask_threshold: float = 0.5,
    *,
    source_valid_mask: Optional[Tensor] = None,
    target_valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Resize ``[Q, H, W]`` logits and return boolean ``[Q, h, w]`` masks.

    Bilinear interpolation uses ``align_corners=False``.  If SAM validity
    masks are supplied, masked-normalized interpolation prevents padding
    logits from affecting the visible boundary.  Topology construction uses
    the same inclusive ``probability >= mask_threshold`` contract as the
    online decoder; formal RefSeg post-processing keeps its separate strict
    comparison.
    """

    if not isinstance(query_mask_logits, Tensor):
        raise TypeError("query_mask_logits must be a torch.Tensor")
    if query_mask_logits.ndim != 3:
        raise ValueError(
            "query_mask_logits must have shape [Q, H, W], "
            f"got {tuple(query_mask_logits.shape)}"
        )
    if query_mask_logits.shape[0] <= 0:
        raise ValueError("query_mask_logits must contain at least one query")
    if query_mask_logits.shape[1] <= 0 or query_mask_logits.shape[2] <= 0:
        raise ValueError("query_mask_logits must have non-empty spatial dimensions")
    if query_mask_logits.is_floating_point() and not bool(torch.isfinite(query_mask_logits).all()):
        raise ValueError("query_mask_logits contains NaN or Inf")

    size = _normalize_mask_size(mask_size)
    threshold = _as_finite_float(mask_threshold, "mask_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"mask_threshold must be in [0, 1], got {threshold}")

    if (source_valid_mask is None) != (target_valid_mask is None):
        raise ValueError(
            "source_valid_mask and target_valid_mask must be supplied together"
        )
    resolved_target_valid = None
    if source_valid_mask is None:
        resized_logits = F.interpolate(
            query_mask_logits.to(dtype=torch.float32).unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    else:
        expected_source = tuple(query_mask_logits.shape[-2:])
        if source_valid_mask.shape != expected_source:
            raise ValueError(
                "source_valid_mask must match the native Query grid: "
                f"{tuple(source_valid_mask.shape)} != {expected_source}"
            )
        if target_valid_mask.shape != size:
            raise ValueError(
                "target_valid_mask must match mask_size: "
                f"{tuple(target_valid_mask.shape)} != {size}"
            )
        for name, valid_mask in (
            ("source_valid_mask", source_valid_mask),
            ("target_valid_mask", target_valid_mask),
        ):
            if valid_mask.device != query_mask_logits.device:
                raise ValueError(
                    f"{name} and query_mask_logits must share one device"
                )
            if valid_mask.dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")
        resized, _ = masked_normalized_resize(
            query_mask_logits.to(dtype=torch.float32).unsqueeze(0),
            source_valid_mask[None, None],
            size,
            target_valid_mask=target_valid_mask[None, None],
        )
        resized_logits = resized.squeeze(0)
        resolved_target_valid = target_valid_mask
    probabilities = resized_logits.sigmoid()
    binary_masks = probabilities.ge(threshold)
    if resolved_target_valid is not None:
        binary_masks = binary_masks & resolved_target_valid.unsqueeze(0)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("mask preprocessing produced NaN or Inf")
    return binary_masks


def compute_pairwise_mask_iou(query_binary_masks: Tensor) -> Tensor:
    """Compute vectorized float32 IoU for binary masks shaped ``[Q, H, W]``.

    Empty-mask pairs have IoU zero off the diagonal. Every diagonal value is
    forced to one, including the diagonal of an empty mask.
    """

    if not isinstance(query_binary_masks, Tensor):
        raise TypeError("query_binary_masks must be a torch.Tensor")
    if query_binary_masks.ndim != 3:
        raise ValueError(
            "query_binary_masks must have shape [Q, H, W], "
            f"got {tuple(query_binary_masks.shape)}"
        )
    if query_binary_masks.shape[0] <= 0:
        raise ValueError("query_binary_masks must contain at least one query")
    if query_binary_masks.shape[1] <= 0 or query_binary_masks.shape[2] <= 0:
        raise ValueError("query_binary_masks must have non-empty spatial dimensions")
    if query_binary_masks.is_floating_point() and not bool(torch.isfinite(query_binary_masks).all()):
        raise ValueError("query_binary_masks contains NaN or Inf")

    masks = query_binary_masks.ne(0).reshape(query_binary_masks.shape[0], -1).to(torch.float32)
    intersections = masks @ masks.transpose(0, 1)
    areas = masks.sum(dim=1)
    unions = areas[:, None] + areas[None, :] - intersections
    pairwise_iou = torch.where(
        unions > 0,
        intersections / unions.clamp_min(1.0),
        torch.zeros_like(intersections),
    )
    pairwise_iou.fill_diagonal_(1.0)

    if pairwise_iou.dtype != torch.float32:
        raise RuntimeError(f"pairwise IoU must be float32, got {pairwise_iou.dtype}")
    if not bool(torch.isfinite(pairwise_iou).all()):
        raise RuntimeError("pairwise IoU contains NaN or Inf")
    return pairwise_iou


def _validate_pairwise_iou(pairwise_iou: Tensor) -> int:
    if not isinstance(pairwise_iou, Tensor):
        raise TypeError("pairwise_iou must be a torch.Tensor")
    if pairwise_iou.ndim != 2 or pairwise_iou.shape[0] != pairwise_iou.shape[1]:
        raise ValueError(
            "pairwise_iou must be square with shape [Q, Q], "
            f"got {tuple(pairwise_iou.shape)}"
        )
    if not pairwise_iou.is_floating_point():
        raise TypeError(f"pairwise_iou must have floating dtype, got {pairwise_iou.dtype}")
    if not bool(torch.isfinite(pairwise_iou).all()):
        raise ValueError("pairwise_iou contains NaN or Inf")
    if not bool(torch.allclose(pairwise_iou, pairwise_iou.transpose(0, 1), atol=1e-6, rtol=0.0)):
        raise ValueError("pairwise_iou must be symmetric")
    if pairwise_iou.numel() and (
        bool((pairwise_iou < -1e-6).any()) or bool((pairwise_iou > 1.0 + 1e-6).any())
    ):
        raise ValueError("pairwise_iou values must be in [0, 1]")
    return int(pairwise_iou.shape[0])


def connected_component_roots_from_adjacency(adjacency: Tensor) -> Tensor:
    """Return minimum-query roots for undirected connected components.

    Args:
        adjacency: Boolean-like ``[Q,Q]`` or batched ``[B,Q,Q]`` adjacency.

    Returns:
        Integer physical-root IDs with shape ``[Q]`` or ``[B,Q]``. The
        implementation is device-native and deterministic. Offline
        diagnostics and the trainable decoder both call this function so
        their connected-component rule cannot drift apart.
    """

    if not isinstance(adjacency, Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    if adjacency.ndim not in (2, 3) or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError(
            "adjacency must have shape [Q,Q] or [B,Q,Q], "
            f"got {tuple(adjacency.shape)}"
        )
    if adjacency.shape[-1] <= 0:
        raise ValueError("adjacency must contain at least one query")
    if adjacency.is_floating_point() and not bool(torch.isfinite(adjacency).all()):
        raise ValueError("adjacency contains NaN or Inf")

    unbatched = adjacency.ndim == 2
    graph = adjacency.unsqueeze(0) if unbatched else adjacency
    graph = graph.to(dtype=torch.bool)
    graph = graph | graph.transpose(-1, -2)
    graph = graph.clone()

    batch_size, num_queries, _ = graph.shape
    diagonal = torch.arange(num_queries, device=graph.device)
    graph[:, diagonal, diagonal] = True
    labels = diagonal.unsqueeze(0).expand(batch_size, -1).clone()
    sentinel = torch.full(
        (1,),
        num_queries,
        dtype=torch.long,
        device=graph.device,
    )

    # Minimum-label propagation is equivalent to deterministic union-find,
    # while remaining on the input device during decoder forward.
    for _ in range(num_queries):
        candidates = labels[:, None, :].expand(-1, num_queries, -1)
        next_labels = torch.where(graph, candidates, sentinel).amin(dim=-1)
        if torch.equal(next_labels, labels):
            break
        labels = next_labels

    return labels[0] if unbatched else labels


def _partition_error(
    groups: Sequence[Sequence[int]],
    num_queries: int,
    query_to_group: Optional[Tensor],
    require_deterministic_order: bool,
) -> Optional[str]:
    if num_queries < 0:
        return f"num_queries must be non-negative, got {num_queries}"
    if not isinstance(groups, Sequence):
        return "groups must be a sequence of groups"

    flattened: List[int] = []
    previous_min = -1
    for group_id, group in enumerate(groups):
        if not isinstance(group, Sequence):
            return f"group {group_id} is not a sequence"
        if not group:
            return f"group {group_id} is empty"

        normalized: List[int] = []
        for query_id in group:
            if isinstance(query_id, bool) or not isinstance(query_id, int):
                return f"group {group_id} contains a non-integer query ID: {query_id!r}"
            if not 0 <= query_id < num_queries:
                return f"group {group_id} contains out-of-range query ID {query_id}"
            normalized.append(query_id)

        if require_deterministic_order:
            if normalized != sorted(normalized):
                return f"group {group_id} members are not sorted"
            if normalized[0] <= previous_min:
                return "groups are not ordered by ascending minimum query ID"
            previous_min = normalized[0]
        flattened.extend(normalized)

    expected = list(range(num_queries))
    if sorted(flattened) != expected:
        return (
            "groups must contain every query ID exactly once; "
            f"expected {expected}, got {sorted(flattened)}"
        )

    if query_to_group is not None:
        if not isinstance(query_to_group, Tensor):
            return "query_to_group must be a torch.Tensor"
        if query_to_group.ndim != 1 or query_to_group.numel() != num_queries:
            return (
                f"query_to_group must have shape [{num_queries}], "
                f"got {tuple(query_to_group.shape)}"
            )
        if query_to_group.dtype == torch.bool or query_to_group.is_floating_point():
            return "query_to_group must have an integer dtype"
        assignments = query_to_group.detach().cpu().tolist()
        for group_id, group in enumerate(groups):
            for query_id in group:
                if assignments[query_id] != group_id:
                    return (
                        f"query_to_group[{query_id}]={assignments[query_id]}, "
                        f"but the query is in group {group_id}"
                    )
    return None


def validate_partition(
    groups: Sequence[Sequence[int]],
    num_queries: int,
    query_to_group: Optional[Tensor] = None,
    *,
    require_deterministic_order: bool = False,
    raise_on_error: bool = True,
) -> bool:
    """Validate that groups form a complete, mutually exclusive partition.

    Returns ``True`` for a valid partition. Invalid input raises ``ValueError``
    by default; set ``raise_on_error=False`` to receive ``False`` instead.

    For compatibility with the standalone analysis driver, the positional
    order ``(query_to_group, groups, num_queries)`` is accepted as well.
    """

    if isinstance(groups, Tensor):
        legacy_query_to_group = groups
        legacy_groups = num_queries
        legacy_num_queries = query_to_group
        if not isinstance(legacy_num_queries, int):
            raise TypeError(
                "legacy validate_partition order must be "
                "(query_to_group, groups, num_queries)"
            )
        groups = legacy_groups
        num_queries = legacy_num_queries
        query_to_group = legacy_query_to_group

    error = _partition_error(groups, int(num_queries), query_to_group, require_deterministic_order)
    if error is not None and raise_on_error:
        raise ValueError(error)
    return error is None


def is_valid_partition(
    groups: Sequence[Sequence[int]],
    num_queries: int,
    query_to_group: Optional[Tensor] = None,
) -> bool:
    """Non-raising shorthand for :func:`validate_partition`."""

    return validate_partition(
        groups,
        num_queries,
        query_to_group,
        raise_on_error=False,
    )


def connected_components_from_iou(pairwise_iou: Tensor, threshold: float) -> GroupingResult:
    """Build deterministic connected components from a pairwise IoU matrix."""

    num_queries = _validate_pairwise_iou(pairwise_iou)
    threshold = _as_finite_float(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    physical_roots = connected_component_roots_from_adjacency(
        pairwise_iou >= threshold
    )
    components: Dict[int, List[int]] = {}
    for query_id, root_id in enumerate(physical_roots.detach().cpu().tolist()):
        components.setdefault(int(root_id), []).append(query_id)
    groups = sorted((sorted(group) for group in components.values()), key=lambda group: group[0])

    query_to_group = torch.empty(num_queries, dtype=torch.long, device=pairwise_iou.device)
    for group_id, group in enumerate(groups):
        query_to_group[group] = group_id
    group_sizes = [len(group) for group in groups]

    validate_partition(
        groups,
        num_queries,
        query_to_group,
        require_deterministic_order=True,
        raise_on_error=True,
    )
    return GroupingResult(query_to_group, groups, group_sizes)


def group_queries_at_thresholds(
    pairwise_iou: Tensor,
    thresholds: Sequence[float] = (0.6, 0.7, 0.8),
) -> Dict[float, GroupingResult]:
    """Reuse one pairwise IoU matrix to group queries at several thresholds."""

    _validate_pairwise_iou(pairwise_iou)
    results: Dict[float, GroupingResult] = {}
    for raw_threshold in thresholds:
        threshold = _as_finite_float(raw_threshold, "threshold")
        if threshold in results:
            raise ValueError(f"duplicate grouping threshold: {threshold}")
        results[threshold] = connected_components_from_iou(pairwise_iou, threshold)
    return results


def _normalized_group(group: Sequence[int], num_queries: int, name: str = "group") -> List[int]:
    if not isinstance(group, Sequence) or not group:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized: List[int] = []
    for query_id in group:
        if isinstance(query_id, bool) or not isinstance(query_id, int):
            raise TypeError(f"{name} contains a non-integer query ID: {query_id!r}")
        if not 0 <= query_id < num_queries:
            raise ValueError(f"{name} contains out-of-range query ID {query_id}")
        normalized.append(query_id)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate query IDs")
    return sorted(normalized)


def select_group_medoid_ids(pairwise_iou: Tensor, groups: Sequence[Sequence[int]]) -> List[int]:
    """Select one mask-IoU medoid per group, breaking ties by query ID."""

    num_queries = _validate_pairwise_iou(pairwise_iou)
    validate_partition(groups, num_queries, raise_on_error=True)

    medoid_query_ids: List[int] = []
    for group_id, raw_group in enumerate(groups):
        group = _normalized_group(raw_group, num_queries, f"group {group_id}")
        if len(group) == 1:
            medoid_query_ids.append(group[0])
            continue

        indices = torch.tensor(group, dtype=torch.long, device=pairwise_iou.device)
        within_group = pairwise_iou.index_select(0, indices).index_select(1, indices)
        mean_to_other_members = (
            within_group.sum(dim=1) - torch.diagonal(within_group)
        ) / float(len(group) - 1)
        maximum = mean_to_other_members.max()
        tied_local_ids = torch.nonzero(mean_to_other_members == maximum, as_tuple=False).flatten()
        tied_query_ids = [group[int(local_id)] for local_id in tied_local_ids.detach().cpu().tolist()]
        medoid_query_ids.append(min(tied_query_ids))
    return medoid_query_ids


def select_group_medoids(
    pairwise_iou: Tensor,
    groups: Sequence[Sequence[int]],
    query_binary_masks: Optional[Tensor] = None,
    query_class_logits: Optional[Tensor] = None,
) -> Union[GroupMedoidResult, List[int]]:
    """Return medoid IDs plus optional tensors gathered at those IDs.

    The analysis driver historically calls this helper as
    ``select_group_medoids(groups, pairwise_iou)`` and expects only the list of
    IDs. That order remains supported. The canonical
    ``select_group_medoids(pairwise_iou, groups, masks, logits)`` form returns
    :class:`GroupMedoidResult`.
    """

    ids_only = not isinstance(pairwise_iou, Tensor) and isinstance(groups, Tensor)
    if ids_only:
        pairwise_iou, groups = groups, pairwise_iou

    num_queries = _validate_pairwise_iou(pairwise_iou)
    medoid_query_ids = select_group_medoid_ids(pairwise_iou, groups)
    if ids_only and query_binary_masks is None and query_class_logits is None:
        return medoid_query_ids

    def gather_optional(tensor: Optional[Tensor], name: str) -> Optional[Tensor]:
        if tensor is None:
            return None
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")
        if tensor.ndim < 1 or tensor.shape[0] != num_queries:
            raise ValueError(
                f"{name} must have leading query dimension {num_queries}, "
                f"got {tuple(tensor.shape)}"
            )
        indices = torch.tensor(medoid_query_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, indices)

    return GroupMedoidResult(
        medoid_query_ids,
        gather_optional(query_binary_masks, "query_binary_masks"),
        gather_optional(query_class_logits, "query_class_logits"),
    )


def group_min_pair_iou(
    pairwise_iou: Tensor,
    group: Sequence[int],
) -> Optional[float]:
    """Return a non-singleton group's minimum off-diagonal pairwise IoU."""

    num_queries = _validate_pairwise_iou(pairwise_iou)
    group = _normalized_group(group, num_queries)
    if len(group) == 1:
        return None
    indices = torch.tensor(group, dtype=torch.long, device=pairwise_iou.device)
    within_group = pairwise_iou.index_select(0, indices).index_select(1, indices)
    upper_triangle = torch.triu_indices(
        len(group),
        len(group),
        offset=1,
        device=pairwise_iou.device,
    )
    return float(within_group[upper_triangle[0], upper_triangle[1]].min().item())


def medoid_min_iou(
    pairwise_iou: Tensor,
    group: Sequence[int],
    medoid_query_id: int,
) -> Optional[float]:
    """Return a medoid's minimum IoU to the other members of its group."""

    num_queries = _validate_pairwise_iou(pairwise_iou)
    group = _normalized_group(group, num_queries)
    if medoid_query_id not in group:
        raise ValueError(f"medoid query ID {medoid_query_id} is not in its group")
    if len(group) == 1:
        return None
    other_query_ids = [query_id for query_id in group if query_id != medoid_query_id]
    indices = torch.tensor(other_query_ids, dtype=torch.long, device=pairwise_iou.device)
    return float(pairwise_iou[medoid_query_id].index_select(0, indices).min().item())


def is_bridge_group(
    pairwise_iou: Tensor,
    group: Sequence[int],
    threshold: float,
) -> bool:
    """Return whether a connected component contains a below-threshold pair."""

    threshold = _as_finite_float(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    minimum = group_min_pair_iou(pairwise_iou, group)
    return minimum is not None and minimum < threshold


def compute_group_bridge_diagnostics(
    pairwise_iou: Tensor,
    groups: Sequence[Sequence[int]],
    threshold: float,
    medoid_query_ids: Optional[Sequence[int]] = None,
) -> BridgeDiagnostics:
    """Compute bridge flags and minimum-IoU values for all non-singletons.

    Also accepts the analysis driver's positional order
    ``(groups, medoid_query_ids, pairwise_iou, threshold)``.
    """

    if not isinstance(pairwise_iou, Tensor) and isinstance(threshold, Tensor):
        legacy_groups = pairwise_iou
        legacy_medoid_query_ids = groups
        legacy_pairwise_iou = threshold
        legacy_threshold = medoid_query_ids
        pairwise_iou = legacy_pairwise_iou
        groups = legacy_groups
        threshold = legacy_threshold
        medoid_query_ids = legacy_medoid_query_ids

    num_queries = _validate_pairwise_iou(pairwise_iou)
    validate_partition(groups, num_queries, raise_on_error=True)
    threshold = _as_finite_float(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    if medoid_query_ids is None:
        medoid_query_ids = select_group_medoid_ids(pairwise_iou, groups)
    else:
        medoid_query_ids = list(medoid_query_ids)
        if len(medoid_query_ids) != len(groups):
            raise ValueError(
                "medoid_query_ids must contain one ID per group, "
                f"got {len(medoid_query_ids)} IDs for {len(groups)} groups"
            )

    bridge_group_flags: List[bool] = []
    bridge_group_ids: List[int] = []
    non_singleton_group_ids: List[int] = []
    medoid_min_ious: List[float] = []
    group_min_pair_ious: List[float] = []

    for group_id, raw_group in enumerate(groups):
        group = _normalized_group(raw_group, num_queries, f"group {group_id}")
        minimum_pair_iou = group_min_pair_iou(pairwise_iou, group)
        bridge = minimum_pair_iou is not None and minimum_pair_iou < threshold
        bridge_group_flags.append(bridge)
        if bridge:
            bridge_group_ids.append(group_id)
        if minimum_pair_iou is None:
            continue

        minimum_medoid_iou = medoid_min_iou(
            pairwise_iou,
            group,
            int(medoid_query_ids[group_id]),
        )
        if minimum_medoid_iou is None:
            raise RuntimeError("non-singleton group unexpectedly has no medoid minimum IoU")
        non_singleton_group_ids.append(group_id)
        medoid_min_ious.append(minimum_medoid_iou)
        group_min_pair_ious.append(minimum_pair_iou)

    return BridgeDiagnostics(
        bridge_group_flags,
        bridge_group_ids,
        non_singleton_group_ids,
        medoid_min_ious,
        group_min_pair_ious,
    )


def _flatten_query_scores(query_scores: Tensor) -> Tensor:
    if not isinstance(query_scores, Tensor):
        raise TypeError("query_scores must be a torch.Tensor")
    if query_scores.ndim == 2 and query_scores.shape[1] == 1:
        query_scores = query_scores[:, 0]
    if query_scores.ndim != 1 or query_scores.numel() <= 0:
        raise ValueError(
            "query_scores must have shape [Q] or [Q, 1], "
            f"got {tuple(query_scores.shape)}"
        )
    if not query_scores.is_floating_point():
        query_scores = query_scores.to(torch.float32)
    if not bool(torch.isfinite(query_scores).all()):
        raise ValueError("query_scores contains NaN or Inf")
    return query_scores


def select_group_top_score_query(
    query_scores: Tensor,
    groups: Sequence[Sequence[int]],
    query_masks: Optional[Tensor] = None,
) -> TopScoreSelection:
    """Select the highest-score group and that group's highest-score query.

    A group's score is the maximum original query score among its members.
    Ties are resolved by the smallest top-scoring query ID, which exactly
    preserves ``torch.argmax(query_scores)`` semantics across group boundaries.
    """

    query_scores = _flatten_query_scores(query_scores)
    num_queries = int(query_scores.numel())
    validate_partition(groups, num_queries, raise_on_error=True)

    group_scores: List[Tensor] = []
    group_top_query_ids: List[int] = []
    for group_id, raw_group in enumerate(groups):
        group = _normalized_group(raw_group, num_queries, f"group {group_id}")
        indices = torch.tensor(group, dtype=torch.long, device=query_scores.device)
        member_scores = query_scores.index_select(0, indices)
        local_top_id = int(torch.argmax(member_scores).item())
        group_scores.append(member_scores[local_top_id])
        group_top_query_ids.append(group[local_top_id])

    stacked_group_scores = torch.stack(group_scores)
    maximum_group_score = stacked_group_scores.max()
    tied_group_ids = torch.nonzero(
        stacked_group_scores == maximum_group_score,
        as_tuple=False,
    ).flatten().detach().cpu().tolist()
    selected_group_id = min(tied_group_ids, key=lambda group_id: group_top_query_ids[group_id])
    selected_query_id = group_top_query_ids[selected_group_id]

    global_selected_query_id = int(torch.argmax(query_scores).item())
    if selected_query_id != global_selected_query_id:
        raise RuntimeError(
            "group top-score selection did not preserve global query selection: "
            f"group result {selected_query_id}, global result {global_selected_query_id}"
        )

    selected_query_mask: Optional[Tensor] = None
    if query_masks is not None:
        if not isinstance(query_masks, Tensor):
            raise TypeError("query_masks must be a torch.Tensor or None")
        if query_masks.ndim < 1 or query_masks.shape[0] != num_queries:
            raise ValueError(
                f"query_masks must have leading query dimension {num_queries}, "
                f"got {tuple(query_masks.shape)}"
            )
        selected_query_mask = query_masks[selected_query_id]

    return TopScoreSelection(
        selected_group_id,
        selected_query_id,
        stacked_group_scores,
        group_top_query_ids,
        selected_query_mask,
    )


def selected_group_top_score_identity(
    query_scores: Tensor,
    groups: Sequence[Sequence[int]],
    *,
    selected_query_id: Optional[int] = None,
    query_masks: Optional[Tensor] = None,
    selected_query_mask: Optional[Tensor] = None,
) -> bool:
    """Check group top-score selection against an original selected query."""

    flattened_scores = _flatten_query_scores(query_scores)
    expected_query_id = (
        int(torch.argmax(flattened_scores).item())
        if selected_query_id is None
        else int(selected_query_id)
    )
    selection = select_group_top_score_query(flattened_scores, groups, query_masks)
    if selection.selected_query_id != expected_query_id:
        return False
    if selected_query_mask is not None:
        if selection.selected_query_mask is None:
            raise ValueError("query_masks is required when selected_query_mask is provided")
        return bool(torch.equal(selection.selected_query_mask, selected_query_mask))
    return True


def select_top_score_group_query(
    groups: Sequence[Sequence[int]],
    query_scores: Tensor,
) -> Tuple[int, int]:
    """Compatibility wrapper returning ``(group_id, query_id)``."""

    selection = select_group_top_score_query(query_scores, groups)
    return selection.selected_group_id, selection.selected_query_id


# Readable aliases for callers that prefer query/topology-specific names.
preprocess_query_masks = resize_and_binarize_masks
resize_and_binarize_query_masks = resize_and_binarize_masks
compute_pairwise_iou = compute_pairwise_mask_iou
pairwise_mask_iou = compute_pairwise_mask_iou
group_queries = connected_components_from_iou
build_topology_groups = connected_components_from_iou
compute_group_medoids = select_group_medoids
analyze_bridge_groups = compute_group_bridge_diagnostics
selected_group_top_score_query = select_group_top_score_query


__all__ = [
    "BridgeDiagnostics",
    "GroupingResult",
    "GroupMedoidResult",
    "TopScoreSelection",
    "analyze_bridge_groups",
    "build_topology_groups",
    "compute_group_bridge_diagnostics",
    "compute_group_medoids",
    "compute_pairwise_iou",
    "compute_pairwise_mask_iou",
    "connected_component_roots_from_adjacency",
    "connected_components_from_iou",
    "group_min_pair_iou",
    "group_queries",
    "group_queries_at_thresholds",
    "is_bridge_group",
    "is_valid_partition",
    "medoid_min_iou",
    "pairwise_mask_iou",
    "preprocess_query_masks",
    "resize_and_binarize_masks",
    "resize_and_binarize_query_masks",
    "select_group_medoid_ids",
    "select_group_medoids",
    "select_group_top_score_query",
    "select_top_score_group_query",
    "selected_group_top_score_identity",
    "selected_group_top_score_query",
    "validate_partition",
]
