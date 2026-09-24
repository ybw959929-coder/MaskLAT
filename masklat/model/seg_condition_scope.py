"""Source-local SEG/condition data contracts for the staged MaskLAT path.

This module is deliberately independent from :mod:`masklat.model.masklat`.  It
contains only tensor/list bookkeeping and validation, so a future model path
can adopt the contract without changing the historical training or evaluation
path.

Terminology
-----------
``source``
    One item in the original dataloader batch (the ``B`` dimension).
``SEG row``
    One effective segmentation request after expanding a source by its
    ``<SEG>`` occurrences.  Rows are always source-major and occurrence-major.
``local condition``
    A ``<COND>`` occurrence belonging to the same source/SEG scope.  Conditions
    from another source are never placed in a row, so they cannot accidentally
    become classification negatives.

The supported task contracts are intentionally strict:

* ``refseg``/``reaseg``: source ``b`` must have ``C_b == S_b > 0``.  SEG
  occurrence ``j`` receives condition occurrence ``j`` plus background.  Its
  target is also chosen by occurrence ``j``; legacy source-level class labels
  are validated for shape/range but are not used to choose the occurrence.
* ``genseg``/``ovseg``/``gcgseg``/``vgdseg``/``intseg``: source ``b`` must
  have one SEG and at least one condition.  The SEG receives all of that
  source's conditions plus background; target labels remain source-local and
  may repeat.
* ``imgconv``: no SEG, condition, or segmentation target is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor


ORDINAL_SEGMENTATION_TASKS = frozenset({"refseg", "reaseg"})
SINGLE_SEGMENTATION_TASKS = frozenset(
    {"genseg", "ovseg", "gcgseg", "vgdseg", "intseg"}
)
NO_SEGMENTATION_TASKS = frozenset({"imgconv"})
SUPPORTED_TASKS = (
    ORDINAL_SEGMENTATION_TASKS
    | SINGLE_SEGMENTATION_TASKS
    | NO_SEGMENTATION_TASKS
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _normalise_task_name(task_name: str) -> str:
    if not isinstance(task_name, str) or not task_name.strip():
        raise TypeError("task_name must be a non-empty string")
    normalised = task_name.strip().lower()
    if normalised not in SUPPORTED_TASKS:
        raise ValueError(
            f"unsupported SEG/condition task {task_name!r}; "
            f"supported tasks are {sorted(SUPPORTED_TASKS)}"
        )
    return normalised


def _normalise_counts(values: Sequence[int], *, name: str) -> Tuple[int, ...]:
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {tuple(values.shape)}")
        values = values.detach().cpu().tolist()
    else:
        values = tuple(values)
    try:
        counts = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of non-negative integers") from exc
    for source_index, (raw_value, count) in enumerate(zip(values, counts)):
        if not isinstance(raw_value, Integral) or isinstance(raw_value, bool) or count < 0:
            raise ValueError(
                f"{name}[{source_index}] must be a non-negative integer, "
                f"got {raw_value!r}"
            )
    return counts


def _require_rank_two_embeddings(
    embeddings: Sequence[Tensor],
    *,
    name: str,
) -> Tuple[Tensor, ...]:
    if embeddings is None:
        raise TypeError(f"{name} cannot be None")
    tensors = tuple(embeddings)
    for source_index, tensor in enumerate(tensors):
        if not isinstance(tensor, Tensor) or tensor.ndim != 2:
            shape = tuple(tensor.shape) if isinstance(tensor, Tensor) else None
            raise ValueError(
                f"{name}[{source_index}] must be a [N, D] tensor, got {shape}"
            )
        if not tensor.is_floating_point():
            raise TypeError(
                f"{name}[{source_index}] must have floating dtype, got {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"{name}[{source_index}] contains NaN or Inf"
            )
    return tensors


def extract_indexed_embeddings(
    hidden_states: Tensor,
    index_ids: Tensor,
    *,
    field_name: str = "indexed embeddings",
) -> Tuple[Tensor, ...]:
    """Mean-pool hidden states by non-negative occurrence ID without losing B.

    Args:
        hidden_states: Tensor shaped ``[B, L, D]``.
        index_ids: Integer tensor shaped ``[B, L]``.  ``-1`` means that a
            token does not belong to this field; non-negative IDs must be the
            contiguous occurrence ordinals ``0..N_b-1`` for each source.
        field_name: Context included in validation errors.

    Returns:
        A tuple of length exactly ``B``.  Entry ``b`` is shaped ``[N_b, D]``;
        a source with no occurrence is preserved as ``[0, D]`` rather than
        being dropped.
    """

    if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
        shape = tuple(hidden_states.shape) if isinstance(hidden_states, Tensor) else None
        raise ValueError(f"hidden_states must be [B, L, D], got {shape}")
    if not isinstance(index_ids, Tensor) or index_ids.ndim != 2:
        shape = tuple(index_ids.shape) if isinstance(index_ids, Tensor) else None
        raise ValueError(f"index_ids must be [B, L], got {shape}")
    if tuple(index_ids.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(
            "hidden_states/index_ids B,L mismatch: "
            f"{tuple(hidden_states.shape[:2])} != {tuple(index_ids.shape)}"
        )
    if not hidden_states.is_floating_point():
        raise TypeError(
            f"hidden_states for {field_name} must have floating dtype, "
            f"got {hidden_states.dtype}"
        )
    if index_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"index_ids for {field_name} must have integer dtype, got {index_ids.dtype}"
        )
    if not torch.isfinite(hidden_states).all():
        raise FloatingPointError(f"hidden_states for {field_name} contain NaN or Inf")

    pooled_sources = []
    hidden_width = int(hidden_states.shape[-1])
    for source_index in range(int(hidden_states.shape[0])):
        source_hidden = hidden_states[source_index]
        source_ids = index_ids[source_index]
        if source_ids.device != source_hidden.device:
            source_ids = source_ids.to(device=source_hidden.device)
        if bool((source_ids < -1).any()):
            minimum = int(source_ids.min().item())
            raise ValueError(
                f"{field_name} IDs for source {source_index} must be >= -1, "
                f"got minimum {minimum}"
            )

        occurrence_ids = torch.unique(source_ids[source_ids >= 0], sorted=True)
        if occurrence_ids.numel() == 0:
            pooled_sources.append(source_hidden.new_empty((0, hidden_width)))
            continue

        expected = torch.arange(
            occurrence_ids.numel(),
            dtype=occurrence_ids.dtype,
            device=occurrence_ids.device,
        )
        if not torch.equal(occurrence_ids, expected):
            raise ValueError(
                f"{field_name} IDs for source {source_index} must be contiguous "
                f"occurrence ordinals 0..N-1, got {occurrence_ids.tolist()}"
            )
        pooled_sources.append(
            torch.stack(
                [source_hidden[source_ids == occurrence_id].mean(dim=0) for occurrence_id in occurrence_ids],
                dim=0,
            )
        )

    return tuple(pooled_sources)


@dataclass(frozen=True)
class SegConditionScope:
    """Deterministic source-to-SEG-to-condition mapping metadata."""

    task_name: str
    condition_counts: Tuple[int, ...]
    segment_counts: Tuple[int, ...]
    seg_to_sample: Tensor
    seg_to_local_index: Tensor
    seg_to_condition_indices: Tuple[Tensor, ...]

    @property
    def num_sources(self) -> int:
        return len(self.condition_counts)

    @property
    def num_segments(self) -> int:
        return int(self.seg_to_sample.numel())

    @property
    def local_condition_counts(self) -> Tuple[int, ...]:
        return tuple(int(indices.numel()) for indices in self.seg_to_condition_indices)


def build_seg_condition_scope(
    task_name: str,
    condition_counts: Sequence[int],
    segment_counts: Sequence[int],
    *,
    device: Optional[torch.device] = None,
) -> SegConditionScope:
    """Build source-major SEG rows and each row's source-local Cond indices."""

    task = _normalise_task_name(task_name)
    cond_counts = _normalise_counts(condition_counts, name="condition_counts")
    seg_counts = _normalise_counts(segment_counts, name="segment_counts")
    if len(cond_counts) != len(seg_counts):
        raise ValueError(
            "condition_counts and segment_counts must preserve the same B: "
            f"{len(cond_counts)} != {len(seg_counts)}"
        )

    for source_index, (condition_count, segment_count) in enumerate(
        zip(cond_counts, seg_counts)
    ):
        if task in ORDINAL_SEGMENTATION_TASKS:
            if condition_count <= 0 or segment_count <= 0 or condition_count != segment_count:
                raise ValueError(
                    f"task {task!r} source {source_index} requires C_b == S_b > 0; "
                    f"got C_b={condition_count}, S_b={segment_count}"
                )
        elif task in SINGLE_SEGMENTATION_TASKS:
            if condition_count <= 0 or segment_count != 1:
                raise ValueError(
                    f"task {task!r} source {source_index} requires C_b > 0 and S_b == 1; "
                    f"got C_b={condition_count}, S_b={segment_count}"
                )
        elif condition_count != 0 or segment_count != 0:
            raise ValueError(
                f"task {task!r} source {source_index} permits no Cond or SEG; "
                f"got C_b={condition_count}, S_b={segment_count}"
            )

    mapping_device = torch.device("cpu") if device is None else torch.device(device)
    seg_to_sample = []
    seg_to_local_index = []
    seg_to_condition_indices = []
    for source_index, (condition_count, segment_count) in enumerate(
        zip(cond_counts, seg_counts)
    ):
        for local_seg_index in range(segment_count):
            seg_to_sample.append(source_index)
            seg_to_local_index.append(local_seg_index)
            if task in ORDINAL_SEGMENTATION_TASKS:
                local_condition_indices = [local_seg_index]
            else:
                local_condition_indices = list(range(condition_count))
            seg_to_condition_indices.append(
                torch.tensor(
                    local_condition_indices,
                    dtype=torch.long,
                    device=mapping_device,
                )
            )

    return SegConditionScope(
        task_name=task,
        condition_counts=cond_counts,
        segment_counts=seg_counts,
        seg_to_sample=torch.tensor(
            seg_to_sample,
            dtype=torch.long,
            device=mapping_device,
        ),
        seg_to_local_index=torch.tensor(
            seg_to_local_index,
            dtype=torch.long,
            device=mapping_device,
        ),
        seg_to_condition_indices=tuple(seg_to_condition_indices),
    )


@dataclass(frozen=True)
class PackedSegConditions:
    """Padded SEG-local foreground conditions with a final BG column."""

    scope: SegConditionScope
    embeddings: Tensor
    valid_mask: Tensor
    foreground_counts: Tensor


def pack_seg_local_conditions(
    condition_embeddings: Sequence[Tensor],
    scope: SegConditionScope,
    background_embedding: Tensor,
) -> PackedSegConditions:
    """Pack each SEG row's local Conds and one learned background embedding.

    The background embedding is always stored in the final physical column.
    Any columns between a row's foreground conditions and that final column
    are padding and are marked ``False`` by ``valid_mask``.
    """

    sources = _require_rank_two_embeddings(
        condition_embeddings,
        name="condition_embeddings",
    )
    if len(sources) != scope.num_sources:
        raise ValueError(
            "condition_embeddings must preserve the source batch B: "
            f"{len(sources)} != {scope.num_sources}"
        )
    if not isinstance(background_embedding, Tensor):
        raise TypeError("background_embedding must be a tensor")
    if not background_embedding.is_floating_point():
        raise TypeError(
            "background_embedding must have floating dtype, got "
            f"{background_embedding.dtype}"
        )
    if background_embedding.ndim == 2 and background_embedding.shape[0] == 1:
        background = background_embedding[0]
    elif background_embedding.ndim == 1:
        background = background_embedding
    else:
        raise ValueError(
            "background_embedding must be [D] or [1, D], got "
            f"{tuple(background_embedding.shape)}"
        )
    hidden_width = int(background.shape[0])
    # MaskLAT checkpoints initialize the segmentor-side learned BG embedding in
    # the segmentor dtype (commonly BF16), while an FP16 S3 autocast produces
    # Cond/SEG states in FP16.  They are semantically one local classifier
    # table, so use the Cond table as the runtime compute dtype and cast BG at
    # this boundary.  ``Tensor.to`` remains differentiable: gradients flow
    # back to the original learned BG parameter in its storage dtype.
    if sources:
        target_device = sources[0].device
        target_dtype = sources[0].dtype
        for source_index, source in enumerate(sources[1:], start=1):
            if source.device != target_device or source.dtype != target_dtype:
                raise ValueError(
                    "all condition_embeddings must share device/dtype; "
                    f"condition_embeddings[0]=({target_device}, "
                    f"{target_dtype}), condition_embeddings[{source_index}]="
                    f"({source.device}, {source.dtype})"
                )
        background = background.to(
            device=target_device,
            dtype=target_dtype,
        )
    if not torch.isfinite(background).all():
        raise FloatingPointError(
            "background_embedding contains NaN or Inf after runtime casting"
        )

    for source_index, (source, expected_count) in enumerate(
        zip(sources, scope.condition_counts)
    ):
        if tuple(source.shape) != (expected_count, hidden_width):
            raise ValueError(
                f"condition_embeddings[{source_index}] must be "
                f"[{expected_count}, {hidden_width}], got {tuple(source.shape)}"
            )

    row_counts = scope.local_condition_counts
    max_foreground = max(row_counts, default=0)
    if scope.num_segments == 0:
        return PackedSegConditions(
            scope=scope,
            embeddings=background.new_empty((0, 1, hidden_width)),
            valid_mask=torch.empty(
                (0, 1),
                dtype=torch.bool,
                device=background.device,
            ),
            foreground_counts=torch.empty(
                (0,),
                dtype=torch.long,
                device=background.device,
            ),
        )

    packed_rows = []
    valid_rows = []
    for source_index, local_indices in zip(
        scope.seg_to_sample.tolist(), scope.seg_to_condition_indices
    ):
        source = sources[source_index]
        indices = local_indices.to(device=source.device)
        foreground = source.index_select(0, indices)
        pad_count = max_foreground - int(foreground.shape[0])
        padding = background.new_zeros((pad_count, hidden_width))
        packed_rows.append(
            torch.cat((foreground, padding, background.unsqueeze(0)), dim=0)
        )
        valid_rows.append(
            torch.cat(
                (
                    torch.ones(
                        foreground.shape[0],
                        dtype=torch.bool,
                        device=background.device,
                    ),
                    torch.zeros(
                        pad_count,
                        dtype=torch.bool,
                        device=background.device,
                    ),
                    torch.ones(1, dtype=torch.bool, device=background.device),
                ),
                dim=0,
            )
        )

    return PackedSegConditions(
        scope=scope,
        embeddings=torch.stack(packed_rows, dim=0),
        valid_mask=torch.stack(valid_rows, dim=0),
        foreground_counts=torch.tensor(
            row_counts,
            dtype=torch.long,
            device=background.device,
        ),
    )


def build_and_pack_seg_conditions(
    task_name: str,
    condition_embeddings: Sequence[Tensor],
    segment_embeddings: Sequence[Tensor],
    background_embedding: Tensor,
) -> PackedSegConditions:
    """Validate extracted source tensors, build the scope, and pack Cond+BG."""

    if not isinstance(background_embedding, Tensor):
        raise TypeError("background_embedding must be a tensor")
    cond_sources = _require_rank_two_embeddings(
        condition_embeddings,
        name="condition_embeddings",
    )
    seg_sources = _require_rank_two_embeddings(
        segment_embeddings,
        name="segment_embeddings",
    )
    if len(cond_sources) != len(seg_sources):
        raise ValueError(
            "condition_embeddings and segment_embeddings must preserve the same B: "
            f"{len(cond_sources)} != {len(seg_sources)}"
        )
    scope_device = (
        cond_sources[0].device
        if cond_sources
        else background_embedding.device
    )
    scope = build_seg_condition_scope(
        task_name,
        [int(source.shape[0]) for source in cond_sources],
        [int(source.shape[0]) for source in seg_sources],
        device=scope_device,
    )
    packed = pack_seg_local_conditions(cond_sources, scope, background_embedding)
    hidden_width = int(packed.embeddings.shape[-1])
    for source_index, source in enumerate(seg_sources):
        if source.shape[1] != hidden_width:
            raise ValueError(
                f"segment_embeddings[{source_index}] must have hidden width "
                f"{hidden_width}, got {source.shape[1]}"
            )
        if (
            source.device != packed.embeddings.device
            or source.dtype != packed.embeddings.dtype
        ):
            raise ValueError(
                f"segment_embeddings[{source_index}] must match packed Cond "
                f"device/dtype ({packed.embeddings.device}, "
                f"{packed.embeddings.dtype}), got ({source.device}, "
                f"{source.dtype})"
            )
    return packed


@dataclass(frozen=True)
class SegRegroupedTargets:
    """Mask/class targets in the exact source-major SEG-row order."""

    scope: SegConditionScope
    mask_labels: Tuple[Tensor, ...]
    class_labels: Tuple[Tensor, ...]
    source_target_indices: Tuple[Tensor, ...]


def _validate_source_masks(mask: Tensor, *, source_index: int) -> Tensor:
    if not isinstance(mask, Tensor) or mask.ndim != 3:
        shape = tuple(mask.shape) if isinstance(mask, Tensor) else None
        raise ValueError(
            f"mask_labels[{source_index}] must be [T, H, W], got {shape}"
        )
    if mask.dtype.is_floating_point and not torch.isfinite(mask).all():
        raise FloatingPointError(f"mask_labels[{source_index}] contains NaN or Inf")
    return mask


def _validate_source_labels(
    labels: Tensor,
    *,
    source_index: int,
    target_count: int,
    condition_count: int,
) -> Tensor:
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        shape = tuple(labels.shape) if isinstance(labels, Tensor) else None
        raise ValueError(
            f"class_labels[{source_index}] must be [T], got {shape}"
        )
    if int(labels.numel()) != target_count:
        raise ValueError(
            f"source {source_index} mask/class target count mismatch: "
            f"{target_count} != {int(labels.numel())}"
        )
    if labels.numel() == 0:
        return labels.to(dtype=torch.long)
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"non-empty class_labels[{source_index}] must have integer dtype, "
            f"got {labels.dtype}"
        )
    labels = labels.to(dtype=torch.long)
    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < 0 or maximum >= condition_count:
        raise ValueError(
            f"class_labels[{source_index}] must be source-local Cond IDs in "
            f"[0, {condition_count}), got min={minimum}, max={maximum}"
        )
    return labels


def regroup_targets_by_seg(
    scope: SegConditionScope,
    mask_labels: Optional[Sequence[Tensor]],
    class_labels: Optional[Sequence[Tensor]],
) -> SegRegroupedTargets:
    """Regroup source-level mask/class targets into effective SEG rows.

    Ref/Rea pairing is occurrence-ordinal: row ``(b, j)`` receives target
    ``j`` and its row-local foreground class is zero.  The input class labels
    are therefore *not* used as a lookup key (important when two expressions
    for the same instance have duplicate legacy labels).

    For one-SEG tasks, every source target remains in the source's sole row and
    its source-local condition ID is preserved, including repeated IDs.
    """

    task = scope.task_name
    if mask_labels is None:
        masks = None
    else:
        masks = tuple(mask_labels)
        if len(masks) != scope.num_sources:
            raise ValueError(
                "mask_labels must preserve the source batch B: "
                f"{len(masks)} != {scope.num_sources}"
            )
    if class_labels is None:
        labels = None
    else:
        labels = tuple(class_labels)
        if len(labels) != scope.num_sources:
            raise ValueError(
                "class_labels must preserve the source batch B: "
                f"{len(labels)} != {scope.num_sources}"
            )

    if task in NO_SEGMENTATION_TASKS:
        if masks is not None:
            for source_index, source_masks in enumerate(masks):
                source_masks = _validate_source_masks(
                    source_masks,
                    source_index=source_index,
                )
                if source_masks.shape[0] != 0:
                    raise ValueError(
                        f"task {task!r} source {source_index} permits no masks"
                    )
        if labels is not None:
            for source_index, source_labels in enumerate(labels):
                if not isinstance(source_labels, Tensor) or source_labels.ndim != 1:
                    shape = (
                        tuple(source_labels.shape)
                        if isinstance(source_labels, Tensor)
                        else None
                    )
                    raise ValueError(
                        f"class_labels[{source_index}] must be [0], got {shape}"
                    )
                if source_labels.numel() != 0:
                    raise ValueError(
                        f"task {task!r} source {source_index} permits no class labels"
                    )
        return SegRegroupedTargets(scope, (), (), ())

    if masks is None:
        raise TypeError(f"task {task!r} requires mask_labels")
    if task in SINGLE_SEGMENTATION_TASKS and labels is None:
        raise TypeError(f"task {task!r} requires class_labels")

    row_masks = []
    row_labels = []
    row_target_indices = []
    for source_index, source_masks in enumerate(masks):
        source_masks = _validate_source_masks(
            source_masks,
            source_index=source_index,
        )
        target_count = int(source_masks.shape[0])
        source_labels = None
        if labels is not None:
            source_labels = _validate_source_labels(
                labels[source_index],
                source_index=source_index,
                target_count=target_count,
                condition_count=scope.condition_counts[source_index],
            )

        if task in ORDINAL_SEGMENTATION_TASKS:
            expected_targets = scope.segment_counts[source_index]
            if target_count != expected_targets:
                raise ValueError(
                    f"task {task!r} source {source_index} requires one target "
                    f"per SEG occurrence: T_b={target_count}, S_b={expected_targets}"
                )
            label_device = (
                source_labels.device if source_labels is not None else source_masks.device
            )
            for local_seg_index in range(expected_targets):
                row_masks.append(source_masks[local_seg_index : local_seg_index + 1])
                row_labels.append(
                    torch.zeros(1, dtype=torch.long, device=label_device)
                )
                row_target_indices.append(
                    torch.tensor(
                        [local_seg_index],
                        dtype=torch.long,
                        device=source_masks.device,
                    )
                )
        else:
            # One SEG row owns all targets from this source.  Repeated local
            # Cond IDs intentionally remain repeated.
            assert source_labels is not None
            row_masks.append(source_masks)
            row_labels.append(source_labels)
            row_target_indices.append(
                torch.arange(
                    target_count,
                    dtype=torch.long,
                    device=source_masks.device,
                )
            )

    if len(row_masks) != scope.num_segments:
        raise RuntimeError(
            "internal SEG target row order mismatch: "
            f"built {len(row_masks)}, expected {scope.num_segments}"
        )
    return SegRegroupedTargets(
        scope=scope,
        mask_labels=tuple(row_masks),
        class_labels=tuple(row_labels),
        source_target_indices=tuple(row_target_indices),
    )


__all__ = [
    "NO_SEGMENTATION_TASKS",
    "ORDINAL_SEGMENTATION_TASKS",
    "PackedSegConditions",
    "SINGLE_SEGMENTATION_TASKS",
    "SUPPORTED_TASKS",
    "SegConditionScope",
    "SegRegroupedTargets",
    "build_and_pack_seg_conditions",
    "build_seg_condition_scope",
    "extract_indexed_embeddings",
    "pack_seg_local_conditions",
    "regroup_targets_by_seg",
]
