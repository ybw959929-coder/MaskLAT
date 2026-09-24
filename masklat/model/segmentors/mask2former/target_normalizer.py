"""Canonical target handling for topology-group segmentation losses.

The existing data pipelines expose segmentation targets in a few equivalent
forms (``masks``/``condition_ids`` and
``mask_labels``/``class_labels``).  This module gives the topology-group loss
a single, validated representation without depending on MMEngine or
Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


_MASK_KEYS = ("masks", "mask_labels")
_LABEL_KEYS = ("condition_ids", "target_condition_ids", "labels", "class_labels")


@dataclass(frozen=True)
class NormalizedTarget:
    """One image's canonical segmentation targets.

    Attributes:
        masks: Float tensor shaped ``[N, H, W]`` with values in ``[0, 1]``.
        condition_ids: Long tensor shaped ``[N]``.  IDs index foreground
            condition columns; the final background column is not a target.
        task_name: Canonical lower-case MaskLAT task name.
    """

    masks: Tensor
    condition_ids: Tensor
    task_name: str = "unspecified"

    @property
    def labels(self) -> Tensor:
        """Read-only compatibility alias used by Mask2Former-style callers."""

        return self.condition_ids

    @property
    def target_condition_ids(self) -> Tensor:
        """Read-only alias matching the user-facing target contract."""

        return self.condition_ids

    @property
    def num_masks(self) -> int:
        return int(self.condition_ids.numel())

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "masks": self.masks,
            "condition_ids": self.condition_ids,
            "task_name": self.task_name,
        }


@dataclass(frozen=True)
class NormalizedTargetBatch(Sequence[NormalizedTarget]):
    """Immutable, sequence-compatible batch of normalized targets."""

    targets: Tuple[NormalizedTarget, ...]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: Union[int, slice]):
        return self.targets[index]

    def __iter__(self) -> Iterator[NormalizedTarget]:
        return iter(self.targets)

    @property
    def num_masks(self) -> int:
        return sum(target.num_masks for target in self.targets)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], what: str) -> Any:
    present = [key for key in keys if key in mapping]
    if not present:
        raise KeyError(f"target mapping is missing {what}; expected one of {tuple(keys)}")
    if len(present) > 1:
        first = mapping[present[0]]
        if any(mapping[key] is not first for key in present[1:]):
            raise ValueError(f"target mapping contains conflicting aliases for {what}: {present}")
    return mapping[present[0]]


def _is_tensor_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, Tensor) for item in value
    )


def _looks_like_batched_pair(value: Any) -> bool:
    """Conservatively recognize ``(mask_batch, label_batch)`` input."""

    if not isinstance(value, tuple) or len(value) != 2:
        return False
    masks, labels = value
    if isinstance(masks, Tensor) and isinstance(labels, Tensor):
        return masks.ndim in (2, 3, 4) and labels.ndim in (0, 1, 2)
    if not (_is_tensor_sequence(masks) and _is_tensor_sequence(labels)):
        return False
    if len(masks) != len(labels):
        return False
    return all(mask.ndim in (2, 3) for mask in masks) and all(
        label.ndim in (0, 1) for label in labels
    )


class UnifiedTargetNormalizer(nn.Module):
    """Normalize heterogeneous segmentation targets to one strict schema.

    Supported inputs are:

    * a sequence of mappings containing ``masks``/``condition_ids`` or
      ``mask_labels``/``class_labels``;
    * a sequence of :class:`NormalizedTarget` objects;
    * ``(mask_batch, label_batch)``;
    * separate ``targets=mask_batch, class_labels=label_batch`` arguments;
    * a batch mapping with list-valued mask and label entries.

    ``embed_masks`` follows MaskLAT's global-condition convention ``[B, C]``:
    valid foreground columns and the final background column are ``True``.
    When supplied, condition IDs are checked against both the validity mask
    and the reserved final background column.
    """

    def __init__(
        self,
        *,
        allow_soft_masks: bool = True,
        validate_finite: bool = True,
        background_is_last: bool = True,
    ) -> None:
        super().__init__()
        self.allow_soft_masks = bool(allow_soft_masks)
        self.validate_finite = bool(validate_finite)
        self.background_is_last = bool(background_is_last)

    @staticmethod
    def _tensor_pair_to_batch(
        masks: Tensor,
        labels: Tensor,
    ) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
        if masks.ndim == 4 or labels.ndim == 2:
            if masks.ndim != 4 or labels.ndim != 2:
                raise ValueError(
                    "batched tensor targets require masks [B,N,H,W] and "
                    f"condition IDs [B,N], got {tuple(masks.shape)} and "
                    f"{tuple(labels.shape)}"
                )
            if masks.shape[0] != labels.shape[0]:
                raise ValueError(
                    "batched tensor target sizes differ: "
                    f"{masks.shape[0]} != {labels.shape[0]}"
                )
            return tuple(masks.unbind(0)), tuple(labels.unbind(0))
        return (masks,), (labels,)

    @staticmethod
    def _coerce_batch(
        targets: Any,
        class_labels: Optional[Any],
        *,
        infer_single_foreground: bool,
    ) -> Tuple[Sequence[Any], Sequence[Any]]:
        if class_labels is not None:
            masks_batch = targets
            labels_batch = class_labels
            if isinstance(masks_batch, Tensor) and isinstance(
                labels_batch, Tensor
            ):
                return UnifiedTargetNormalizer._tensor_pair_to_batch(
                    masks_batch, labels_batch
                )
            if isinstance(masks_batch, Tensor):
                masks_batch = (masks_batch,)
            if isinstance(labels_batch, Tensor):
                labels_batch = (labels_batch,)
            if not isinstance(masks_batch, (list, tuple)) or not isinstance(
                labels_batch, (list, tuple)
            ):
                raise TypeError(
                    "separate masks and class_labels must be tensors or sequences"
                )
            if len(masks_batch) != len(labels_batch):
                raise ValueError(
                    "masks and class_labels batch lengths differ: "
                    f"{len(masks_batch)} != {len(labels_batch)}"
                )
            return masks_batch, labels_batch

        if isinstance(targets, NormalizedTargetBatch):
            return (
                tuple(target.masks for target in targets),
                tuple(target.condition_ids for target in targets),
            )
        if isinstance(targets, NormalizedTarget):
            return (targets.masks,), (targets.condition_ids,)
        if isinstance(targets, Mapping):
            masks = _first_present(targets, _MASK_KEYS, "masks")
            try:
                labels = _first_present(
                    targets, _LABEL_KEYS, "condition IDs"
                )
            except KeyError:
                labels = None
            if labels is None:
                if not infer_single_foreground:
                    raise KeyError(
                        "target mapping is missing unambiguous condition IDs"
                    )
                if isinstance(masks, Tensor) and masks.ndim == 4:
                    return tuple(masks.unbind(0)), (None,) * masks.shape[0]
                if isinstance(masks, (list, tuple)):
                    return masks, (None,) * len(masks)
                return (masks,), (None,)
            if isinstance(masks, Tensor) and isinstance(labels, Tensor):
                return UnifiedTargetNormalizer._tensor_pair_to_batch(
                    masks, labels
                )
            if isinstance(masks, (list, tuple)) and isinstance(
                labels, (list, tuple)
            ):
                if len(masks) != len(labels):
                    raise ValueError(
                        "batched masks and condition IDs must have equal "
                        f"lengths, got {len(masks)} and {len(labels)}"
                    )
                return masks, labels
            return (masks,), (labels,)
        if _looks_like_batched_pair(targets):
            masks_batch, labels_batch = targets
            if isinstance(masks_batch, Tensor) and isinstance(
                labels_batch, Tensor
            ):
                return UnifiedTargetNormalizer._tensor_pair_to_batch(
                    masks_batch, labels_batch
                )
            return masks_batch, labels_batch
        if isinstance(targets, Tensor) and infer_single_foreground:
            if targets.ndim == 4:
                return tuple(targets.unbind(0)), (None,) * targets.shape[0]
            return (targets,), (None,)
        if not isinstance(targets, (list, tuple)):
            raise TypeError(
                "targets must be a target mapping, target sequence, or "
                "(mask_batch, label_batch) pair"
            )

        masks_batch = []
        labels_batch = []
        for index, target in enumerate(targets):
            if isinstance(target, NormalizedTarget):
                masks_batch.append(target.masks)
                labels_batch.append(target.labels)
            elif isinstance(target, Mapping):
                masks_batch.append(_first_present(target, _MASK_KEYS, "masks"))
                try:
                    labels_batch.append(
                        _first_present(target, _LABEL_KEYS, "condition IDs")
                    )
                except KeyError:
                    if not infer_single_foreground:
                        raise
                    labels_batch.append(None)
            elif isinstance(target, (list, tuple)) and len(target) == 2:
                masks_batch.append(target[0])
                labels_batch.append(target[1])
            elif isinstance(target, Tensor) and infer_single_foreground:
                masks_batch.append(target)
                labels_batch.append(None)
            else:
                raise TypeError(
                    f"target {index} must be a mapping, NormalizedTarget, "
                    "or (masks, labels) pair"
                )
        return masks_batch, labels_batch

    def _normalize_one(
        self,
        masks: Any,
        labels: Any,
        *,
        index: int,
        task_name: str,
        device: Optional[torch.device],
        mask_dtype: torch.dtype,
    ) -> NormalizedTarget:
        context = f"task {task_name!r} sample {index}"
        masks = torch.as_tensor(masks, device=device)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        elif masks.ndim == 1 and masks.numel() == 0:
            masks = masks.reshape(0, 0, 0)
        if labels is None:
            labels = torch.zeros(masks.shape[0], dtype=torch.long, device=masks.device)
        else:
            labels = torch.as_tensor(labels, device=device)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        if masks.ndim != 3:
            raise ValueError(
                f"{context} masks must be [N,H,W], got {tuple(masks.shape)}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"{context} condition IDs must be [N], got {tuple(labels.shape)}"
            )
        if masks.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{context} mask/condition count mismatch: "
                f"{masks.shape[0]} != {labels.shape[0]}"
            )
        if masks.shape[0] > 0 and (masks.shape[1] <= 0 or masks.shape[2] <= 0):
            raise ValueError(
                f"{context} non-empty masks need positive H,W, got "
                f"{tuple(masks.shape)}"
            )
        if labels.is_floating_point() and labels.numel() > 0:
            rounded = labels.round()
            if not bool(torch.equal(labels, rounded)):
                raise ValueError(f"{context} condition IDs must be integer-valued")

        masks = masks.to(dtype=mask_dtype)
        labels = labels.to(dtype=torch.long)
        if labels.numel() > 0 and bool((labels < 0).any()):
            raise ValueError(f"{context} condition IDs must be non-negative")
        if self.validate_finite:
            if masks.is_floating_point() and not bool(torch.isfinite(masks).all()):
                raise FloatingPointError(f"{context} masks contain NaN or Inf")
            if labels.is_floating_point() and not bool(torch.isfinite(labels).all()):
                raise FloatingPointError(
                    f"{context} condition IDs contain NaN or Inf"
                )
        if masks.numel() > 0:
            if bool((masks < 0).any()) or bool((masks > 1).any()):
                raise ValueError(f"{context} masks must have values in [0,1]")
            if not self.allow_soft_masks:
                is_binary = masks.eq(0) | masks.eq(1)
                if not bool(is_binary.all()):
                    raise ValueError(
                        f"{context} masks must be binary when "
                        "allow_soft_masks=False"
                    )
        return NormalizedTarget(
            masks=masks.contiguous(),
            condition_ids=labels.contiguous(),
            task_name=task_name,
        )

    def _validate_classes(
        self,
        normalized: Sequence[NormalizedTarget],
        *,
        num_classes: Optional[Union[int, Sequence[int]]],
        embed_masks: Optional[Tensor],
        task_name: str,
    ) -> None:
        batch_size = len(normalized)
        if embed_masks is not None:
            if embed_masks.ndim != 2:
                raise ValueError(
                    f"embed_masks must be [B,C], got {tuple(embed_masks.shape)}"
                )
            if embed_masks.shape[0] != batch_size:
                raise ValueError(
                    "embed_masks batch size does not match targets: "
                    f"{embed_masks.shape[0]} != {batch_size}"
                )
            embed_masks = embed_masks.to(dtype=torch.bool)
            if embed_masks.shape[1] == 0:
                raise ValueError("embed_masks must contain at least one class column")
            if self.background_is_last and not bool(embed_masks[:, -1].all()):
                raise ValueError(
                    "the final background column must be valid for every sample"
                )

        if isinstance(num_classes, int):
            class_counts: Optional[Sequence[int]] = (int(num_classes),) * batch_size
        elif num_classes is None:
            class_counts = None
        else:
            class_counts = tuple(int(value) for value in num_classes)
            if len(class_counts) != batch_size:
                raise ValueError(
                    "num_classes sequence length does not match targets: "
                    f"{len(class_counts)} != {batch_size}"
                )

        for index, target in enumerate(normalized):
            labels = target.labels
            if labels.numel() == 0:
                continue
            if class_counts is not None:
                count = class_counts[index]
                if count <= 0 or bool((labels >= count).any()):
                    raise ValueError(
                        f"task {task_name!r} sample {index} has condition ID "
                        f"outside [0,{count - 1}]"
                    )
                if self.background_is_last and bool((labels == count - 1).any()):
                    raise ValueError(
                        f"task {task_name!r} sample {index} uses reserved "
                        f"background condition {count - 1}"
                    )
            if embed_masks is not None:
                class_count = embed_masks.shape[1]
                if bool((labels >= class_count).any()):
                    raise ValueError(
                        f"task {task_name!r} sample {index} has condition ID "
                        f"outside [0,{class_count - 1}]"
                    )
                sample_embed_mask = embed_masks[index].to(device=labels.device)
                if not bool(sample_embed_mask.gather(0, labels).all()):
                    raise ValueError(
                        f"task {task_name!r} sample {index} references a "
                        "padded/invalid condition column"
                    )
                if self.background_is_last and bool(
                    (labels == class_count - 1).any()
                ):
                    raise ValueError(
                        f"task {task_name!r} sample {index} uses reserved "
                        "background condition "
                        f"{class_count - 1}"
                    )

    def forward(
        self,
        targets: Any = None,
        class_labels: Optional[Any] = None,
        *,
        task_name: Optional[str] = None,
        mask_labels: Optional[Any] = None,
        batch_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        mask_dtype: torch.dtype = torch.float32,
        num_classes: Optional[Union[int, Sequence[int]]] = None,
        embed_masks: Optional[Tensor] = None,
    ) -> Optional[NormalizedTargetBatch]:
        if not torch.empty((), dtype=mask_dtype).is_floating_point():
            raise TypeError(f"mask_dtype must be floating point, got {mask_dtype}")
        normalized_task_name = (
            None if task_name is None else str(task_name).strip().lower()
        )
        if normalized_task_name is None:
            if isinstance(targets, NormalizedTarget):
                normalized_task_name = targets.task_name
            elif isinstance(targets, NormalizedTargetBatch):
                existing_names = {target.task_name for target in targets}
                if len(existing_names) == 1:
                    normalized_task_name = next(iter(existing_names))
        canonical_task_name = normalized_task_name or "unspecified"
        if targets is not None and mask_labels is not None:
            raise ValueError("pass masks through either targets or mask_labels, not both")
        if mask_labels is not None:
            targets = mask_labels
        if targets is None:
            if normalized_task_name == "imgconv":
                return None
            if batch_size is None:
                raise ValueError(
                    f"task {task_name!r} has no targets; pass batch_size for an "
                    "explicit empty mask batch"
                )
            empty_targets = tuple(
                NormalizedTarget(
                    masks=torch.empty(
                        0,
                        0,
                        0,
                        dtype=mask_dtype,
                        device=device,
                    ),
                    condition_ids=torch.empty(
                        0,
                        dtype=torch.long,
                        device=device,
                    ),
                    task_name=canonical_task_name,
                )
                for _ in range(int(batch_size))
            )
            empty_batch = NormalizedTargetBatch(empty_targets)
            self._validate_classes(
                empty_batch,
                num_classes=num_classes,
                embed_masks=embed_masks,
                task_name=canonical_task_name,
            )
            return empty_batch

        single_foreground_by_count = (
            isinstance(num_classes, int) and int(num_classes) == 2
        )
        single_foreground_by_mask = False
        if embed_masks is not None and embed_masks.ndim == 2:
            single_foreground_by_mask = bool(
                embed_masks.to(dtype=torch.bool).sum(dim=1).eq(2).all()
            )
        infer_single_foreground = (
            class_labels is None
            and (
                single_foreground_by_count
                or single_foreground_by_mask
            )
        )
        resolved_device = torch.device(device) if device is not None else None
        try:
            masks_batch, labels_batch = self._coerce_batch(
                targets,
                class_labels,
                infer_single_foreground=infer_single_foreground,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise type(error)(
                f"task {canonical_task_name!r}: {error}"
            ) from error
        if batch_size is not None and len(masks_batch) != int(batch_size):
            raise ValueError(
                f"target batch size {len(masks_batch)} does not match expected "
                f"{int(batch_size)}"
            )
        normalized = tuple(
            self._normalize_one(
                masks,
                labels,
                index=index,
                task_name=canonical_task_name,
                device=resolved_device,
                mask_dtype=mask_dtype,
            )
            for index, (masks, labels) in enumerate(
                zip(masks_batch, labels_batch)
            )
        )
        self._validate_classes(
            normalized,
            num_classes=num_classes,
            embed_masks=embed_masks,
            task_name=canonical_task_name,
        )
        return NormalizedTargetBatch(normalized)


__all__ = [
    "NormalizedTarget",
    "NormalizedTargetBatch",
    "UnifiedTargetNormalizer",
]
