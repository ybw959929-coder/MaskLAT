"""Formal, side-effect-free evaluation updates for topology group decoding.

The public helper in this module consumes the *raw* ``MaskLATSegmentorOutput``
returned before task post-processing.  It only updates a caller-owned
``GroupDiagnostics`` accumulator: it never changes model outputs, never
selects a formal prediction with GT, and never prints.

The GT-dependent definitions follow ``tools/analyze_mask_topology_groups.py``:

* a good query has binary mask IoU >= 0.7;
* good groups are connected components containing a good query;
* group purity is the fraction of good queries among a good group's members;
* a bridge is a non-singleton whose minimum within-group pair IoU is below
  the grouping threshold;
* RefSeg/ReaSeg routing ranks groups by the effective sample's formal
  target-Cond softmax probability (the text condition is part of the model
  input; GT masks are never used for this ranking);
* the selected-group and global Query oracles use GT only in explicitly
  oracle-labelled diagnostic variants.

``global_group_oracle_member_mask`` is defined as the best, GT-selected group
among the learned quality representative of every group.  This measures the
group-routing ceiling while keeping the learned member selector fixed.
``global_200q_oracle_query_mask`` remains the unconstrained Query oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from masklat.model.segmentors.mask2former.spatial_validity import (
        masked_normalized_resize,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )
except ImportError:  # Standalone path-loading used by focused unit tests.
    from spatial_validity import (  # type: ignore
        masked_normalized_resize,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )


REF_TASKS = frozenset({"refseg", "reaseg"})


@dataclass(frozen=True)
class TopologyEvaluationUpdate:
    """Objects produced while updating one raw evaluation batch."""

    task_name: str
    normalized_targets: Any
    matching: Any
    batch_size: int


def _field(container: Any, name: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(name, default)
    value = getattr(container, name, default)
    if value is not default:
        return value
    metainfo = getattr(container, "metainfo", None)
    if isinstance(metainfo, Mapping):
        return metainfo.get(name, default)
    return default


def _resolve_task_name(task_name: Optional[str], data_samples: Any) -> str:
    if task_name is None:
        task_names = _field(data_samples, "task_names")
        if isinstance(task_names, str):
            task_name = task_names
        elif isinstance(task_names, Sequence) and task_names:
            names = {str(value).strip().lower() for value in task_names}
            if len(names) != 1:
                raise ValueError(
                    "topology diagnostics require a homogeneous task batch, "
                    f"got {sorted(names)}"
                )
            task_name = next(iter(names))
    resolved = "unspecified" if task_name is None else str(task_name).strip().lower()
    if not resolved:
        raise ValueError("task_name must be non-empty")
    return resolved


def _default_normalizer():
    from masklat.model.segmentors.mask2former.target_normalizer import (
        UnifiedTargetNormalizer,
    )

    return UnifiedTargetNormalizer()


def _default_matcher():
    from masklat.model.segmentors.mask2former.group_matcher import (
        HierarchicalGroupMatcher,
    )

    # Evaluation diagnostics must be deterministic.  Full-resolution
    # matching reuses the formal matcher costs without random point sampling.
    return HierarchicalGroupMatcher(use_sample_point=False)


def _as_batch(value: Any, *, masks: bool) -> Sequence[Tensor]:
    if isinstance(value, Tensor):
        if masks and value.ndim == 4:
            return tuple(value.unbind(0))
        if not masks and value.ndim == 2:
            return tuple(value.unbind(0))
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError("evaluation targets must be tensors or tensor sequences")


def _expand_ref_targets(
    mask_labels: Any,
    class_labels: Any,
    *,
    effective_batch_size: int,
    num_classes: int,
) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
    """Repeat one original-image target per local referring condition."""

    mask_batch = _as_batch(mask_labels, masks=True)
    if class_labels is None:
        label_batch = (None,) * len(mask_batch)
    else:
        label_batch = _as_batch(class_labels, masks=False)
    if len(mask_batch) != len(label_batch):
        raise ValueError("Ref target mask/label source batches differ")

    expanded_masks = []
    expanded_labels = []
    condition_offset = 0
    for sample_index, (sample_masks, sample_labels) in enumerate(
        zip(mask_batch, label_batch)
    ):
        sample_masks = torch.as_tensor(sample_masks)
        if sample_masks.ndim == 2:
            sample_masks = sample_masks.unsqueeze(0)
        if sample_masks.ndim != 3:
            raise ValueError(
                f"Ref sample {sample_index} masks must be [N,H,W], got "
                f"{tuple(sample_masks.shape)}"
            )
        count = int(sample_masks.shape[0])
        if count <= 0:
            raise ValueError(
                f"Ref sample {sample_index} must contain at least one target"
            )
        if sample_labels is None:
            if count != 1:
                raise ValueError(
                    "multi-condition Ref diagnostics require explicit class labels"
                )
            sample_labels = torch.zeros(1, dtype=torch.long)
        else:
            sample_labels = torch.as_tensor(sample_labels)
            if sample_labels.ndim == 0:
                sample_labels = sample_labels.unsqueeze(0)
        if sample_labels.ndim != 1 or sample_labels.numel() != count:
            raise ValueError(
                f"Ref sample {sample_index} label count does not match masks"
            )
        if sample_labels.is_floating_point():
            if not bool(torch.isfinite(sample_labels).all()):
                raise ValueError(
                    "Ref class labels must be finite; "
                    f"sample {sample_index} has {sample_labels.tolist()}"
                )
            if not bool(torch.equal(sample_labels, sample_labels.round())):
                raise ValueError(
                    "Ref class labels must be integer-valued; "
                    f"sample {sample_index} has {sample_labels.tolist()}"
                )
        local_labels = sample_labels.to(torch.long)
        if bool((local_labels < 0).any()) or bool((local_labels >= count).any()):
            raise ValueError(
                "Ref class labels must be local condition indices before "
                f"cond_lens expansion; sample {sample_index} has "
                f"{local_labels.tolist()} for {count} conditions"
            )
        for target_index in range(count):
            expanded_masks.append(sample_masks[target_index : target_index + 1])
            expanded_labels.append(
                local_labels[target_index : target_index + 1]
                + condition_offset
            )
        condition_offset += count

    if len(expanded_masks) != effective_batch_size:
        raise ValueError(
            "Ref cond_lens expansion and raw prediction batch differ: "
            f"{len(expanded_masks)} != {effective_batch_size}"
        )
    if condition_offset + 1 != num_classes:
        raise ValueError(
            "Ref condition table must contain all local conditions plus final "
            f"background: {condition_offset + 1} != {num_classes}"
        )
    return tuple(expanded_masks), tuple(expanded_labels)


def _target_payload(
    data_samples: Any,
    targets: Any,
    *,
    task_name: str,
    effective_batch_size: int,
    num_classes: int,
) -> Any:
    if targets is not None:
        return targets
    mask_labels = _field(data_samples, "mask_labels")
    class_labels = _field(data_samples, "class_labels")
    if mask_labels is None:
        return None
    if task_name in REF_TASKS:
        return _expand_ref_targets(
            mask_labels,
            class_labels,
            effective_batch_size=effective_batch_size,
            num_classes=num_classes,
        )
    return (mask_labels, class_labels) if class_labels is not None else mask_labels


def _validate_embed_masks(final_stage: Any, supplied: Tensor) -> Tensor:
    logits = torch.as_tensor(final_stage.group_class_logits)
    batch_size, _, num_classes = logits.shape
    if num_classes <= 1:
        raise ValueError("topology diagnostics need foreground and background")
    if supplied.shape != (batch_size, num_classes):
        raise ValueError(
            "condition_valid_mask must match [B,C], got "
            f"{tuple(supplied.shape)} != {(batch_size, num_classes)}"
        )
    result = supplied.to(device=logits.device, dtype=torch.bool)
    if not bool(result[:, -1].all()):
        raise ValueError("the final background condition must be valid")
    if not bool(result[:, :-1].any(dim=1).all()):
        raise ValueError("every sample needs at least one valid foreground condition")
    return result


def _size_pair(value: Any, name: str) -> Tuple[int, int]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        value = (value.get("height"), value.get("width"))
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be (height,width), got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def _size_batch(value: Any, name: str) -> Optional[Sequence[Tuple[int, int]]]:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.ndim == 1:
            return (_size_pair(value, name),)
        if value.ndim == 2:
            return tuple(_size_pair(item, name) for item in value)
    if isinstance(value, Mapping):
        return (_size_pair(value, name),)
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(
            isinstance(item, (int, float)) for item in value
        ):
            return (_size_pair(value, name),)
        return tuple(_size_pair(item, name) for item in value)
    raise ValueError(f"{name} must be a size or size sequence")


def _evaluation_sizes(
    data_samples: Any,
    *,
    task_name: str,
    batch_size: int,
) -> Sequence[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    """Resolve SAM valid crop and original output size per effective sample."""

    scaled = _size_batch(_field(data_samples, "scaled_sizes"), "scaled_size")
    original = _size_batch(_field(data_samples, "image_sizes"), "image_size")
    if scaled is None and original is None:
        return ((None, None),) * batch_size
    if scaled is None or original is None or len(scaled) != len(original):
        raise ValueError(
            "formal coordinate restoration requires paired scaled_sizes and "
            "image_sizes"
        )
    pairs = tuple(zip(scaled, original))
    if task_name in REF_TASKS and len(pairs) != batch_size:
        mask_batch = _as_batch(_field(data_samples, "mask_labels"), masks=True)
        if len(mask_batch) != len(pairs):
            raise ValueError(
                "Ref spatial metadata and source target batches differ"
            )
        expanded = []
        for pair, sample_masks in zip(pairs, mask_batch):
            sample_masks = torch.as_tensor(sample_masks)
            count = 1 if sample_masks.ndim == 2 else int(sample_masks.shape[0])
            expanded.extend([pair] * count)
        pairs = tuple(expanded)
    if len(pairs) != batch_size:
        raise ValueError(
            "spatial metadata and effective prediction batches differ: "
            f"{len(pairs)} != {batch_size}"
        )
    return pairs


def _decode_coco_rle_mask(
    encoded: Any,
    *,
    expected_size: Optional[Tuple[int, int]],
    name: str,
) -> Tensor:
    """Decode one COCO RLE/polygon into a CPU boolean ``[H,W]`` tensor."""

    if isinstance(encoded, Tensor):
        decoded = encoded.detach().cpu()
    elif (
        isinstance(encoded, Mapping)
        and "counts" in encoded
    ) or (
        isinstance(encoded, Sequence)
        and not isinstance(encoded, (str, bytes))
    ):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:  # pragma: no cover - project dependency.
            raise ImportError(
                "decoding diagnostic COCO masks requires pycocotools"
            ) from error
        if isinstance(encoded, Mapping):
            rle = dict(encoded)
            size = rle.get("size")
        else:
            rle = encoded
            size = None
        if size is None:
            if expected_size is None:
                raise ValueError(
                    f"{name} needs an explicit original image size"
                )
            size = list(expected_size)
        height, width = _size_pair(size, f"{name} RLE size")
        if (
            expected_size is not None
            and (height, width) != tuple(expected_size)
        ):
            raise ValueError(
                f"{name} declared RLE size {(height, width)} does not match "
                f"original image size {tuple(expected_size)}"
            )
        if isinstance(rle, Mapping):
            rle = dict(rle)
            rle["size"] = [height, width]
            counts = rle["counts"]
            if isinstance(counts, list):
                rle = mask_utils.frPyObjects(rle, height, width)
            elif isinstance(counts, str):
                rle["counts"] = counts.encode("ascii")
        elif len(rle) and isinstance(rle[0], Mapping):
            normalized_rles = []
            for component_index, component in enumerate(rle):
                component = dict(component)
                component_size = component.get("size")
                if component_size is not None:
                    declared_component_size = _size_pair(
                        component_size,
                        f"{name} RLE component {component_index} size",
                    )
                    if declared_component_size != (height, width):
                        raise ValueError(
                            f"{name} RLE component {component_index} declared "
                            f"size {declared_component_size} does not match "
                            f"original image size {(height, width)}"
                        )
                component["size"] = [height, width]
                counts = component.get("counts")
                if isinstance(counts, list):
                    component = mask_utils.frPyObjects(
                        component,
                        height,
                        width,
                    )
                elif isinstance(counts, str):
                    component["counts"] = counts.encode("ascii")
                normalized_rles.append(component)
            rle = normalized_rles
        else:
            rle = mask_utils.frPyObjects(rle, height, width)
        if isinstance(rle, list):
            rle = mask_utils.merge(rle)
        decoded = torch.as_tensor(mask_utils.decode(rle).copy())
    else:
        raise TypeError(
            f"{name} must be a tensor or COCO RLE mapping, got "
            f"{type(encoded).__name__}"
        )
    if decoded.ndim == 3 and decoded.shape[-1] == 1:
        decoded = decoded[..., 0]
    if decoded.ndim != 2:
        raise ValueError(f"{name} must decode to [H,W], got {tuple(decoded.shape)}")
    if expected_size is not None and tuple(decoded.shape) != tuple(expected_size):
        raise ValueError(
            f"{name} shape {tuple(decoded.shape)} does not match "
            f"original image size {tuple(expected_size)}"
        )
    return decoded.to(torch.bool)


def _resolve_diagnostic_gt_masks(
    data_samples: Any,
    supplied: Any,
    *,
    task_name: str,
    batch_size: int,
) -> Sequence[Optional[Tensor]]:
    """Resolve exact original-annotation GT masks for RefSeg/ReaSeg."""

    if task_name not in REF_TASKS:
        return (None,) * batch_size
    image_infos = _field(data_samples, "image_infos")
    if isinstance(image_infos, Mapping):
        image_infos = (image_infos,)
    if not isinstance(image_infos, Sequence) or isinstance(
        image_infos,
        (str, bytes),
    ):
        raise ValueError(
            "RefSeg/ReaSeg diagnostics require image_infos with exact "
            "original annotation geometry"
        )
    image_infos = tuple(image_infos)
    if len(image_infos) != batch_size:
        raise ValueError(
            "RefSeg/ReaSeg diagnostics require one image_info per effective "
            f"sample: {len(image_infos)} != {batch_size}"
        )
    if supplied is None:
        encoded_masks = []
        for sample_index, image_info in enumerate(image_infos):
            if (
                not isinstance(image_info, Mapping)
                or image_info.get("diagnostic_gt_mask") is None
            ):
                raise ValueError(
                    "RefSeg/ReaSeg diagnostics require "
                    "image_infos[*]['diagnostic_gt_mask']; "
                    f"sample {sample_index} is missing it"
                )
            encoded_masks.append(image_info["diagnostic_gt_mask"])
    elif isinstance(supplied, Tensor):
        if supplied.ndim == 2 and batch_size == 1:
            encoded_masks = [supplied]
        elif supplied.ndim == 3 and supplied.shape[0] == batch_size:
            encoded_masks = list(supplied.unbind(0))
        elif supplied.ndim == 4 and supplied.shape[:2] == (batch_size, 1):
            encoded_masks = list(supplied[:, 0].unbind(0))
        else:
            raise ValueError(
                "explicit diagnostic_gt_masks tensor must be [H,W], "
                "[B,H,W], or [B,1,H,W]"
            )
    elif isinstance(supplied, Sequence) and not isinstance(
        supplied,
        (str, bytes),
    ):
        encoded_masks = list(supplied)
    else:
        raise TypeError(
            "explicit diagnostic_gt_masks must be a tensor or sequence"
        )
    if len(encoded_masks) != batch_size:
        raise ValueError(
            "diagnostic GT mask count must equal effective batch: "
            f"{len(encoded_masks)} != {batch_size}"
        )

    decoded_masks = []
    for sample_index, (encoded, image_info) in enumerate(
        zip(encoded_masks, image_infos)
    ):
        if not isinstance(image_info, Mapping):
            raise ValueError(
                f"image_info[{sample_index}] must be a mapping"
            )
        expected_size = _size_pair(
            (image_info.get("height"), image_info.get("width")),
            f"image_info[{sample_index}] size",
        )
        decoded_masks.append(
            _decode_coco_rle_mask(
                encoded,
                expected_size=expected_size,
                name=f"diagnostic_gt_mask[{sample_index}]",
            )
        )
    return tuple(decoded_masks)


def _resolve_ignore_masks(
    data_samples: Any,
    supplied: Any,
    *,
    task_name: str,
    batch_size: int,
) -> Sequence[Optional[Tensor]]:
    """Resolve per-effective-sample original-resolution ReaSeg ignore masks."""

    if task_name != "reaseg":
        return (None,) * batch_size
    image_infos = _field(data_samples, "image_infos")
    if isinstance(image_infos, Mapping):
        image_infos = (image_infos,)
    if image_infos is None:
        image_infos = (None,) * batch_size
    elif not isinstance(image_infos, Sequence) or isinstance(
        image_infos,
        (str, bytes),
    ):
        raise TypeError("ReaSeg image_infos must be a mapping sequence")
    else:
        image_infos = tuple(image_infos)
    if len(image_infos) != batch_size:
        raise ValueError(
            "ReaSeg diagnostics require one image_info per effective sample: "
            f"{len(image_infos)} != {batch_size}"
        )

    if supplied is None:
        encoded_masks = []
        for sample_index, image_info in enumerate(image_infos):
            encoded = (
                image_info.get("diagnostic_ignore_mask")
                if isinstance(image_info, Mapping)
                else None
            )
            if encoded is None and isinstance(image_info, Mapping):
                encoded = image_info.get("ignore_mask")
            if encoded is None:
                raise ValueError(
                    "ReaSeg diagnostics require image_infos[*]"
                    "['diagnostic_ignore_mask']; "
                    f"sample {sample_index} is missing it"
                )
            encoded_masks.append(encoded)
    elif isinstance(supplied, Tensor):
        if supplied.ndim == 2 and batch_size == 1:
            encoded_masks = [supplied]
        elif supplied.ndim == 3 and supplied.shape[0] == batch_size:
            encoded_masks = list(supplied.unbind(0))
        elif (
            supplied.ndim == 4
            and supplied.shape[:2] == (batch_size, 1)
        ):
            encoded_masks = list(supplied[:, 0].unbind(0))
        else:
            raise ValueError(
                "explicit ReaSeg ignore_masks tensor must be [H,W], "
                "[B,H,W], or [B,1,H,W]"
            )
    elif isinstance(supplied, Sequence) and not isinstance(
        supplied,
        (str, bytes),
    ):
        encoded_masks = list(supplied)
    else:
        raise TypeError("explicit ReaSeg ignore_masks must be a tensor sequence")
    if len(encoded_masks) != batch_size:
        raise ValueError(
            "ReaSeg ignore mask count must equal effective batch: "
            f"{len(encoded_masks)} != {batch_size}"
        )

    decoded_masks = []
    for sample_index, (encoded, image_info) in enumerate(
        zip(encoded_masks, image_infos)
    ):
        expected_size = None
        if isinstance(image_info, Mapping):
            height = image_info.get("height")
            width = image_info.get("width")
            if height is not None and width is not None:
                expected_size = _size_pair(
                    (height, width),
                    f"ReaSeg image_info[{sample_index}] size",
                )
        decoded_masks.append(
            _decode_coco_rle_mask(
                encoded,
                expected_size=expected_size,
                name=f"ReaSeg ignore_mask[{sample_index}]",
            )
        )
    return tuple(decoded_masks)


def _query_target_iou(
    query_mask_logits: Tensor,
    target_masks: Tensor,
    *,
    mask_threshold: float,
    ignore_mask: Optional[Tensor] = None,
    sam_canvas_size: Optional[Tuple[int, int]] = None,
    target_masks_are_original: bool = False,
    scaled_size: Optional[Tuple[int, int]] = None,
    output_size: Optional[Tuple[int, int]] = None,
    sam_valid_box: Optional[Tensor] = None,
    query_chunk_size: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return restored original-resolution ``[Q,N]`` intersections/unions/IoUs.

    Query logits are first masked-normalized-resized to the SAM target
    canvas, so their interpolation stencil cannot read SAM padding.  They are
    then cropped to ``scaled_size`` and restored to ``output_size``.  A
    processed GT follows the same crop with nearest-neighbour restoration;
    when ``target_masks_are_original`` is true, the exact original annotation
    is scored directly without interpolation.  Query chunking bounds peak
    memory for Q=200 and 1024-square SAM tensors.
    """

    if query_mask_logits.ndim != 3 or target_masks.ndim != 3:
        raise ValueError("query and target masks must be [Q,H,W] and [N,H,W]")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    num_queries = query_mask_logits.shape[0]
    num_targets = target_masks.shape[0]
    if num_targets == 0:
        empty = torch.empty(
            num_queries,
            0,
            dtype=torch.float64,
            device=query_mask_logits.device,
        )
        return empty, empty, empty
    target_mask_size = tuple(int(value) for value in target_masks.shape[-2:])
    target_canvas_size = (
        target_mask_size
        if sam_canvas_size is None
        else _size_pair(sam_canvas_size, "sam_canvas_size")
    )
    crop_height, crop_width = (
        target_canvas_size
        if scaled_size is None
        else _size_pair(scaled_size, "scaled_size")
    )
    if (
        crop_height > target_canvas_size[0]
        or crop_width > target_canvas_size[1]
    ):
        raise ValueError(
            "scaled_size cannot exceed the SAM target canvas: "
            f"{(crop_height, crop_width)} > {target_canvas_size}"
        )
    expected_box = torch.tensor(
        [
            0.0,
            0.0,
            crop_height / float(target_canvas_size[0]),
            crop_width / float(target_canvas_size[1]),
        ],
        dtype=torch.float32,
        device=query_mask_logits.device,
    ).reshape(1, 4)
    if sam_valid_box is None:
        valid_box = expected_box
    else:
        valid_box = validate_normalized_valid_boxes(
            sam_valid_box.reshape(1, 4),
            batch_size=1,
            device=query_mask_logits.device,
        )
        if not bool(torch.allclose(
            valid_box,
            expected_box,
            rtol=0.0,
            atol=1e-6,
        )):
            raise ValueError(
                "stage SAM valid box disagrees with scaled_size/target "
                f"canvas: {valid_box[0].tolist()} != "
                f"{expected_box[0].tolist()}"
            )
    source_valid_single = valid_mask_from_normalized_boxes(
        valid_box,
        query_mask_logits.shape[-2:],
    )
    target_valid_single = valid_mask_from_normalized_boxes(
        valid_box,
        target_canvas_size,
    )
    resolved_output_size = (
        (
            target_mask_size
            if target_masks_are_original
            else (crop_height, crop_width)
        )
        if output_size is None
        else _size_pair(output_size, "output_size")
    )
    if target_masks_are_original:
        if target_mask_size != resolved_output_size:
            raise ValueError(
                "exact diagnostic GT mask must already use original output "
                f"size: {target_mask_size} != {resolved_output_size}"
            )
        target_values = target_masks.to(
            device=query_mask_logits.device,
            dtype=torch.float32,
        )
    else:
        if (
            target_mask_size[0] < crop_height
            or target_mask_size[1] < crop_width
        ):
            raise ValueError(
                "processed target mask cannot be smaller than scaled crop: "
                f"{target_mask_size} < {(crop_height, crop_width)}"
            )
        target_values = target_masks[:, :crop_height, :crop_width].to(
            device=query_mask_logits.device,
            dtype=torch.float32,
        )
        if tuple(target_values.shape[-2:]) != resolved_output_size:
            target_values = F.interpolate(
                target_values[:, None],
                size=resolved_output_size,
                mode="nearest",
            ).squeeze(1)
    valid_output = torch.ones(
        resolved_output_size,
        dtype=torch.bool,
        device=query_mask_logits.device,
    )
    if ignore_mask is not None:
        ignore_values = torch.as_tensor(
            ignore_mask,
            device=query_mask_logits.device,
        )
        if ignore_values.ndim == 3 and ignore_values.shape[0] == 1:
            ignore_values = ignore_values[0]
        if ignore_values.ndim != 2:
            raise ValueError(
                "ignore_mask must be [H,W] or [1,H,W], got "
                f"{tuple(ignore_values.shape)}"
            )
        ignore_size = tuple(int(value) for value in ignore_values.shape)
        if ignore_size == target_canvas_size:
            ignore_values = ignore_values[
                :crop_height,
                :crop_width,
            ]
        elif ignore_size not in (
            (crop_height, crop_width),
            resolved_output_size,
        ):
            raise ValueError(
                "ignore_mask must use the SAM target, scaled crop, or "
                "original output canvas: "
                f"{ignore_size} not in "
                f"{(target_canvas_size, (crop_height, crop_width), resolved_output_size)}"
            )
        if tuple(ignore_values.shape) != resolved_output_size:
            ignore_values = F.interpolate(
                ignore_values[None, None].to(torch.float32),
                size=resolved_output_size,
                mode="nearest",
            )[0, 0]
        valid_output = ~ignore_values.ge(0.5)
        if not bool(valid_output.any()):
            raise ValueError("ignore_mask excludes every output pixel")

    valid_float = valid_output.flatten().to(torch.float32)
    target_float = target_values.ge(0.5).flatten(1).to(torch.float32)
    target_float = target_float * valid_float.unsqueeze(0)
    target_areas = target_float.sum(dim=1).unsqueeze(0)
    intersection_chunks = []
    query_area_chunks = []
    for start in range(0, num_queries, query_chunk_size):
        query_chunk = query_mask_logits[start : start + query_chunk_size]
        chunk_size = query_chunk.shape[0]
        resized, _ = masked_normalized_resize(
            query_chunk[:, None],
            source_valid_single.expand(chunk_size, -1, -1, -1),
            target_canvas_size,
            target_valid_mask=target_valid_single.expand(
                chunk_size,
                -1,
                -1,
                -1,
            ),
        )
        # MaskLATSegmentor.postprocess_masks_preds performs its padding-aware
        # interpolation in float32, then explicitly casts back to the raw
        # mask-logit dtype before the scaled-size crop and original-image
        # interpolation.  Preserve that boundary exactly: with BF16
        # checkpoints, keeping this diagnostic path in FP32 can flip pixels
        # whose final logit lies close to zero and break formal parity.
        resized = resized.to(dtype=query_chunk.dtype)
        logits = resized.squeeze(1)
        logits = logits[:, :crop_height, :crop_width]
        if tuple(logits.shape[-2:]) != resolved_output_size:
            logits = F.interpolate(
                logits[:, None],
                size=resolved_output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        # Match the strict binary-mask diagnostic convention: a pixel whose
        # probability equals the threshold remains background.  Group
        # construction deliberately has a separate >= threshold contract.
        query_float = logits.sigmoid().gt(mask_threshold).flatten(1).to(
            torch.float32
        )
        query_float = query_float * valid_float.unsqueeze(0)
        intersection_chunks.append(
            torch.matmul(query_float, target_float.transpose(0, 1))
        )
        query_area_chunks.append(query_float.sum(dim=1, keepdim=True))
    intersections = torch.cat(intersection_chunks, dim=0).to(torch.float64)
    query_areas = torch.cat(query_area_chunks, dim=0).to(torch.float64)
    target_areas = target_areas.to(torch.float64)
    unions = query_areas + target_areas - intersections
    ious = torch.where(
        unions > 0,
        intersections / unions.clamp_min(1.0),
        torch.ones_like(unions),
    )
    return intersections, unions, ious


def _group_members(stage: Any, batch_index: int, root: Tensor) -> Tensor:
    return torch.nonzero(
        stage.query_to_group[batch_index].eq(root),
        as_tuple=False,
    ).flatten()


def _group_oracle_iou_and_good_counts(
    stage: Any,
    batch_index: int,
    query_ious: Tensor,
    roots: Tensor,
    *,
    good_iou_threshold: float,
) -> Tuple[Tensor, Tensor]:
    """Reduce Query IoU/goodness to physical roots without Python loops."""

    query_to_group = stage.query_to_group[batch_index]
    if query_ious.shape != query_to_group.shape:
        raise ValueError("query IoUs and query_to_group must both be [Q]")
    group_oracle = torch.full(
        query_ious.shape,
        torch.finfo(query_ious.dtype).min,
        dtype=query_ious.dtype,
        device=query_ious.device,
    )
    group_oracle.scatter_reduce_(
        0,
        query_to_group,
        query_ious,
        reduce="amax",
        include_self=True,
    )
    good_counts = torch.zeros(
        query_ious.shape,
        dtype=torch.long,
        device=query_ious.device,
    )
    good_counts.scatter_add_(
        0,
        query_to_group,
        query_ious.ge(good_iou_threshold).to(torch.long),
    )
    result_iou = group_oracle[roots]
    result_good = good_counts[roots]
    if not bool(torch.isfinite(result_iou).all()):
        raise RuntimeError("a valid Group has no Query IoU")
    return result_iou, result_good


def _stage_bridge_flags(
    stage: Any,
    *,
    group_iou_threshold: float,
) -> Tensor:
    """Compute topology bridge flags without any GT restoration."""

    query_to_group = stage.query_to_group
    pairwise_iou = stage.pairwise_iou
    if query_to_group.ndim != 2 or pairwise_iou.shape != (
        query_to_group.shape[0],
        query_to_group.shape[1],
        query_to_group.shape[1],
    ):
        raise ValueError(
            "bridge diagnostics require query_to_group [B,Q] and "
            "pairwise_iou [B,Q,Q]"
        )
    same_group = query_to_group[:, :, None].eq(
        query_to_group[:, None, :]
    )
    off_diagonal = ~torch.eye(
        query_to_group.shape[1],
        dtype=torch.bool,
        device=query_to_group.device,
    ).unsqueeze(0)
    bad_member = (
        same_group
        & off_diagonal
        & pairwise_iou.lt(group_iou_threshold)
    ).any(dim=2)
    root_bad = torch.zeros_like(query_to_group)
    root_bad.scatter_reduce_(
        1,
        query_to_group,
        bad_member.to(query_to_group.dtype),
        reduce="amax",
        include_self=True,
    )
    return root_bad.to(torch.bool) & stage.group_valid_mask.to(torch.bool)


def _stage_gt_arguments(
    stage: Any,
    normalized_targets: Sequence[Any],
    diagnostic_gt_masks: Sequence[Optional[Tensor]],
    evaluation_sizes: Sequence[
        Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
    ],
    ignore_masks: Sequence[Optional[Tensor]],
    *,
    mask_threshold: float,
    good_iou_threshold: float,
    group_iou_threshold: float,
):
    purities = []
    good_group_counts = []
    duplicate_group_counts = []
    bridge_flags = _stage_bridge_flags(
        stage,
        group_iou_threshold=group_iou_threshold,
    )
    iou_matrices = []
    for batch_index, (
        target,
        diagnostic_gt_mask,
        spatial_sizes,
        ignore_mask,
    ) in enumerate(
        zip(
            normalized_targets,
            diagnostic_gt_masks,
            evaluation_sizes,
            ignore_masks,
        )
    ):
        scaled_size, output_size = spatial_sizes
        scoring_target = (
            target.masks
            if diagnostic_gt_mask is None
            else diagnostic_gt_mask.unsqueeze(0)
        )
        _, _, query_target_iou = _query_target_iou(
            stage.query_mask_logits[batch_index],
            scoring_target,
            mask_threshold=mask_threshold,
            ignore_mask=ignore_mask,
            sam_canvas_size=tuple(target.masks.shape[-2:]),
            target_masks_are_original=diagnostic_gt_mask is not None,
            scaled_size=scaled_size,
            output_size=output_size,
            sam_valid_box=(
                stage.sam_valid_boxes_normalized[batch_index]
            ),
        )
        iou_matrices.append(query_target_iou)
        if query_target_iou.shape[1]:
            best_query_iou = query_target_iou.amax(dim=1)
        else:
            best_query_iou = torch.zeros(
                stage.query_mask_logits.shape[1],
                dtype=torch.float64,
                device=stage.query_mask_logits.device,
            )
        roots = torch.nonzero(
            stage.group_valid_mask[batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        _, group_good_counts = _group_oracle_iou_and_good_counts(
            stage,
            batch_index,
            best_query_iou,
            roots,
            good_iou_threshold=good_iou_threshold,
        )
        good_group_mask = group_good_counts.gt(0)
        sample_good_groups = int(good_group_mask.sum().item())
        sample_purities = (
            group_good_counts[good_group_mask].to(torch.float64)
            / stage.group_sizes[
                batch_index, roots[good_group_mask]
            ].to(torch.float64)
        )
        # Purity is defined over good Groups, not over samples.  A no-good
        # sample contributes no valid purity observation; inserting a
        # synthetic zero would bias the dataset mean and contradict that
        # denominator.
        purities.extend(sample_purities.detach().cpu().tolist())
        good_group_counts.append(sample_good_groups)
        # Fragmentation/duplication is target-local.  Counting the union of
        # good groups would incorrectly label two different GT instances,
        # each covered by one group, as a duplicate.
        sample_duplicate_groups = 0
        for target_index in range(query_target_iou.shape[1]):
            _, target_group_good_counts = (
                _group_oracle_iou_and_good_counts(
                    stage,
                    batch_index,
                    query_target_iou[:, target_index],
                    roots,
                    good_iou_threshold=good_iou_threshold,
                )
            )
            target_good_group_count = int(
                target_group_good_counts.gt(0).sum().item()
            )
            sample_duplicate_groups += max(target_good_group_count - 1, 0)
        duplicate_group_counts.append(sample_duplicate_groups)
    return (
        torch.tensor(purities, dtype=torch.float64),
        torch.tensor(good_group_counts, dtype=torch.float64),
        torch.tensor(duplicate_group_counts, dtype=torch.float64),
        bridge_flags,
        tuple(iou_matrices),
    )


def _core_scalars(core: Any, stage_index: int, final_stage: int):
    if core is None:
        return None, None
    alpha_region = (
        core.region_pooler.alpha_region[stage_index].float().detach()
    )
    alpha_fb = (
        core.feedback.alpha_feedback[stage_index].float().detach()
        if stage_index < final_stage
        else None
    )
    return alpha_region, alpha_fb


def _quality_representatives(raw_outputs: Any, final_stage: Any) -> Tensor:
    representatives = getattr(
        raw_outputs,
        "group_representative_query_indices",
        None,
    )
    if representatives is not None:
        return representatives.to(
            device=final_stage.query_mask_logits.device,
            dtype=torch.long,
        )
    quality = getattr(final_stage, "member_quality_logits", None)
    if quality is None:
        raise ValueError("formal diagnostics require member_quality_logits")
    result = torch.full_like(final_stage.query_to_group, -1)
    for batch_index in range(quality.shape[0]):
        roots = torch.nonzero(
            final_stage.group_valid_mask[batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        for root in roots:
            members = _group_members(final_stage, batch_index, root)
            # torch.argmax deterministically keeps the lowest physical query
            # id because members are returned in ascending order.
            result[batch_index, root] = members[
                quality[batch_index, members].argmax()
            ]
    return result


def _average_ranks(values: Tensor) -> Tensor:
    """Return one-based average ranks, assigning equal values equal ranks."""

    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return values
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rank values must be finite")
    order = torch.argsort(values)
    sorted_values = values[order]
    is_new = torch.ones(
        sorted_values.numel(),
        dtype=torch.bool,
        device=sorted_values.device,
    )
    is_new[1:] = sorted_values[1:].ne(sorted_values[:-1])
    group_ids = is_new.to(torch.long).cumsum(dim=0) - 1
    counts = torch.bincount(group_ids)
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    average_by_group = 0.5 * (
        starts.to(torch.float64) + 1.0 + ends.to(torch.float64)
    )
    sorted_ranks = average_by_group[group_ids]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _spearman(values_x: Tensor, values_y: Tensor) -> Optional[float]:
    """Return tie-aware Spearman correlation, or ``None`` if undefined."""

    values_x = torch.as_tensor(values_x, dtype=torch.float64).reshape(-1)
    values_y = torch.as_tensor(values_y, dtype=torch.float64).reshape(-1)
    if values_x.shape != values_y.shape:
        raise ValueError("Spearman inputs must have equal shapes")
    if values_x.numel() < 2:
        return None
    ranks_x = _average_ranks(values_x)
    ranks_y = _average_ranks(values_y)
    centered_x = ranks_x - ranks_x.mean()
    centered_y = ranks_y - ranks_y.mean()
    denominator = centered_x.square().sum().sqrt()
    denominator = denominator * centered_y.square().sum().sqrt()
    if not bool(denominator > 0):
        return None
    correlation = (centered_x * centered_y).sum() / denominator
    return float(correlation.clamp(-1.0, 1.0).item())


def _update_ref_stagewise_internal(
    diagnostics: Any,
    stage: Any,
    normalized_targets: Sequence[Any],
    embed_masks: Tensor,
    iou_matrices: Sequence[Tensor],
    *,
    good_iou_threshold: float,
    quality_representatives: Optional[Tensor] = None,
) -> None:
    """Collect per-stage internal associations without changing selection."""

    if stage.group_class_logits is None:
        raise ValueError(
            f"Ref stage {stage.stage_index} is missing group_class_logits"
        )
    for batch_index, (target, iou_matrix) in enumerate(
        zip(normalized_targets, iou_matrices)
    ):
        if target.condition_ids.numel() != 1 or iou_matrix.shape[1] != 1:
            raise ValueError(
                "Ref stagewise diagnostics require exactly one GT per "
                "effective sample"
            )
        target_condition = int(target.condition_ids[0].item())
        if not 0 <= target_condition < embed_masks.shape[1] - 1:
            raise ValueError("Ref target condition is outside foreground")
        query_ious = iou_matrix[:, 0]
        roots = torch.nonzero(
            stage.group_valid_mask[batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        group_logits = stage.group_class_logits[
            batch_index, roots
        ].float().masked_fill(
            ~embed_masks[batch_index].unsqueeze(0),
            -1e9,
        )
        group_scores = group_logits.softmax(dim=-1)[:, target_condition]
        selected_position = int(group_scores.argmax().item())
        selected_root = roots[selected_position]
        selected_members = _group_members(
            stage,
            batch_index,
            selected_root,
        )
        group_oracle_ious, good_counts = (
            _group_oracle_iou_and_good_counts(
                stage,
                batch_index,
                query_ious,
                roots,
                good_iou_threshold=good_iou_threshold,
            )
        )
        selected_group_oracle_iou = group_oracle_ious[
            selected_position
        ]
        query_oracle_iou = query_ious.max()
        good_queries = query_ious.ge(good_iou_threshold)
        total_good = int(good_counts.sum().item())
        if total_good:
            concentration_position = int(good_counts.argmax().item())
            good_concentration = (
                float(good_counts[concentration_position].item())
                / total_good
            )
            best_group_purity = (
                float(good_counts[concentration_position].item())
                / int(
                    stage.group_sizes[
                        batch_index,
                        roots[concentration_position],
                    ].item()
                )
            )
        else:
            good_concentration = None
            best_group_purity = None

        wrong_member = None
        member_gap = None
        if quality_representatives is not None:
            quality_query = int(
                quality_representatives[
                    batch_index, selected_root
                ].item()
            )
            if quality_query < 0 or not bool(
                stage.query_to_group[
                    batch_index, quality_query
                ].eq(selected_root)
            ):
                raise RuntimeError(
                    "final quality representative is not in its selected "
                    "group"
                )
            quality_iou = query_ious[quality_query]
            wrong_member = bool(
                selected_group_oracle_iou >= good_iou_threshold
                and quality_iou < good_iou_threshold
            )
            member_gap = selected_group_oracle_iou - quality_iou

        diagnostics.update_stagewise_internal(
            int(stage.stage_index),
            query_oracle_giou=query_oracle_iou,
            no_good_query7=query_oracle_iou < good_iou_threshold,
            selected_group_contains_good7=bool(
                good_queries[selected_members].any()
            ),
            routing_gap_giou=(
                query_oracle_iou - selected_group_oracle_iou
            ),
            cond_iou_spearman=_spearman(
                group_scores,
                group_oracle_ious,
            ),
            good_concentration7=good_concentration,
            best_group_purity7=best_group_purity,
            wrong_member7=wrong_member,
            member_gap_giou=member_gap,
        )


def _update_ref_diagnostics(
    diagnostics: Any,
    raw_outputs: Any,
    final_stage: Any,
    normalized_targets: Sequence[Any],
    matching: Any,
    embed_masks: Tensor,
    final_iou_matrices: Sequence[Tensor],
    evaluation_sizes: Sequence[
        Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
    ],
    diagnostic_gt_masks: Sequence[Optional[Tensor]],
    ignore_masks: Sequence[Optional[Tensor]],
    *,
    mask_threshold: float,
    good_iou_threshold: float,
) -> None:
    raw_class_logits = getattr(raw_outputs, "raw_query_class_logits", None)
    if raw_class_logits is None:
        raise ValueError(
            "Ref diagnostics require group_return_raw_query_outputs=True"
        )
    raw_class_logits = raw_class_logits.to(final_stage.query_mask_logits.device)
    if raw_class_logits.shape[:2] != final_stage.query_mask_logits.shape[:2]:
        raise ValueError("raw Query class and mask batches must match")
    representatives = _quality_representatives(raw_outputs, final_stage)

    for batch_index, (
        target,
        spatial_sizes,
        diagnostic_gt_mask,
        ignore_mask,
    ) in enumerate(
        zip(
            normalized_targets,
            evaluation_sizes,
            diagnostic_gt_masks,
            ignore_masks,
        )
    ):
        if diagnostic_gt_mask is None:
            raise ValueError(
                "RefSeg/ReaSeg variant IoU requires exact original GT"
            )
        target_union = diagnostic_gt_mask.to(
            device=final_stage.query_mask_logits.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        query_ious = final_iou_matrices[batch_index][:, 0]

        if target.condition_ids.numel() == 0:
            raise ValueError("Ref diagnostics require a target condition id")
        target_condition = int(target.condition_ids[0].item())
        if not 0 <= target_condition < embed_masks.shape[1] - 1:
            raise ValueError(
                "Ref target condition id is outside the foreground columns: "
                f"{target_condition}"
            )
        if not bool(embed_masks[batch_index, target_condition]):
            raise ValueError(
                "Ref target condition id points to a padded condition column"
            )
        # Formal RefSeg computes softmax/argmax on the model-output dtype
        # (normally BF16).  Promoting here can break a quantized tie and pick
        # a different Query, so the counterfactual keeps that exact dtype.
        # These are the already-masked logits emitted by
        # MaskLATSegmentor.get_class_prediction.  Do not replace invalid-column
        # values here: formal post-processing consumes this exact tensor.
        raw_logits = raw_class_logits[batch_index]
        raw_probabilities = raw_logits.softmax(dim=-1)
        # Ref expansion can leave several globally valid foreground columns.
        # Formal selection is nevertheless local to this effective sample:
        # score only its normalized target condition, exactly as the
        # single-foreground refseg postprocessor does after expansion.
        raw_scores = raw_probabilities[:, target_condition]
        selected_query = int(raw_scores.argmax().item())

        roots = torch.nonzero(
            final_stage.group_valid_mask[batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        # GroupConditionClassifier already applied the model's finite
        # invalid-condition mask.  Keeping its exact values and dtype makes
        # softmax/argmax identical to the formal RefSeg postprocessor.
        group_logits = final_stage.group_class_logits[batch_index, roots]
        group_probabilities = group_logits.softmax(dim=-1)
        group_scores = group_probabilities[:, target_condition]
        selected_position = int(group_scores.argmax().item())
        selected_root = int(roots[selected_position].item())
        selected_probabilities = group_probabilities[selected_position]
        selected_logits = group_logits[selected_position]
        selected_class = int(selected_probabilities.argmax().item())
        members = _group_members(
            final_stage,
            batch_index,
            torch.tensor(
                selected_root,
                dtype=torch.long,
                device=roots.device,
            ),
        )
        selected_top_class = int(
            members[raw_scores[members].argmax()].item()
        )
        selected_quality = int(
            representatives[batch_index, selected_root].item()
        )
        if selected_quality < 0 or not bool(
            final_stage.query_to_group[batch_index, selected_quality].eq(
                selected_root
            )
        ):
            raise RuntimeError("formal representative is not in its selected group")

        selected_oracle = int(members[query_ious[members].argmax()].item())
        global_query_oracle = int(query_ious.argmax().item())
        global_oracle_root = int(
            final_stage.query_to_group[
                batch_index, global_query_oracle
            ].item()
        )
        quality_candidates = representatives[batch_index, roots]
        if bool((quality_candidates < 0).any()):
            raise RuntimeError("a valid group is missing its quality representative")
        global_group_oracle = int(
            quality_candidates[
                query_ious[quality_candidates].argmax()
            ].item()
        )

        variant_queries = {
            "selected_query_mask": selected_query,
            "selected_group_top_class_query_mask": selected_top_class,
            "selected_group_quality_query_mask": selected_quality,
            "selected_group_oracle_member_mask": selected_oracle,
            "global_group_oracle_member_mask": global_group_oracle,
            "global_200q_oracle_query_mask": global_query_oracle,
        }
        variant_logits = torch.stack(
            [
                final_stage.query_mask_logits[batch_index, query_id]
                for query_id in variant_queries.values()
            ]
        )
        scaled_size, output_size = spatial_sizes
        intersections, unions, variant_ious = _query_target_iou(
            variant_logits,
            target_union,
            mask_threshold=mask_threshold,
            ignore_mask=ignore_mask,
            sam_canvas_size=tuple(target.masks.shape[-2:]),
            target_masks_are_original=True,
            scaled_size=scaled_size,
            output_size=output_size,
            sam_valid_box=(
                final_stage.sam_valid_boxes_normalized[batch_index]
            ),
        )
        variant_stats = {}
        for position, variant in enumerate(variant_queries):
            variant_stats[variant] = (
                intersections[position, 0],
                unions[position, 0],
                variant_ious[position, 0],
            )
        baseline_iou = variant_stats["selected_query_mask"][2]
        for variant, (intersection, union, sample_iou) in variant_stats.items():
            diagnostics.update_refseg_variant(
                variant,
                intersection=intersection,
                union=union,
                sample_iou=sample_iou,
                improved=sample_iou > baseline_iou,
                worsened=sample_iou < baseline_iou,
            )

        good_queries = query_ious.ge(good_iou_threshold)
        selected_group_contains_good = bool(good_queries[members].any())
        group_oracle_ious, good_counts = (
            _group_oracle_iou_and_good_counts(
                final_stage,
                batch_index,
                query_ious,
                roots,
                good_iou_threshold=good_iou_threshold,
            )
        )
        # Roots are physically sorted.  Stable descending argsort therefore
        # preserves the formal lowest-root tie break without one GPU sync per
        # Group.
        ranked_positions = torch.argsort(
            group_scores,
            descending=True,
            stable=True,
        )
        ranked_good = group_oracle_ious[
            ranked_positions
        ].ge(good_iou_threshold)
        first_good_positions = torch.nonzero(
            ranked_good,
            as_tuple=False,
        ).flatten()
        first_good_rank = (
            int(first_good_positions[0].item()) + 1
            if first_good_positions.numel()
            else None
        )
        formal_iou = variant_stats[
            "selected_group_quality_query_mask"
        ][2]
        selected_oracle_iou = variant_stats[
            "selected_group_oracle_member_mask"
        ][2]
        global_query_iou = variant_stats[
            "global_200q_oracle_query_mask"
        ][2]

        sample_match = matching[batch_index]
        target_zero_matches = torch.nonzero(
            sample_match.target_indices.eq(0),
            as_tuple=False,
        ).flatten()
        if target_zero_matches.numel() != 1:
            raise RuntimeError(
                "Ref diagnostics require exactly one final-stage Hungarian "
                "member matched to the sole GT"
            )
        matched_member = int(
            sample_match.member_indices[
                target_zero_matches[0]
            ].item()
        )
        matched_iou = query_ious[matched_member]
        variant_good_queries = good_queries.clone()
        variant_good_queries[matched_member] = False
        variant_available = bool(variant_good_queries.any())
        rescue_opportunity = bool(
            matched_iou < good_iou_threshold and variant_available
        )
        matched_good = bool(matched_iou >= good_iou_threshold)

        total_good = int(good_counts.sum().item())
        if total_good:
            concentration_position = int(good_counts.argmax().item())
            concentration_root = roots[concentration_position]
            good_concentration = (
                float(good_counts[concentration_position].item())
                / total_good
            )
            best_group_purity = (
                float(good_counts[concentration_position].item())
                / int(
                    final_stage.group_sizes[
                        batch_index,
                        concentration_root,
                    ].item()
                )
            )
        else:
            good_concentration = None
            best_group_purity = None

        diagnostics.update_idea_diagnostics(
            variant_available7=variant_available,
            variant_selected7=(
                bool(formal_iou >= good_iou_threshold)
                and selected_quality != matched_member
            ),
            rescue_opportunity7=rescue_opportunity,
            rescue7=(
                bool(formal_iou >= good_iou_threshold)
                if rescue_opportunity
                else None
            ),
            matched_good_lost7=(
                bool(formal_iou < good_iou_threshold)
                if matched_good
                else None
            ),
            good_concentration7=good_concentration,
            best_group_purity7=best_group_purity,
            cond_iou_spearman=_spearman(
                group_scores,
                group_oracle_ious,
            ),
        )
        diagnostics.update_routing(
            selected_cond_probability=selected_probabilities[
                target_condition
            ],
            selected_cond_margin=(
                selected_logits[target_condition]
                - selected_logits[-1]
            ),
            selected_cond_is_gt=(selected_class == target_condition),
            selected_cond_is_bg=(
                selected_class == selected_probabilities.numel() - 1
            ),
            selected_cond_is_other=(
                selected_class not in (
                    target_condition,
                    selected_probabilities.numel() - 1,
                )
            ),
            selected_group_contains_good7=selected_group_contains_good,
            selected_group_same_as_oracle_group=(
                selected_root == global_oracle_root
            ),
            success7=formal_iou >= good_iou_threshold,
            wrong_group7=(
                selected_oracle_iou < good_iou_threshold
                and global_query_iou >= good_iou_threshold
            ),
            wrong_member7=(
                formal_iou < good_iou_threshold
                and selected_oracle_iou >= good_iou_threshold
            ),
            no_good_query7=global_query_iou < good_iou_threshold,
            first_good_group_rank=first_good_rank,
            within_group_member_gap_giou=(
                selected_oracle_iou - formal_iou
            ),
            cross_group_routing_gap_giou=(
                global_query_iou - selected_oracle_iou
            ),
        )


def _update_multigt_diagnostics(
    diagnostics: Any,
    task_name: str,
    final_stage: Any,
    normalized_targets: Sequence[Any],
    matching: Any,
    embed_masks: Tensor,
    final_iou_matrices: Sequence[Tensor],
) -> None:
    target_count = sum(int(target.condition_ids.numel()) for target in normalized_targets)
    condition_count = int(embed_masks[:, :-1].sum().item())
    group_count = int(final_stage.group_valid_mask.sum().item())
    # In the formal multi-GT contract a background group is a valid group left
    # unmatched by Hungarian assignment, not a class-argmax prediction.
    bg_groups = group_count - int(matching.num_matches)
    duplicate_predictions = 0
    same_condition_samples = 0
    recall50_hits = 0
    recall70_hits = 0

    for batch_index, (target, sample_match) in enumerate(
        zip(normalized_targets, matching)
    ):
        labels = target.condition_ids
        same_condition_samples += int(
            labels.numel() > torch.unique(labels).numel()
        )
        roots = torch.nonzero(
            final_stage.group_valid_mask[batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        logits = final_stage.group_class_logits[batch_index, roots].float()
        logits = logits.masked_fill(
            ~embed_masks[batch_index].unsqueeze(0),
            -1e9,
        )
        predicted_conditions = logits.argmax(dim=-1)
        foreground = predicted_conditions[
            predicted_conditions.ne(logits.shape[-1] - 1)
        ]
        if foreground.numel():
            _, counts = torch.unique(foreground, return_counts=True)
            duplicate_predictions += int((counts - 1).clamp_min(0).sum())

        iou_matrix = final_iou_matrices[batch_index]
        for match_index in range(len(sample_match)):
            query_id = sample_match.member_indices[match_index]
            target_id = sample_match.target_indices[match_index]
            matched_iou = iou_matrix[query_id, target_id]
            recall50_hits += int(matched_iou >= 0.5)
            recall70_hits += int(matched_iou >= 0.7)

    diagnostics.update_multigt(
        task_name,
        targets=target_count,
        conditions=condition_count,
        groups=group_count,
        matched_groups=matching.num_matches,
        bg_groups=bg_groups,
        same_cond_multi_gt_samples=same_condition_samples,
        sample_count=len(normalized_targets),
        unmatched_targets=sum(
            int(sample.unmatched_target_indices.numel()) for sample in matching
        ),
        group_recall50_hits=recall50_hits,
        group_recall70_hits=recall70_hits,
        recall_denominator=target_count,
        duplicate_predictions=duplicate_predictions,
        duplicate_denominator=group_count,
    )


@torch.no_grad()
def update_topology_group_diagnostics(
    diagnostics: Any,
    raw_outputs: Any,
    data_samples: Any = None,
    *,
    targets: Any = None,
    task_name: Optional[str] = None,
    embed_masks: Optional[Tensor] = None,
    diagnostic_gt_masks: Any = None,
    ignore_masks: Any = None,
    normalizer: Any = None,
    matcher: Any = None,
    topology_group_core: Any = None,
    require_ground_truth: bool = True,
    collect_all_stage_gt: bool = True,
    mask_threshold: float = 0.5,
    good_iou_threshold: float = 0.7,
    group_iou_threshold: float = 0.7,
) -> TopologyEvaluationUpdate:
    """Update formal topology diagnostics from one raw evaluation batch.

    Args:
        diagnostics: A mutable ``GroupDiagnostics`` instance.
        raw_outputs: Raw ``MaskLATSegmentorOutput`` before task post-processing.
        data_samples: Collated MaskLAT data sample containing GT and task name.
        targets: Optional explicit target payload, useful outside MMEngine.
        embed_masks: Optional exact condition validity mask override.  When
            omitted, ``raw_outputs.condition_valid_mask`` is required.
        diagnostic_gt_masks: Optional exact original-annotation RefSeg/ReaSeg
            GT override.  Without it, every effective sample must provide
            ``image_infos[*]['diagnostic_gt_mask']``.
        ignore_masks: Optional explicit ReaSeg ignore-mask override.  ReaSeg
            otherwise requires one original-resolution COCO RLE at
            ``image_infos[*]['diagnostic_ignore_mask']``.
        normalizer, matcher: Optional injected instances.  Defaults lazily
            instantiate ``UnifiedTargetNormalizer`` and deterministic
            ``HierarchicalGroupMatcher(use_sample_point=False)``.
        collect_all_stage_gt: If true, restore all Query masks and compute GT
            IoUs for stages 0--9.  If false, perform expensive GT restoration
            only for the final stage while retaining cheap topology/activation
            statistics for earlier stages.

    The learned group/quality predictions are fixed before any GT operation.
    GT is used only for diagnostic IoUs, explicitly named oracle variants,
    and Hungarian evaluation summaries.
    """

    stages = getattr(raw_outputs, "group_stage_outputs", None)
    if stages is None:
        raise ValueError("raw outputs do not contain topology group stages")
    stages = tuple(stages)
    if len(stages) != int(diagnostics.num_stages):
        raise ValueError(
            "raw topology stage count and diagnostics differ: "
            f"{len(stages)} != {diagnostics.num_stages}"
        )
    if [int(stage.stage_index) for stage in stages] != list(range(len(stages))):
        raise ValueError("raw topology stages must be ordered st0..stN")
    final_stage = stages[-1]
    if final_stage.group_class_logits is None:
        raise ValueError("final topology stage is missing group_class_logits")
    batch_size = int(final_stage.query_mask_logits.shape[0])
    num_classes = int(final_stage.group_class_logits.shape[-1])
    resolved_task = _resolve_task_name(task_name, data_samples)
    if embed_masks is None:
        embed_masks = getattr(raw_outputs, "condition_valid_mask", None)
    if embed_masks is None:
        raise ValueError(
            "raw outputs must expose condition_valid_mask for formal "
            "diagnostics"
        )
    resolved_embed_masks = _validate_embed_masks(final_stage, embed_masks)

    payload = _target_payload(
        data_samples,
        targets,
        task_name=resolved_task,
        effective_batch_size=batch_size,
        num_classes=num_classes,
    )
    if payload is None:
        if require_ground_truth:
            raise ValueError(
                f"task {resolved_task!r} has no GT for formal diagnostics"
            )
        for stage_index, stage in enumerate(stages):
            alpha_region, alpha_fb = _core_scalars(
                topology_group_core,
                stage_index,
                len(stages) - 1,
            )
            diagnostics.update_stage(
                stage,
                alpha_region=alpha_region,
                alpha_fb=alpha_fb,
            )
        return TopologyEvaluationUpdate(
            task_name=resolved_task,
            normalized_targets=None,
            matching=None,
            batch_size=batch_size,
        )

    normalizer = _default_normalizer() if normalizer is None else normalizer
    normalized_targets = normalizer(
        payload,
        task_name=resolved_task,
        batch_size=batch_size,
        device=final_stage.query_mask_logits.device,
        mask_dtype=torch.float32,
        num_classes=num_classes,
        embed_masks=resolved_embed_masks,
    )
    if normalized_targets is None:
        raise ValueError("formal segmentation diagnostics cannot skip GT targets")
    evaluation_sizes = _evaluation_sizes(
        data_samples,
        task_name=resolved_task,
        batch_size=batch_size,
    )
    resolved_diagnostic_gt_masks = _resolve_diagnostic_gt_masks(
        data_samples,
        diagnostic_gt_masks,
        task_name=resolved_task,
        batch_size=batch_size,
    )
    resolved_ignore_masks = _resolve_ignore_masks(
        data_samples,
        ignore_masks,
        task_name=resolved_task,
        batch_size=batch_size,
    )

    final_iou_matrices = None
    for stage_index, stage in enumerate(stages):
        collect_stage_gt = (
            bool(collect_all_stage_gt)
            or stage_index == len(stages) - 1
        )
        if collect_stage_gt:
            (
                group_purity,
                good_group_count,
                duplicate_group_count,
                bridge_flags,
                iou_matrices,
            ) = _stage_gt_arguments(
                stage,
                normalized_targets,
                resolved_diagnostic_gt_masks,
                evaluation_sizes,
                resolved_ignore_masks,
                mask_threshold=mask_threshold,
                good_iou_threshold=good_iou_threshold,
                group_iou_threshold=group_iou_threshold,
            )
        else:
            group_purity = None
            good_group_count = None
            duplicate_group_count = None
            iou_matrices = None
            bridge_flags = _stage_bridge_flags(
                stage,
                group_iou_threshold=group_iou_threshold,
            )
        alpha_region, alpha_fb = _core_scalars(
            topology_group_core,
            stage_index,
            len(stages) - 1,
        )
        diagnostics.update_stage(
            stage,
            group_purity=group_purity,
            good_group_count7=good_group_count,
            duplicate_group_count7=duplicate_group_count,
            bridge_flags=bridge_flags,
            alpha_region=alpha_region,
            alpha_fb=alpha_fb,
        )
        if resolved_task in REF_TASKS and iou_matrices is not None:
            _update_ref_stagewise_internal(
                diagnostics,
                stage,
                normalized_targets,
                resolved_embed_masks,
                iou_matrices,
                good_iou_threshold=good_iou_threshold,
                quality_representatives=(
                    _quality_representatives(raw_outputs, final_stage)
                    if stage_index == len(stages) - 1
                    else None
                ),
            )
        if stage_index == len(stages) - 1:
            if iou_matrices is None:  # pragma: no cover - guarded above.
                raise RuntimeError("final-stage GT IoUs were not collected")
            final_iou_matrices = iou_matrices
    if final_iou_matrices is None:  # pragma: no cover - non-empty stages above.
        raise RuntimeError("final-stage IoUs were not computed")

    matcher = _default_matcher() if matcher is None else matcher
    matching = matcher(
        final_stage,
        normalized_targets,
        embed_masks=resolved_embed_masks,
    )
    if resolved_task in REF_TASKS:
        _update_ref_diagnostics(
            diagnostics,
            raw_outputs,
            final_stage,
            normalized_targets,
            matching,
            resolved_embed_masks,
            final_iou_matrices,
            evaluation_sizes,
            resolved_diagnostic_gt_masks,
            resolved_ignore_masks,
            mask_threshold=mask_threshold,
            good_iou_threshold=good_iou_threshold,
        )
    _update_multigt_diagnostics(
        diagnostics,
        resolved_task,
        final_stage,
        normalized_targets,
        matching,
        resolved_embed_masks,
        final_iou_matrices,
    )
    return TopologyEvaluationUpdate(
        task_name=resolved_task,
        normalized_targets=normalized_targets,
        matching=matching,
        batch_size=batch_size,
    )


__all__ = [
    "TopologyEvaluationUpdate",
    "update_topology_group_diagnostics",
]
