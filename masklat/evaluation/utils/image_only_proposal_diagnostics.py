"""Pure helpers for image-only, stage-wise mask-proposal diagnostics.

The helpers in this module never call the language model and never alter the
segmentation model.  Ground truth is used only after Query masks have been
predicted, binarized, and grouped.
"""

from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _image_size_tuple(value, *, name: str) -> Tuple[int, int]:
    if isinstance(value, Mapping):
        value = (value.get("height"), value.get("width"))
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain (height, width), got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def _canonical_mask_payload(value) -> str:
    """Return a deterministic representation for duplicate-key validation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_unique_image_manifest(records: Sequence[Mapping]) -> list[Dict]:
    """Collapse expression records to unique images and unique annotations.

    The representative dataset index is the first deterministic expression
    record for an image.  Repeated expressions for the same annotation are
    removed using ``(image_id, annotation_id)`` only.  Different annotation
    ids remain distinct even when their encoded masks happen to be identical.
    """

    images: "OrderedDict[str, Dict]" = OrderedDict()
    for dataset_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"dataset record {dataset_index} is not a mapping")
        image_info = record.get("image_info")
        if not isinstance(image_info, Mapping):
            raise ValueError(f"dataset record {dataset_index} has no image_info")
        if "image_id" not in image_info or "annotation_id" not in image_info:
            raise ValueError(
                f"dataset record {dataset_index} lacks image_id/annotation_id"
            )
        image_id = str(image_info["image_id"])
        annotation_id = str(image_info["annotation_id"])
        image_size = _image_size_tuple(
            record.get("image_size", (image_info.get("height"), image_info.get("width"))),
            name=f"record[{dataset_index}].image_size",
        )
        image_file = str(record.get("image_file", image_info.get("file_name", "")))
        if not image_file:
            raise ValueError(f"dataset record {dataset_index} has no image file")
        encoded_mask = image_info.get("diagnostic_gt_mask")
        if encoded_mask is None:
            annotations = record.get("annotations")
            if not isinstance(annotations, Sequence) or len(annotations) != 1:
                raise ValueError(
                    f"dataset record {dataset_index} has no exact diagnostic GT"
                )
            encoded_mask = annotations[0].get("segmentation")
        if encoded_mask is None:
            raise ValueError(
                f"dataset record {dataset_index} has an empty diagnostic GT"
            )

        entry = images.get(image_id)
        if entry is None:
            entry = {
                "image_id": image_id,
                "image_file": image_file,
                "image_size": image_size,
                "representative_index": int(dataset_index),
                "annotations": OrderedDict(),
            }
            images[image_id] = entry
        elif entry["image_file"] != image_file or entry["image_size"] != image_size:
            raise ValueError(
                f"image {image_id} has inconsistent file/size across expressions"
            )

        annotation = {
            "annotation_id": annotation_id,
            "segmentation": copy.deepcopy(encoded_mask),
            # Preserve how many referring-expression records point at this
            # unique GT.  The image-only forward is still executed only once,
            # while this multiplicity lets downstream diagnostics reproduce
            # the expression-weighted ``goodQ`` convention used by the formal
            # RefSeg diagnostic.
            "expression_count": 1,
        }
        previous = entry["annotations"].get(annotation_id)
        if previous is None:
            entry["annotations"][annotation_id] = annotation
        elif _canonical_mask_payload(previous["segmentation"]) != _canonical_mask_payload(
            annotation["segmentation"]
        ):
            raise ValueError(
                "the same (image_id, annotation_id) has inconsistent masks: "
                f"({image_id}, {annotation_id})"
            )
        else:
            previous["expression_count"] = int(previous["expression_count"]) + 1

    manifest = []
    for entry in images.values():
        annotations = list(entry["annotations"].values())
        if not annotations:
            raise ValueError(f"image {entry['image_id']} has no unique GT instances")
        manifest.append({**entry, "annotations": annotations})
    if not manifest:
        raise ValueError("the RefSeg dataset contains no unique images")
    return manifest


def _as_metric_matrix(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def good_query_counts_per_gt(iou, threshold: float = 0.7) -> np.ndarray:
    """Count high-IoU Query masks independently for every GT instance."""

    matrix = _as_metric_matrix(iou, name="iou")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be finite and in [0,1], got {threshold}"
        )
    if bool((matrix < -1.0e-9).any()) or bool((matrix > 1.0 + 1.0e-9).any()):
        raise ValueError("IoU values must lie in [0,1]")
    return np.count_nonzero(matrix >= threshold, axis=0).astype(
        np.int64,
        copy=False,
    )


def maximum_cardinality_iou_matches(iou, threshold: float) -> int:
    """Return maximum one-to-one matches on the ``IoU >= threshold`` graph."""

    matrix = _as_metric_matrix(iou, name="iou")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be finite and in [0,1], got {threshold}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0
    edges = matrix >= threshold
    rows, columns = linear_sum_assignment(-edges.astype(np.int8))
    return int(edges[rows, columns].sum())


def proposal_oracle_statistics(
    iou,
    intersections,
    unions,
    gt_areas,
    *,
    thresholds: Sequence[float] = (0.5, 0.7),
) -> Dict:
    """Score a GT-free proposal set against unique GT with one-to-one oracles.

    Hungarian matching maximizes total IoU.  A GT without a positive-IoU
    matched proposal is treated as having an empty prediction: IoU and
    intersection are zero while its full area contributes to the cIoU union.
    Threshold recalls are solved independently as maximum-cardinality
    bipartite matching problems.
    """

    iou = _as_metric_matrix(iou, name="iou")
    intersections = _as_metric_matrix(intersections, name="intersections")
    unions = _as_metric_matrix(unions, name="unions")
    if intersections.shape != iou.shape or unions.shape != iou.shape:
        raise ValueError("IoU/intersection/union matrices must have the same shape")
    if bool((iou < -1.0e-9).any()) or bool((iou > 1.0 + 1.0e-9).any()):
        raise ValueError("IoU values must lie in [0,1]")
    if bool((intersections < 0).any()) or bool((unions < 0).any()):
        raise ValueError("intersection/union values must be non-negative")
    gt_areas = np.asarray(gt_areas, dtype=np.float64).reshape(-1)
    if gt_areas.shape[0] != iou.shape[1]:
        raise ValueError(
            f"gt_areas length {gt_areas.shape[0]} != target count {iou.shape[1]}"
        )
    if not np.isfinite(gt_areas).all() or bool((gt_areas <= 0).any()):
        raise ValueError("every GT area must be finite and positive")

    target_count = int(iou.shape[1])
    matched_intersection = 0.0
    matched_union = 0.0
    matched_iou = 0.0
    positive_targets = set()
    if iou.shape[0] > 0 and target_count > 0:
        rows, columns = linear_sum_assignment(-iou)
        for row, column in zip(rows.tolist(), columns.tolist()):
            value = float(iou[row, column])
            if value <= 0.0:
                continue
            positive_targets.add(int(column))
            matched_intersection += float(intersections[row, column])
            matched_union += float(unions[row, column])
            matched_iou += value
    unmatched_targets = set(range(target_count)) - positive_targets
    if unmatched_targets:
        matched_union += float(gt_areas[list(sorted(unmatched_targets))].sum())

    independent_best = (
        iou.max(axis=0)
        if iou.shape[0] > 0
        else np.zeros(target_count, dtype=np.float64)
    )
    threshold_matches = {
        f"r{int(round(float(threshold) * 100)):02d}_matches": (
            maximum_cardinality_iou_matches(iou, float(threshold))
        )
        for threshold in thresholds
    }
    independent_matches = {
        f"ind_r{int(round(float(threshold) * 100)):02d}_matches": int(
            (independent_best >= float(threshold)).sum()
        )
        for threshold in thresholds
    }
    return {
        "intersection_sum": matched_intersection,
        "union_sum": matched_union,
        "iou_sum": matched_iou,
        "gt_count": target_count,
        "independent_best_iou_sum": float(independent_best.sum()),
        **threshold_matches,
        **independent_matches,
    }


def group_member_matrices(
    query_iou,
    query_intersections,
    query_unions,
    groups: Sequence[Sequence[int]],
):
    """Build Group×GT oracle-member matrices with deterministic ties."""

    query_iou = _as_metric_matrix(query_iou, name="query_iou")
    query_intersections = _as_metric_matrix(
        query_intersections,
        name="query_intersections",
    )
    query_unions = _as_metric_matrix(query_unions, name="query_unions")
    if query_intersections.shape != query_iou.shape or query_unions.shape != query_iou.shape:
        raise ValueError("query metric matrices must have the same shape")
    num_queries, num_targets = query_iou.shape
    flattened = [int(query_id) for group in groups for query_id in group]
    if sorted(flattened) != list(range(num_queries)):
        raise ValueError("groups must partition every Query exactly once")

    group_iou = np.zeros((len(groups), num_targets), dtype=np.float64)
    group_intersections = np.zeros_like(group_iou)
    group_unions = np.zeros_like(group_iou)
    member_ids = np.zeros((len(groups), num_targets), dtype=np.int64)
    for group_id, group in enumerate(groups):
        members = np.asarray(sorted(map(int, group)), dtype=np.int64)
        local_iou = query_iou[members]
        # np.argmax returns the first maximum and members are sorted, giving
        # the required smallest-Query-id tie break.
        local_best = np.argmax(local_iou, axis=0)
        selected = members[local_best]
        columns = np.arange(num_targets, dtype=np.int64)
        group_iou[group_id] = query_iou[selected, columns]
        group_intersections[group_id] = query_intersections[selected, columns]
        group_unions[group_id] = query_unions[selected, columns]
        member_ids[group_id] = selected
    return group_iou, group_intersections, group_unions, member_ids


__all__ = [
    "build_unique_image_manifest",
    "group_member_matrices",
    "maximum_cardinality_iou_matches",
    "proposal_oracle_statistics",
]
