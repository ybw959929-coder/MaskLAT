"""Read-only, original-resolution Query/GT diagnostics for decoder st1--st9.

These are oracle proposal-quality measurements, not the model's chosen-mask
accuracy. All Queries are included, without class filtering or matching. The
production RefSeg restoration and binarization helpers are imported lazily so
that the aggregation code is usable without loading the model dependency tree.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Integral

import torch


THRESHOLDS = (0.5, 0.7, 0.9)
_THRESHOLD_KEYS = tuple(str(value) for value in THRESHOLDS)
_STAGES = tuple(f"st{index}" for index in range(1, 10))
_SCHEMA_VERSION = 1


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _size_pair(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain height and width")
    return tuple(_positive_integer(item, f"{name} component") for item in value)


def _binary_gt(gt_mask, image_size):
    gt = torch.as_tensor(gt_mask).detach()
    if gt.ndim != 2 or tuple(gt.shape) != image_size:
        raise ValueError("GT must be a two-dimensional original-resolution mask")
    if gt.is_complex() or not torch.isfinite(gt).all():
        raise ValueError("GT contains non-finite or complex values")
    if not torch.all((gt == 0) | (gt == 1)):
        raise ValueError("GT must be binary (0/1 or bool), not an instance-label map")
    gt = gt.bool()
    if not bool(gt.any()):
        raise ValueError("the single referring GT must not be empty")
    return gt


def summarize_iou_stages(stage_ious):
    """Summarize nine [Q] IoU vectors in st1--st9 order, on the CPU.

    Query IDs denote persistent decoder slots. ``transition_0p7`` compares the
    same IDs at adjacent layers, not a rematching of masks. The first stage has
    no transition because st0 is intentionally outside this report.
    """
    if not isinstance(stage_ious, (list, tuple)) or len(stage_ious) != 9:
        raise ValueError("exactly nine IoU vectors (st1 through st9) are required")
    vectors = []
    query_count = None
    for stage, values in zip(_STAGES, stage_ious):
        tensor = torch.as_tensor(values)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError(f"{stage} IoU values must be real numbers, not bool/complex")
        # JSON lists must be rebuilt directly as float64. Going via the default
        # float32 would turn a true 7/10 IoU into 0.699999988 and cross >=0.7.
        tensor = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError(f"{stage} IoU must have nonempty shape [queries]")
        if not torch.isfinite(tensor).all() or not torch.all((tensor >= 0) & (tensor <= 1)):
            raise ValueError(f"{stage} IoU must be finite and within [0, 1]")
        if query_count is None:
            query_count = tensor.numel()
        if tensor.numel() != query_count:
            raise ValueError("Query count must remain unchanged across st1--st9")
        vectors.append(tensor)

    rows = []
    previous_good = None
    for stage_index, (stage, values) in enumerate(zip(_STAGES, vectors), start=1):
        good_ids = {
            key: torch.nonzero(values >= threshold, as_tuple=False).flatten().tolist()
            for key, threshold in zip(_THRESHOLD_KEYS, THRESHOLDS)
        }
        current_good = set(good_ids["0.7"])
        transition = None
        if previous_good is not None:
            added = sorted(current_good - previous_good)
            lost = sorted(previous_good - current_good)
            transition = {
                "added_query_ids": added,
                "lost_query_ids": lost,
                "added_count": len(added),
                "lost_count": len(lost),
                "net_count_change": len(added) - len(lost),
            }
        top_k = min(5, int(query_count))
        rows.append({
            "stage": stage,
            "stage_index": stage_index,
            "num_queries": int(query_count),
            "per_query_iou": values.tolist(),
            "good_query_ids": good_ids,
            "good_query_count": {key: len(ids) for key, ids in good_ids.items()},
            "coverage": {key: bool(ids) for key, ids in good_ids.items()},
            "best_iou": float(values.max()),
            "oracle_top5_mean_iou": float(values.topk(top_k).values.mean()),
            "oracle_top_k": top_k,
            "transition_0p7": transition,
        })
        previous_good = current_good
    return rows


@torch.no_grad()
def evaluate_stage_query_gt(
    stage_logits,
    segmentor,
    gt_mask,
    image_size,
    scaled_size,
    final_selected_query=None,
    final_formal_mask=None,
    chunk_size=16,
    *,
    sam_valid_boxes_normalized,
):
    """Measure all native Queries against the current expression's one GT.

    ``stage_logits`` is the native length-ten sequence including st0, each
    tensor [1, Q, H, W]. Input dtype and device are preserved through formal
    mask restoration. ``final_formal_mask`` accepts the actual formal label
    image (foreground == 1, background 255), or a bool foreground image.
    ``sam_valid_boxes_normalized`` is the actual native builder's [1, 4] valid
    region in top/left/bottom/right order, reused for every decoder stage.
    Supplying the final selected Query and mask together enables an exact
    equality guard against the formal prediction, before publishing results.

    Call only after the model forward and temporary observation hooks finish.
    No class scores, masks, model parameters, or Query selection are changed.
    """
    image_size = _size_pair(image_size, "image_size")
    scaled_size = _size_pair(scaled_size, "scaled_size")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    gt = _binary_gt(gt_mask, image_size)
    boxes = sam_valid_boxes_normalized
    if (not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point()
            or tuple(boxes.shape) != (1, 4) or not torch.isfinite(boxes).all()
            or not torch.all((boxes >= 0) & (boxes <= 1))
            or not bool((boxes[:, 0] < boxes[:, 2]).all())
            or not bool((boxes[:, 1] < boxes[:, 3]).all())):
        raise ValueError("native SAM valid boxes must be finite [1, 4] normalized TLBR tensors")
    boxes = boxes.detach()
    if not isinstance(stage_logits, (list, tuple)) or len(stage_logits) != 10:
        raise ValueError("native stage_logits must include exactly st0 through st9")
    query_count = None
    for index, logits in enumerate(stage_logits):
        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise ValueError(f"st{index} logits must be a floating-point tensor")
        if logits.ndim != 4 or logits.shape[0] != 1 or min(logits.shape[1:]) < 1:
            raise ValueError(f"st{index} logits must have shape [1, queries, height, width]")
        if not torch.isfinite(logits).all():
            raise ValueError(f"st{index} logits contain non-finite values")
        if logits.device != boxes.device:
            raise ValueError("native SAM valid boxes and stage logits must be on the same device")
        if query_count is None:
            query_count = logits.shape[1]
        if logits.shape[1] != query_count:
            raise ValueError("Query count must remain unchanged across decoder layers")

    if (final_selected_query is None) != (final_formal_mask is None):
        raise ValueError("provide final_selected_query and final_formal_mask together")
    formal_foreground = None
    if final_selected_query is not None:
        if (isinstance(final_selected_query, bool)
                or not isinstance(final_selected_query, Integral)
                or not 0 <= final_selected_query < query_count):
            raise ValueError("final selected Query ID is outside the decoder Query range")
        final_selected_query = int(final_selected_query)
        formal = torch.as_tensor(final_formal_mask).detach().cpu()
        if formal.ndim != 2 or tuple(formal.shape) != image_size:
            raise ValueError("final formal segmentation must have original image dimensions")
        if formal.is_complex() or not torch.isfinite(formal).all():
            raise ValueError("final formal segmentation must be finite and real")
        if not torch.all((formal == 0) | (formal == 1) | (formal == 255)):
            raise ValueError("formal segmentation must use bool or 0/1/255 labels")
        formal_foreground = formal == 1

    from masklat.dataset.process_fns.postprocess_fns.refseg_process_fn import (
        binarize_refseg_query_masks,
        restore_refseg_query_masks,
    )

    stage_ious = []
    final_selected_mask = None
    for stage_index in range(1, 10):
        logits = stage_logits[stage_index].detach()[0]
        current_gt = gt.to(device=logits.device)
        gt_area = current_gt.sum(dtype=torch.int64)
        chunks = []
        for begin in range(0, query_count, chunk_size):
            end = min(begin + chunk_size, query_count)
            sam_logits = segmentor.postprocess_masks_preds(
                (logits[begin:end].unsqueeze(0),),
                sam_valid_boxes_normalized=boxes,
            )[0][0]
            restored_logits = restore_refseg_query_masks(sam_logits, image_size, scaled_size)
            if (restored_logits.ndim != 3
                    or tuple(restored_logits.shape) != (end - begin, *image_size)
                    or not torch.isfinite(restored_logits).all()):
                raise ValueError("formal mask restoration returned invalid Query logits")
            masks = binarize_refseg_query_masks(restored_logits, 0.5)
            if masks.dtype != torch.bool or tuple(masks.shape) != tuple(restored_logits.shape):
                raise ValueError("formal mask binarization must return bool Query masks")
            intersection = (masks & current_gt).sum(dim=(-2, -1), dtype=torch.int64)
            union = masks.sum(dim=(-2, -1), dtype=torch.int64) + gt_area - intersection
            if not torch.all(union > 0):
                raise ValueError("nonempty referring GT must produce positive mask unions")
            chunks.append((intersection.to(torch.float64) / union).cpu())
            if (stage_index == 9 and final_selected_query is not None
                    and begin <= final_selected_query < end):
                final_selected_mask = masks[final_selected_query - begin].detach().cpu().clone()
            del sam_logits, restored_logits, masks, intersection, union
        stage_ious.append(torch.cat(chunks))

    check = None
    if final_selected_query is not None:
        if final_selected_mask is None or not torch.equal(final_selected_mask, formal_foreground):
            raise ValueError(
                "restored st9 selected Query does not exactly match the formal foreground mask; "
                "do not publish Query/GT statistics from a different restoration path"
            )
        check = {"query_id": final_selected_query, "passed": True, "comparison": "exact_binary_mask"}
    return {
        "schema_version": _SCHEMA_VERSION,
        "thresholds": list(THRESHOLDS),
        "iou_comparison": ">=",
        "mask_binarization": "sigmoid(logits) > 0.5 (native RefSeg)",
        "stage0_omitted": True,
        "class_filter_applied": False,
        "image_size_hw": list(image_size),
        "scaled_size_hw": list(scaled_size),
        "sam_valid_boxes_normalized_tlbr": boxes.detach().cpu().tolist(),
        "gt_foreground_pixels": int(gt.sum()),
        "rows": summarize_iou_stages(stage_ious),
        "final_selected_query_check": check,
    }


def _validated_record_rows(record):
    if not isinstance(record, dict):
        raise ValueError("each Query/GT record must be a mapping")
    for key in ("dataset", "dataset_index", "image_id", "query_gt"):
        if key not in record:
            raise ValueError(f"Query/GT record is missing {key}")
    if not isinstance(record["dataset"], str) or not record["dataset"].strip():
        raise ValueError("dataset name must be nonempty text")
    if (isinstance(record["dataset_index"], bool)
            or not isinstance(record["dataset_index"], Integral)
            or record["dataset_index"] < 0):
        raise ValueError("dataset_index must be a nonnegative integer")
    result = record["query_gt"]
    if not isinstance(result, dict) or result.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unknown Query/GT result schema")
    if result.get("thresholds") != list(THRESHOLDS):
        raise ValueError("Query/GT records must use the same 0.5/0.7/0.9 thresholds")
    source_rows = result.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 9:
        raise ValueError("each Query/GT record must contain all nine decoder stages")
    for index, row in enumerate(source_rows, start=1):
        if not isinstance(row, dict) or row.get("stage") != f"st{index}" or row.get("stage_index") != index:
            raise ValueError("Query/GT stages must be unique and ordered st1 through st9")
        if "per_query_iou" not in row:
            raise ValueError("per-Query IoUs are required for auditable aggregation")
    rows = summarize_iou_stages([row["per_query_iou"] for row in source_rows])
    for supplied, recomputed in zip(source_rows, rows):
        for key in ("num_queries", "good_query_ids", "good_query_count", "coverage", "transition_0p7"):
            if supplied.get(key) != recomputed[key]:
                raise ValueError(f"inconsistent {supplied['stage']} {key}: does not match per-Query IoUs")
        for key in ("best_iou", "oracle_top5_mean_iou"):
            value = supplied.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, recomputed[key], rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"inconsistent {supplied['stage']} {key}")
    check = result.get("final_selected_query_check")
    if check is not None and (not isinstance(check, dict) or check.get("passed") is not True):
        raise ValueError("cannot aggregate a failed final formal-mask equality check")
    return rows


def _aggregate_rows(samples):
    sample_count = len(samples)
    if sample_count < 1:
        raise ValueError("cannot summarize an empty Query/GT sample collection")
    rows = []
    for stage_offset, stage in enumerate(_STAGES):
        stage_rows = [sample[stage_offset] for sample in samples]
        count_sum = {
            key: sum(row["good_query_count"][key] for row in stage_rows)
            for key in _THRESHOLD_KEYS
        }
        transition = None
        if stage_offset > 0:
            transition = {}
            for key in ("added_count", "lost_count", "net_count_change"):
                total = sum(row["transition_0p7"][key] for row in stage_rows)
                transition[f"{key}_sum"] = total
                transition[f"{key}_mean"] = total / sample_count
        rows.append({
            "stage": stage,
            "stage_index": stage_offset + 1,
            "sample_count": sample_count,
            "num_queries_mean": sum(row["num_queries"] for row in stage_rows) / sample_count,
            "num_queries_min": min(row["num_queries"] for row in stage_rows),
            "num_queries_max": max(row["num_queries"] for row in stage_rows),
            "good_query_count_sum": count_sum,
            "good_query_count_mean": {key: value / sample_count for key, value in count_sum.items()},
            "coverage_sample_count": {
                key: sum(row["coverage"][key] for row in stage_rows) for key in _THRESHOLD_KEYS
            },
            "coverage_percent": {
                key: 100.0 * sum(row["coverage"][key] for row in stage_rows) / sample_count
                for key in _THRESHOLD_KEYS
            },
            "best_iou_percent_mean": 100.0 * sum(row["best_iou"] for row in stage_rows) / sample_count,
            "oracle_top5_mean_iou_percent_mean": (
                100.0 * sum(row["oracle_top5_mean_iou"] for row in stage_rows) / sample_count
            ),
            "oracle_top_k_min": min(row["oracle_top_k"] for row in stage_rows),
            "oracle_top_k_max": max(row["oracle_top_k"] for row in stage_rows),
            "transition_0p7": transition,
        })
    return {"sample_count": sample_count, "rows": rows}


def aggregate_query_gt(records):
    """Return nine-row per-dataset and sample-weighted pooled summaries.

    Each record must identify ``dataset``, ``dataset_index`` and ``image_id``,
    and contain the result of ``evaluate_stage_query_gt`` as ``query_gt``.
    Statistics are recomputed from saved per-Query IoUs, and contradictory
    counts/transitions are rejected instead of silently averaged.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("at least one Query/GT record is required")
    groups = defaultdict(list)
    all_rows = []
    seen = set()
    checked_samples = 0
    for record in records:
        rows = _validated_record_rows(record)
        identity = (record["dataset"], int(record["dataset_index"]))
        if identity in seen:
            raise ValueError(f"duplicate Query/GT sample record: {identity}")
        seen.add(identity)
        groups[record["dataset"]].append(rows)
        all_rows.append(rows)
        checked_samples += record["query_gt"].get("final_selected_query_check") is not None
    return {
        "schema_version": _SCHEMA_VERSION,
        "thresholds": list(THRESHOLDS),
        "metadata": {
            "unit": "one sampled image with one referring expression/GT",
            "good_query_definition": "original-resolution binary mask IoU >= threshold",
            "query_counts_are_not_distinct_objects": True,
            "no_class_filter_or_hungarian_matching": True,
            "oracle_metrics_are_not_final_selected_mask_accuracy": True,
            "top5_definition": "five highest GT IoUs, NOT attention Top5",
            "transition_definition": "persistent Query IDs crossing IoU 0.7 since previous stage",
            "pooled_weighting": "equal weight per sampled image/expression",
            "final_formal_mask_equality_checked_samples": int(checked_samples),
            "sample_count": len(records),
        },
        "datasets": {name: _aggregate_rows(samples) for name, samples in sorted(groups.items())},
        "pooled": _aggregate_rows(all_rows),
    }


def format_query_gt_markdown(summary):
    """Format each dataset and the pooled sample set as one readable nine-row table."""
    if not isinstance(summary, dict) or summary.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unknown Query/GT summary schema")
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict) or not datasets or "pooled" not in summary:
        raise ValueError("Query/GT summary needs datasets and pooled results")
    lines = [
        "# st1–st9 Query 与正确 GT mask 的逐层对照", "",
        "这里的“合格 Query 数”是同一个表达对应的 GT，有多少个 Query 的 mask 达到 IoU 阈值；"
        "不是图中不同物体的数量，也不代表模型最终选中了这些 Query。", "",
        "表内合格数量、新增/丢失及净增均为每张图的平均值；总数量保存在 JSON。"
        "覆盖率表示至少有一个合格 Query 的样本占比。所有 Query 均参与，不按分类分数筛选。", "",
        "最佳 IoU 与 Top5 IoU 是使用 GT 找出的候选质量上限，不是最终预测指标；"
        "此处 Top5 按 GT IoU 排序，与注意力 Top5 不同。", "",
    ]
    groups = list(sorted(datasets.items())) + [("全部抽样图片（合并）", summary["pooled"])]
    for name, group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("rows"), list) or len(group["rows"]) != 9:
            raise ValueError("each Query/GT summary table must contain exactly nine stages")
        safe_name = str(name).replace("\n", " ").replace("\r", " ")
        lines.extend([
            f"## {safe_name}", "",
            "| 层 | 图数 | Query 数/图 | 合格数 ≥0.5 | 合格数 ≥0.7 | 合格数 ≥0.9 | 净增 ≥0.7 | 新增/丢失 ≥0.7 | 覆盖率 ≥0.7 | 最佳 IoU | Top5 IoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for stage, row in zip(_STAGES, group["rows"]):
            if row.get("stage") != stage:
                raise ValueError("summary rows must be ordered st1 through st9")
            counts = row["good_query_count_mean"]
            transition = row["transition_0p7"]
            net = "—" if transition is None else f"{transition['net_count_change_mean']:+.2f}"
            changed = ("—" if transition is None else
                       f"{transition['added_count_mean']:.2f} / {transition['lost_count_mean']:.2f}")
            lines.append(
                f"| {stage} | {row['sample_count']} | {row['num_queries_mean']:.1f} "
                f"| {counts['0.5']:.2f} | {counts['0.7']:.2f} | {counts['0.9']:.2f} "
                f"| {net} | {changed} | {row['coverage_percent']['0.7']:.1f}% "
                f"| {row['best_iou_percent_mean']:.2f}% "
                f"| {row['oracle_top5_mean_iou_percent_mean']:.2f}% |"
            )
        lines.append("")
    lines.extend([
        "判读：先看“合格数 ≥0.7”是否逐层增加，再看新增/丢失，避免数量不变但 Query 身份已替换。"
        "同时看覆盖率和最佳 IoU：更多 Query 可能只是同一好 mask 的重复，并不一定带来更高分割质量。", "",
        "说明：mask 沿用原生 RefSeg 恢复到原图尺寸，再按 sigmoid(logit) > 0.5 二值化；"
        "IoU 合格条件为 ≥ 阈值。st1 的增减留空，不与未展示的 st0 比较。"
        "少于 5 个 Query 时 Top5 对现有全部 Query 取均值。", "",
        "这是抽样诊断，不是全量 val 的正式评测结果。", "",
    ])
    return "\n".join(lines)
