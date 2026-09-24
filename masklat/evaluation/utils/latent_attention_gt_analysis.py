"""Offline joins of saved latent attention, native Query IoU, and final selection.

There is no inference or model mutation in this module. A sampled image with one
referring expression is the statistical unit; its 64 latent slots are not 64
independent samples. st9 is deliberately kept distinct: it stores P(latent|Query),
not the P(Query|latent) distribution available at st3 through st8.
"""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral, Real

import torch


THRESHOLDS = (0.5, 0.7, 0.9)
STAGES = ("st3_block0", "st3", "st4", "st5", "st6", "st7", "st8", "st9")
_FORWARD = "latent_reads_query"
_REVERSE = "query_reads_latent"
_NORMALIZATION_ATOL = 0.02


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value, name, minimum=0.0, maximum=1.0):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return float(value)


def _float64_tensor(values, name):
    try:
        original = torch.as_tensor(values)
        if original.is_complex() or original.dtype == torch.bool:
            raise ValueError(f"{name} must contain real numbers, not bool/complex")
        # Construct Python/JSON values directly as FP64: 0.7 must not become
        # 0.699999988 by an intermediate FP32 conversion.
        result = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid {name}") from error
    if not torch.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _validated_ious(query_gt):
    if not isinstance(query_gt, dict) or query_gt.get("schema_version") != 1:
        raise ValueError("unknown Query/GT schema")
    if query_gt.get("thresholds") != list(THRESHOLDS):
        raise ValueError("Query/GT thresholds must be 0.5, 0.7, 0.9")
    if (query_gt.get("iou_comparison") != ">="
            or query_gt.get("mask_binarization") != "sigmoid(logits) > 0.5 (native RefSeg)"
            or query_gt.get("class_filter_applied") is not False
            or query_gt.get("stage0_omitted") is not True):
        raise ValueError("Query/GT must use the saved native RefSeg mask/IoU comparison contract")
    rows = query_gt.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("Query/GT must contain all nine ordered stages")
    vectors, count, previous_good = {}, None, None
    for index, row in enumerate(rows, 1):
        stage = f"st{index}"
        if not isinstance(row, dict) or row.get("stage") != stage or row.get("stage_index") != index:
            raise ValueError("Query/GT stages must be ordered st1 through st9")
        values = _float64_tensor(row.get("per_query_iou"), f"{stage} per_query_iou")
        if values.ndim != 1 or values.numel() < 5:
            raise ValueError("Query/GT vectors must have shape [Q] with at least five Queries")
        if count is None:
            count = values.numel()
        if values.numel() != count or row.get("num_queries") != count:
            raise ValueError("Query count must be consistent across all nine stages")
        ids = {str(t): torch.nonzero(values >= t).flatten().tolist() for t in THRESHOLDS}
        current_good = set(ids["0.7"])
        transition = None if previous_good is None else {
            "added_query_ids": sorted(current_good - previous_good),
            "lost_query_ids": sorted(previous_good - current_good),
            "added_count": len(current_good - previous_good),
            "lost_count": len(previous_good - current_good),
            "net_count_change": len(current_good) - len(previous_good),
        }
        expected = {
            "good_query_ids": ids,
            "good_query_count": {k: len(v) for k, v in ids.items()},
            "coverage": {k: bool(v) for k, v in ids.items()},
            "transition_0p7": transition,
            "oracle_top_k": 5,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"inconsistent {stage} {key}")
        for key, value in (("best_iou", float(values.max())),
                           ("oracle_top5_mean_iou", float(values.topk(5).values.mean()))):
            supplied = _finite_number(row.get(key), f"{stage} {key}")
            if not math.isclose(supplied, value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"inconsistent {stage} {key}")
        vectors[stage] = values
        previous_good = current_good
    check = query_gt.get("final_selected_query_check")
    if (not isinstance(check, dict) or check.get("passed") is not True
            or check.get("comparison") != "exact_binary_mask"):
        raise ValueError("a passed final formal-mask equality check is required")
    selected = _integer(check.get("query_id"), "final selected Query ID")
    if selected >= count:
        raise ValueError("final selected Query ID is out of range")
    return vectors, selected


def _selection_from_vector(values, selected):
    oracle = int(values.argmax())  # Equal best IoUs use the smallest Query ID.
    selected_iou, oracle_iou = float(values[selected]), float(values[oracle])
    selected_good = {str(t): selected_iou >= t for t in THRESHOLDS}
    oracle_good = {str(t): oracle_iou >= t for t in THRESHOLDS}
    return {
        "selected_query_id": selected,
        "selected_iou": selected_iou,
        "oracle_query_id": oracle,
        "oracle_iou": oracle_iou,
        "oracle_tied_query_ids": torch.nonzero(values == values[oracle]).flatten().tolist(),
        "gap": oracle_iou - selected_iou,
        "selected_matches_oracle_iou": selected_iou == oracle_iou,
        "selected_good": selected_good,
        "oracle_good": oracle_good,
        "missed_good": {key: oracle_good[key] and not selected_good[key] for key in selected_good},
    }


def analyze_final_selection(query_gt):
    """Audit the actually chosen st9 mask against all st9 Query masks.

    The saved Query/GT producer must have verified exact binary-mask equality
    between this Query's restored mask and the native formal model prediction.
    """
    vectors, selected = _validated_ious(query_gt)
    return _selection_from_vector(vectors["st9"], selected)


def _validated_attention(stage, expected_stage, query_count):
    if not isinstance(stage, dict) or stage.get("stage") != expected_stage:
        raise ValueError("attention stages must be unique and in the documented order")
    expected_kind = _REVERSE if expected_stage == "st9" else _FORWARD
    if stage.get("weight_kind") != expected_kind:
        raise ValueError(f"{expected_stage} has the wrong attention direction")
    weights = _float64_tensor(stage.get("weights"), f"{expected_stage} weights")
    if weights.shape != (64, query_count):
        raise ValueError(f"{expected_stage} attention must have shape [64, {query_count}]")
    axis = 0 if expected_kind == _REVERSE else 1
    sums = weights.sum(dim=axis)
    maximum_error = float((sums - 1).abs().max())
    if maximum_error > _NORMALIZATION_ATOL:
        raise ValueError(f"{expected_stage} attention is not normalized on its documented axis")
    # Native BF16/FP16 probabilities are preserved, NOT silently renormalized.
    top_ids = weights.argsort(dim=1, descending=True, stable=True)[:, :5]
    try:
        supplied_ids = torch.as_tensor(stage.get("top5_query_ids")).detach().cpu()
    except (TypeError, RuntimeError) as error:
        raise ValueError("saved top5 Query IDs are invalid") from error
    if (supplied_ids.dtype == torch.bool or supplied_ids.is_floating_point()
            or supplied_ids.is_complex() or supplied_ids.shape != (64, 5)
            or not torch.equal(supplied_ids.to(torch.int64), top_ids)):
        raise ValueError("saved top5 Query IDs disagree with stable attention ranking")
    top_weights = weights.gather(1, top_ids)
    supplied_weights = _float64_tensor(stage.get("top5_weights"), "saved top5 weights")
    if supplied_weights.shape != (64, 5) or not torch.equal(supplied_weights, top_weights):
        raise ValueError("saved top5 weights disagree with full attention weights")
    return weights, top_ids, top_weights, expected_kind, maximum_error


def _mean(values):
    return math.fsum(values) / len(values)


def _stage_summary(stage, latent_rows):
    forward = stage != "st9"
    unique = set(q for row in latent_rows for q in row["top5_query_ids"])
    return {
        "stage": stage,
        "gt_stage": "st3" if stage == "st3_block0" else stage,
        "weight_kind": _FORWARD if forward else _REVERSE,
        "top1_iou_percent_mean": 100 * _mean([r["top1_iou"] for r in latent_rows]),
        "top5_mean_iou_percent_mean": 100 * _mean([r["top5_mean_iou"] for r in latent_rows]),
        "top5_max_iou_percent_mean": 100 * _mean([r["top5_max_iou"] for r in latent_rows]),
        "top5_good_query_count_0p7_mean": _mean([r["top5_good_query_count_0p7"] for r in latent_rows]),
        "slot_hit_rate_0p7_percent_mean": 100 * _mean([int(r["top5_any_good_0p7"]) for r in latent_rows]),
        "unique_top5_query_count_mean": float(len(unique)),
        "top5_weight_mass_mean": _mean([r["top5_weight_mass"] for r in latent_rows]) if forward else None,
        "good_attention_mass_0p7_mean": _mean([r["good_attention_mass_0p7"] for r in latent_rows]) if forward else None,
    }


def analyze_sample_attention_gt(query_gt, attention_stages):
    """Join one sample's eight saved attention stages to exact native GT IoUs.

    This is a Query-slot-ID join, never an IoU rematching of low-resolution
    preview masks. Every top5 IoU refers to that stage's original-resolution
    Query mask. The block0 view uses the same st3 Query masks as the st3 view.
    """
    vectors, selected = _validated_ious(query_gt)
    if not isinstance(attention_stages, (list, tuple)) or len(attention_stages) != len(STAGES):
        raise ValueError("all eight saved attention stages are required")
    count = vectors["st9"].numel()
    stage_rows = []
    for name, source_stage in zip(STAGES, attention_stages):
        weights, top_ids, top_weights, kind, norm_error = _validated_attention(source_stage, name, count)
        gt_stage = "st3" if name == "st3_block0" else name
        values = vectors[gt_stage]
        top_ious = values[top_ids]
        good_mass = weights[:, values >= 0.7].sum(dim=1) if kind == _FORWARD else None
        latent_rows = []
        for latent in range(64):
            ious = top_ious[latent].tolist()
            good_count = sum(value >= 0.7 for value in ious)
            latent_rows.append({
                "latent_id": latent,
                "top5_query_ids": top_ids[latent].tolist(),
                "top5_weights": top_weights[latent].tolist(),
                "top5_ious": ious,
                "top1_iou": ious[0],
                "top5_mean_iou": _mean(ious),
                "top5_max_iou": max(ious),
                "top5_good_query_count_0p7": good_count,
                "top5_any_good_0p7": good_count > 0,
                "top5_weight_mass": float(top_weights[latent].sum()) if kind == _FORWARD else None,
                "good_attention_mass_0p7": float(good_mass[latent]) if good_mass is not None else None,
            })
        stage_rows.append({
            **_stage_summary(name, latent_rows),
            "num_latents": 64,
            "num_queries": int(count),
            "normalization_axis": "latent" if kind == _REVERSE else "query",
            "normalization_max_abs_error": norm_error,
            "weights_renormalized": False,
            "direction_note": (
                "P(latent|Query): ranked Queries depending on a latent; NOT latent reading Queries"
                if kind == _REVERSE else "P(Query|latent): ranked Queries read by a latent"
            ),
            "latent_rows": latent_rows,
        })
    return {
        "schema_version": 1,
        "final_selection": _selection_from_vector(vectors["st9"], selected),
        "stages": stage_rows,
    }


def _validate_selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("missing final selection analysis")
    _integer(selection.get("selected_query_id"), "selected Query ID")
    _integer(selection.get("oracle_query_id"), "oracle Query ID")
    selected = _finite_number(selection.get("selected_iou"), "selected IoU")
    oracle = _finite_number(selection.get("oracle_iou"), "oracle IoU")
    gap = _finite_number(selection.get("gap"), "oracle gap")
    if oracle < selected or not math.isclose(gap, oracle - selected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("inconsistent oracle gap")
    for field, expected in (
        ("selected_good", {str(t): selected >= t for t in THRESHOLDS}),
        ("oracle_good", {str(t): oracle >= t for t in THRESHOLDS}),
        ("missed_good", {str(t): oracle >= t and selected < t for t in THRESHOLDS}),
    ):
        value = selection.get(field)
        if value != expected or not all(isinstance(v, bool) for v in value.values()):
            raise ValueError(f"inconsistent final selection {field}")
    if selection.get("selected_matches_oracle_iou") is not (selected == oracle):
        raise ValueError("inconsistent oracle equality flag")
    return selection


def _validate_stage_result(row, expected_stage):
    if (not isinstance(row, dict) or row.get("stage") != expected_stage
            or row.get("gt_stage") != ("st3" if expected_stage == "st3_block0" else expected_stage)
            or row.get("weight_kind") != (_REVERSE if expected_stage == "st9" else _FORWARD)):
        raise ValueError("invalid analysis stage/direction")
    latent_rows = row.get("latent_rows")
    query_count = _integer(row.get("num_queries"), "Query count", minimum=5)
    if row.get("num_latents") != 64 or not isinstance(latent_rows, list) or len(latent_rows) != 64:
        raise ValueError("analysis stages must contain all 64 latent rows")
    for index, latent in enumerate(latent_rows):
        if not isinstance(latent, dict) or latent.get("latent_id") != index:
            raise ValueError("latent rows must be ordered 0..63")
        ids, weights, ious = (latent.get(key) for key in ("top5_query_ids", "top5_weights", "top5_ious"))
        if not all(isinstance(v, list) and len(v) == 5 for v in (ids, weights, ious)):
            raise ValueError("each latent needs exactly five IDs, weights, and IoUs")
        ids = [_integer(v, "top5 Query ID") for v in ids]
        if len(set(ids)) != 5 or max(ids) >= query_count:
            raise ValueError("top5 Query IDs must be unique and in range")
        weights = [_finite_number(v, "top5 weight") for v in weights]
        ious = [_finite_number(v, "top5 IoU") for v in ious]
        if any(weights[j] < weights[j + 1] or (weights[j] == weights[j + 1] and ids[j] > ids[j + 1]) for j in range(4)):
            raise ValueError("top5 weights and tie IDs must be descending/stable")
        count = sum(v >= 0.7 for v in ious)
        expected = {"top1_iou": ious[0], "top5_mean_iou": _mean(ious), "top5_max_iou": max(ious),
                    "top5_good_query_count_0p7": count, "top5_any_good_0p7": count > 0}
        for key, value in expected.items():
            if latent.get(key) != value:
                raise ValueError(f"inconsistent latent {key}")
        if expected_stage == "st9":
            if latent.get("top5_weight_mass") is not None or latent.get("good_attention_mass_0p7") is not None:
                raise ValueError("st9 cannot report a per-latent Query probability mass")
        else:
            mass = _finite_number(latent.get("top5_weight_mass"), "top5 mass", maximum=1.02)
            _finite_number(latent.get("good_attention_mass_0p7"), "good attention mass", maximum=1.02)
            if not math.isclose(mass, math.fsum(weights), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("inconsistent top5 mass")
    expected_summary = _stage_summary(expected_stage, latent_rows)
    for key, value in expected_summary.items():
        actual = row.get(key)
        if isinstance(value, float):
            if not isinstance(actual, Real) or isinstance(actual, bool) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"inconsistent stage summary {key}")
        elif actual != value:
            raise ValueError(f"inconsistent stage summary {key}")
    return expected_summary


def _aggregate_group(analyses):
    count = len(analyses)
    selected = [a["final_selection"] for a in analyses]
    keys = [str(t) for t in THRESHOLDS]
    good_counts = {key: sum(s["oracle_good"][key] for s in selected) for key in keys}
    missed_counts = {key: sum(s["missed_good"][key] for s in selected) for key in keys}
    final = {
        "sample_count": count,
        "selected_iou_percent_mean": 100 * _mean([s["selected_iou"] for s in selected]),
        "oracle_iou_percent_mean": 100 * _mean([s["oracle_iou"] for s in selected]),
        "gap_percentage_points_mean": 100 * _mean([s["gap"] for s in selected]),
        "selected_matches_oracle_iou_percent": 100 * _mean([int(s["selected_matches_oracle_iou"]) for s in selected]),
        "selected_good_percent": {k: 100 * sum(s["selected_good"][k] for s in selected) / count for k in keys},
        "selected_good_sample_count": {k: sum(s["selected_good"][k] for s in selected) for k in keys},
        "oracle_good_sample_count": good_counts,
        "oracle_good_percent": {k: 100 * good_counts[k] / count for k in keys},
        "missed_good_sample_count": missed_counts,
        "missed_good_percent": {k: 100 * missed_counts[k] / count for k in keys},
        "missed_good_given_oracle_good_percent": {
            k: 100 * missed_counts[k] / good_counts[k] if good_counts[k] else None for k in keys
        },
    }
    rows = []
    if analyses[0]["stages"]:
        for index, stage in enumerate(STAGES):
            summaries = [_validate_stage_result(a["stages"][index], stage) for a in analyses]
            result = {"stage": stage, "gt_stage": summaries[0]["gt_stage"],
                      "weight_kind": summaries[0]["weight_kind"], "sample_count": count}
            for key, value in summaries[0].items():
                if key not in result:
                    result[key] = None if value is None else _mean([r[key] for r in summaries])
            rows.append(result)
    return {"sample_count": count, "final_selection": final, "stages": rows}


def aggregate_attention_gt(records):
    """Pool sample analyses, with equal image/expression (not latent) weight.

    Each record has dataset, dataset_index, image_id, and analysis. Selection-only
    records use the same analysis envelope with stages=[]; mixing full and
    selection-only records is rejected to avoid changing denominators silently.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("at least one analyzed sample is required")
    groups, analyses, seen, seen_images = defaultdict(list), [], set(), set()
    attention_available = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("analysis records must be mappings")
        name = record.get("dataset")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dataset must be nonempty text")
        index = _integer(record.get("dataset_index"), "dataset_index")
        image_id = record.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, (str, Integral)) or str(image_id) == "":
            raise ValueError("image_id must identify the original source image")
        identity, image_identity = (name, index), (name, str(image_id))
        if identity in seen or image_identity in seen_images:
            raise ValueError("duplicate dataset sample or original image")
        seen.add(identity)
        seen_images.add(image_identity)
        analysis = record.get("analysis")
        if not isinstance(analysis, dict) or analysis.get("schema_version") != 1:
            raise ValueError("unknown attention/GT analysis schema")
        _validate_selection(analysis.get("final_selection"))
        stages = analysis.get("stages")
        if not isinstance(stages, list) or len(stages) not in (0, len(STAGES)):
            raise ValueError("analysis must include either all eight stages or selection only")
        has_attention = bool(stages)
        if attention_available is None:
            attention_available = has_attention
        if has_attention != attention_available:
            raise ValueError("do not mix attention and selection-only samples")
        groups[name].append(analysis)
        analyses.append(analysis)
    return {
        "schema_version": 1,
        "thresholds": list(THRESHOLDS),
        "metadata": {
            "sample_count": len(records),
            "attention_available": attention_available,
            "unit": "one original image with one referring expression/GT",
            "pooled_weighting": "equal per image/expression; slot-average within image first",
            "top5_definition": "attention-ranked Query IDs, NOT GT-oracle ranking",
            "good_attention_mass_definition": "all Query attention weights with IoU >= 0.7, not only top5",
            "st9_direction": "P(latent|Query), NOT P(Query|latent); no row probability-mass metric",
            "final_selection_definition": "native selected st9 Query, verified against formal binary mask",
            "mask_definition": "original-resolution Query/GT IoU from saved native RefSeg restoration",
            "block0_gt_definition": "st3_block0 and st3 use the same st3 Query mask IDs/IoUs",
            "not_causal_evidence": True,
        },
        "datasets": {name: _aggregate_group(samples) for name, samples in sorted(groups.items())},
        "pooled": _aggregate_group(analyses),
    }
