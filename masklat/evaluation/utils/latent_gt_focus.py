"""Join grouped latent<-Query attention to original-resolution Query/GT IoU.

This module measures association, not causal contribution.  A Query is called
GT-good when its restored binary mask reaches the requested IoU threshold
against the single referring-expression GT mask.  Attention mass is meaningful
only for row-normalized ``p(Query | latent)`` weights.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Integral, Real

import torch


STAGES = ("st3_block0", "st3", "st4", "st5", "st6", "st7", "st8", "st9")
GT_THRESHOLDS = (0.5, 0.7, 0.9)
MASS_THRESHOLDS = (0.25, 0.5, 0.75)
PRIMARY_GT_THRESHOLD = 0.7
SCHEMA_VERSION = 1
NORMALIZATION_ATOL = 0.005


def _finite_vector(values, name):
    try:
        result = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{name} is not a numeric vector") from error
    if result.ndim != 1 or result.numel() < 5:
        raise ValueError(f"{name} must have shape [Q] with at least five Queries")
    if not torch.isfinite(result).all() or not torch.all((result >= 0) & (result <= 1)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return result


def query_iou_vector(query_gt, attention_stage):
    """Return the exact Query/GT vector matching an attention stage.

    The builder's two st3 attention views both select the same st3 Query mask
    slots.  Later views use their same-named decoder stage.
    """
    if attention_stage not in STAGES:
        raise ValueError(f"unknown attention stage: {attention_stage}")
    if not isinstance(query_gt, dict) or query_gt.get("schema_version") != 1:
        raise ValueError("unknown Query/GT result schema")
    rows = query_gt.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("Query/GT result must contain ordered st1 through st9 rows")
    expected = [f"st{index}" for index in range(1, 10)]
    if [row.get("stage") if isinstance(row, dict) else None for row in rows] != expected:
        raise ValueError("Query/GT rows must be uniquely ordered st1 through st9")
    gt_stage = "st3" if attention_stage == "st3_block0" else attention_stage
    row = rows[int(gt_stage[2:]) - 1]
    values = _finite_vector(row.get("per_query_iou"), f"{gt_stage} per_query_iou")
    if row.get("num_queries") != values.numel():
        raise ValueError(f"{gt_stage} Query count disagrees with its IoU vector")
    return values


def analyze_latent_gt_focus_stage(stage, query_ious, *, top_k=5):
    """Analyze one real grouped latent<-Query attention matrix.

    ``stage`` is one compact/native atlas stage.  No attention probabilities
    are renormalized.  Stable ties use ascending Query ID.
    """
    if not isinstance(stage, dict) or stage.get("stage") not in STAGES:
        raise ValueError("stage must be one documented attention stage")
    name = stage["stage"]
    if stage.get("weight_kind") != "latent_reads_query":
        raise ValueError(
            f"{name} is not latent<-Query attention; GT attention mass would be invalid"
        )
    weights = torch.as_tensor(stage.get("weights"), dtype=torch.float64).detach().cpu()
    if weights.ndim != 2 or weights.shape[0] != 64 or weights.shape[1] < top_k:
        raise ValueError("grouped attention must have shape [64, Q] with Q >= top_k")
    if not torch.isfinite(weights).all() or not torch.all((weights >= 0) & (weights <= 1)):
        raise ValueError("attention weights must be finite probabilities in [0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or not 1 <= top_k <= weights.shape[1]:
        raise ValueError("top_k must be an integer within the Query count")
    sums = weights.sum(dim=1)
    normalization_error = float((sums - 1).abs().max())
    if normalization_error > NORMALIZATION_ATOL:
        raise ValueError(
            f"{name} latent<-Query rows do not sum to one; max error={normalization_error:.6g}"
        )
    ious = _finite_vector(query_ious, f"{name} Query/GT IoU")
    if ious.numel() != weights.shape[1]:
        raise ValueError("attention Query count and Query/GT IoU count disagree")

    top_ids = weights.argsort(dim=1, descending=True, stable=True)[:, :top_k]
    top_weights = weights.gather(1, top_ids)
    top_ious = ious[top_ids]
    good_masks = {str(threshold): ious >= threshold for threshold in GT_THRESHOLDS}
    good_counts = {key: int(mask.sum()) for key, mask in good_masks.items()}
    good_mass = {key: weights[:, mask].sum(dim=1) for key, mask in good_masks.items()}
    primary_key = str(PRIMARY_GT_THRESHOLD)
    primary_top_good = top_ious >= PRIMARY_GT_THRESHOLD

    latent_rows = []
    for latent_id in range(64):
        threshold_counts = {
            str(threshold): int((top_ious[latent_id] >= threshold).sum())
            for threshold in GT_THRESHOLDS
        }
        masses = {key: float(values[latent_id]) for key, values in good_mass.items()}
        primary_count = threshold_counts[primary_key]
        latent_rows.append({
            "latent_id": latent_id,
            "top_query_ids": top_ids[latent_id].tolist(),
            "top_weights": top_weights[latent_id].tolist(),
            "top_query_gt_ious": top_ious[latent_id].tolist(),
            "top_gt_good_counts": threshold_counts,
            "gt_good_attention_mass": masses,
            "primary_top5_good_count": primary_count,
            "primary_top5_hit": primary_count >= 1,
            "primary_top5_multi_hit": primary_count >= 2,
            "primary_gt_attention_mass": masses[primary_key],
        })

    primary_mass = good_mass[primary_key]
    hit_count = int(primary_top_good.any(dim=1).sum())
    multi_count = int((primary_top_good.sum(dim=1) >= 2).sum())
    covered_good_ids = sorted(set(top_ids[primary_top_good].tolist()))
    ranked_latents = sorted(range(64), key=lambda index: (-float(primary_mass[index]), index))
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": name,
        "gt_stage": "st3" if name == "st3_block0" else name,
        "weight_kind": "latent_reads_query",
        "attention_interpretation": "p(Query | latent)",
        "num_latents": 64,
        "num_queries": int(weights.shape[1]),
        "top_k": int(top_k),
        "normalization_max_abs_error": normalization_error,
        "weights_renormalized": False,
        "gt_thresholds": list(GT_THRESHOLDS),
        "primary_gt_threshold": PRIMARY_GT_THRESHOLD,
        "mass_thresholds": list(MASS_THRESHOLDS),
        "good_query_count": good_counts,
        "primary_good_query_ids": torch.nonzero(good_masks[primary_key]).flatten().tolist(),
        "primary_good_query_count": good_counts[primary_key],
        "primary_multi_query_evaluable": good_counts[primary_key] >= 2,
        "latent_top5_hit_count": hit_count,
        "latent_top5_multi_hit_count": multi_count,
        "latent_top5_hit_percent": 100.0 * hit_count / 64,
        "latent_top5_multi_hit_percent": 100.0 * multi_count / 64,
        "distinct_primary_good_queries_covered_by_top5": len(covered_good_ids),
        "primary_good_query_ids_covered_by_top5": covered_good_ids,
        "latent_mass_threshold_count": {
            str(threshold): int((primary_mass >= threshold).sum())
            for threshold in MASS_THRESHOLDS
        },
        "primary_gt_attention_mass_mean": float(primary_mass.mean()),
        "primary_gt_attention_mass_max": float(primary_mass.max()),
        "ranked_latent_ids_by_primary_gt_mass": ranked_latents,
        "latent_rows": latent_rows,
    }


def analyze_sample_latent_gt_focus(query_gt, stages):
    if not isinstance(stages, (list, tuple)) or len(stages) != len(STAGES):
        raise ValueError("all eight attention stages are required")
    if [stage.get("stage") if isinstance(stage, dict) else None for stage in stages] != list(STAGES):
        raise ValueError("attention stages must be ordered st3_block0, st3, ..., st9")
    return {
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "association_not_causality": True,
            "good_query": "original-resolution binary Query mask IoU with referring GT >= threshold",
            "top5": "five Query IDs with largest p(Query|latent), stable Query-ID tie break",
            "gt_mass": "sum of full attention distribution over every GT-good Query, not only Top5",
            "block0_gt_stage": "st3",
        },
        "stages": [
            analyze_latent_gt_focus_stage(stage, query_iou_vector(query_gt, stage["stage"]))
            for stage in stages
        ],
    }


def _mean(values):
    return math.fsum(values) / len(values)


def _validate_identity(record):
    if not isinstance(record, dict):
        raise ValueError("focus records must be mappings")
    name = record.get("dataset")
    index = record.get("dataset_index")
    if not isinstance(name, str) or not name:
        raise ValueError("dataset must be nonempty text")
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("dataset_index must be a nonnegative integer")
    analysis = record.get("latent_gt_focus")
    if not isinstance(analysis, dict) or analysis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("missing latent GT focus analysis")
    rows = analysis.get("stages")
    if not isinstance(rows, list) or [row.get("stage") for row in rows] != list(STAGES):
        raise ValueError("focus analysis must contain all eight ordered stages")
    return name, int(index), rows


def _aggregate_group(samples):
    count = len(samples)
    rows = []
    for offset, stage in enumerate(STAGES):
        values = [sample[offset] for sample in samples]
        evaluable = [row for row in values if row["primary_multi_query_evaluable"]]
        row = {
            "stage": stage,
            "gt_stage": "st3" if stage == "st3_block0" else stage,
            "sample_count": count,
            "primary_good_query_count_mean": _mean([v["primary_good_query_count"] for v in values]),
            "samples_with_any_primary_good_query": sum(v["primary_good_query_count"] >= 1 for v in values),
            "samples_with_multiple_primary_good_queries": len(evaluable),
            "latent_top5_hit_count_mean": _mean([v["latent_top5_hit_count"] for v in values]),
            "latent_top5_hit_percent_mean": _mean([v["latent_top5_hit_percent"] for v in values]),
            "samples_with_any_latent_top5_hit": sum(v["latent_top5_hit_count"] >= 1 for v in values),
            "latent_top5_multi_hit_count_mean": _mean([v["latent_top5_multi_hit_count"] for v in values]),
            "latent_top5_multi_hit_percent_mean": _mean([v["latent_top5_multi_hit_percent"] for v in values]),
            "samples_with_any_latent_top5_multi_hit": sum(v["latent_top5_multi_hit_count"] >= 1 for v in values),
            "multi_evaluable_sample_count": len(evaluable),
            "multi_evaluable_samples_with_any_multi_hit": sum(
                v["latent_top5_multi_hit_count"] >= 1 for v in evaluable
            ),
            "distinct_primary_good_queries_covered_mean": _mean([
                v["distinct_primary_good_queries_covered_by_top5"] for v in values
            ]),
            "primary_gt_attention_mass_mean": _mean([
                v["primary_gt_attention_mass_mean"] for v in values
            ]),
            "primary_gt_attention_mass_max_mean": _mean([
                v["primary_gt_attention_mass_max"] for v in values
            ]),
            "latent_mass_threshold_count_mean": {
                str(threshold): _mean([
                    v["latent_mass_threshold_count"][str(threshold)] for v in values
                ]) for threshold in MASS_THRESHOLDS
            },
        }
        row["multi_evaluable_success_percent"] = (
            100.0 * row["multi_evaluable_samples_with_any_multi_hit"] / len(evaluable)
            if evaluable else None
        )
        rows.append(row)
    return {"sample_count": count, "rows": rows}


def aggregate_latent_gt_focus(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("at least one latent GT focus record is required")
    groups, all_rows, seen = defaultdict(list), [], set()
    for record in records:
        name, index, rows = _validate_identity(record)
        identity = (name, index)
        if identity in seen:
            raise ValueError(f"duplicate focus sample: {identity}")
        seen.add(identity)
        groups[name].append(rows)
        all_rows.append(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_gt_threshold": PRIMARY_GT_THRESHOLD,
        "mass_thresholds": list(MASS_THRESHOLDS),
        "metadata": {
            "unit": "one sampled image/expression; latent statistics averaged within image first",
            "pooled_weighting": "equal image/expression weight",
            "association_not_causal_attribution": True,
            "multi_hit_denominator": "only samples having at least two IoU>=0.7 Queries",
            "st3_block0_uses_st3_query_masks": True,
        },
        "datasets": {name: _aggregate_group(samples) for name, samples in sorted(groups.items())},
        "pooled": _aggregate_group(all_rows),
    }


def _format_group(title, group):
    lines = [f"## {title}", "", (
        "| 层 | 图数 | GT Query 数/图 | Top5 命中 latent/64 | Top5 聚合>=2个 GT Query 的 latent/64 "
        "| 可判定图中存在多 Query 聚合 | GT 质量均值 | 最强 latent GT 质量 | 质量>=25%/50%/75% 的 latent 数 |"
    ), "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in group["rows"]:
        success = ("—" if row["multi_evaluable_success_percent"] is None else
                   f"{row['multi_evaluable_samples_with_any_multi_hit']}/{row['multi_evaluable_sample_count']} "
                   f"({row['multi_evaluable_success_percent']:.1f}%)")
        mass = row["latent_mass_threshold_count_mean"]
        lines.append(
            f"| {row['stage']} | {row['sample_count']} | {row['primary_good_query_count_mean']:.2f} | "
            f"{row['latent_top5_hit_count_mean']:.2f} | {row['latent_top5_multi_hit_count_mean']:.2f} | "
            f"{success} | {100 * row['primary_gt_attention_mass_mean']:.2f}% | "
            f"{100 * row['primary_gt_attention_mass_max_mean']:.2f}% | "
            f"{mass['0.25']:.2f}/{mass['0.5']:.2f}/{mass['0.75']:.2f} |"
        )
    return lines


def format_latent_gt_focus_markdown(summary):
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown latent GT focus summary schema")
    lines = [
        "# 64 latent 对同一 GT 的 Query 聚合诊断", "",
        "主标准：Query 的原图二值 mask 与当前 referring GT 的 IoU ≥ 0.7。",
        "`Top5 命中`看注意力最高的五个 Query；`GT 质量`把该 latent 对全部合格 Query 的注意力相加。",
        "只有一张图至少存在两个合格 Query 时，才判断它是否出现“一个 latent 同时聚合多个同 GT Query”。",
        "该报告证明读取关系是否形成 GT 聚合，不等价于因果消融。", "",
    ]
    for name, group in summary["datasets"].items():
        lines.extend(_format_group(name, group))
        lines.append("")
    lines.extend(_format_group("合并（按图等权）", summary["pooled"]))
    lines.extend([
        "", "## 判读", "",
        "- `Top5 聚合>=2个`持续大于 0，说明至少部分 latent 的高权重候选来自同一 GT 的多个 Query。",
        "- `GT 质量`或质量阈值计数随层提高，说明 latent 对正确目标 Query 集合的读取在增强。",
        "- 若某层合格 Query 本来不足 2 个，该图不能用于否定多 Query 聚合，因此单独从分母排除。",
    ])
    return "\n".join(lines) + "\n"
