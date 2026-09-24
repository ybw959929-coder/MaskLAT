"""Intuitive diversity summaries for saved 64-latent Query distributions."""
from __future__ import annotations

from collections import defaultdict
import math

import torch


STAGES = ("st3_block0", "st3", "st4", "st5", "st6", "st7", "st8", "st9")


def summarize_latent_query_diversity(weights, *, top_k=5, weight_kind="latent_reads_query"):
    """Summarize one image/stage without treating 64 latents as samples."""
    values = torch.as_tensor(weights).detach().double().cpu()
    if values.ndim != 2 or values.shape[0] != 64 or values.shape[1] < top_k:
        raise ValueError("weights must be [64,Q] with Q >= top_k")
    if top_k < 1 or not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("top_k/weights are invalid")
    top_ids = values.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    ordered_patterns = {tuple(row.tolist()) for row in top_ids}
    set_patterns = {tuple(sorted(row.tolist())) for row in top_ids}
    union = set(top_ids.flatten().tolist())
    jaccards, identical = [], []
    for left in range(64):
        a = set(top_ids[left].tolist())
        for right in range(left + 1, 64):
            b = set(top_ids[right].tolist())
            jaccards.append(len(a & b) / len(a | b))
            identical.append(a == b)
    result = {
        "latent_count": 64,
        "query_count": int(values.shape[1]),
        "top_k": int(top_k),
        "weight_kind": weight_kind,
        "unique_query_ids_in_all_topk": len(union),
        "unique_ordered_topk_patterns": len(ordered_patterns),
        "unique_unordered_topk_sets": len(set_patterns),
        "mean_pairwise_topk_jaccard_percent": 100.0 * sum(jaccards) / len(jaccards),
        "identical_topk_set_pair_percent": 100.0 * sum(identical) / len(identical),
        "topk_mass_percent_mean": None,
        "mean_pairwise_full_distribution_overlap_percent": None,
        "attention_entropy_normalized_percent_mean": None,
    }
    row_sums = values.sum(-1)
    if weight_kind == "latent_reads_query" and torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=0.02, rtol=0.0
    ):
        probabilities = values / row_sums[:, None].clamp_min(1e-30)
        pairs = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1)
        overlap = 1.0 - 0.5 * torch.cdist(probabilities, probabilities, p=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        result.update(
            topk_mass_percent_mean=100.0 * float(
                probabilities.gather(1, top_ids).sum(-1).mean()
            ),
            mean_pairwise_full_distribution_overlap_percent=100.0 * float(
                overlap[pairs].mean()
            ),
            attention_entropy_normalized_percent_mean=100.0 * float(
                entropy.mean() / max(math.log(values.shape[1]), 1e-30)
            ),
        )
    return result


def aggregate_latent_query_diversity(records):
    """Average per-image summaries within each dataset/stage and across images."""
    if not records:
        raise ValueError("at least one image record is required")
    grouped = defaultdict(list)
    for record in records:
        dataset = record.get("dataset")
        stages = record.get("diversity")
        if not isinstance(dataset, str) or not isinstance(stages, list):
            raise ValueError("records require dataset and diversity lists")
        if tuple(row.get("stage") for row in stages) != STAGES:
            raise ValueError("every image requires ordered st3_block0 and st3..st9 summaries")
        for row in stages:
            grouped[(dataset, row["stage"])].append(row)

    def reduce_rows(rows):
        result = {"image_count": len(rows), "weight_kind": rows[0]["weight_kind"]}
        if any(row["weight_kind"] != result["weight_kind"] for row in rows):
            raise ValueError("mixed attention directions within a stage")
        for key in (
            "unique_query_ids_in_all_topk", "unique_ordered_topk_patterns",
            "unique_unordered_topk_sets", "mean_pairwise_topk_jaccard_percent",
            "identical_topk_set_pair_percent", "topk_mass_percent_mean",
            "mean_pairwise_full_distribution_overlap_percent",
            "attention_entropy_normalized_percent_mean",
        ):
            values = [row[key] for row in rows]
            result[key] = None if values[0] is None else sum(values) / len(values)
        return result

    datasets = {}
    for dataset in sorted({key[0] for key in grouped}):
        datasets[dataset] = {
            stage: reduce_rows(grouped[(dataset, stage)]) for stage in STAGES
        }
    pooled = {}
    for stage in STAGES:
        pooled[stage] = reduce_rows([
            row for (dataset, current), rows in grouped.items() if current == stage for row in rows
        ])
    return {"schema_version": 1, "datasets": datasets, "pooled": pooled}


def format_latent_query_diversity_markdown(summary):
    """Make a compact Chinese table whose direction is unambiguous."""
    if summary.get("schema_version") != 1:
        raise ValueError("unknown diversity summary schema")
    labels = {
        "refcoco_val_refseg": "RefCOCO val",
        "refcoco+_val_refseg": "RefCOCO+ val",
        "refcocog_val_refseg": "RefCOCOg val",
    }
    sections = [
        "# 64 个 latent 的 Top-5 Query 区分度",
        "",
        "读法：`Top5 重合` 越低、`不同 Top5 组` 越多，说明 latent 选择的 Query 越有区分度；"
        "`完整分布重合` 是全部 Query 权重的重合率，100% 表示完全相同。每张图先统计一次，再按图等权平均。",
    ]
    groups = [(labels.get(name, name), rows) for name, rows in summary["datasets"].items()]
    groups.append(("合并（按图等权）", summary["pooled"]))
    for label, stages in groups:
        sections.extend([
            "", f"## {label}", "",
            "| 层 | 图数 | 64×Top5 共用不同 Query 数 | 不同 Top5 组数 | Top5 两两重合 | 完全相同 Top5 对 | 完整注意力分布重合 | Top5 权重和 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for stage in STAGES:
            row = stages[stage]
            def percent(key):
                value = row[key]
                return "N/A" if value is None else f"{value:.2f}%"
            sections.append(
                f"| {stage} | {row['image_count']} | "
                f"{row['unique_query_ids_in_all_topk']:.2f} | "
                f"{row['unique_unordered_topk_sets']:.2f} / 64 | "
                f"{row['mean_pairwise_topk_jaccard_percent']:.2f}% | "
                f"{row['identical_topk_set_pair_percent']:.2f}% | "
                f"{percent('mean_pairwise_full_distribution_overlap_percent')} | "
                f"{percent('topk_mass_percent_mean')} |"
            )
    sections.extend([
        "", "## 判断标准", "",
        "若训练后的 st3–st9 仍接近 `不同 Top5 组数=1`、`Top5 两两重合=100%`、"
        "`完整注意力分布重合≈100%`，就仍是所有 latent 读取几乎相同 Query；"
        "反之，只要这些值在多张图上稳定拉开，才能说分组辅助损失确实让读取关系产生了区分。",
    ])
    return "\n".join(sections) + "\n"


__all__ = [
    "STAGES", "summarize_latent_query_diversity",
    "aggregate_latent_query_diversity", "format_latent_query_diversity_markdown",
]
