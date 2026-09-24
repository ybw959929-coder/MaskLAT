"""Offline, human-readable reports for read-only latent/GT diagnostics.

This module formats already-computed statistics. It neither runs the model nor
loads masks, and it never substitutes oracle IoU for the model's actual choice.
"""
from __future__ import annotations

from html import escape
import math
from numbers import Integral, Real


_STAGES = ("st3_block0", "st3", "st4", "st5", "st6", "st7", "st8", "st9")
_DATASET_LABELS = {
    "refcoco_val_refseg": "RefCOCO",
    "refcoco+_val_refseg": "RefCOCO+",
    "refcocog_val_refseg": "RefCOCOg",
}


def _cell(value):
    """Escape Markdown table syntax, including untrusted source expressions."""
    return (escape(str(value), quote=False).replace("\\", "&#92;")
            .replace("|", "&#124;").replace("`", "&#96;")
            .replace("*", "&#42;").replace("_", "&#95;")
            .replace("[", "&#91;").replace("]", "&#93;")
            .replace("\r\n", " ").replace("\r", " ").replace("\n", " "))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _percent(value, name, *, fraction=False):
    value = _number(value, name)
    if fraction:
        value *= 100
    if not 0 <= value <= 100 + 1e-6:
        raise ValueError(f"{name} must be within [0, {'1' if fraction else '100'}]")
    return f"{value:.2f}%"


def _schema(value):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("expected a schema_version=1 diagnostic dictionary")


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def _ordered_stages(stages):
    if not isinstance(stages, list):
        raise ValueError("stages must be a list")
    if not stages:
        return []
    if len(stages) != len(_STAGES) or {row.get("stage") for row in stages} != set(_STAGES):
        raise ValueError("attention report requires st3_block0 and st3 through st9 exactly once")
    by_name = {row["stage"]: row for row in stages}
    ordered = [by_name[name] for name in _STAGES]
    for row in ordered:
        expected_direction = "query_reads_latent" if row["stage"] == "st9" else "latent_reads_query"
        if row["weight_kind"] != expected_direction:
            raise ValueError(f"unexpected attention direction for {row['stage']}")
    return ordered


def _stage_label(name):
    return "st3 block0（额外）" if name == "st3_block0" else "st9 *" if name == "st9" else name


def _dataset_label(name):
    return _cell(_DATASET_LABELS.get(name, name))


_NOTES = """## 怎么读这份报告

- 这里的 Top5 按注意力权重排序，不是上一份表按 GT IoU 排序的“最好五个”。相同 Query ID 在同一张图的不同 latent 中，表示同一个 Query，不是新的候选或新的图片。
- st3 block0 是额外观察点；st3 是 builder 最后一个 block。两者以同一组 st3 Query mask 计算 GT IoU。st4～st9 采用各层对应的 Query mask。
- st4～st8 的实际 Query 写回 value 位于 Query self-attention 之后、FFN 之前，而这里用同一 Query ID 在该层 FFN 之后输出的 mask 衡量 GT IoU。这是回看该 Query 的候选质量，不是声称注意力直接读取了这张最终 mask。
- st3～st8 展示 latent 读取/写回聚合 Query 的关系。st9 * 没有 Query→latent 写回，展示反方向“哪些 Query 最依赖这个 latent”的 Top5，不能解释成 latent 在该层主动读取它们。st9 仍可能经过 Cond/VLM 刷新。
- “latent 命中好 Query 比例”是每张图 64 个 latent 中，Top5 至少有一个 IoU≥0.7 Query 的比例，不是图像覆盖率。“不同 Query 数”是该图全部 64×5 个位置去重后的数量。
- 先对每张图的 64 个 latent 求均值，再在图像之间平均；合并表按图像等权，不把 64 个 latent 当成 64 张独立图片。同一 COCO 原图可能出现在不同数据集，合并表不是互不重复图像的独立样本统计。
- 最终选中 IoU 使用模型真正输出的 mask；oracle 是最终层所有 Query 中与 GT 最接近的候选。差距以百分点计，不是相对百分比。错失好候选指 oracle IoU≥0.7、但实际选中 IoU<0.7。
- 注意力与 GT IoU 的关系只能说明关联，不能证明因果：低 GT IoU 的 Query 也可能提供背景或关系信息。注意力相似也不等于 latent 特征相同。
"""


def format_summary_markdown(summary):
    """Return Chinese Markdown for dataset and image-weighted pooled results."""
    _schema(summary)
    datasets = summary["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("summary must contain at least one dataset")
    groups = [(name, group) for name, group in datasets.items()]
    groups.append(("合并（按图像等权）", summary["pooled"]))
    parts = ["# Latent 关注了什么，以及最终有没有选中好 mask", _NOTES.rstrip()]
    if summary.get("metadata", {}).get("synthetic", False):
        parts.insert(1, "**合成测试数据，仅用于验证流程，不是真实模型结果。**")
    parts.append("## 一、模型最终选择与最佳候选")
    final_rows = []
    for name, group in groups:
        final = group["final_selection"]
        sample_count = _integer(group["sample_count"], "sample_count")
        if sample_count != _integer(final["sample_count"], "final_selection.sample_count"):
            raise ValueError("group and final-selection sample counts must agree")
        missed = _integer(final["missed_good_sample_count"]["0.7"], "missed_good_sample_count")
        oracle_count = _integer(final["oracle_good_sample_count"]["0.7"], "oracle_good_sample_count")
        if not 0 <= missed <= oracle_count <= sample_count:
            raise ValueError("missed/oracle-good counts are inconsistent")
        conditional = final["missed_good_given_oracle_good_percent"]["0.7"]
        if oracle_count == 0:
            if conditional is not None:
                raise ValueError("conditional percentage must be null when no oracle-good sample exists")
            success = "N/A（0 张有好候选）"
        else:
            # Counts are shown explicitly: this is not the unconditional success rate.
            _percent(conditional, "missed_good_given_oracle_good_percent")
            success = f"{100 * (oracle_count - missed) / oracle_count:.2f}%（{oracle_count - missed}/{oracle_count}）"
        gap = _number(final["gap_percentage_points_mean"], "gap_percentage_points_mean")
        if gap < -1e-6:
            raise ValueError("oracle minus selected gap must not be negative")
        final_rows.append([
            _dataset_label(name), sample_count,
            _percent(final["selected_iou_percent_mean"], "selected_iou_percent_mean"),
            _percent(final["oracle_iou_percent_mean"], "oracle_iou_percent_mean"),
            f"{gap:.2f} pp",
            _percent(final["oracle_good_percent"]["0.7"], "oracle_good_percent"),
            _percent(final["selected_good_percent"]["0.7"], "selected_good_percent"),
            missed, success,
        ])
    parts.append(_table([
        "数据集", "图数", "实际选中 IoU", "最佳候选 IoU", "差距（百分点）",
        "有好候选 ≥0.7", "实际选对 ≥0.7", "错失好候选图数", "有好候选时选对比例",
    ], final_rows))
    parts.append("## 二、注意力 Top5 与 GT 的关系")
    for name, group in groups:
        parts.append(f"### {_dataset_label(name)}（{_integer(group['sample_count'], 'sample_count')} 张）")
        stages = _ordered_stages(group["stages"])
        if not stages:
            parts.append("未采集注意力。本组只有最终选择统计，不能由此判断 latent 的 Top5。")
            continue
        rows = []
        for stage in stages:
            if _integer(stage["sample_count"], "stage.sample_count") != group["sample_count"]:
                raise ValueError("attention stage sample count must match group")
            rows.append([
                _stage_label(stage["stage"]),
                _percent(stage["top1_iou_percent_mean"], "top1_iou_percent_mean"),
                _percent(stage["top5_mean_iou_percent_mean"], "top5_mean_iou_percent_mean"),
                _percent(stage["top5_max_iou_percent_mean"], "top5_max_iou_percent_mean"),
                f"{_number(stage['top5_good_query_count_0p7_mean'], 'top5_good_query_count_0p7_mean'):.2f} / 5",
                _percent(stage["slot_hit_rate_0p7_percent_mean"], "slot_hit_rate_0p7_percent_mean"),
                f"{_number(stage['unique_top5_query_count_mean'], 'unique_top5_query_count_mean'):.2f}",
            ])
        parts.append(_table([
            "层", "关注 Top1 IoU", "关注 Top5 平均 IoU", "Top5 中最好 IoU",
            "5 个里好 Query 数 ≥0.7", "latent 命中好 Query 比例", "64 latent 合计不同 Query 数",
        ], rows))
    parts.append("## 建议先看什么\n\n先看最终实际选中与最佳候选的差距，再看注意力 Top5 是否包含好候选。好候选数量、注意力权重和最终选择是三个不同问题；本表不自动把差距归因于 latent 或 decoder。")
    return "\n\n".join(parts) + "\n"


def format_sample_markdown(source, analysis):
    """Return per-image details, including every latent's five Query ID/IoU pairs."""
    _schema(analysis)
    if not isinstance(source, dict):
        raise ValueError("source must be a dictionary")
    parts = ["# 单张图：注意力选择与 GT 对照"]
    if source.get("synthetic", False) or analysis.get("synthetic", False):
        parts.append("**合成测试数据，仅用于验证流程，不是真实模型结果。**")
    source_rows = [[label, _cell(source.get(key, "未提供"))] for key, label in (
        ("dataset", "数据集"), ("image_id", "原图 ID"), ("image_file", "原图文件"),
        ("expression", "对应表达"),
    )]
    parts.append(_table(["项目", "值"], source_rows))
    final = analysis["final_selection"]
    chosen_id = _integer(final["selected_query_id"], "selected_query_id")
    oracle_id = _integer(final["oracle_query_id"], "oracle_query_id")
    gap = _number(final["gap"], "gap")
    if gap < -1e-8:
        raise ValueError("oracle minus selected gap must not be negative")
    parts.append(_table(
        ["实际选中 Query", "实际 mask IoU", "最佳候选 Query", "最佳 IoU", "差距（百分点）", "是否错失好候选 ≥0.7"],
        [[f"Q{chosen_id}", _percent(final["selected_iou"], "selected_iou", fraction=True),
          f"Q{oracle_id}", _percent(final["oracle_iou"], "oracle_iou", fraction=True),
          f"{gap * 100:.2f} pp", "是" if final["missed_good"]["0.7"] else "否"]],
    ))
    parts.append("下表每格为 **Query ID；注意力权重；该 Query mask 的 GT IoU**。顺序按注意力排序，不按 GT IoU 排序。64 行是同一张图的 64 个 latent，不是 64 张图片。")
    stages = _ordered_stages(analysis["stages"])
    if not stages:
        parts.append("未采集注意力；本图仅保存最终选择统计。")
    for stage in stages:
        parts.append(f"## {_stage_label(stage['stage'])}")
        if stage["stage"] == "st9":
            parts.append("此处是 Query→latent 依赖权重：为每个 latent 排出最依赖它的 Query，不是 latent 主动读取 Query。每行的五个百分比不是一份总和为 100% 的 latent 读取分布；st9 仍可能有 Cond/VLM 刷新。")
        else:
            parts.append("此处是 latent 对 Query 的读取/写回聚合关系；每行只展示权重最大的 5 个 Query，未将 Top5 权重重新归一化。")
        if stage["stage"] in ("st3", "st3_block0"):
            parts.append("GT IoU 来自同一组 st3 Query mask；block0 与最后一个 block 可选中不同 Query。")
        latent_rows = stage["latent_rows"]
        if len(latent_rows) != 64:
            raise ValueError("per-stage detail must contain all 64 latent rows")
        by_id = {}
        for latent in latent_rows:
            latent_id = _integer(latent["latent_id"], "latent_id")
            if latent_id in by_id:
                raise ValueError("duplicate latent ID")
            by_id[latent_id] = latent
        if set(by_id) != set(range(64)):
            raise ValueError("latent IDs must be 0 through 63")
        rows = []
        for latent_id in range(64):
            latent = by_id[latent_id]
            ids, weights, ious = (latent[key] for key in ("top5_query_ids", "top5_weights", "top5_ious"))
            if not len(ids) == len(weights) == len(ious) == 5:
                raise ValueError("each latent must contain exactly five Query ID/weight/IoU triplets")
            if len(set(ids)) != 5:
                raise ValueError("Top5 Query IDs within a latent must be distinct")
            cells = [f"L{latent_id:02d}"]
            for query_id, weight, iou in zip(ids, weights, ious):
                cells.append(f"Q{_integer(query_id, 'query_id')}；权重 {_percent(weight, 'weight', fraction=True)}；IoU {_percent(iou, 'iou', fraction=True)}")
            rows.append(cells)
        parts.append(_table(["latent", "Top1", "Top2", "Top3", "Top4", "Top5"], rows))
    parts.append(_NOTES.rstrip())
    return "\n\n".join(parts) + "\n"


def format_report_html(title, markdown_text):
    """Return a standalone, escaped offline page; no Markdown/JS dependency.

    Preformatted Markdown deliberately preserves the same readable data table
    as the .md file. No source text is interpreted as active HTML or a URL.
    """
    if not isinstance(title, str) or not isinstance(markdown_text, str):
        raise ValueError("HTML title and report must be strings")
    return ("<!doctype html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
            f"<title>{escape(title)}</title>"
            "<style>body{margin:24px;color:#202124;background:#fff;font:16px/1.65 sans-serif}"
            "pre{white-space:pre;overflow-x:auto;padding:16px;border:1px solid #ddd;"
            "font:14px/1.7 ui-monospace,Consolas,monospace}"
            "@media print{pre{white-space:pre-wrap;overflow:visible}}</style></head><body>"
            f"<h1>{escape(title)}</h1><p>离线报告，与 Markdown 文件内容相同。横向滚动可查看完整表格。</p>"
            f"<pre>{escape(markdown_text)}</pre></body></html>\n")
