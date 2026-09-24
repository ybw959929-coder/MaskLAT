"""Inference-only intervention on latent -> Query residual updates.

Never subtract the residual from a rounded output: return a fresh copy of
the original Query input, and preserve the ORIGINAL relation-logit object.
The latter is still consumed by Query -> latent writeback downstream.
"""
from contextlib import contextmanager
import math

import torch


STAGES = ("entry_st3", "st4", "st5", "st6", "st7", "st8", "st9")
DATASETS = ("refcoco_val_refseg", "refcoco+_val_refseg", "refcocog_val_refseg", "val_reaseg")


class QueryReadIntervention:
    """Temporary hooks; no parameter, production forward or config edits."""

    def __init__(self, model):
        self.model = model
        bridge = model.segmentor.st3_transport_bridge
        if type(bridge).__name__ != "BipartiteLatentTransportBridge":
            raise ValueError("requires the native BipartiteLatentTransportBridge")
        if len(bridge.transport_stages) != 6:
            raise ValueError("requires all six transport stages st4..st9")
        self.modules = dict(zip(STAGES, [bridge.entry_query_read] + [
            stage.query_read for stage in bridge.transport_stages
        ]))
        if len({id(m) for m in self.modules.values()}) != 7:
            raise ValueError("expected seven distinct QueryLatentRead modules")
        for stage, module in self.modules.items():
            if type(module).__name__ != "QueryLatentRead":
                raise ValueError(f"{stage} is not a native QueryLatentRead")
        found = {id(m) for m in model.modules() if type(m).__name__ == "QueryLatentRead"}
        if found != {id(m) for m in self.modules.values()}:
            raise ValueError("unaccounted QueryLatentRead modules; refusing partial ablation")
        self.active = False

    @contextmanager
    def run(self, *, block):
        if type(block) is not bool:
            raise TypeError("block must be boolean")
        if self.active:
            raise RuntimeError("nested Query-read interventions are forbidden")
        if any(m.training for m in self.model.modules()):
            raise ValueError("intervention is evaluation-only; call model.eval()")
        self.active = True
        audit = {"block": block, "calls": dict.fromkeys(STAGES, 0)}
        handles = []

        def make_hook(stage):
            def hook(module, args, kwargs, output):
                if torch.is_grad_enabled() or module.training:
                    raise RuntimeError("intervention must run under no_grad in eval mode")
                query = kwargs.get("query_states", args[0] if args else None)
                if (not torch.is_tensor(query) or query.ndim != 3
                        or not isinstance(output, tuple) or len(output) != 2
                        or output[0].shape != query.shape):
                    raise ValueError("unexpected native QueryLatentRead I/O")
                relation = output[1]
                if relation.ndim != 3 or relation.shape[:2] != query.shape[:2]:
                    raise ValueError("unexpected shared Query-latent relation shape")
                audit["calls"][stage] += 1
                # Baseline returns the exact native tuple, without arithmetic.
                return (query.clone(), relation) if block else output
            return hook

        try:
            for stage, module in self.modules.items():
                handles.append(module.register_forward_hook(make_hook(stage), with_kwargs=True))
            yield audit
        finally:
            for handle in handles:
                handle.remove()
            self.active = False


def validate_pair_audit(normal, blocked):
    if normal.get("block") is not False or blocked.get("block") is not True:
        raise ValueError("expected normal and blocked audits")
    if (set(normal["calls"]) != set(STAGES) or set(blocked["calls"]) != set(STAGES)
            or normal["calls"] != blocked["calls"]
            or any(type(n) is not int or n <= 0 for n in normal["calls"].values())):
        raise ValueError("every pair must execute all seven branches equally, at least once")


def validate_coverage(shards, length):
    indices = [row["dataset_index"] for shard in shards for row in shard]
    if len(indices) != length or set(indices) != set(range(length)):
        raise ValueError("paired full-val samples are missing or duplicated")
    for shard in shards:
        for row in shard:
            validate_pair_audit(row["normal_audit"], row["blocked_audit"])
    return sorted((row for shard in shards for row in shard), key=lambda row: row["dataset_index"])


def foreground_metrics(summary, name, expected_count, world):
    foreground = "reason" if name == "val_reaseg" else "refer"
    if (summary.get("schema_version") != 3 or summary.get("data_name") != name
            or summary.get("num_predictions") != expected_count):
        raise ValueError("missing or incomplete corrected all-rank metric summary")
    metrics = summary.get("metrics", {}).get(foreground, {})
    audit = summary.get("aggregation_audit", {}).get("correct_all_ranks", {})
    if (audit.get("world_size") != world or audit.get("num_predictions") != expected_count
            or audit.get("foreground_class") != foreground):
        raise ValueError("invalid all-rank aggregation audit")
    result = {}
    for key in ("cIoU", "gIoU"):
        value, audited = metrics.get(key), audit.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 100
                or isinstance(audited, bool) or not isinstance(audited, (int, float))
                or not math.isclose(value, audited, rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f"invalid corrected {foreground}/{key}")
        result[key] = float(value)
    return result


def comparison_row(name, normal, blocked, records, world):
    if not records:
        raise ValueError("cannot report empty paired evaluation")
    a = foreground_metrics(normal, name, len(records), world)
    b = foreground_metrics(blocked, name, len(records), world)
    return dict(
        dataset=name, samples=len(records), normal=a, blocked=b,
        drop_pp={key: a[key] - b[key] for key in a},
        changed_masks=sum(bool(r["mask_changed"]) for r in records),
        changed_masks_percent=100 * sum(bool(r["mask_changed"]) for r in records) / len(records),
    )


def format_comparison(rows):
    lines = [
        "# latent -> query 分支成对关闭测试", "",
        "下降 = 正常 - 关闭（百分点）；正数表示关闭后变差，负数表示关闭后变好。", "",
        "| 数据集 | 样本数 | 正常 cIoU | 关闭 cIoU | 下降 pp | 正常 gIoU | 关闭 gIoU | 下降 pp | mask 改变比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        a, b, d = row["normal"], row["blocked"], row["drop_pp"]
        lines.append(
            f"| {row['dataset']} | {row['samples']} | {a['cIoU']:.2f} | {b['cIoU']:.2f} | {d['cIoU']:+.2f} | "
            f"{a['gIoU']:.2f} | {b['gIoU']:.2f} | {d['gIoU']:+.2f} | {row['changed_masks_percent']:.2f}% |")
    lines.extend(["", "这是同一已训练模型的推理期干预，不是删除分支后重新训练的消融。",
                  "下降支持当前模型依赖此路径；不能单独证明共享信息机制或相对原始 MaskLAT 的涨点来源。", ""])
    return "\n".join(lines)
