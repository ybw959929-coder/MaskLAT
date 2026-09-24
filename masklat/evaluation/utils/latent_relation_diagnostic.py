"""Inference-only tests of st4--st8 Query-to-latent writeback correspondence.

Permuting rows of W(A @ V) is exactly permuting rows of A before the shared
affine projection W. Hook ONLY this increment, before addition to persistent
latents. Query-read logits/updates, latent residual/SA/FFN and Cond stay native.
This is NOT a trained independent-relation architecture comparison.
"""
from contextlib import contextmanager
import math
from statistics import mean

import torch

from .latent_shared_query_read import SharedQueryReadIntervention
from .latent_query_read_ablation import foreground_metrics


WRITEBACK_SITES = tuple(f"writeback.st{i}" for i in range(4, 9))
DEFAULT_SHUFFLE_SEEDS = (17, 29, 43)


def variant_names(seeds):
    seeds = tuple(seeds)
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)):
        raise ValueError("use exactly three distinct shuffle seeds in [0, 2**31)")
    return ("normal",) + tuple(f"shuffle_{s}" for s in seeds) + ("no_writeback",)


def slot_permutation(seed, stage, slots=64):
    """Fixed per seed AND stage, same on all ranks/images; never consumes global RNG."""
    if type(seed) is not int or not 0 <= seed < 2**31 or stage not in range(4, 9) or slots != 64:
        raise ValueError("expected nonnegative seed, st4..st8 and 64 slots")
    generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + stage)
    for _ in range(10000):
        order = torch.randperm(slots, generator=generator)
        if bool((order != torch.arange(slots)).all()):
            return order
    raise RuntimeError("failed to construct a no-fixed-point permutation")


def change_statistics(native, changed):
    if (not torch.is_tensor(native) or native.ndim != 3 or native.shape[1] != 64
            or min(native.shape) <= 0 or not native.is_floating_point()
            or not torch.is_tensor(changed) or changed.shape != native.shape
            or changed.dtype != native.dtype or changed.device != native.device
            or not torch.isfinite(native).all() or not torch.isfinite(changed).all()):
        raise ValueError("write increments must be matching finite floating [B,64,D]")
    before = native.double() if native.dtype == torch.float64 else native.float()
    delta = changed.to(before.dtype) - before
    norms = before.flatten(1).norm(dim=1)
    delta_norms = delta.flatten(1).norm(dim=1)
    # Zero native messages necessarily remain zero for both of our interventions.
    ratios = delta_norms / norms.clamp_min(torch.finfo(before.dtype).tiny)
    return dict(rows=len(native), relative_l2_mean=float(ratios.mean()),
                relative_l2_max=float(ratios.max()), zero_native_rows=int((norms == 0).sum()),
                native_rms=float(before.square().mean(dim=(1, 2)).sqrt().mean()),
                change_rms=float(delta.square().mean(dim=(1, 2)).sqrt().mean()),
                changed_element_fraction=float((native != changed).float().mean()))


class RelationDiagnostic:
    def __init__(self, model, seeds=DEFAULT_SHUFFLE_SEEDS):
        self.variants = variant_names(seeds)
        self.model, self.active = model, False
        # Reuse strict native bridge/builder, latent count and extra-module guards.
        layout = SharedQueryReadIntervention(model, scope="writeback")
        self.reads, self.feedback = layout.reads, layout.feedback
        self.read_names = tuple(self.reads)
        if self.read_names != WRITEBACK_SITES:
            raise ValueError("diagnostic must target exactly st4..st8 writebacks")
        bridge = model.segmentor.st3_transport_bridge
        self.preserved = dict(self.feedback)
        for index, stage in enumerate(bridge.transport_stages[:5], start=4):
            self.preserved[f"latent_sa.st{index}"] = stage.latent_self_attention
            self.preserved[f"latent_ffn.st{index}"] = stage.latent_ffn
        for name, refresher in getattr(bridge, "late_condition_refreshers", {}).items():
            self.preserved[f"cond_refresh.{name}"] = refresher
        if len({id(m) for m in self.preserved.values()}) != len(self.preserved):
            raise ValueError("unexpected shared preserved module")
        self.permutations = {
            f"shuffle_{seed}": {name: slot_permutation(seed, stage)
                               for stage, name in zip(range(4, 9), self.read_names)}
            for seed in seeds
        }

    def contract(self):
        return dict(variants=list(self.variants), writeback_sites=list(self.read_names),
                    preserved_sites=list(self.preserved),
                    permutations={mode: {site: p.tolist() for site, p in sites.items()}
                                  for mode, sites in self.permutations.items()},
                    permutation_convention="output slot l receives native increment at permutation[l]",
                    unchanged="builder, st3 entry, all Query feedback/logits, st9, Cond, latent residual/SA/FFN",
                    strength_definition="local counterfactual: changed vs native write increment for the SAME current inputs; not vs baseline trajectory")

    @contextmanager
    def run(self, variant):
        if variant not in self.variants:
            raise ValueError("unknown relation-diagnostic variant")
        if self.active:
            raise RuntimeError("nested relation intervention")
        if any(module.training for module in self.model.modules()):
            raise ValueError("requires model.eval()")
        audit = dict(variant=variant, calls=dict.fromkeys(self.read_names, 0),
                     preserved_calls=dict.fromkeys(self.preserved, 0),
                     changes={name: [] for name in self.read_names})
        handles = []
        self.active = True

        def ensure_eval(module):
            if module.training or torch.is_grad_enabled():
                raise RuntimeError("diagnostic requires eval and no_grad")

        def write_hook(name):
            def hook(module, args, output):
                ensure_eval(module)
                if variant == "normal":
                    changed = output
                elif variant == "no_writeback":
                    changed = torch.zeros_like(output)
                else:
                    changed = output.index_select(1, self.permutations[variant][name].to(output.device))
                audit["changes"][name].append(change_statistics(output, changed))
                audit["calls"][name] += 1
                return changed  # normal returns exact object; never change latent residual
            return hook

        def audit_hook(name):
            def hook(module, args, output):
                ensure_eval(module)
                audit["preserved_calls"][name] += 1
                return output
            return hook

        try:
            for name, (module, _) in self.reads.items():
                handles.append(module.register_forward_hook(write_hook(name)))
            for name, module in self.preserved.items():
                handles.append(module.register_forward_hook(audit_hook(name)))
            yield audit
        finally:
            for handle in handles:
                handle.remove()
            self.active = False


def validate_audits(audits, contract):
    variants = contract["variants"]
    if set(audits) != set(variants) or variants[0] != "normal":
        raise ValueError("missing or unexpected variant audits")
    normal = audits["normal"]
    for mode in variants:
        audit = audits[mode]
        if audit["variant"] != mode:
            raise ValueError("mismatched variant label")
        for key, expected in (("calls", contract["writeback_sites"]),
                              ("preserved_calls", contract["preserved_sites"])):
            if (set(audit[key]) != set(expected) or audit[key] != normal[key]
                    or any(type(v) is not int or v <= 0 for v in audit[key].values())):
                raise ValueError("selected/preserved sites must all execute equally")
        if set(audit["changes"]) != set(WRITEBACK_SITES):
            raise ValueError("missing writeback strength measurements")
        for name, stats in audit["changes"].items():
            if len(stats) != audit["calls"][name]:
                raise ValueError("missing per-call changes")
            if [s["rows"] for s in stats] != [s["rows"] for s in normal["changes"][name]]:
                raise ValueError("paired writeback batch sizes differ")
            for s in stats:
                if (type(s["rows"]) is not int or s["rows"] <= 0
                        or type(s["zero_native_rows"]) is not int
                        or not 0 <= s["zero_native_rows"] <= s["rows"]):
                    raise ValueError("invalid writeback row counts")
                for key in ("relative_l2_mean", "relative_l2_max", "native_rms", "change_rms", "changed_element_fraction"):
                    if not math.isfinite(s[key]) or s[key] < 0:
                        raise ValueError("nonfinite/negative intervention strength")
                if (not s["relative_l2_mean"] <= s["relative_l2_max"] + 1e-7
                        or s["relative_l2_max"] > 2.0001 or s["changed_element_fraction"] > 1):
                    raise ValueError("invalid norm-preserving permutation strength")
                if mode == "normal" and (s["change_rms"] != 0 or s["changed_element_fraction"] != 0):
                    raise ValueError("normal path changed")
                if mode == "no_writeback":
                    expected = 1 - s["zero_native_rows"] / s["rows"]
                    if not math.isclose(s["relative_l2_mean"], expected, abs_tol=1e-6):
                        raise ValueError("no-writeback must remove the entire native increment")


def validate_coverage(shards, length, contract):
    records = [row for shard in shards for row in shard]
    if length <= 0 or len(records) != length or {r["dataset_index"] for r in records} != set(range(length)):
        raise ValueError("missing/duplicate full-val records")
    for row in records:
        validate_audits(row["audits"], contract)
        expected = set(contract["variants"]) - {"normal"}
        if set(row["mask_changes"]) != expected:
            raise ValueError("missing paired mask comparisons")
        for change in row["mask_changes"].values():
            value = change["pixel_disagreement_fraction"]
            if type(change["mask_changed"]) is not bool or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid paired mask comparison")
    return sorted(records, key=lambda r: r["dataset_index"])


def comparison_rows(name, summaries, records, world, contract):
    if not records or set(summaries) != set(contract["variants"]):
        raise ValueError("incomplete metric variants")
    normal = foreground_metrics(summaries["normal"], name, len(records), world)
    rows = []
    for mode in contract["variants"]:
        result = foreground_metrics(summaries[mode], name, len(records), world)
        strengths = {}
        for site in WRITEBACK_SITES:
            # Equal weight per dataset record, even if one record has several SEG rows/calls.
            per_record = []
            for record in records:
                stats = record["audits"][mode]["changes"][site]
                count = sum(s["rows"] for s in stats)
                per_record.append(sum(s["relative_l2_mean"] * s["rows"] for s in stats) / count)
            strengths[site] = dict(relative_l2_percent=100 * mean(per_record),
                                   maximum_record_relative_l2_percent=100 * max(per_record))
        changed = 0 if mode == "normal" else sum(r["mask_changes"][mode]["mask_changed"] for r in records)
        rows.append(dict(dataset=name, variant=mode, samples=len(records), normal=normal, metrics=result,
                         drop_pp={k: normal[k] - result[k] for k in normal},
                         changed_masks_percent=100 * changed / len(records), strength_by_stage=strengths))
    return rows


def format_comparison(rows):
    lines = ["# st4–st8 共享关系写回诊断", "",
             "下降 = 本次正常 − 干预（百分点）；不是共享 R vs 独立 R 的重训对照。", "",
             "| 数据集 | 模式 | 样本数 | cIoU | 下降 pp | gIoU | 下降 pp | mask 改变比例 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m, d = row["metrics"], row["drop_pp"]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['samples']} | {m['cIoU']:.2f} | {d['cIoU']:+.2f} | "
                     f"{m['gIoU']:.2f} | {d['gIoU']:+.2f} | {row['changed_masks_percent']:.2f}% |")
    lines.extend(["", "## 写回增量实际变化（相对 L2，%）", "",
                  "同一次前向、相同当前输入下：100 × ||干预增量−原增量|| / ||原增量||；零增量记0。不是注意力差异或准确率。",
                  "先在记录内按 SEG 行加权，再对数据集记录等权平均；逐记录详情见 paired JSON。", "",
                  "| 数据集 | 模式 | st4 | st5 | st6 | st7 | st8 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        values = " | ".join(f"{row['strength_by_stage'][s]['relative_l2_percent']:.4f}" for s in WRITEBACK_SITES)
        lines.append(f"| {row['dataset']} | {row['variant']} | {values} |")
    for name in dict.fromkeys(row["dataset"] for row in rows):
        shuffled = [r for r in rows if r["dataset"] == name and r["variant"].startswith("shuffle_")]
        if shuffled:
            parts = []
            for metric in ("cIoU", "gIoU"):
                drops = [r["drop_pp"][metric] for r in shuffled]
                parts.append(f"{metric}下降均值 {mean(drops):+.3f} pp，范围 [{min(drops):+.3f}, {max(drops):+.3f}]")
            lines.extend(["", f"{name} 三次错配：" + "；".join(parts) + "。这是排列敏感性，不是训练随机种子置信区间。"])
    lines.extend(["", "normal=正常；shuffle_数字=固定种子错配写回；no_writeback=只关闭 Query 写回增量。",
                  "全部保留 st3 入口、所有 latent→Query 反馈、latent 残差/SA/FFN 与 Cond 交互；st9 不改。",
                  "错配几乎不改变增量时，不掉点不能说明对应关系无用；关闭写回不等于 latent 完全不更新。",
                  "掉点只支持当前模型对该干预敏感，不能证明共享 R 优于独立训练的双向 R。", ""])
    return "\n".join(lines)
