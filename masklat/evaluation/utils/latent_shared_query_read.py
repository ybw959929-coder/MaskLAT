"""Share Query-read messages across latent slots, NOT latent feature vectors.

For shared values V and a shared affine output projection W, mean_l(W(A_l V))
equals W(mean_l(A_l) V). The same identity holds independently in every MHA
head, then through concatenation/out_proj. We therefore average only native
readout increments BEFORE their residual addition, without reimplementing
attention, averaging logits, mixing heads, or changing reverse routing.
Finite-precision reduction can introduce ordinary rounding differences.
"""
from contextlib import contextmanager

import torch
from torch import nn

from .latent_query_read_ablation import STAGES as FEEDBACK_STAGES, foreground_metrics


def shared_increment(increment):
    if (not torch.is_tensor(increment) or increment.ndim != 3
            or min(increment.shape) <= 0 or not increment.is_floating_point()
            or not torch.isfinite(increment).all()):
        raise ValueError("read increment must be finite nonempty floating [B,L,D]")
    reduction = increment if increment.dtype == torch.float64 else increment.float()
    # Centered reduction leaves an already-identical readout bitwise unchanged,
    # avoiding spurious changes from summing the same FP32 value 64 times.
    anchor = reduction[:, :1]
    mean = (anchor + (reduction - anchor).mean(dim=1, keepdim=True)).to(increment.dtype)
    return mean.expand_as(increment).clone()


class SharedQueryReadIntervention:
    def __init__(self, model, scope="all"):
        if scope not in ("all", "builder", "writeback"):
            raise ValueError("scope must be all, builder or writeback")
        self.model, self.scope, self.active = model, scope, False
        segmentor = model.segmentor
        builder = segmentor.st3_transport_proposal_builder
        bridge = segmentor.st3_transport_bridge
        if type(bridge).__name__ != "BipartiteLatentTransportBridge" or len(bridge.transport_stages) != 6:
            raise ValueError("requires native six-stage bipartite transport")
        if type(builder).__name__ == "St3BipartiteLatentBuilder":
            builders = [("st3", builder)]
        elif type(builder).__name__ == "St123LatentCascadeBuilder":
            builders = list(builder.stages.items())
            if [name for name, _ in builders] != ["st1", "st2", "st3"]:
                raise ValueError("cascade builders must be st1,st2,st3")
        else:
            raise ValueError("unsupported native latent builder")
        self.reads = {}
        for name, stage_builder in builders:
            if stage_builder.num_latents != 64 or not stage_builder.pre_vlm_blocks:
                raise ValueError("requires 64 latent slots and nonempty native builder blocks")
            if scope in ("all", "builder"):
                for index, block in enumerate(stage_builder.pre_vlm_blocks):
                    module = block.cross_attention
                    if (type(block).__name__ != "GateFreeLatentBlock"
                            or not isinstance(module, nn.MultiheadAttention)
                            or not module.batch_first):
                        raise ValueError("builder must use native batch-first gate-free MHA")
                    self.reads[f"builder.{name}.block{index}"] = (module, "mha")
        for index, stage in enumerate(bridge.transport_stages):
            if stage.enable_latent_writeback is not (index < 5):
                raise ValueError("requires st4..st8 writeback and read-only st9")
            if index < 5 and scope in ("all", "writeback"):
                if not isinstance(stage.latent_output, nn.Linear):
                    raise ValueError("writeback must use shared affine latent_output")
                self.reads[f"writeback.st{index + 4}"] = (stage.latent_output, "linear")
        self.read_names = tuple(self.reads)
        if len({id(module) for module, _ in self.reads.values()}) != len(self.reads):
            raise ValueError("read sites unexpectedly share the same module")
        modules = [bridge.entry_query_read] + [stage.query_read for stage in bridge.transport_stages]
        self.feedback = dict(zip(FEEDBACK_STAGES, modules))
        if len({id(m) for m in modules}) != 7 or any(type(m).__name__ != "QueryLatentRead" for m in modules):
            raise ValueError("expected all seven distinct native Query reads")
        found = {id(m) for m in model.modules() if type(m).__name__ == "QueryLatentRead"}
        if found != {id(m) for m in modules}:
            raise ValueError("unexpected additional Query feedback module")

    @contextmanager
    def run(self, *, shared):
        if type(shared) is not bool:
            raise TypeError("shared must be bool")
        if self.active:
            raise RuntimeError("nested shared-read intervention")
        if any(module.training for module in self.model.modules()):
            raise ValueError("shared-read intervention requires model.eval()")
        audit = dict(shared=shared, scope=self.scope, read_names=list(self.read_names),
                     calls=dict.fromkeys(self.read_names, 0),
                     feedback_calls=dict.fromkeys(FEEDBACK_STAGES, 0),
                     removed_relative_l2={name: [] for name in self.read_names})
        handles = []
        self.active = True

        def ensure_eval(module):
            if module.training or torch.is_grad_enabled():
                raise RuntimeError("shared-read intervention requires eval and no_grad")

        def read_hook(name, kind):
            def hook(module, args, output):
                ensure_eval(module)
                if kind == "mha":
                    if not isinstance(output, tuple) or len(output) != 2 or output[1] is not None:
                        raise ValueError("native builder must use need_weights=False")
                    increment = output[0]
                else:
                    increment = output
                mean = shared_increment(increment)
                if increment.shape[1] != 64:
                    raise ValueError("expected exactly 64 latent Query-read messages")
                numerator = (increment.float() - mean.float()).norm()
                denominator = increment.float().norm().clamp_min(1e-12)
                audit["removed_relative_l2"][name].append(float(numerator / denominator))
                audit["calls"][name] += 1
                if not shared:
                    return output  # exact native object; no baseline arithmetic
                return (mean, output[1]) if kind == "mha" else mean
            return hook

        def feedback_hook(name):
            def hook(module, args, output):
                ensure_eval(module)
                audit["feedback_calls"][name] += 1
                return output  # KEEP query update and relation logits unchanged
            return hook

        try:
            for name, (module, kind) in self.reads.items():
                handles.append(module.register_forward_hook(read_hook(name, kind)))
            for name, module in self.feedback.items():
                handles.append(module.register_forward_hook(feedback_hook(name)))
            yield audit
        finally:
            for handle in handles:
                handle.remove()
            self.active = False


def validate_pair_audit(normal, shared):
    if normal.get("shared") is not False or shared.get("shared") is not True:
        raise ValueError("expected native/shared-read pair")
    names = normal.get("read_names", [])
    if (not names or len(set(names)) != len(names) or shared.get("read_names") != names
            or normal.get("scope") != shared.get("scope")):
        raise ValueError("inconsistent read scope")
    for key, expected in (("calls", set(names)), ("feedback_calls", set(FEEDBACK_STAGES))):
        if (set(normal[key]) != expected or set(shared[key]) != expected
                or normal[key] != shared[key]
                or any(type(v) is not int or v <= 0 for v in normal[key].values())):
            raise ValueError("all selected reads and seven preserved feedbacks must execute equally")
    for audit in (normal, shared):
        if set(audit["removed_relative_l2"]) != set(names):
            raise ValueError("missing read-message difference audit")
        for name in names:
            values = audit["removed_relative_l2"][name]
            if len(values) != audit["calls"][name] or any(not 0 <= v <= 1.01 for v in values):
                raise ValueError("invalid read-message difference audit")


def validate_coverage(shards, length):
    records = [row for shard in shards for row in shard]
    if len(records) != length or {r["dataset_index"] for r in records} != set(range(length)):
        raise ValueError("shared-read evaluation has missing or duplicate records")
    for row in records:
        validate_pair_audit(row["normal_audit"], row["shared_audit"])
    return sorted(records, key=lambda row: row["dataset_index"])


def comparison_row(name, normal, shared, records, world):
    if not records:
        raise ValueError("empty evaluation")
    a = foreground_metrics(normal, name, len(records), world)
    b = foreground_metrics(shared, name, len(records), world)
    return dict(dataset=name, samples=len(records), normal=a, shared=b,
                drop_pp={key: a[key] - b[key] for key in a},
                scope=records[0]["normal_audit"]["scope"],
                changed_masks_percent=100 * sum(bool(r["mask_changed"]) for r in records) / len(records))


def format_comparison(rows):
    lines = ["# 64 latent 共享 Query 读取测试", "",
             "下降 = 正常 − 共享读取（百分点）。latent 自身特征与反向反馈没有被平均或关闭。", "",
             "| 数据集 | 范围 | 样本数 | 正常 cIoU | 共享 cIoU | 下降 pp | 正常 gIoU | 共享 gIoU | 下降 pp | mask 改变比例 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        a, b, d = row["normal"], row["shared"], row["drop_pp"]
        lines.append(f"| {row['dataset']} | {row['scope']} | {row['samples']} | "
                     f"{a['cIoU']:.2f} | {b['cIoU']:.2f} | {d['cIoU']:+.2f} | "
                     f"{a['gIoU']:.2f} | {b['gIoU']:.2f} | {d['gIoU']:+.2f} | {row['changed_masks_percent']:.2f}% |")
    lines.extend(["", "不掉点仅支持所测读取差异对当前结果不必要，不证明64个特征相同或一个latent足够。", ""])
    return "\n".join(lines)
