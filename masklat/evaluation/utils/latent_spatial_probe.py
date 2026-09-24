"""Temporary, read-only hooks for the native bipartite latent forward.

No production module, attention argument, tensor, or parameter is replaced.
All numerical analysis runs after the production forward has completed.
Support maps are readout proxies, not attribution through residuals/SA/LLM.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from .latent_spatial_probe_math import (
    reconstruct_mha_weights, routing_statistics, soft_iou_matrix,
    spatial_support, support_statistics,
)
from masklat.model.segmentors.mask2former.spatial_validity import (
    valid_mask_from_normalized_boxes,
)


def parameter_stamp(model):
    """Lightweight mutation guard; not an expensive full-weights checksum."""
    return {
        name: (id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
    }


def sample_indices(length, maximum, seed, rank=0, world_size=1):
    """Fixed global random subset, then rank-stride sharding without padding."""
    if length <= 0 or maximum <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid dataset/sample/rank dimensions")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    selected = indices[:min(length, maximum)].tolist()
    return selected, selected[rank::world_size]


def prediction_snapshot(formal, raw):
    """Keep final native RefSeg tensors for an off/on forward comparison."""
    classes = getattr(raw, "raw_query_class_logits", None)
    masks = getattr(raw, "raw_query_mask_logits", None)
    if classes is None or masks is None:
        raise ValueError("native raw Query class/mask logits are required")
    if classes.ndim != 3 or classes.shape[0] != 1 or classes.shape[-1] != 2:
        raise ValueError("first-step probe supports one image, one SEG and one condition + BG")
    if masks.ndim != 4 or masks.shape[:2] != classes.shape[:2] or len(formal) != 1:
        raise ValueError("unexpected native RefSeg output dimensions")
    return {
        "classes": classes.detach().float().cpu(),
        "masks": masks.detach().float().cpu(),
        # Selection must match native RefSeg's original-dtype softmax.
        "query_index": int(classes.detach().softmax(-1)[0, :, 0].argmax()),
        "segmentation": formal[0]["segmentation"].detach().cpu(),
    }


def compare_predictions(baseline, observed, atol=1e-5, rtol=1e-5):
    result = {"atol": atol, "rtol": rtol}
    for key in ("classes", "masks"):
        x, y = baseline[key], observed[key]
        if x.shape != y.shape:
            raise ValueError(f"off/on {key} shape mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"non-finite off/on {key}")
        result[f"{key}_max_abs_delta"] = float((x - y).abs().max())
        result[f"{key}_allclose"] = bool(torch.allclose(x, y, atol=atol, rtol=rtol))
        result[f"{key}_bitwise_equal"] = bool(torch.equal(x, y))
    result["selected_query_equal"] = baseline["query_index"] == observed["query_index"]
    result["formal_segmentation_equal"] = bool(torch.equal(
        baseline["segmentation"], observed["segmentation"]))
    result["passed"] = all(result[k] for k in (
        "classes_allclose", "masks_allclose", "selected_query_equal", "formal_segmentation_equal"))
    return result


def aggregate_records(records):
    """Per-dataset/stage summaries; item-weighted stats use explicit sums/counts."""
    groups = defaultdict(list)
    for record in records:
        for stage, values in record["stages"].items():
            groups[(record["dataset"], stage)].append(values)
    datasets = {}
    for (dataset, stage), rows in sorted(groups.items()):
        summary = {"sample_count": len(rows)}
        for key in sorted(set().union(*(row.keys() for row in rows))):
            if key.endswith("_mean") and key[:-5] + "_count" in rows[0]:
                prefix = key[:-5]
                count = sum(row[prefix + "_count"] for row in rows)
                total = sum(row[prefix + "_sum"] for row in rows)
                summary[key] = total / count if count else None
                summary[prefix + "_count"] = count
            elif key.endswith(("_min", "_max")):
                values = [row[key] for row in rows if isinstance(row.get(key), (float, int))]
                summary[key] = (min(values) if key.endswith("_min") else max(values)) if values else None
            elif key.endswith(("_count", "_sum")):
                continue
            elif all(row.get(key) is None or (isinstance(row.get(key), (float, int))
                     and not isinstance(row.get(key), bool) and math.isfinite(row[key])) for row in rows):
                # Ratios and normalized scalars without item counts: equal sample weights.
                values = [row[key] for row in rows if row.get(key) is not None]
                summary[key + "_sample_mean"] = sum(values) / len(values) if values else None
                summary[key + "_valid_sample_count"] = len(values)
        datasets.setdefault(dataset, {})[stage] = summary
    checks = [r["verification"] for r in records if r.get("verification") is not None]
    return {
        "datasets": datasets, "sample_count": len(records),
        "verification": {
            "sample_count": len(checks), "all_passed": bool(checks) and all(c["passed"] for c in checks),
            "all_bitwise_equal": bool(checks) and all(
                c["classes_bitwise_equal"] and c["masks_bitwise_equal"] for c in checks),
            "max_class_abs_delta": max((c["classes_max_abs_delta"] for c in checks), default=None),
            "max_mask_abs_delta": max((c["masks_max_abs_delta"] for c in checks), default=None),
        },
    }


class LatentSpatialProbe:
    """One inference pass only. Supports native single-st3/cascade + transport.

    Late Cond refresh is allowed if already present and trained in the supplied
    checkpoint. Deepstack, direct-QC, topology and other bridges are rejected.
    First-step scope: batch=1, one SEG row, one condition plus background.
    """

    def __init__(self, model, support_size=32):
        if isinstance(support_size, bool) or not isinstance(support_size, int) or support_size < 1:
            raise ValueError("support_size must be a positive integer")
        if any(module.training for module in model.modules()):
            raise ValueError("probe requires model.eval() for every module")
        self.model, self.size = model, support_size
        segmentor = model.segmentor
        cfg = segmentor.dec_config
        if not getattr(cfg, "use_st3_bipartite_latent_transport", False):
            raise ValueError("only the native st3 bipartite latent path is supported")
        if getattr(cfg, "use_st123_latent_deepstack", False):
            raise ValueError("LLM-layer deepstack is not supported by this spatial probe")
        self.builder = segmentor.st3_transport_proposal_builder
        self.bridge = segmentor.st3_transport_bridge
        bridge_name = type(self.bridge).__name__ if self.bridge is not None else None
        supported_bridges = {
            "BipartiteLatentTransportBridge",
            "GroupSupervisedLatentTransportBridge",
        }
        if bridge_name not in supported_bridges:
            raise ValueError(
                "a native bipartite or group-supervised latent transport bridge is required"
            )
        if type(self.builder).__name__ == "St123LatentCascadeBuilder":
            self.builders = list(self.builder.stages.items())
            if [name for name, _ in self.builders] != ["st1", "st2", "st3"]:
                raise ValueError("cascade must contain st1, st2, st3 in that order")
        elif type(self.builder).__name__ in {
            "St3BipartiteLatentBuilder",
            "GroupSupervisedLatentBuilder",
        }:
            self.builders = [("st3", self.builder)]
        else:
            raise ValueError(f"unsupported builder: {type(self.builder).__name__}")
        self.predictor = segmentor.decoder.decoder.mask_predictor
        if len(self.bridge.transport_stages) != 6:
            raise ValueError("expected six transport stages st4 through st9")
        self.topology = {
            "builder_type": type(self.builder).__name__,
            "builder_stages": [name for name, _ in self.builders],
            "latent_count": int(self.builders[-1][1].num_latents),
            "late_condition_refresh_stages": list(self.bridge.late_condition_refresh_stages),
            "latent_writeback_stages": [i + 4 for i, stage in enumerate(self.bridge.transport_stages)
                                        if stage.enable_latent_writeback],
        }
        self.handles, self.mask_logits, self.builder_outputs = [], [], {}
        self.attention_inputs, self.relations = {}, {}
        self.used = False

    @staticmethod
    def _unique(store, key, value):
        if key in store:
            raise RuntimeError(f"probe expected one call to {key}; repeated calls are unsupported")
        store[key] = value

    def _builder_hook(self, name):
        def hook(module, args, kwargs, output):
            mask = kwargs.get("mask_logits")
            if mask is None or mask.ndim != 4 or mask.shape[0] != 1:
                raise ValueError("builder probe requires one source image and keyword mask_logits")
            self._unique(self.builder_outputs, name, (
                mask.detach(), output.sam_valid_boxes_normalized.detach()))
        return hook

    def _mha_hook(self, name):
        def hook(module, args, kwargs):
            def arg(key, position, default=None):
                return kwargs.get(key, args[position] if len(args) > position else default)
            if arg("attn_mask", 5) is not None or arg("is_causal", 7, False):
                raise ValueError("masked/causal builder CA is unsupported")
            if arg("need_weights", 4, True):
                raise ValueError("native builder must keep need_weights=False")
            q, k, padding = arg("query", 0), arg("key", 1), arg("key_padding_mask", 3)
            self._unique(self.attention_inputs, name, (
                module, q.detach(), k.detach(), None if padding is None else padding.detach()))
        return hook

    def _relation_hook(self, name):
        def hook(module, args, output):
            relation = output[1]
            if relation.ndim != 3 or relation.shape[0] != 1:
                raise ValueError("first-step probe requires exactly one SEG row per sample")
            self._unique(self.relations, name, relation.detach())
        return hook

    def __enter__(self):
        if self.used:
            raise RuntimeError("use a new LatentSpatialProbe for each sample")
        self.used = True
        try:
            self.handles.append(self.predictor.register_forward_hook(
                lambda module, args, output: self.mask_logits.append(output[0].detach())))
            for name, builder in self.builders:
                self.handles.append(builder.register_forward_hook(self._builder_hook(name), with_kwargs=True))
                if not builder.pre_vlm_blocks:
                    raise ValueError("builder has no pre-VLM cross-attention blocks")
                for i, block in enumerate(builder.pre_vlm_blocks):
                    self.handles.append(block.cross_attention.register_forward_pre_hook(
                        self._mha_hook(f"{name}.block{i}"), with_kwargs=True))
            self.handles.append(self.bridge.entry_query_read.register_forward_hook(self._relation_hook("entry_st3")))
            for i, stage in enumerate(self.bridge.transport_stages):
                self.handles.append(stage.query_read.register_forward_hook(self._relation_hook(f"st{i + 4}")))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    @torch.no_grad()
    def finalize(self, keep_example=False):
        if self.handles:
            raise RuntimeError("exit probe context before doing shadow computations")
        if len(self.mask_logits) != 10:
            raise ValueError(f"expected st0--st9 mask calls, got {len(self.mask_logits)}")
        if list(self.builder_outputs) != self.topology["builder_stages"]:
            raise ValueError("builder execution does not match supplied topology")
        if list(self.relations) != ["entry_st3"] + [f"st{i}" for i in range(4, 10)]:
            raise ValueError("transport execution does not match entry + st4--st9")
        boxes = self.builder_outputs["st3"][1]
        if boxes.shape != (1, 4):
            raise ValueError("first-step spatial metadata must describe exactly one source image")
        stages, examples = {}, {}
        maps, valid = {}, None
        # Resize each stage's predicted probabilities with fractional valid-area
        # coverage. Never include SAM padding as background in area statistics.
        for t, logits in enumerate(self.mask_logits):
            if logits.ndim != 4 or logits.shape[0] != 1:
                raise ValueError("one image/SEG row required at every mask stage")
            validity = valid_mask_from_normalized_boxes(boxes.to(logits.device), tuple(logits.shape[-2:]))
            uniform = logits.new_full((1, 1, logits.shape[1]), 1 / logits.shape[1], dtype=torch.float32)
            _, maps[t], stage_valid = spatial_support(uniform, logits.float().sigmoid(), validity, self.size)
            if valid is not None and not torch.equal(valid, stage_valid):
                raise ValueError("mask stages must use identical spatial validity")
            valid = stage_valid
        initial = None
        for name, builder in self.builders:
            t = int(name[2:])
            logits, stage_boxes = self.builder_outputs[name]
            if not torch.equal(boxes, stage_boxes) or not torch.equal(logits, self.mask_logits[t]):
                raise ValueError(f"{name} builder mask/geometry differs from the captured decoder stage")
            for i, _ in enumerate(builder.pre_vlm_blocks):
                key = f"{name}.block{i}"
                if key not in self.attention_inputs:
                    raise ValueError(f"missing actual builder CA input: {key}")
                weights = reconstruct_mha_weights(*self.attention_inputs[key])
                if weights.shape[:2] != (1, self.topology["latent_count"]) or weights.shape[2] != maps[t].shape[1]:
                    raise ValueError("builder attention does not index the captured proposal masks")
                with torch.autocast(device_type=weights.device.type, enabled=False):
                    support = torch.bmm(weights.float(), maps[t].float().flatten(2)).reshape(
                        1, weights.shape[1], self.size, self.size).clamp(0, 1)
                entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
                stats = support_statistics(support, valid)
                stats["builder_read_entropy_normalized"] = float(entropy.mean() / max(math.log(weights.shape[-1]), 1e-30))
                stages[f"builder.{key}"] = stats
                if keep_example:
                    examples[f"builder.{key}"] = {"support": support.cpu(), "weights_fp32_reconstructed": weights.cpu()}
                if name == "st3":
                    initial = support  # Last st3 CA readout, NOT total feature provenance.
        support = initial
        for name, relation in self.relations.items():
            t = 3 if name == "entry_st3" else int(name[2:]) - 1
            if relation.shape != (1, maps[t].shape[1], support.shape[1]):
                raise ValueError("Query/latent relationship dimensions disagree with support maps")
            bias = soft_iou_matrix(maps[t], support, valid)
            stages[f"transport.{name}"] = {
                **support_statistics(support, valid), **routing_statistics(relation, bias)}
            if keep_example:
                examples[f"transport.{name}"] = {
                    "support_before_read": support.cpu(), "query_masks_before_read": maps[t].cpu(),
                    "actual_relation_logits": relation.float().cpu(), "offline_soft_iou": bias.cpu()}
            if name != "entry_st3":
                current_stage = t + 1
                module = self.bridge.transport_stages[current_stage - 4]
                if module.enable_latent_writeback:
                    # Shadow bookkeeping: actual shared E write weights with the
                    # just-produced stage masks. This is retrospective: real
                    # writeback values are Query-SA outputs BEFORE Query FFN,
                    # whereas these masks are predicted AFTER Query FFN.
                    # These maps NEVER enter the model.
                    with torch.autocast(device_type=relation.device.type, enabled=False):
                        weights = relation.float().softmax(dim=1).transpose(1, 2)
                        support = torch.bmm(weights, maps[current_stage].float().flatten(2)).reshape_as(support).clamp(0, 1)
        if keep_example:
            examples["valid_area_coverage"] = valid.cpu()
            examples["sam_valid_boxes_normalized_tlbr"] = boxes.cpu()
        self.mask_logits.clear()
        self.builder_outputs.clear()
        self.attention_inputs.clear()
        self.relations.clear()
        return stages, examples
