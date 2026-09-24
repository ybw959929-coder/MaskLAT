"""Read-only Query-mask atlas capture for the native st3--st9 latent path.

The st3 maps describe reconstructed, head-averaged builder cross-attention.
The st4--st8 maps describe the executed latent-writeback distribution.  A
read-only stage (normally st9) instead displays the executed Query-read
distribution with its axes transposed; it is NOT a latent-read distribution.

Only hooks are installed, and all reconstruction/cropping happens after their
removal.  No model tensor, attention argument, or production forward is changed.
"""
from __future__ import annotations

import torch

from .latent_spatial_probe import LatentSpatialProbe
from .latent_spatial_probe_math import reconstruct_mha_weights
from masklat.model.segmentors.mask2former.spatial_validity import (
    masked_normalized_grid_sample,
    valid_mask_from_normalized_boxes,
    validate_normalized_valid_boxes,
)


@torch.no_grad()
def crop_query_probabilities(mask_logits, normalized_box):
    """Return CPU FP32 ``[Q,Hvalid,Wvalid]`` probabilities without SAM padding.

    Resample the exact continuous valid rectangle at approximately its native
    mask-grid resolution.  Pixel centers are used, never floor/ceil slicing
    that would include a padded border.  Mask-normalized interpolation excludes
    all padded source pixels, including at fractional valid-box boundaries.

    This is a visualization of native low-resolution probability maps, not the
    formal RefSeg logit-upsample/crop/threshold path used to compute metrics.
    Original-image orientation assumes verified resize-and-pad-only geometry.
    """
    if not isinstance(mask_logits, torch.Tensor) or mask_logits.ndim != 4:
        raise ValueError("mask_logits must be [1,Q,H,W]")
    if mask_logits.shape[0] != 1 or any(n < 1 for n in mask_logits.shape):
        raise ValueError("mask_logits must contain exactly one nonempty image")
    logits = mask_logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("mask_logits must be finite")
    boxes = validate_normalized_valid_boxes(
        normalized_box.detach().float().cpu(), batch_size=1)
    height, width = logits.shape[-2:]
    top, left, bottom, right = boxes[0].tolist()
    if (top, left, bottom, right) == (0.0, 0.0, 1.0, 1.0):
        return logits[0].sigmoid().contiguous()
    cropped_height = max(1, int(round((bottom - top) * height)))
    cropped_width = max(1, int(round((right - left) * width)))
    valid = valid_mask_from_normalized_boxes(boxes, (height, width))
    y = top + (torch.arange(cropped_height, dtype=torch.float32) + 0.5) * (
        (bottom - top) / cropped_height)
    x = left + (torch.arange(cropped_width, dtype=torch.float32) + 0.5) * (
        (right - left) / cropped_width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((2 * xx - 1, 2 * yy - 1), dim=-1).unsqueeze(0)
    with torch.autocast(device_type="cpu", enabled=False):
        probabilities, coverage = masked_normalized_grid_sample(
            logits.sigmoid(), valid, grid)
    if not bool(coverage.gt(1e-6).all()):
        raise ValueError("cropped atlas contains points without valid SAM coverage")
    return probabilities[0].clamp(0, 1).contiguous()


class LatentQueryAtlasProbe(LatentSpatialProbe):
    """Single-sample temporary capture; call ``finalize_atlas`` after exit.

    Each returned record contains ``stage``, CPU ``weights[L,Q]``, CPU
    ``mask_prob[Q,H,W]``, ``weight_kind``, and JSON-serializable ``metadata``.
    Slots and Query IDs remain in their native order; top-k is a renderer task.
    """

    def __init__(self, model):
        # Reuse audited native topology checks and read-only hook lifecycle.
        # The inherited support resolution is unused by this atlas finalizer.
        super().__init__(model, support_size=32)

    @torch.no_grad()
    def finalize_atlas(self, include_first_block=True):
        if self.handles:
            raise RuntimeError("exit probe context before finalizing the atlas")
        if len(self.mask_logits) != 10:
            raise ValueError(f"expected st0--st9 mask calls, got {len(self.mask_logits)}")
        if list(self.builder_outputs) != self.topology["builder_stages"]:
            raise ValueError("builder execution does not match supplied topology")
        if list(self.relations) != ["entry_st3"] + [f"st{i}" for i in range(4, 10)]:
            raise ValueError("transport execution does not match entry + st4--st9")
        masks, boxes = self.builder_outputs["st3"]
        if boxes.shape != (1, 4):
            raise ValueError("atlas requires exactly one source-image valid box")
        if not torch.equal(masks, self.mask_logits[3]):
            raise ValueError("st3 builder masks differ from captured decoder masks")
        boxes_cpu = validate_normalized_valid_boxes(boxes.detach().float().cpu(), batch_size=1)
        last_builder = self.builders[-1][1]
        last_block = len(last_builder.pre_vlm_blocks) - 1
        blocks = [0, last_block] if include_first_block and last_block > 0 else [last_block]
        records = []
        cropped_masks = {}

        def make_record(stage, formal_stage, weights, kind, metadata):
            if formal_stage not in cropped_masks:
                cropped_masks[formal_stage] = crop_query_probabilities(
                    self.mask_logits[formal_stage], boxes_cpu)
            mask_prob = cropped_masks[formal_stage]
            expected = (1, self.topology["latent_count"], mask_prob.shape[0])
            if tuple(weights.shape) != expected:
                raise ValueError(f"{stage} attention shape {tuple(weights.shape)} != {expected}")
            weights = weights.detach().float().cpu()[0].contiguous()
            if not torch.isfinite(weights).all() or bool(((weights < 0) | (weights > 1)).any()):
                raise ValueError(f"{stage} contains invalid attention probabilities")
            # Production BF16 softmax is deliberately retained, including its
            # rounding.  Do not renormalize it merely to make the atlas sum 1.
            normalized_axis = -1 if kind == "latent_reads_query" else 0
            sums = weights.sum(normalized_axis)
            if not torch.allclose(sums, torch.ones_like(sums), atol=0.02, rtol=0.0):
                raise ValueError(f"{stage} attention has an invalid normalization axis")
            metadata = {
                "stage": stage,
                "formal_decoder_stage": formal_stage,
                "latent_count": self.topology["latent_count"],
                "query_count": int(mask_prob.shape[0]),
                "sam_valid_boxes_normalized_tlbr": boxes_cpu.tolist(),
                "native_mask_grid_size": list(self.mask_logits[formal_stage].shape[-2:]),
                "cropped_mask_grid_size": list(mask_prob.shape[-2:]),
                "mask_display": "FP32 sigmoid then valid-box masked-normalized resampling; fixed probability scale",
                "geometry_requirement": "verified original -> axis-aligned positive resize -> SAM padding only",
                "query_index_semantics": "persistent decoder Query index, not a guaranteed object identity",
                "latent_index_semantics": "persistent slot order; feature trajectories are not captured",
                **metadata,
            }
            records.append({
                "stage": stage, "weights": weights, "mask_prob": mask_prob,
                "weight_kind": kind, "metadata": metadata,
            })

        for block_index in blocks:
            key = f"st3.block{block_index}"
            if key not in self.attention_inputs:
                raise ValueError(f"missing actual builder CA input: {key}")
            weights = reconstruct_mha_weights(*self.attention_inputs[key])
            stage = "st3" if block_index == last_block else f"st3_block{block_index}"
            make_record(stage, 3, weights, "latent_reads_query", {
                "attention_source": f"builder.{key}",
                "attention_normalization": "each latent sums to one across Queries; heads averaged after softmax",
                "attention_numerics": "FP32 reconstruction from actual normalized MHA inputs and trained Q/K projections; fused mixed-precision rounding may differ",
                "latent_query_writeback": True,
                "late_condition_refresh": False,
                "query_mask_timing": "st3 masks used by the same builder proposal memory",
                "notes": "Pre-LLM builder cross-attention only, not the complete latent feature or its residual/SA/FFN attribution.",
            })

        for formal_stage in range(4, 10):
            stage = f"st{formal_stage}"
            module = self.bridge.transport_stages[formal_stage - 4]
            relation = self.relations[stage]
            if not torch.isfinite(relation).all():
                raise ValueError(f"{stage} contains non-finite relation logits")
            writeback = bool(module.enable_latent_writeback)
            # Match production routing_weights: retain the relation's dtype
            # for softmax, and upcast ONLY after computing probabilities.
            with torch.autocast(device_type=relation.device.type, enabled=False):
                weights = relation.softmax(dim=1 if writeback else -1).transpose(1, 2)
            if writeback:
                kind = "latent_reads_query"
                normalization = "each latent sums to one across Queries: softmax(E, dim=Query).T"
                timing = "same-stage post-FFN predicted masks label persistent Queries; actual writeback values are post-Query-SA, pre-Query-FFN"
                notes = "Executed Query-to-latent writeback weights, matched retrospectively to same-stage candidate masks; these are not exact masks of the writeback value features."
            else:
                kind = "query_reads_latent"
                normalization = "each Query sums to one across latents: softmax(E, dim=latent).T; rows DO NOT sum to one"
                timing = "same-stage post-FFN predicted masks label persistent Queries; relation was evaluated at stage entry before Query-SA/FFN"
                notes = "NO Query-to-latent writeback here. Top Queries are those reading this latent most strongly, not Queries attended by this latent. Not directly comparable to preceding latent-read percentages. A configured Cond/VLM refresh can still change the latent on entry."
            make_record(stage, formal_stage, weights, kind, {
                "attention_source": f"transport.{stage}.query_read shared relation",
                "attention_normalization": normalization,
                "attention_numerics": f"softmax of captured actual relation logits in production dtype {relation.dtype}; then FP32 storage",
                "latent_query_writeback": writeback,
                "late_condition_refresh": formal_stage in self.topology["late_condition_refresh_stages"],
                "query_mask_timing": timing,
                "notes": notes,
            })

        self.mask_logits.clear()
        self.builder_outputs.clear()
        self.attention_inputs.clear()
        self.relations.clear()
        return records


__all__ = ["LatentQueryAtlasProbe", "crop_query_probabilities"]
