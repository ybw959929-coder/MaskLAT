"""Mask-group supervision for latent readout."""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from mmengine.logging import MessageHub

from .masklat import MaskLATModel
from .segmentors.group_supervised_latent_segmentor import GroupSupervisedLatentSegmentor


class GroupSupervisedLatentMaskLATModel(MaskLATModel):
    segmentor_type = GroupSupervisedLatentSegmentor
    group_supervision_s3_stages = tuple(range(3, 10))

    def __init__(
        self, *, group_loss_weight=0.1, group_loss_warmup_steps=500,
        group_mask_iou_threshold=0.7, group_mask_size=64,
        group_mask_min_area=4, group_mask_threshold=0.5,
        group_loss_backprop_enabled=True, **kwargs,
    ):
        weight = float(group_loss_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("group_loss_weight must be finite and non-negative")
        if isinstance(group_loss_warmup_steps, bool) or int(group_loss_warmup_steps) != group_loss_warmup_steps or group_loss_warmup_steps < 0:
            raise ValueError("group_loss_warmup_steps must be a non-negative integer")
        if not isinstance(group_loss_backprop_enabled, bool):
            raise TypeError("group_loss_backprop_enabled must be boolean")
        if kwargs.get("latent_s3_pretrained_pth") is not None:
            raise ValueError("latent_s3_pretrained_pth is not supported")
        if kwargs.get("late_condition_refresh_finetune", False) or kwargs.get("latent_s2_post_vlm_decoder_finetune", False):
            raise ValueError("group supervision requires full joint optimization")
        segmentor_config = kwargs.get("segmentor")
        if not isinstance(segmentor_config, dict):
            raise TypeError("group supervision requires a segmentor config dictionary")
        segmentor_config = deepcopy(segmentor_config)
        segmentor_config["type"] = self.segmentor_type
        segmentor_config["group_loss_config"] = dict(
            mask_threshold=float(group_mask_threshold),
            group_iou_threshold=float(group_mask_iou_threshold),
            spatial_size=int(group_mask_size),
            min_mask_area=int(group_mask_min_area),
        )
        kwargs["segmentor"] = segmentor_config
        super().__init__(**kwargs)
        if not isinstance(self.segmentor, self.segmentor_type):
            raise RuntimeError("the grouped segmentor was not constructed")
        self.group_loss_weight = weight
        self.group_loss_warmup_steps = int(group_loss_warmup_steps)
        self.group_loss_backprop_enabled = group_loss_backprop_enabled
        self._group_fallback_iteration = 0

    def _adapt_s2_latent_token_count(self, checkpoint_state):
        result = super()._adapt_s2_latent_token_count(checkpoint_state)
        marker = result.get(
            "segmentor.st3_transport_proposal_builder.group_supervision_version"
        )
        if marker is None or int(torch.as_tensor(marker).item()) != 1:
            raise ValueError(
                "the checkpoint does not contain grouped-builder parameters"
            )
        return result

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        if self.segmentor._group_supervision_records is not None:
            raise RuntimeError("nested group-supervised model forwards are unsupported")
        if mode == "loss":
            self.segmentor._group_supervision_records = {}
        try:
            return super().forward(data_dict, data_samples, mode=mode, **kwargs)
        finally:
            self.segmentor._group_supervision_records = None

    def _current_group_weight(self):
        try:
            iteration = MessageHub.get_current_instance().get_info("iter")
        except KeyError:
            iteration = None
        if iteration is None:
            iteration = self._group_fallback_iteration
        iteration = int(iteration)
        factor = 1.0 if self.group_loss_warmup_steps == 0 else min(
            max(iteration + 1, 0) / self.group_loss_warmup_steps, 1.0
        )
        return self.group_loss_weight * factor

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        result = super().compute_loss(data_dict, data_samples, **kwargs)
        records = self.segmentor._group_supervision_records
        if records is None:
            raise RuntimeError("call forward(mode='loss') to collect the ST3 group supervision")
        if self.st3_latent_alignment_pretrain and set(records) != {3}:
            raise RuntimeError("S2 must include ST3 group supervision and no post-VLM stages")
        expected_s3_stages = set(self.group_supervision_s3_stages)
        if not self.st3_latent_alignment_pretrain and records and set(records) != expected_s3_stages:
            expected = ",".join(f"ST{stage}" for stage in self.group_supervision_s3_stages)
            raise RuntimeError(f"segmentation S3 must supervise exactly {expected}")
        zero = result["loss"].new_zeros(())
        active = [
            record["loss"] for record in records.values()
            if bool(record["valid_sample_mask"].any())
        ]
        raw_auxiliary = (
            torch.stack(active).mean() if active else
            sum((record["loss"] for record in records.values()), zero)
        )
        weight = self._current_group_weight()
        if self.group_loss_backprop_enabled:
            weighted = raw_auxiliary * weight
            result["loss"] = result["loss"] + weighted
        else:
            weighted = zero
        result["aux_group_raw"] = raw_auxiliary.detach()
        result["aux_group_weight"] = zero + weight
        result["aux_group_weighted"] = weighted.detach()
        result["aux_group_backprop_enabled"] = zero + int(
            self.group_loss_backprop_enabled
        )
        result["aux_group_active_stages"] = zero + len(active)
        for stage in self.group_supervision_s3_stages:
            record = records.get(stage)
            prefix = f"grp_st{stage}"
            result[prefix + "_groups"] = (
                record["num_groups"].float().mean() if record is not None else zero
            ).detach()
            result[prefix + "_selected"] = (
                record["num_selected_groups"].float().mean() if record is not None else zero
            ).detach()
            result[prefix + "_kl"] = (
                record["loss"] if record is not None else zero
            ).detach()
        self._group_fallback_iteration += 1
        return result

__all__ = ["GroupSupervisedLatentMaskLATModel"]
