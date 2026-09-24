from __future__ import annotations

from collections import OrderedDict

import torch

from .group_supervised_latent import GroupSupervisedLatentMaskLATModel
from .segmentors.grouped_st123_segmentor import GroupedST123Segmentor
from .segmentors.mask2former.group_supervised_latent_transport import (
    FinalConditionLatentRead,
)
from .segmentors.mask2former.grouped_st123_builder import GroupedST123Builder


class GroupedST123Model(GroupSupervisedLatentMaskLATModel):
    """MaskLAT with recurrent grouped builders at decoder stages 1, 2 and 3."""

    segmentor_type = GroupedST123Segmentor
    group_supervision_s3_stages = tuple(range(1, 10))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        builder = self.segmentor.st3_transport_proposal_builder
        bridge = self.segmentor.st3_transport_bridge
        if not isinstance(builder, GroupedST123Builder):
            raise RuntimeError("the grouped ST1-ST3 builder is not installed")
        if tuple(builder.stages) != ("st1", "st2", "st3"):
            raise RuntimeError("the prefix path must contain stages 1, 2 and 3")
        if len({id(stage) for stage in builder.stages.values()}) != 3:
            raise RuntimeError("each prefix stage must own independent parameters")
        if builder.stages["st1"].learned_latents is None:
            raise RuntimeError("stage 1 must own the learned latent seed")
        if any(
            builder.stages[name].learned_latents is not None
            for name in ("st2", "st3")
        ):
            raise RuntimeError("stages 2 and 3 must consume the preceding state")
        if not isinstance(bridge.final_condition_reader, FinalConditionLatentRead):
            raise RuntimeError("the final condition-to-latent readout is required")

    def _adapt_s2_latent_token_count(self, checkpoint_state):
        source = super()._adapt_s2_latent_token_count(checkpoint_state)
        source_prefix = "segmentor.st3_transport_proposal_builder."
        target_state = self.segmentor.st3_transport_proposal_builder.state_dict()
        adapted = OrderedDict(
            (key, value)
            for key, value in source.items()
            if not key.startswith(source_prefix)
        )
        missing = []
        mismatched = []

        for relative_key, target in target_state.items():
            pieces = relative_key.split(".", 2)
            if (
                len(pieces) != 3
                or pieces[0] != "stages"
                or pieces[1] not in {"st1", "st2", "st3"}
            ):
                raise RuntimeError(
                    f"unexpected grouped builder state key: {relative_key}"
                )
            source_key = source_prefix + pieces[2]
            target_key = source_prefix + relative_key
            if source_key not in source:
                missing.append(source_key)
                continue
            value = source[source_key]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(target.shape)
            ):
                mismatched.append(
                    (
                        source_key,
                        tuple(getattr(value, "shape", ())),
                        tuple(target.shape),
                    )
                )
                continue
            adapted[target_key] = value

        if missing or mismatched:
            raise RuntimeError(
                "initialization requires one complete grouped builder; "
                f"missing={missing[:12]}, mismatched={mismatched[:12]}"
            )
        return adapted


__all__ = ["GroupedST123Model"]
