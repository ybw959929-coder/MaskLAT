"""Opt-in ST3 builder supervision without changing the legacy builder.

Only the last pre-VLM block's REAL multi-head cross-attention is supervised.
Its head-averaged probabilities are obtained from the same forward operation
that produces the latent update, not from the unused affinity projection.
Hooks are scoped to this one builder call and always removed in ``finally``;
there is no persistent attention/graph cache and inference installs no hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .latent_mask_group_loss import MaskGroupAuxiliaryLoss
from .spatial_validity import valid_mask_from_normalized_boxes
from .st3_bipartite_latent_transport import St3BipartiteLatentBuilder
from .st3_proposal_latent_bridge import St3LatentProposal


@dataclass
class GroupSupervisedLatentProposal(St3LatentProposal):
    group_supervision: Optional[Dict[str, Any]] = None


class GroupSupervisedLatentBuilder(St3BipartiteLatentBuilder):
    def __init__(self, *, group_loss_config=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_group_auxiliary = MaskGroupAuxiliaryLoss(
            **(group_loss_config or {})
        )
        # Included under the already authorized S2 builder checkpoint prefix.
        # S3 can reject an accidental legacy S2 handoff with identical shapes.
        self.register_buffer("group_supervision_version", torch.tensor(1, dtype=torch.long))
        self.group_supervision_enabled = True

    def forward(self, **kwargs):
        supervise = (
            self.training and torch.is_grad_enabled()
            and self.group_supervision_enabled
        )
        if not supervise:
            proposal = super().forward(**kwargs)
            return GroupSupervisedLatentProposal(**vars(proposal))

        captured = []
        attention = self.pre_vlm_blocks[-1].cross_attention

        def request_weights(module, args, call_kwargs):
            call_kwargs = dict(call_kwargs)
            call_kwargs["need_weights"] = True
            call_kwargs["average_attn_weights"] = True
            return args, call_kwargs

        def capture_weights(module, args, output):
            if output[1] is None:
                raise RuntimeError("ST3 group supervision requires real attention weights")
            captured.append(output[1])

        before = attention.register_forward_pre_hook(request_weights, with_kwargs=True)
        after = attention.register_forward_hook(capture_weights)
        try:
            proposal = super().forward(**kwargs)
        finally:
            before.remove()
            after.remove()
        if len(captured) != 1:
            raise RuntimeError("the final ST3 builder attention must execute exactly once")
        masks = kwargs["mask_logits"]
        valid_pixels = valid_mask_from_normalized_boxes(
            proposal.sam_valid_boxes_normalized, masks.shape[-2:]
        )
        result = self.mask_group_auxiliary(
            captured[0], masks,
            spatial_valid_mask=valid_pixels,
            query_valid_mask=proposal.query_geometry_valid_mask,
        )
        return GroupSupervisedLatentProposal(
            **vars(proposal), group_supervision=result
        )


__all__ = ["GroupSupervisedLatentBuilder", "GroupSupervisedLatentProposal"]
