from __future__ import annotations

from .group_supervised_latent_segmentor import GroupSupervisedLatentSegmentor
from .mask2former.grouped_st123_builder import (
    GroupedST123Builder,
    GroupedST123Proposal,
)


class GroupedST123Segmentor(GroupSupervisedLatentSegmentor):
    """Expose grouped supervision from each prefix stage."""

    proposal_builder_type = GroupedST123Builder
    incompatible_builder_flags = (
        "use_st123_latent_deepstack",
        "use_stagewise_latent_trifusion",
        "use_latent_s2_post_vlm_transport",
        "use_proposal_mediated_tripartite_transport",
    )

    def prepare_st3_latent_execution(self, **kwargs):
        bundle = super().prepare_st3_latent_execution(**kwargs)
        proposal = bundle.proposal
        if not isinstance(proposal, GroupedST123Proposal):
            raise RuntimeError("the segmentor requires a grouped ST1-ST3 proposal")
        records = proposal.stage_group_supervision
        if records is None or set(records) != {1, 2, 3}:
            raise RuntimeError("group supervision is required at stages 1, 2 and 3")
        if records[3] is not proposal.group_supervision:
            raise RuntimeError("the final proposal must expose stage-3 supervision")
        self._record_group_supervision(1, records[1])
        self._record_group_supervision(2, records[2])
        return bundle


__all__ = ["GroupedST123Segmentor"]
