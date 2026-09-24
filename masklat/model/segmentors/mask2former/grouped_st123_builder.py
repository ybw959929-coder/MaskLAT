from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from torch import Tensor, nn

from .group_supervised_latent_builder import (
    GroupSupervisedLatentBuilder,
    GroupSupervisedLatentProposal,
)
from .st123_latent_cascade import St123LatentCascadeBuilder


@dataclass
class GroupedST123Proposal(GroupSupervisedLatentProposal):
    stage_group_supervision: Optional[
        Dict[int, Optional[Dict[str, Any]]]
    ] = None


class GroupedST123Builder(St123LatentCascadeBuilder):
    """Build a recurrent latent state from the first three decoder stages."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        sam_feature_dim: int,
        siglip_feature_dim: int,
        llm_hidden_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        mask_threshold: float = 0.5,
        fourier_features: int = 64,
        use_sam_region: bool = True,
        use_siglip_region: bool = True,
        use_xywh: bool = True,
        pre_vlm_depth: int = 2,
        group_loss_config=None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_dim = int(hidden_dim)
        self.num_latents = int(num_latents)
        self.stages = nn.ModuleDict()

        for stage_index in (1, 2, 3):
            builder = GroupSupervisedLatentBuilder(
                hidden_dim=hidden_dim,
                sam_feature_dim=sam_feature_dim,
                siglip_feature_dim=siglip_feature_dim,
                llm_hidden_dim=llm_hidden_dim,
                num_latents=num_latents,
                num_heads=num_heads,
                mask_threshold=mask_threshold,
                fourier_features=fourier_features,
                use_sam_region=use_sam_region,
                use_siglip_region=use_siglip_region,
                use_xywh=use_xywh,
                pre_vlm_depth=pre_vlm_depth,
                group_loss_config=group_loss_config,
            )
            if stage_index != 1:
                builder.register_parameter("learned_latents", None)
            del builder.latent_query
            del builder.proposal_key
            if stage_index != 3:
                builder.vlm_projector = nn.Identity()
            self.stages[f"st{stage_index}"] = builder

    def forward(
        self,
        *,
        query_states: Sequence[Tensor],
        mask_logits: Sequence[Tensor],
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> GroupedST123Proposal:
        if isinstance(query_states, Tensor) or isinstance(mask_logits, Tensor):
            raise TypeError("query_states and mask_logits must be stage sequences")
        if len(query_states) != 3 or len(mask_logits) != 3:
            raise ValueError("the prefix builder requires stages 1, 2 and 3")

        latent_seed = None
        stage_records = {}
        proposal = None
        for stage_index, (queries, masks) in enumerate(
            zip(query_states, mask_logits), start=1
        ):
            proposal = self.stages[f"st{stage_index}"](
                query_states=queries,
                mask_logits=masks,
                sam_mask_features=sam_mask_features,
                siglip_spatial_features=siglip_spatial_features,
                spatial_metadata=spatial_metadata,
                siglip_valid_mask=siglip_valid_mask,
                latent_seed=latent_seed,
                compute_affinity=False,
            )
            stage_records[stage_index] = proposal.group_supervision
            latent_seed = proposal.latent_features

        if proposal is None:
            raise RuntimeError("the prefix builder produced no proposal")
        return GroupedST123Proposal(
            **vars(proposal), stage_group_supervision=stage_records
        )


__all__ = ["GroupedST123Builder", "GroupedST123Proposal"]
