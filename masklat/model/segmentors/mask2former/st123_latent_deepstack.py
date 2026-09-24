"""Opt-in recurrent st1/st2/st3 proposal latents for Phi input deepstack.

All three stages reuse the baseline's mask pooling, box coordinates, feature
sources and two gate-free latent blocks.  Only the first stage owns learned
latent seeds.  Every stage owns its own proposal encoder and LLM projection.
The final proposal remains compatible with the unchanged st3 post-VLM bridge;
the first-stage projection occupies the ordinary latent token positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from torch import Tensor, nn

from .st3_bipartite_latent_transport import St3BipartiteLatentBuilder
from .st3_proposal_latent_bridge import St3LatentProposal


@dataclass
class St123LatentDeepstackProposal(St3LatentProposal):
    stage_latent_features: Tuple[Tensor, Tensor, Tensor]
    stage_vlm_tokens: Tuple[Tensor, Tensor, Tensor]


class St123LatentDeepstackBuilder(nn.Module):
    """Carry one latent bank through three independent proposal builders."""

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
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_latents = int(num_latents)
        self.stages = nn.ModuleDict()
        for stage_index in (1, 2, 3):
            builder = St3BipartiteLatentBuilder(
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
            )
            if stage_index != 1:
                builder.register_parameter("learned_latents", None)
            # Unlike the legacy builder, do not retain unused affinity heads.
            # This also keeps S2 DDP safe after requires_grad_(True).
            del builder.latent_query
            del builder.proposal_key
            self.stages[f"st{stage_index}"] = builder

    @property
    def llm_hidden_dim(self) -> int:
        return self.stages["st1"].llm_hidden_dim

    @property
    def siglip_projector(self) -> nn.Linear:
        # Property access does not register the same child under two prefixes.
        return self.stages["st1"].siglip_projector

    def freeze_unused_affinity_heads(self) -> None:
        """Match the S2 builder contract; this variant has no affinity heads."""

        for stage in self.stages.values():
            for name in ("latent_query", "proposal_key"):
                head = getattr(stage, name, None)
                if head is not None:
                    head.requires_grad_(False)

    def forward(
        self,
        *,
        query_states: Sequence[Tensor],
        mask_logits: Sequence[Tensor],
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> St123LatentDeepstackProposal:
        if isinstance(query_states, Tensor) or isinstance(mask_logits, Tensor):
            raise TypeError("st123 query_states/mask_logits must be stage sequences")
        if len(query_states) != 3 or len(mask_logits) != 3:
            raise ValueError("st123 requires exactly formal st1, st2 and st3")
        outputs = []
        latent_seed = None
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
            outputs.append(proposal)
            # Preserve the full cross-stage graph, including under frozen S2
            # proposal sources; only the image-only decoder is under no_grad.
            latent_seed = proposal.latent_features
        final = outputs[-1]
        stage_latents = tuple(item.latent_features for item in outputs)
        stage_tokens = tuple(item.packed_group_tokens for item in outputs)
        return St123LatentDeepstackProposal(
            proposal_features=final.proposal_features,
            latent_features=final.latent_features,
            affinity_logits=final.affinity_logits,
            query_boxes_cxcywh=final.query_boxes_cxcywh,
            query_geometry_valid_mask=final.query_geometry_valid_mask,
            query_mask_on_siglip=final.query_mask_on_siglip,
            siglip_valid_mask=final.siglip_valid_mask,
            sam_valid_boxes_normalized=final.sam_valid_boxes_normalized,
            packed_group_tokens=stage_tokens[0],
            packed_group_valid_mask=final.packed_group_valid_mask,
            stage_latent_features=stage_latents,
            stage_vlm_tokens=stage_tokens,
        )


__all__ = ["St123LatentDeepstackBuilder", "St123LatentDeepstackProposal"]
