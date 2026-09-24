"""Recurrent st1--st3 proposal latents with only final Z3 sent to the VLM.

This path deliberately has no Phi deepstack outputs or layer-injection hooks.
The ordinary st3 proposal and post-VLM transport interfaces are unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from torch import Tensor, nn

from .st3_bipartite_latent_transport import St3BipartiteLatentBuilder
from .st3_proposal_latent_bridge import St3LatentProposal


class St123LatentCascadeBuilder(nn.Module):
    """Carry Z1 -> Z2 -> Z3 through independent builders, project only Z3."""

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
            del builder.latent_query
            del builder.proposal_key
            if stage_index != 3:
                # The shared builder always produces a packed token field.
                # It is only an internal placeholder at st1/st2: registering
                # real projections here would leave unused trainable weights.
                builder.vlm_projector = nn.Identity()
            self.stages[f"st{stage_index}"] = builder

    @property
    def llm_hidden_dim(self) -> int:
        # st1/st2 have no LLM projection; only the real st3 head defines width.
        return self.stages["st3"].llm_hidden_dim

    @property
    def siglip_projector(self) -> nn.Linear:
        return self.stages["st1"].siglip_projector

    def freeze_unused_affinity_heads(self) -> None:
        """Compatibility with builder callers; these heads are not registered."""

    def forward(
        self,
        *,
        query_states: Sequence[Tensor],
        mask_logits: Sequence[Tensor],
        sam_mask_features: Tensor,
        siglip_spatial_features: Tensor,
        spatial_metadata: Dict[str, Tensor],
        siglip_valid_mask: Optional[Tensor] = None,
    ) -> St3LatentProposal:
        if isinstance(query_states, Tensor) or isinstance(mask_logits, Tensor):
            raise TypeError("st123 cascade query_states/mask_logits must be stage sequences")
        if len(query_states) != 3 or len(mask_logits) != 3:
            raise ValueError("st123 cascade requires exactly formal st1, st2 and st3")
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
            # Do not detach: final Z3 supervision must reach all three stages.
            latent_seed = proposal.latent_features
        # This is the ordinary final-stage proposal, not a deepstack proposal.
        # Its packed_group_tokens are E3 only (64 tokens, not three banks).
        return proposal


__all__ = ["St123LatentCascadeBuilder"]
