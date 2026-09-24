"""MaskLAT training configuration with a Phi-3 language backbone."""

from copy import deepcopy
from os import getenv

from mmengine.config import read_base

from masklat.model.grouped_st123 import GroupedST123Model


with read_base():
    from ._base_.masklat_phi3_group_supervised_latent_s3_full_refcoco2000_4node import *


model = deepcopy(model)
checkpoint = getenv("MASKLAT_PHI3_S2_CHECKPOINT")
if not checkpoint:
    raise ValueError("MASKLAT_PHI3_S2_CHECKPOINT is required")

model.update(
    type=GroupedST123Model,
    s2_pretrained_pth=checkpoint,
    group_loss_weight=0.1,
    group_loss_warmup_steps=500,
)
model["segmentor"]["decoder"]["config"].update(
    use_st123_latent_cascade=True,
    use_st123_latent_deepstack=False,
    use_st3_bipartite_latent_transport=True,
    use_latent_s2_post_vlm_transport=False,
    st3_transport_enable_latent_writeback=True,
    st3_transport_late_condition_refresh_stages=(),
)
