






from copy import deepcopy
from os import getenv

from mmengine.config import read_base
from mmengine.hooks import LoggerHook

from masklat.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from masklat.model.group_supervised_latent import GroupSupervisedLatentMaskLATModel

with read_base():
    from .masklat_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_2node8_refseg1000_final8_st3_bipartite_latent_transport_no_sam_vlm_finetune import *


model = deepcopy(model)
train_dataloader = deepcopy(train_dataloader)
optim_wrapper = deepcopy(optim_wrapper)
train_cfg = deepcopy(train_cfg)
param_scheduler = deepcopy(param_scheduler)
default_hooks = deepcopy(default_hooks)

batch_size = 4
accumulative_counts = 1
world_size = 32
global_batch = batch_size * accumulative_counts * world_size
lr = 4.0e-5
sample_ratio = 1.0
max_epochs = 2

group_supervised_s2_pretrained_pth = getenv("GROUP_SUPERVISED_S2_PRETRAINED_PTH") or (
    work_dir
    + "s2_latent_align_pretrain/"
    + "phi3_group_supervised_readout_s2_4node8gpu_mb4_ga2_gb256_lr1e-3_1epoch/"
    + "pytorch_model.bin"
)
model.update(
    dict(
        type=GroupSupervisedLatentMaskLATModel,
        s2_pretrained_pth=group_supervised_s2_pretrained_pth,
        s2g_pretrained_pth=None,
        latent_s3_pretrained_pth=None,
        st3_latent_alignment_pretrain=False,
        late_condition_refresh_finetune=False,
        latent_s2_post_vlm_decoder_finetune=False,
        group_loss_weight=0.1,
        group_loss_warmup_steps=500,
        group_mask_iou_threshold=0.7,
        group_mask_size=64,
        group_mask_min_area=4,
        group_mask_threshold=0.5,
    )
)
model["segmentor"]["decoder"]["config"].update(
    dict(
        use_st3_bipartite_latent_transport=True,
        use_st123_latent_deepstack=False,
        use_latent_s2_post_vlm_transport=False,
        st3_transport_enable_latent_writeback=True,
        st3_transport_late_condition_refresh_stages=(),
    )
)
train_dataloader["batch_size"] = batch_size
train_dataloader["sampler"]["per_device_batch_size"] = batch_size * accumulative_counts
train_dataloader["sampler"]["sample_ratio"] = sample_ratio
optim_wrapper["accumulative_counts"] = accumulative_counts
optim_wrapper["optimizer"]["lr"] = lr
train_cfg["max_epochs"] = max_epochs
param_scheduler[0]["end"] = warmup_ratio * max_epochs
param_scheduler[1]["begin"] = warmup_ratio * max_epochs
param_scheduler[1]["end"] = max_epochs
default_hooks["logger"].update(dict(type=LoggerHook, interval=50))
default_hooks["checkpoint"].update(
    dict(by_epoch=False, interval=500, max_keep_ckpts=2, save_last=True)
)




custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "visual_projector", "segmentor"],
        display_params=False,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=PTCheckpointHook, clean_pth=False),
]
visualizer = None

assert global_batch == 128
assert model["s1_pretrained_pth"] == s1_pretrained_pth
assert model["s2_pretrained_pth"] == group_supervised_s2_pretrained_pth
assert model["latent_s3_pretrained_pth"] is None
assert model["freeze_llm"] is False
assert model["freeze_visual_encoder"] is False
assert model["use_st3_latent_for_imgconv"] is False
assert model["inject_sam_vit_tokens_to_vlm"] is False
assert optim_wrapper["dtype"] == "float16"
assert train_dataloader["sampler"]["sample_ratio"] == 1.0
assert train_cfg["max_epochs"] == 2
assert optim_wrapper["paramwise_cfg"]["custom_keys"]["segmentor.encoder"]["lr_mult"] == 0.1
assert optim_wrapper["paramwise_cfg"]["custom_keys"]["visual_encoder"]["lr_mult"] == 0.1
