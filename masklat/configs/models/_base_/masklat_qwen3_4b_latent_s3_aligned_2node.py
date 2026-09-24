





from mmengine.config import read_base

with read_base():
    from .masklat_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu8_1node_refseg1000_final8_st3_bipartite_latent_transport_no_sam_vlm_finetune import *


batch_size = 4
accumulative_counts = 1
lr = 4.0e-5
max_epochs = 2

train_dataloader["batch_size"] = batch_size
train_dataloader["sampler"]["per_device_batch_size"] = (
    batch_size * accumulative_counts
)
train_dataloader["sampler"]["sample_ratio"] = 1.0
optim_wrapper["accumulative_counts"] = accumulative_counts
optim_wrapper["optimizer"]["lr"] = lr
train_cfg["max_epochs"] = max_epochs
param_scheduler[0]["end"] = warmup_ratio * max_epochs
param_scheduler[1]["begin"] = warmup_ratio * max_epochs
param_scheduler[1]["end"] = max_epochs


default_hooks["logger"]["interval"] = 50
default_hooks["checkpoint"].update(
    dict(by_epoch=False, interval=500, max_keep_ckpts=2, save_last=True)
)
custom_hooks[0].update(dict(interval=500, eval_at_start=True))

assert batch_size * 16 * accumulative_counts == 64
assert train_dataloader["batch_size"] == 4
assert train_dataloader["sampler"]["per_device_batch_size"] == 4
assert optim_wrapper["accumulative_counts"] == 1
assert optim_wrapper["optimizer"]["lr"] == 4.0e-5
assert train_cfg["max_epochs"] == 2
assert param_scheduler[1]["end"] == 2
assert model["inject_sam_vit_tokens_to_vlm"] is False
assert model["use_st3_latent_for_imgconv"] is False
assert model["sampler_input_feat"] == "pixel_values"
assert custom_hooks[0]["eval_at_start"] is True
assert custom_hooks[0]["interval"] == 500
assert default_hooks["checkpoint"]["interval"] == 500
