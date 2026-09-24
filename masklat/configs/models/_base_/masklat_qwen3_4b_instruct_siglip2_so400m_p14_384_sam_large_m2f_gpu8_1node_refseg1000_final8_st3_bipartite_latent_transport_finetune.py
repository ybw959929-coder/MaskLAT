

from mmengine.config import read_base

with read_base():
    from .masklat_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_2node8_refseg1000_final8_st3_bipartite_latent_transport_finetune import *


accumulative_counts = 2
optim_wrapper["accumulative_counts"] = accumulative_counts
train_dataloader["sampler"]["per_device_batch_size"] = (
    batch_size * accumulative_counts
)

assert batch_size == 4
assert optim_wrapper["accumulative_counts"] == 2
assert train_dataloader["sampler"]["per_device_batch_size"] == 8
assert custom_hooks[0]["eval_at_start"] is False
assert custom_hooks[0]["interval"] == 1000
assert len(custom_hooks[0]["final_datasets"]) == 8
assert default_hooks["checkpoint"]["interval"] == 500
assert default_hooks["checkpoint"]["max_keep_ckpts"] == 2
