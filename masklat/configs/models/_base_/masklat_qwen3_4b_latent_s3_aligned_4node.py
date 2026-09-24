






from mmengine.config import read_base

with read_base():
    from .masklat_qwen3_4b_latent_s3_aligned_2node import *


_reference_global_batch = 64
_global_batch = batch_size * 32 * accumulative_counts
lr = 4.0e-5
optim_wrapper["optimizer"]["lr"] = lr

assert batch_size == 4
assert accumulative_counts == 1
assert _global_batch == 128
assert lr == 4.0e-5
assert optim_wrapper["optimizer"]["lr"] == lr
assert train_cfg["max_epochs"] == 2
assert train_dataloader["sampler"]["per_device_batch_size"] == 4
