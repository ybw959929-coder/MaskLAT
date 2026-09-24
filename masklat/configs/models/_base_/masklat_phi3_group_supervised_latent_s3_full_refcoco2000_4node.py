





from copy import deepcopy
from os import getenv

from mmengine.config import read_base
from masklat.engine.hooks import PeriodicRefSegEvalHook

with read_base():
    from .masklat_phi3_group_supervised_latent_s3_4node import *


model = deepcopy(model)
default_hooks = deepcopy(default_hooks)
custom_hooks = deepcopy(custom_hooks)



group_supervised_s2_pretrained_pth = getenv("GROUP_SUPERVISED_S2_PRETRAINED_PTH") or (
    work_dir + "s2_latent_align_pretrain/"
    + "phi3_group_supervised_readout_s2_1node8gpu_mb4_ga8_gb256_lr1e-3_1epoch/"
    + "pytorch_model.bin"
)
model["s2_pretrained_pth"] = group_supervised_s2_pretrained_pth

save_steps = 1000
periodic_eval_interval = 2000
default_hooks["checkpoint"].update(
    dict(
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=2,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
    )
)

_periodic_pairs = [
    (dataset, evaluator)
    for dataset, evaluator in zip(val_datasets, val_evaluators)
    if evaluator["data_name"] == "refcoco_val_refseg"
]
assert len(val_datasets) == len(val_evaluators)
assert len(_periodic_pairs) == 1
periodic_refcoco_dataset = deepcopy(_periodic_pairs[0][0])
periodic_refcoco_evaluator = deepcopy(_periodic_pairs[0][1])
assert periodic_refcoco_dataset["data_name"] == "refcoco_val_refseg"
periodic_refcoco_evaluator.update(terminal_summary_only=True)
custom_hooks.insert(
    0,
    dict(
        type=PeriodicRefSegEvalHook,
        dataset=periodic_refcoco_dataset,
        evaluator=periodic_refcoco_evaluator,
        interval=periodic_eval_interval,
        dataloader_num_workers=1,
        eval_at_start=False,
        final_datasets=None,
        final_evaluators=None,
    ),
)

assert world_size == 32 and global_batch == 128
assert batch_size == train_dataloader["batch_size"] == 4
assert accumulative_counts == optim_wrapper["accumulative_counts"] == 1
assert train_dataloader["sampler"]["per_device_batch_size"] == 4
assert sample_ratio == train_dataloader["sampler"]["sample_ratio"] == 1.0
assert max_epochs == train_cfg["max_epochs"] == 2
assert lr == optim_wrapper["optimizer"]["lr"] == 4e-5
assert param_scheduler[0]["end"] == 0.06
assert param_scheduler[1]["begin"] == 0.06 and param_scheduler[1]["end"] == 2
assert model["latent_s3_pretrained_pth"] is None
assert len(custom_hooks) == 4
assert visualizer is None
