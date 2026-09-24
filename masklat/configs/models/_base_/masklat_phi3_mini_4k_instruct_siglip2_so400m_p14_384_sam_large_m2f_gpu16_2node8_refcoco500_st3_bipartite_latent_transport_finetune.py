





from mmengine.config import read_base

with read_base():
    from .masklat_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_st3_bipartite_latent_transport_finetune import *

from masklat.engine.hooks import ConciseLoggerHook, PeriodicRefSegEvalHook


step_interval = 500
save_total_limit = 2
logging_interval = 50

refcoco_val_datasets = [
    dataset
    for dataset in val_datasets
    if dataset["data_name"] == "refcoco_val_refseg"
]
refcoco_val_evaluators = [
    evaluator
    for evaluator in val_evaluators
    if evaluator["data_name"] == "refcoco_val_refseg"
]
assert len(refcoco_val_datasets) == 1
assert len(refcoco_val_evaluators) == 1
periodic_refcoco_val_evaluator = deepcopy(refcoco_val_evaluators[0])
periodic_refcoco_val_evaluator["terminal_summary_only"] = True

default_hooks["checkpoint"].update(
    dict(
        by_epoch=False,
        interval=step_interval,
        max_keep_ckpts=save_total_limit,
        save_last=True,
    )
)
default_hooks["logger"].update(
    dict(
        type=ConciseLoggerHook,
        interval=logging_interval,
    )
)

custom_hooks = [
    dict(
        type=PeriodicRefSegEvalHook,
        dataset=deepcopy(refcoco_val_datasets[0]),
        evaluator=periodic_refcoco_val_evaluator,
        interval=step_interval,
        dataloader_num_workers=1,
        eval_at_start=False,
    ),
    custom_hooks[-1],
]

assert custom_hooks[0]["eval_at_start"] is False
assert default_hooks["checkpoint"]["max_keep_ckpts"] == 2
