









from mmengine.config import read_base

with read_base():
    from .masklat_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_2node8_refcoco500_st3_bipartite_latent_transport_finetune import *


periodic_eval_interval = 1000
final_refseg_data_names = (
    "refcoco_val_refseg",
    "refcoco_testA_refseg",
    "refcoco_testB_refseg",
    "refcoco+_val_refseg",
    "refcoco+_testA_refseg",
    "refcoco+_testB_refseg",
    "refcocog_val_refseg",
    "refcocog_test_refseg",
)

_final_refseg_pairs = [
    (dataset, evaluator)
    for dataset, evaluator in zip(val_datasets, val_evaluators)
    if evaluator["data_name"] in final_refseg_data_names
]
_final_refseg_by_name = {
    evaluator["data_name"]: (dataset, evaluator)
    for dataset, evaluator in _final_refseg_pairs
}
assert len(_final_refseg_pairs) == len(final_refseg_data_names)
assert len(_final_refseg_by_name) == len(final_refseg_data_names)
assert tuple(_final_refseg_by_name) == final_refseg_data_names

final_refseg_datasets = [
    deepcopy(_final_refseg_by_name[name][0])
    for name in final_refseg_data_names
]
final_refseg_evaluators = [
    deepcopy(_final_refseg_by_name[name][1])
    for name in final_refseg_data_names
]
for _evaluator in final_refseg_evaluators:
    _evaluator["terminal_summary_only"] = True

custom_hooks[0].update(
    dict(
        interval=periodic_eval_interval,
        final_datasets=final_refseg_datasets,
        final_evaluators=final_refseg_evaluators,
        final_output_subdir="final_refseg",
    )
)

assert custom_hooks[0]["eval_at_start"] is False
assert custom_hooks[0]["interval"] == 1000
assert len(custom_hooks[0]["final_datasets"]) == 8
assert len(custom_hooks[0]["final_evaluators"]) == 8

assert default_hooks["checkpoint"]["interval"] == 500
assert default_hooks["checkpoint"]["max_keep_ckpts"] == 2
