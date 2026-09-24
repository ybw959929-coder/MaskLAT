







from copy import deepcopy
from os import getenv
from mmengine.config import read_base
from mmengine.hooks import LoggerHook
from masklat.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook, PeriodicRefSegEvalHook
from masklat.model.group_supervised_latent import GroupSupervisedLatentMaskLATModel

with read_base():
    from .masklat_qwen3_4b_latent_s3_aligned_4node import *


_native_text_max_length = max_length
_aligned_text_max_length = int(4096 - (384 / 14) ** 2 - 1024) - num_proposal_latents


def _align_text_budget(value):
    """Copy every nested dataset/tokenizer/hook root without rewriting LLM capacity."""
    if isinstance(value, dict):
        return {
            key: (_aligned_text_max_length if key == "max_length" and child == _native_text_max_length
                  else _align_text_budget(child))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_align_text_budget(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_align_text_budget(child) for child in value)
    return value


for _name, _value in list(globals().items()):
    if not _name.startswith("_") and isinstance(_value, (dict, list, tuple)):
        globals()[_name] = _align_text_budget(_value)
max_length = latent_text_max_length = _aligned_text_max_length
_base_text_max_length = max_length + num_proposal_latents

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

qwen_group_supervised_s2_pretrained_pth = getenv("QWEN_GROUP_SUPERVISED_S2_PRETRAINED_PTH") or (
    work_dir + "s2_latent_align_pretrain/"
    + "qwen3_4b_group_supervised_readout_s2_4node8gpu_mb4_ga2_gb256_lr1e-3_1epoch/"
    + "pytorch_model.bin"
)


s2_pretrained_pth = s2_latent_pretrained_pth = qwen_group_supervised_s2_pretrained_pth
model.update(
    type=GroupSupervisedLatentMaskLATModel,
    s2_pretrained_pth=qwen_group_supervised_s2_pretrained_pth,
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
model["segmentor"]["decoder"]["config"].update(
    use_st3_bipartite_latent_transport=True,
    use_st123_latent_deepstack=False,
    use_latent_s2_post_vlm_transport=False,
    st3_transport_enable_latent_writeback=True,
    st3_transport_late_condition_refresh_stages=(),
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
logging_interval = 50
save_steps = 1000
save_total_limit = 2
periodic_eval_interval = 2000
default_hooks["logger"].update(type=LoggerHook, interval=logging_interval)
default_hooks["checkpoint"].update(
    by_epoch=False,
    interval=save_steps,
    max_keep_ckpts=save_total_limit,
    save_last=True,
    save_optimizer=True,
    save_param_scheduler=True,
)

_periodic_pairs = [
    (dataset, evaluator) for dataset, evaluator in zip(val_datasets, val_evaluators)
    if evaluator["data_name"] == "refcoco_val_refseg"
]
assert len(val_datasets) == len(val_evaluators) and len(_periodic_pairs) == 1
periodic_refcoco_dataset = deepcopy(_periodic_pairs[0][0])
periodic_refcoco_evaluator = deepcopy(_periodic_pairs[0][1])
periodic_refcoco_evaluator.update(terminal_summary_only=True)

custom_hooks = [
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
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "visual_projector", "segmentor"],
        display_params=False,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=PTCheckpointHook, clean_pth=False),
]
visualizer = None

assert llm_name_or_path == init_dir + "Qwen3-4B-Instruct-2507"
assert model["llm"]["pretrained_model_name_or_path"] == llm_name_or_path
assert model["tokenizer"]["pretrained_model_name_or_path"] == llm_name_or_path
assert model["llm"]["attn_implementation"] == "flash_attention_2"
assert max_length == 2255 and num_proposal_latents == 64
assert world_size == 32 and global_batch == 128
assert batch_size == train_dataloader["batch_size"] == 4
assert accumulative_counts == optim_wrapper["accumulative_counts"] == 1
assert train_dataloader["sampler"]["per_device_batch_size"] == 4
assert sample_ratio == train_dataloader["sampler"]["sample_ratio"] == 1.0
assert max_epochs == train_cfg["max_epochs"] == 2
assert lr == optim_wrapper["optimizer"]["lr"] == 4e-5
assert param_scheduler[0]["end"] == 0.06
assert param_scheduler[1]["begin"] == 0.06 and param_scheduler[1]["end"] == 2
assert model["s1_pretrained_pth"] == s1_pretrained_pth
assert model["s2_pretrained_pth"] == qwen_group_supervised_s2_pretrained_pth
assert model["latent_s3_pretrained_pth"] is None
assert model["freeze_llm"] is False and model["freeze_visual_encoder"] is False
assert model["freeze_segmentor_encoder"] is False
assert model["use_st3_latent_for_imgconv"] is False
assert model["inject_sam_vit_tokens_to_vlm"] is False
assert model["sampler_input_feat"] == "pixel_values"
assert optim_wrapper["dtype"] == "float16"
assert optim_wrapper["paramwise_cfg"]["custom_keys"]["segmentor.encoder"]["lr_mult"] == 0.1
assert optim_wrapper["paramwise_cfg"]["custom_keys"]["visual_encoder"]["lr_mult"] == 0.1
assert len(train_dataloader["dataset"]["datasets"]) == 11
assert all(d["max_length"] == 2255 and d["single_conversation"] is False
           for d in train_dataloader["dataset"]["datasets"])
assert next(d for d in train_dataloader["dataset"]["datasets"]
            if d["data_name"] == "llava_imgconv")["exclude_pure_text"] is False
assert periodic_refcoco_dataset["max_length"] == 2255
assert periodic_refcoco_dataset["data_name"] == "refcoco_val_refseg"
assert len(custom_hooks) == 4 and visualizer is None
