

from os import getenv

from mmengine.config import read_base

with read_base():
    from .masklat_qwen3_4b_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu8_1node_refseg1000_final8_st3_bipartite_latent_transport_finetune import *


sample_ratio = float(getenv("S3_SAMPLE_RATIO", "1.0"))
if not 0.0 < sample_ratio <= 1.0:
    raise ValueError(
        f"S3_SAMPLE_RATIO must be in (0, 1], got {sample_ratio}"
    )
train_dataloader["sampler"]["sample_ratio"] = sample_ratio

s2_latent_pretrained_pth = getenv("S2_LATENT_PRETRAINED_PTH") or (
    init_dir
    + "MaskLAT/s2_latent_align_pretrain/"
    + "masklat_qwen3_4b_instruct_siglip2_so400m_p14_384_"
    + "sam_large_m2f_gpu16_st3_bipartite_latent_align_pretrain/"
    + "pytorch_model.bin"
)
model.update(
    dict(
        s2_pretrained_pth=s2_latent_pretrained_pth,
        s2g_pretrained_pth=None,
        st3_group_alignment_pretrain=False,
        st3_latent_alignment_pretrain=False,
        inject_sam_vit_tokens_to_vlm=False,
        use_st3_latent_for_imgconv=False,
        sampler_input_feat="pixel_values",
    )
)

for _training_dataset in train_dataloader["dataset"]["datasets"]:
    _training_dataset["single_conversation"] = False
    if _training_dataset["data_name"] == "llava_imgconv":
        _training_dataset["exclude_pure_text"] = False


max_epochs = 2
train_cfg["max_epochs"] = max_epochs
param_scheduler[0]["end"] = warmup_ratio * max_epochs
param_scheduler[1]["begin"] = warmup_ratio * max_epochs
param_scheduler[1]["end"] = max_epochs

logging_interval = 100
save_steps = 1000
default_hooks["logger"]["interval"] = logging_interval
default_hooks["checkpoint"].update(
    dict(
        by_epoch=False,
        interval=save_steps,
        save_last=True,
    )
)

assert "Qwen3-4B-Instruct-2507" in llm_name_or_path
assert optim_wrapper["accumulative_counts"] == 2
assert model["s2_pretrained_pth"] == s2_latent_pretrained_pth
assert model["inject_sam_vit_tokens_to_vlm"] is False
assert model["use_st3_latent_for_imgconv"] is False
assert model["sampler_input_feat"] == "pixel_values"
assert train_dataloader["sampler"]["sample_ratio"] == sample_ratio
assert all(
    dataset["single_conversation"] is False
    for dataset in train_dataloader["dataset"]["datasets"]
)
assert next(
    dataset
    for dataset in train_dataloader["dataset"]["datasets"]
    if dataset["data_name"] == "llava_imgconv"
)["exclude_pure_text"] is False
assert train_cfg["max_epochs"] == 2
assert param_scheduler[1]["end"] == 2
assert custom_hooks[0]["eval_at_start"] is False
assert custom_hooks[0]["interval"] == 1000
assert len(custom_hooks[0]["final_datasets"]) == 8
assert default_hooks["logger"]["interval"] == 100
assert default_hooks["checkpoint"]["by_epoch"] is False
assert default_hooks["checkpoint"]["interval"] == 1000
