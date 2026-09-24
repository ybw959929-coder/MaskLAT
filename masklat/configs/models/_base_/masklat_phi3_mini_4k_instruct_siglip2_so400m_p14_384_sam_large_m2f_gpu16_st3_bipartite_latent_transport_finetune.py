







from mmengine.config import read_base

with read_base():
    from .masklat_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune import *


num_proposal_latents = 64
_base_text_max_length = max_length
latent_text_max_length = _base_text_max_length - num_proposal_latents
if latent_text_max_length <= 0:
    raise ValueError("the VLM context leaves no room after latent tokens")


def _reserve_latent_context(value):
    """Reserve exactly 64 positions in every inherited VLM sequence."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key == "max_length" and child == _base_text_max_length:
                value[key] = latent_text_max_length
            else:
                _reserve_latent_context(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reserve_latent_context(child)


for _config_root in (
    train_datasets,
    train_dataloader,
    val_datasets,
    vis_datasets,
    custom_hooks,
):
    _reserve_latent_context(_config_root)

max_length = latent_text_max_length

s1_pretrained_pth = (
    init_dir
    + "MaskLAT/s1_seg_finetune/"
    + "masklat_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin"
)
s2_pretrained_pth = (
    init_dir
    + "MaskLAT/s2_align_pretrain/"
    + "masklat_phi3_mini_4k_instruct_siglip2_so400m_p14_384_"
    + "sam_large_e1_gpu16_align_pretrain/pytorch_model.bin"
)

model.update(
    dict(
        s1_pretrained_pth=s1_pretrained_pth,
        s2_pretrained_pth=s2_pretrained_pth,
    )
)
model["segmentor"]["decoder"]["config"].update(
    dict(
        use_topology_group_decoder=False,
        use_st3_group_vlm_refiner=False,
        use_st3_proposal_latent_bridge=False,
        use_st3_bipartite_latent_transport=True,
        st3_latent_num_tokens=num_proposal_latents,
        st3_latent_num_heads=8,
        st3_latent_mask_threshold=0.5,
        st3_latent_fourier_features=64,
        st3_transport_pre_vlm_depth=2,
        st3_transport_post_vlm_depth=1,
        st3_transport_decoder_depth=6,
        st3_transport_output_init_std=1.0e-3,
        st3_transport_require_expected_shapes=True,
        group_log_diagnostics=False,
    )
)

optim_wrapper["paramwise_cfg"]["custom_keys"].update(
    {
        "segmentor.st3_transport_proposal_builder": dict(
            lr_mult=1.0,
            decay_mult=1.0,
        ),
        "segmentor.st3_transport_bridge": dict(
            lr_mult=1.0,
            decay_mult=1.0,
        ),
    }
)

_transport_config = model["segmentor"]["decoder"]["config"]
assert _transport_config["use_topology_group_decoder"] is False
assert _transport_config["use_st3_group_vlm_refiner"] is False
assert _transport_config["use_st3_proposal_latent_bridge"] is False
assert _transport_config["use_st3_bipartite_latent_transport"] is True
assert _transport_config["st3_latent_num_tokens"] == 64
assert _transport_config["st3_transport_pre_vlm_depth"] == 2
assert _transport_config["st3_transport_post_vlm_depth"] == 1
assert _transport_config["st3_transport_decoder_depth"] == 6

assert optim_wrapper["dtype"] == "float16"
