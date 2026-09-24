import logging
import math
import os
import os.path as osp
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.dist import get_rank
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import get_parameter_dtype
from xtuner.model.modules import dispatch_modules
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.model.utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
    traverse_dict,
)
from xtuner.registry import BUILDER
from xtuner.utils.device import get_device, get_torch_device

from ..model.modules import (
    ConnectorConfig,
    ConnectorModel,
    DynamicProjectorConfig,
    DynamicProjectorModel,
    SamplerConfig,
    SamplerModel,
)
from ..utils.constants import (
    DEFAULT_CLS_TOKEN,
    DEFAULT_PEND_TOKEN,
    DEFAULT_PSTART_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_TASKS,
)
from ..utils.misc import data_sample_to_device
from .seg_condition_scope import (
    build_and_pack_seg_conditions,
    extract_indexed_embeddings,
    regroup_targets_by_seg,
)
from .segmentors.mask2former import (
    AlignedSamSiglipTokenProjector,
    SamSiglipFusionProjector,
)
from .st123_phi_deepstack import St123PhiDeepstackMixin
from .utils import prepare_inputs_labels_for_multimodal
from .utils.input_process import build_group_bidirectional_causal_mask


def _mask_text_only_imgconv_groups(group_valid_mask, image_files):
    """Disable latent insertion for ImgConv rows without a real image."""

    if not isinstance(group_valid_mask, torch.Tensor) or (
        group_valid_mask.ndim != 2
    ):
        raise ValueError("group_valid_mask must be a [batch, groups] tensor")
    if not isinstance(image_files, (list, tuple)):
        raise TypeError("ImgConv image_files must be a list or tuple")
    if len(image_files) != group_valid_mask.shape[0]:
        raise ValueError(
            "ImgConv image_files must match the latent batch: "
            f"{len(image_files)} != {group_valid_mask.shape[0]}"
        )

    has_image = []
    for image_file in image_files:
        if image_file is None:
            has_image.append(False)
        elif isinstance(image_file, str) and image_file:
            has_image.append(True)
        else:
            raise TypeError(
                "each ImgConv image_file must be None or a non-empty string"
            )
    image_mask = torch.tensor(
        has_image,
        dtype=torch.bool,
        device=group_valid_mask.device,
    )
    return group_valid_mask.bool() & image_mask[:, None]


@dataclass
class MaskLATOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None


class MaskLATModel(St123PhiDeepstackMixin, BaseModel):
    def __init__(
        self,
        llm=None,
        tokenizer=None,
        visual_encoder=None,
        postprocess_fn=None,
        segmentor=None,
        special_tokens=None,
        freeze_llm=False,
        freeze_visual_encoder=False,
        freeze_segmentor_encoder=False,
        freeze_segmentor_connector=False,
        visual_select_layer=-2,
        visual_select_indx=0,  # 1 for clip, 0 for siglip
        seg_select_layers=[8, 16, 24, 32],
        extract_seg_embeds=True,
        s1_pretrained_pth=None,
        s2_pretrained_pth=None,
        s2_latent_token_resize=None,
        s2_latent_expand_noise_std: float = 1.0e-3,
        latent_s3_pretrained_pth=None,
        s2g_pretrained_pth=None,
        st3_group_alignment_pretrain=False,
        st3_latent_alignment_pretrain=False,
        late_condition_refresh_finetune=False,
        latent_s2_post_vlm_decoder_finetune=False,
        projector_depth=2,
        downsample_ratio=0.5,
        llm_lora=None,
        visual_encoder_lora=None,
        segmentor_lora=None,
        connector_type=None,
        connector_hidden_dim=256,
        connector_scale_factor=[4, 2, 1, 0.5],
        sampler_type="naive",
        sampler_input_feat="extra_pixel_values",
        cond_type: Literal["phrase", "cls", "all"] = "phrase",
        use_dual_encoder=False,
        inject_sam_vit_tokens_to_vlm=True,
        use_st3_latent_for_imgconv=True,
        stagewise_visual_select_layers=(6, 12, 18, -2),
        stagewise_sam_select_layers=(6, 12, 18, 24),
        stagewise_gate_init_bias=-2.0,
        use_vision_sampler=False,
        use_activation_checkpointing=True,
        max_position_embeddings=None,
        llm_loss_weight: float = 1.0,
        seg_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_segmentor_encoder = freeze_segmentor_encoder
        self.freeze_segmentor_connector = freeze_segmentor_connector
        self.st3_group_alignment_pretrain = bool(
            st3_group_alignment_pretrain
        )
        self.st3_latent_alignment_pretrain = bool(
            st3_latent_alignment_pretrain
        )
        self.late_condition_refresh_finetune = bool(
            late_condition_refresh_finetune
        )
        self.latent_s2_post_vlm_decoder_finetune = bool(
            latent_s2_post_vlm_decoder_finetune
        )
        if s2_latent_token_resize not in (None, "from_64"):
            raise ValueError(
                "s2_latent_token_resize must be None or 'from_64'"
            )
        if (
            not math.isfinite(float(s2_latent_expand_noise_std))
            or float(s2_latent_expand_noise_std) <= 0.0
        ):
            raise ValueError(
                "s2_latent_expand_noise_std must be finite and positive"
            )
        self.s2_latent_token_resize = s2_latent_token_resize
        self.s2_latent_expand_noise_std = float(
            s2_latent_expand_noise_std
        )
        self._latent_s3_handoff_loaded = False
        self._late_condition_refresh_loaded_explicitly = False
        if not isinstance(inject_sam_vit_tokens_to_vlm, bool):
            raise TypeError("inject_sam_vit_tokens_to_vlm must be boolean")
        self.inject_sam_vit_tokens_to_vlm = inject_sam_vit_tokens_to_vlm
        if not isinstance(use_st3_latent_for_imgconv, bool):
            raise TypeError("use_st3_latent_for_imgconv must be boolean")
        self.use_st3_latent_for_imgconv = use_st3_latent_for_imgconv
        if not isinstance(stagewise_visual_select_layers, (tuple, list)) or (
            len(stagewise_visual_select_layers) != 4
        ):
            raise ValueError(
                "stagewise_visual_select_layers must contain four SigLIP hidden-state indices"
            )
        self.stagewise_visual_select_layers = tuple(
            int(index) for index in stagewise_visual_select_layers
        )
        if not isinstance(stagewise_sam_select_layers, (tuple, list)) or (
            len(stagewise_sam_select_layers) != 4
        ):
            raise ValueError(
                "stagewise_sam_select_layers must contain four SAM hidden-state indices"
            )
        self.stagewise_sam_select_layers = tuple(
            int(index) for index in stagewise_sam_select_layers
        )
        self.stagewise_gate_init_bias = float(stagewise_gate_init_bias)
        self._stagewise_deepstack_hook_handles = []
        self._active_stagewise_generation_latents = None
        self._active_stagewise_generation_group_ids = None

        assert (
            llm is not None or visual_encoder is not None or segmentor is not None
        ), "llm, visual_encoder, and segmentor cannot be all None"

        if isinstance(llm, dict):
            llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
        self.llm = self._build_from_cfg_or_module(llm)
        self.tokenizer = self._build_from_cfg_or_module(tokenizer)
        self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)
        self.segmentor = self._build_from_cfg_or_module(segmentor)

        if self.llm is not None:
            self.llm.config.use_cache = False
            dispatch_modules(self.llm)

        self.postprocess_fn = postprocess_fn
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        if self.visual_encoder is not None:
            self.projector_depth = projector_depth
            visual_projector_config = DynamicProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=self.projector_depth,
            )
            self.visual_projector = DynamicProjectorModel(visual_projector_config).to(self.visual_encoder.dtype)

        if self.segmentor is not None:
            if self.llm is not None and self.segmentor.decoder is not None:
                llm_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.llm.config.hidden_size,
                    llm_hidden_size=self.segmentor.dec_config.hidden_size,
                    depth=self.projector_depth,
                )
                self.llm_projector = DynamicProjectorModel(llm_projector_config).to(self.llm.dtype)

            if use_dual_encoder and self.segmentor.encoder is not None:
                seg_projector_config = DynamicProjectorConfig(
                    visual_hidden_size=self.segmentor.enc_config.hidden_size,
                    llm_hidden_size=self.llm.config.hidden_size,
                    downsample_ratio=downsample_ratio,
                    depth=self.projector_depth,
                )
                self.seg_projector = DynamicProjectorModel(seg_projector_config).to(self.segmentor.dtype)

            if extract_seg_embeds and connector_type is not None and self.segmentor.pixel_decoder is not None:
                seg_select_layers = seg_select_layers[-self.segmentor.dec_config.num_feature_levels :]
                connector_config = ConnectorConfig(
                    segmentor_encoder_channels=[self.segmentor.enc_config.hidden_size]
                    * self.segmentor.dec_config.num_feature_levels,
                    hidden_channels=connector_hidden_dim,
                    scale_factor=connector_scale_factor[-self.segmentor.dec_config.num_feature_levels :],
                    connector_type=connector_type,
                )
                self.seg_connector = ConnectorModel(connector_config).to(self.segmentor.dtype)

            if use_vision_sampler and self.segmentor.decoder is not None:
                sampler_config = SamplerConfig(
                    sampler_type=sampler_type,
                    num_sample_point=256,
                    input_dim=self.llm.config.hidden_size,
                    output_dim=self.segmentor.dec_config.hidden_size,
                )
                self.vision_sampler = SamplerModel(sampler_config).to(self.segmentor.dtype)

            if self.segmentor.decoder is not None and self.segmentor.open_cls:
                self.bg_embeds = nn.Embedding(1, self.segmentor.dec_config.hidden_size).to(self.segmentor.dtype)

            if getattr(self.segmentor.dec_config, "use_topology_group_decoder", False):
                if self.visual_encoder is None:
                    raise ValueError(
                        "topology group decoder requires the existing SigLIP visual encoder"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "SigLIP topology features must retain patch token 0; "
                        f"visual_select_indx={visual_select_indx}"
                    )
                self.segmentor.configure_topology_group_decoder(
                    siglip_hidden_dim=int(self.visual_encoder.config.hidden_size)
                )

            if getattr(
                self.segmentor.dec_config,
                "use_st3_group_vlm_refiner",
                False,
            ):
                if getattr(
                    self.segmentor.dec_config,
                    "use_topology_group_decoder",
                    False,
                ):
                    raise ValueError(
                        "st3 Group-in-VLM and legacy topology decoding are "
                        "mutually exclusive"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "st3 Group-in-VLM requires both the LLM and SigLIP "
                        "visual encoder"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "st3 Group-in-VLM must retain every SigLIP patch "
                        f"token; visual_select_indx={visual_select_indx}"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "st3 Group-in-VLM requires extract_seg_embeds=True "
                        "and the SAM multi-scale connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "the first st3 Group-in-VLM experiment requires the "
                        "existing open-vocabulary Cond+BG classifier"
                    )
                attention_impl = getattr(
                    self.llm.config,
                    "_attn_implementation",
                    getattr(self.llm.config, "attn_implementation", None),
                )
                if attention_impl not in (None, "eager", "sdpa"):
                    raise ValueError(
                        "Group-block bidirectional attention requires the "
                        "LLM eager or SDPA attention implementation; "
                        "FlashAttention-2 cannot express this custom block "
                        f"mask, got {attention_impl!r}"
                    )
                # XTuner dispatches the attention modules before this point.
                # Validate the concrete post-dispatch classes as well as the
                # config string so a dependency/version change cannot silently
                # route the 4-D Group block mask into FlashAttention-2.
                attention_classes = sorted(
                    {
                        type(module).__name__
                        for name, module in self.llm.named_modules()
                        if name.endswith("self_attn")
                    }
                )
                flash_classes = [
                    name
                    for name in attention_classes
                    if "flashattention" in name.lower()
                    or "flash_attn" in name.lower()
                ]
                if flash_classes:
                    raise ValueError(
                        "st3 Group-in-VLM requires an eager/SDPA attention "
                        "module after XTuner dispatch, but found "
                        f"{flash_classes}; configured implementation="
                        f"{attention_impl!r}"
                    )
                if get_rank() == 0:
                    print_log(
                        "st3 Group-in-VLM attention backend: "
                        f"config={attention_impl!r}, "
                        f"classes={attention_classes or ['unknown']}",
                        logger="current",
                    )
                self.segmentor.configure_st3_group_vlm_refiner(
                    siglip_hidden_dim=int(
                        self.visual_encoder.config.hidden_size
                    ),
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                )

            if getattr(
                self.segmentor.dec_config,
                "use_st3_proposal_latent_bridge",
                False,
            ):
                if getattr(
                    self.segmentor.dec_config,
                    "use_topology_group_decoder",
                    False,
                ) or getattr(
                    self.segmentor.dec_config,
                    "use_st3_group_vlm_refiner",
                    False,
                ):
                    raise ValueError(
                        "the st3 proposal latent bridge is mutually exclusive "
                        "with all topology/hard-Group paths"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "the st3 proposal latent bridge requires the LLM and "
                        "SigLIP visual encoder"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "proposal MaskPool must retain all SigLIP patch tokens"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "the proposal latent bridge requires the trained SAM "
                        "multi-scale connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "the proposal latent bridge requires open-vocabulary "
                        "local Cond+BG classification"
                    )
                self.segmentor.configure_st3_proposal_latent_bridge(
                    siglip_hidden_dim=int(
                        self.visual_encoder.config.hidden_size
                    ),
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                )

            if getattr(
                self.segmentor.dec_config,
                "use_st3_bipartite_latent_transport",
                False,
            ):
                if any(
                    bool(
                        getattr(
                            self.segmentor.dec_config,
                            name,
                            False,
                        )
                    )
                    for name in (
                        "use_topology_group_decoder",
                        "use_st3_group_vlm_refiner",
                        "use_st3_proposal_latent_bridge",
                    )
                ):
                    raise ValueError(
                        "bipartite latent transport is mutually exclusive "
                        "with all topology, hard-Group, and gated paths"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "bipartite latent transport requires the LLM and "
                        "SigLIP visual encoder"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "proposal MaskPool must retain all SigLIP patch tokens"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "bipartite latent transport requires the trained SAM "
                        "multi-scale connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "bipartite latent transport requires open-vocabulary "
                        "local Cond+BG classification"
                    )
                self.segmentor.configure_st3_bipartite_latent_transport(
                    siglip_hidden_dim=int(
                        self.visual_encoder.config.hidden_size
                    ),
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                )
                if self._uses_st123_latent_deepstack():
                    if getattr(self.llm.config, "model_type", None) != "phi3":
                        raise ValueError("st123 latent deepstack requires Phi-3")
                    if self.inject_sam_vit_tokens_to_vlm:
                        raise ValueError(
                            "st123 latent deepstack requires "
                            "inject_sam_vit_tokens_to_vlm=False"
                        )
                    if self.s2_latent_token_resize is not None:
                        raise ValueError(
                            "st123 latent deepstack uses exactly 64 latents "
                            "and cannot load the old S2 capacity ablation"
                        )
                    self._install_st123_phi_deepstack_hooks()

            if getattr(
                self.segmentor.dec_config,
                "use_latent_s2_post_vlm_transport",
                False,
            ):
                incompatible = (
                    "use_topology_group_decoder",
                    "use_st3_group_vlm_refiner",
                    "use_st3_proposal_latent_bridge",
                    "use_st3_bipartite_latent_transport",
                    "use_stagewise_latent_trifusion",
                    "use_proposal_mediated_tripartite_transport",
                    "use_geometry_aligned_visual_fusion",
                )
                enabled = [
                    name for name in incompatible
                    if bool(getattr(self.segmentor.dec_config, name, False))
                ]
                if enabled:
                    raise ValueError(
                        "latent-S2 post-VLM transport is mutually exclusive "
                        f"with {enabled}"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "latent-S2 post-VLM transport requires Phi-3 and SigLIP"
                    )
                if getattr(self.llm.config, "model_type", None) != "phi3":
                    raise ValueError(
                        "the first post-VLM comparison supports Phi-3 only"
                    )
                if self.inject_sam_vit_tokens_to_vlm:
                    raise ValueError(
                        "latent S2 owns the VLM visual carriers; set "
                        "inject_sam_vit_tokens_to_vlm=False"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "proposal MaskPool must retain every SigLIP patch token"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "latent-S2 post-VLM transport requires the SAM connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "latent-S2 post-VLM transport requires local Cond+BG "
                        "classification"
                    )
                self.segmentor.configure_latent_s2_post_vlm_transport(
                    siglip_hidden_dim=int(
                        self.visual_encoder.config.hidden_size
                    ),
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                )

            if getattr(
                self.segmentor.dec_config,
                "use_stagewise_latent_trifusion",
                False,
            ):
                incompatible = (
                    "use_topology_group_decoder",
                    "use_st3_group_vlm_refiner",
                    "use_st3_proposal_latent_bridge",
                    "use_st3_bipartite_latent_transport",
                )
                enabled = [
                    name for name in incompatible
                    if bool(getattr(self.segmentor.dec_config, name, False))
                ]
                if enabled:
                    raise ValueError(
                        "stagewise latent trifusion is mutually exclusive with "
                        f"{enabled}"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "stagewise latent trifusion requires Phi-3 and SigLIP"
                    )
                if getattr(self.llm.config, "model_type", None) != "phi3":
                    raise ValueError(
                        "the first stagewise implementation supports Phi-3 only"
                    )
                if self.inject_sam_vit_tokens_to_vlm:
                    raise ValueError(
                        "stagewise fusion replaces independent SAM VLM tokens; "
                        "set inject_sam_vit_tokens_to_vlm=False"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "stagewise SigLIP features must retain every patch token"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "stagewise latent trifusion requires the SAM connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "stagewise latent trifusion requires local Cond+BG classification"
                    )
                encoder_layer_specs = (
                    (
                        "SigLIP",
                        self.stagewise_visual_select_layers,
                        self.visual_encoder.config,
                    ),
                    (
                        "SAM",
                        self.stagewise_sam_select_layers,
                        self.segmentor.enc_config,
                    ),
                )
                for encoder_name, selected_layers, encoder_config in (
                    encoder_layer_specs
                ):
                    num_hidden_layers = getattr(
                        encoder_config,
                        "num_hidden_layers",
                        None,
                    )
                    if num_hidden_layers is None:
                        raise ValueError(
                            f"{encoder_name} config must expose num_hidden_layers"
                        )
                    hidden_state_count = int(num_hidden_layers) + 1
                    invalid_layers = [
                        index
                        for index in selected_layers
                        if not -hidden_state_count <= index < hidden_state_count
                    ]
                    if invalid_layers:
                        raise ValueError(
                            f"stagewise {encoder_name} layers {invalid_layers} "
                            f"escape {hidden_state_count} hidden states"
                        )
                siglip_hidden_state_count = (
                    int(self.visual_encoder.config.num_hidden_layers) + 1
                )
                if not -siglip_hidden_state_count <= int(
                    visual_select_layer
                ) < siglip_hidden_state_count:
                    raise ValueError(
                        "visual_select_layer escapes the SigLIP hidden states"
                    )
                resolved_base_layer = int(visual_select_layer) % (
                    siglip_hidden_state_count
                )
                resolved_fusion_layer = self.stagewise_visual_select_layers[-1] % (
                    siglip_hidden_state_count
                )
                if resolved_base_layer != resolved_fusion_layer:
                    raise ValueError(
                        "stagewise dense fusion must use the same final "
                        "SigLIP hidden state as the spatial proposal path"
                    )
                sam_hidden_dim = int(self.segmentor.enc_config.hidden_size)
                siglip_hidden_dim = int(self.visual_encoder.config.hidden_size)
                llm_hidden_dim = int(self.llm.config.hidden_size)
                self.segmentor.configure_stagewise_latent_trifusion(
                    siglip_hidden_dim=siglip_hidden_dim,
                    sam_hidden_dim=sam_hidden_dim,
                    llm_hidden_dim=llm_hidden_dim,
                )
                self.stagewise_visual_fusion = SamSiglipFusionProjector(
                    siglip_hidden_dim=siglip_hidden_dim,
                    sam_hidden_dim=sam_hidden_dim,
                    llm_hidden_dim=llm_hidden_dim,
                ).to(self.visual_encoder.dtype)
                # The joint SAM+SigLIP merger replaces the standalone
                # SigLIP projector for this opt-in path. Keep the legacy
                # module available to old configs without optimizing an
                # unused bypass in S2/S3.
                self.visual_projector.requires_grad_(False)
                self.stagewise_deepstack_gates = nn.Parameter(
                    torch.full(
                        (3,),
                        self.stagewise_gate_init_bias,
                        dtype=torch.float32,
                    )
                )
                self._install_stagewise_phi3_deepstack_hooks()

            if getattr(
                self.segmentor.dec_config,
                "use_proposal_mediated_tripartite_transport",
                False,
            ):
                incompatible = (
                    "use_topology_group_decoder",
                    "use_st3_group_vlm_refiner",
                    "use_st3_proposal_latent_bridge",
                    "use_st3_bipartite_latent_transport",
                    "use_stagewise_latent_trifusion",
                )
                enabled = [
                    name for name in incompatible
                    if bool(getattr(self.segmentor.dec_config, name, False))
                ]
                if enabled:
                    raise ValueError(
                        "geometry proposal transport is mutually exclusive "
                        f"with {enabled}"
                    )
                if not bool(getattr(
                    self.segmentor.dec_config,
                    "use_geometry_aligned_visual_fusion",
                    False,
                )):
                    raise ValueError(
                        "proposal transport requires geometry-aligned visual fusion"
                    )
                if self.llm is None or self.visual_encoder is None:
                    raise ValueError(
                        "geometry proposal transport requires the LLM and SigLIP"
                    )
                if self.inject_sam_vit_tokens_to_vlm:
                    raise ValueError(
                        "geometry fusion replaces independent SAM VLM tokens; "
                        "set inject_sam_vit_tokens_to_vlm=False"
                    )
                if visual_select_indx != 0:
                    raise ValueError(
                        "geometry alignment must retain every SigLIP patch token"
                    )
                if not extract_seg_embeds or not hasattr(self, "seg_connector"):
                    raise ValueError(
                        "geometry proposal transport requires the SAM connector"
                    )
                if not self.segmentor.open_cls:
                    raise ValueError(
                        "geometry proposal transport requires local Cond+BG classification"
                    )
                fusion_dim = int(self.segmentor.dec_config.geometry_fusion_dim)
                self.segmentor.configure_geometry_proposal_transport(
                    fused_feature_dim=fusion_dim,
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                )
                self.geometry_visual_fusion = AlignedSamSiglipTokenProjector(
                    siglip_hidden_dim=int(self.visual_encoder.config.hidden_size),
                    sam_hidden_dim=int(self.segmentor.enc_config.hidden_size),
                    fusion_dim=fusion_dim,
                    llm_hidden_dim=int(self.llm.config.hidden_size),
                    target_size=int(self.segmentor.dec_config.geometry_target_size),
                    output_size=int(self.segmentor.dec_config.geometry_output_size),
                ).to(self.visual_encoder.dtype)
                # Keep the legacy projector in the state layout for old
                # configs, but this experiment has one fused visual stream.
                self.visual_projector.requires_grad_(False)

        if self.st3_group_alignment_pretrain and not self._uses_st3_group_vlm_refiner():
            raise ValueError(
                "st3_group_alignment_pretrain requires "
                "use_st3_group_vlm_refiner=True"
            )
        if self.st3_latent_alignment_pretrain and not (
            self._uses_st3_bipartite_latent_transport()
            or self._uses_stagewise_latent_trifusion()
            or self._uses_geometry_proposal_transport()
        ):
            raise ValueError(
                "st3_latent_alignment_pretrain requires "
                "a bipartite, stagewise, or geometry proposal path"
            )
        if self.st3_group_alignment_pretrain and self.st3_latent_alignment_pretrain:
            raise ValueError(
                "hard-Group and latent S2 alignment modes are mutually exclusive"
            )
        if self.late_condition_refresh_finetune:
            if not self._uses_st3_bipartite_latent_transport():
                raise ValueError(
                    "late_condition_refresh_finetune requires bipartite "
                    "latent transport"
                )
            refresh_stages = tuple(getattr(
                self.segmentor.dec_config,
                "st3_transport_late_condition_refresh_stages",
                (),
            ))
            if refresh_stages != (6, 9):
                raise ValueError(
                    "the incremental experiment requires exactly st6/st9 "
                    f"condition refreshers, got {refresh_stages}"
                )
            if self.st3_group_alignment_pretrain or (
                self.st3_latent_alignment_pretrain
            ):
                raise ValueError(
                    "late condition-refresh finetuning is mutually exclusive "
                    "with S2 alignment modes"
                )
        if self.latent_s2_post_vlm_decoder_finetune:
            if not self._uses_latent_s2_post_vlm_transport():
                raise ValueError(
                    "latent_s2_post_vlm_decoder_finetune requires "
                    "use_latent_s2_post_vlm_transport=True"
                )
            if self.st3_group_alignment_pretrain or (
                self.st3_latent_alignment_pretrain
            ) or self.late_condition_refresh_finetune:
                raise ValueError(
                    "post-VLM decoder finetuning is mutually exclusive with "
                    "alignment and late-refresh modes"
                )
            if latent_s3_pretrained_pth is None:
                raise ValueError(
                    "post-VLM decoder finetuning must initialize from the "
                    "final latent S3 checkpoint; set latent_s3_pretrained_pth"
                )
        if self._uses_st3_latent_path():
            configured_latents = int(
                getattr(self.segmentor.dec_config, "st3_latent_num_tokens", 0)
            )
            resize_authorized = self.s2_latent_token_resize == "from_64"
            if configured_latents != 64 and not resize_authorized:
                raise ValueError(
                    "the latent-only SAM/VLM alignment contract requires "
                    "exactly 64 latent tokens unless the explicit S2 "
                    "capacity ablation is enabled; got "
                    f"{configured_latents}"
                )
            if resize_authorized and configured_latents not in (32, 128):
                raise ValueError(
                    "the S2 capacity ablation supports exactly 32 or 128 "
                    f"latent tokens, got {configured_latents}"
                )
            if self.inject_sam_vit_tokens_to_vlm and not hasattr(
                self,
                "seg_projector",
            ):
                raise ValueError(
                    "inject_sam_vit_tokens_to_vlm=True requires use_dual_encoder "
                    "and a configured seg_projector"
                )

        if self.freeze_llm and self.llm is not None:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder and self.visual_encoder is not None:
            self.visual_encoder.requires_grad_(False)
        if self.freeze_segmentor_encoder and self.segmentor is not None:
            self.segmentor.encoder.requires_grad_(False)
        if self.freeze_segmentor_connector and self.segmentor is not None:
            self.seg_connector.requires_grad_(False)
        if (
            hasattr(self, "seg_projector")
            and not self.inject_sam_vit_tokens_to_vlm
        ):
            # Keep the reversible branch in the module/state layout, but do
            # not optimize a projector whose output is intentionally omitted.
            self.seg_projector.requires_grad_(False)
            if get_rank() == 0:
                print_log(
                    "direct SAM-ViT token injection into the VLM is disabled; "
                    "SAM features remain available to the segmentor and "
                    "st3 latent builder",
                    logger="current",
                )

        if use_activation_checkpointing:
            # For backward compatibility
            if self.llm is not None:
                if hasattr(self.llm, "enable_input_require_grads"):
                    self.llm.enable_input_require_grads()
                else:
                    self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            if self.visual_encoder is not None:
                if hasattr(self.visual_encoder, "enable_input_require_grads"):
                    self.visual_encoder.enable_input_require_grads()
                else:
                    self.visual_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                self.visual_projector.enable_input_require_grads()

            if self.segmentor is not None:
                if hasattr(self.segmentor, "enable_input_require_grads"):
                    self.segmentor.enable_input_require_grads()
                else:
                    self.segmentor.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                if hasattr(self, "seg_projector"):
                    self.seg_projector.enable_input_require_grads()
                if hasattr(self, "llm_projector"):
                    self.llm_projector.enable_input_require_grads()
                if hasattr(self, "seg_connector"):
                    self.seg_connector.enable_input_require_grads()
            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

        if self._uses_st3_group_vlm_refiner():
            st3_refiner = self.segmentor.st3_vlm_refiner
            refiner_checkpointing = bool(
                getattr(st3_refiner, "gradient_checkpointing", False)
            )
            refiner_checkpoint_function = getattr(
                st3_refiner,
                "_gradient_checkpointing_func",
                None,
            )
            if use_activation_checkpointing and (
                not refiner_checkpointing
                or not callable(refiner_checkpoint_function)
            ):
                raise RuntimeError(
                    "st3 Group refiner activation checkpointing was not "
                    "enabled by MaskLATSegmentor; refusing a silently high-memory "
                    "training path"
                )
            if get_rank() == 0:
                print_log(
                    "st3 Group refiner activation checkpointing: "
                    f"{refiner_checkpointing}",
                    logger="current",
                )

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.use_segmentor_encoder_lora = segmentor_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora, use_activation_checkpointing)
        if self.use_segmentor_encoder_lora:
            self._prepare_segmentor_for_lora(segmentor_lora, use_activation_checkpointing)

        state_dict = super().state_dict()
        if s1_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s1_pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)

            matched_keys = [k for k in pretrained_state_dict.keys() if k in state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in state_dict.keys()]
            missed_keys = [k for k in state_dict.keys() if k not in pretrained_state_dict.keys()]
            print_log(f"Load s1_pretrained_pth from {s1_pretrained_pth}", logger="current")
            print_log(f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}", logger="current")
            if len(mismatched_keys) > 0:
                print_log(f"Mismatched keys: {mismatched_keys}", logger="current", level=logging.WARNING)
            if len(missed_keys) > 0:
                print_log(f"Missed keys: {missed_keys}", logger="current", level=logging.WARNING)

        if s2_pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(s2_pretrained_pth)
            pretrained_state_dict = self._adapt_s2_latent_token_count(
                pretrained_state_dict
            )
            if self.late_condition_refresh_finetune:
                bridge_prefix = "segmentor.st3_transport_bridge."
                required_sources = (
                    bridge_prefix + "latent_input_norm.weight",
                    bridge_prefix + "post_vlm_blocks.0.cross_query_norm.weight",
                )
                missing_sources = [
                    key for key in required_sources
                    if key not in pretrained_state_dict
                ]
                if missing_sources:
                    raise RuntimeError(
                        "the incremental st6/st9 run must initialize from a "
                        "trained latent S3 checkpoint; missing keys "
                        f"{missing_sources} in {s2_pretrained_pth}"
                    )
                self._late_condition_refresh_loaded_explicitly = any(
                    key.startswith(
                        bridge_prefix + "late_condition_refreshers."
                    )
                    for key in pretrained_state_dict
                )
            if (
                self._uses_st3_latent_path()
                and not self.inject_sam_vit_tokens_to_vlm
            ):
                required_latent_s2_prefixes = (
                    (
                        "stagewise_visual_fusion.siglip_norm.",
                        "stagewise_visual_fusion.sam_norm.",
                        "stagewise_visual_fusion.fusion_merger.",
                        "stagewise_deepstack_gates",
                        "segmentor.stagewise_latent_builder.",
                    )
                    if self._uses_stagewise_latent_trifusion()
                    else (
                        (
                            "geometry_visual_fusion.",
                            "segmentor.geometry_proposal_builder.",
                            "segmentor.geometry_proposal_builder."
                            "geometry_area_encoder.",
                            "segmentor.geometry_proposal_builder."
                            "prefix_fused_latent_stages.",
                            "segmentor.geometry_proposal_builder."
                            "prefix_transport_stages.",
                        )
                        if self._uses_geometry_proposal_transport()
                        else (
                            "visual_projector.",
                            "segmentor.st3_transport_proposal_builder.",
                        )
                    )
                )
                missing_latent_s2_prefixes = [
                    prefix
                    for prefix in required_latent_s2_prefixes
                    if not any(
                        key.startswith(prefix)
                        for key in pretrained_state_dict
                    )
                ]
                if missing_latent_s2_prefixes:
                    raise RuntimeError(
                        "the no-direct-SAM latent model requires a true latent "
                        "S2 checkpoint containing the joint visual projector "
                        "and 64-token proposal builder; missing prefixes "
                        f"{missing_latent_s2_prefixes} in {s2_pretrained_pth}"
                    )
            if self._uses_st123_latent_deepstack():
                self._validate_st123_s2_checkpoint(pretrained_state_dict)
            self.load_state_dict(pretrained_state_dict, strict=False)

            matched_keys = [k for k in pretrained_state_dict.keys() if k in state_dict.keys()]
            mismatched_keys = [k for k in pretrained_state_dict.keys() if k not in state_dict.keys()]
            missed_keys = [k for k in state_dict.keys() if k not in pretrained_state_dict.keys()]
            print_log(f"Load s2_pretrained_pth from {s2_pretrained_pth}", logger="current")
            print_log(f"Matched keys: {len(matched_keys)} / {len(pretrained_state_dict.keys())}", logger="current")
            if len(mismatched_keys) > 0:
                print_log(f"Mismatched keys: {mismatched_keys}", logger="current", level=logging.WARNING)
            if len(missed_keys) > 0:
                print_log(f"Missed keys: {missed_keys}", logger="current", level=logging.WARNING)

        if latent_s3_pretrained_pth is not None:
            self._load_latent_s3_post_vlm_checkpoint(
                latent_s3_pretrained_pth
            )

        if s2g_pretrained_pth is not None:
            self._load_s2g_proposal_checkpoint(s2g_pretrained_pth)

        if self.st3_group_alignment_pretrain:
            self._freeze_for_st3_group_alignment()
        if self.st3_latent_alignment_pretrain:
            self._freeze_for_st3_latent_alignment()
        if self.late_condition_refresh_finetune:
            self._freeze_for_late_condition_refresh()
        if self.latent_s2_post_vlm_decoder_finetune:
            self._freeze_for_latent_s2_post_vlm_decoder()

        self._finite_diag_initial_query_summary = None
        if self._finite_diagnostics_enabled():
            query_weight = self._topology_query_feature_parameter()
            self._finite_diag_initial_query_summary = (
                self._finite_tensor_summary(query_weight)
                if query_weight is not None
                else "parameter unavailable"
            )
            self._audit_topology_query_parameter(
                "model_init_after_s1_s2_load"
            )

        self.visual_select_layer = visual_select_layer
        self.visual_select_indx = visual_select_indx
        self.seg_select_layers = seg_select_layers
        self.extract_seg_embeds = extract_seg_embeds
        self.sampler_input_feat = sampler_input_feat
        if not isinstance(self.sampler_input_feat, str):
            raise TypeError("sampler_input_feat must be a string")
        if (
            hasattr(self, "vision_sampler")
            and not self.inject_sam_vit_tokens_to_vlm
            and self.sampler_input_feat == "extra_pixel_values"
        ):
            raise ValueError(
                "inject_sam_vit_tokens_to_vlm=False requires the vision "
                "sampler to use projected SigLIP pixel_values; sampling "
                "extra_pixel_values would reintroduce SAM-derived embeddings "
                "at <region> positions in the LLM input"
            )
        self.cond_type = cond_type
        self.llm_loss_weight = llm_loss_weight
        self.seg_loss_weight = seg_loss_weight

    @property
    def device(self):
        return get_device()

    @property
    def dtype(self):
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def _validate_st123_s2_checkpoint(self, checkpoint_state):
        """Require every trained stage/projector tensor before loading new S2.

        The old and new builders deliberately occupy the same outer prefix so
        the existing freeze/save contracts still apply. Checking that prefix
        alone cannot distinguish their incompatible architectures.
        """
        required = {}
        for prefix, module in (
            ("visual_projector.", self.visual_projector),
            (
                "segmentor.st3_transport_proposal_builder.",
                self.segmentor.st3_transport_proposal_builder,
            ),
        ):
            required.update(
                (prefix + key, value) for key, value in module.state_dict().items()
            )
        missing = [key for key in required if key not in checkpoint_state]
        mismatched = [
            key for key, target in required.items()
            if key in checkpoint_state and (
                not isinstance(checkpoint_state[key], torch.Tensor)
                or checkpoint_state[key].shape != target.shape
            )
        ]
        if missing or mismatched:
            raise RuntimeError(
                "st123 deepstack requires its own complete retrained S2 "
                "checkpoint (all three builders and the SigLIP projector); "
                "a single-st3 S2 checkpoint is not compatible. "
                f"Missing {len(missing)} tensors: {missing[:8]}; "
                f"incompatible shapes: {mismatched[:8]}"
            )

    def _adapt_s2_latent_token_count(self, checkpoint_state):
        """Adapt only the learned 64-token S2 seed for capacity ablations."""

        if self.s2_latent_token_resize is None:
            return checkpoint_state
        key = (
            "segmentor.st3_transport_proposal_builder.learned_latents"
        )
        if key not in checkpoint_state:
            raise RuntimeError(
                "the requested S2 latent-capacity ablation cannot find "
                f"{key} in the S2 checkpoint"
            )
        source = checkpoint_state[key]
        if not isinstance(source, torch.Tensor) or source.ndim != 2:
            raise RuntimeError(
                f"invalid S2 learned latent tensor for {key}: "
                f"{type(source).__name__}"
            )
        target_count = int(self.segmentor.dec_config.st3_latent_num_tokens)
        if source.shape[0] != 64:
            raise RuntimeError(
                "s2_latent_token_resize='from_64' requires a 64-token S2 "
                f"checkpoint, got {tuple(source.shape)}"
            )
        if target_count == 32:
            adapted = source[:target_count].clone()
            policy = "first 32 trained S2 seeds"
        elif target_count == 128:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1024)
            noise = torch.randn(
                source.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).mul_(self.s2_latent_expand_noise_std)
            expanded = source.detach().to(device="cpu", dtype=torch.float32)
            expanded = expanded + noise
            adapted = torch.cat(
                (source, expanded.to(dtype=source.dtype)),
                dim=0,
            )
            policy = (
                "64 trained S2 seeds plus a deterministically perturbed copy"
            )
        else:
            raise RuntimeError(
                "the S2 latent-capacity ablation supports target counts "
                f"32 or 128, got {target_count}"
            )
        adapted_state = OrderedDict(checkpoint_state)
        adapted_state[key] = adapted
        print_log(
            "Adapt S2 learned latents "
            f"{tuple(source.shape)} -> {tuple(adapted.shape)} using {policy}",
            logger="current",
        )
        return adapted_state

    def _load_s2g_proposal_checkpoint(self, checkpoint_path):
        """Load only the independently pretrained st3 proposal/token path."""

        if not self._uses_st3_group_vlm_refiner():
            raise ValueError(
                "s2g_pretrained_pth requires use_st3_group_vlm_refiner=True"
            )
        proposal_builder = getattr(
            self.segmentor,
            "st3_proposal_builder",
            None,
        )
        if proposal_builder is None:
            raise RuntimeError(
                "the st3 proposal builder must exist before loading S2-G"
            )
        state_dict = guess_load_checkpoint(checkpoint_path)
        accepted_prefixes = (
            "segmentor.st3_proposal_builder.",
            "module.segmentor.st3_proposal_builder.",
            "model.segmentor.st3_proposal_builder.",
        )
        proposal_state = {}
        for key, value in state_dict.items():
            for prefix in accepted_prefixes:
                if key.startswith(prefix):
                    proposal_state[key[len(prefix) :]] = value
                    break
        if not proposal_state:
            raise RuntimeError(
                "S2-G checkpoint does not contain any "
                "segmentor.st3_proposal_builder parameters: "
                f"{checkpoint_path}"
            )
        proposal_builder.load_state_dict(proposal_state, strict=True)
        print_log(
            "Load S2-G proposal/token alignment from "
            f"{checkpoint_path}; matched keys={len(proposal_state)}",
            logger="current",
        )

    def _load_latent_s3_post_vlm_checkpoint(self, checkpoint_path):
        """Overlay a complete latent S3 and remap its shared VLM handoff.

        Exact-name parameters (LLM, both vision backbones, projectors,
        proposal builder, and the original Mask2Former decoder) are restored
        directly.  The trained S3 handoff is copied from
        ``st3_transport_bridge`` into the comparison bridge.  Its old
        st4--st9 transport stages are intentionally not copied: those are the
        only modules replaced and optimized by this incremental experiment.
        """

        if not self.latent_s2_post_vlm_decoder_finetune or not (
            self._uses_latent_s2_post_vlm_transport()
        ):
            raise ValueError(
                "latent_s3_pretrained_pth is reserved for the post-VLM "
                "incremental experiment"
            )
        checkpoint_state = guess_load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint_state, dict):
            raise TypeError("the final latent S3 checkpoint must be a state dict")

        required_prefixes = (
            "llm.",
            "visual_encoder.",
            "segmentor.encoder.",
            "segmentor.decoder.",
            "visual_projector.",
            "llm_projector.",
            "seg_connector.",
            "segmentor.st3_transport_proposal_builder.",
            "segmentor.st3_transport_bridge.latent_input_norm.",
            "segmentor.st3_transport_bridge.post_vlm_blocks.",
            "segmentor.st3_transport_bridge.entry_query_read.",
        )
        missing_prefixes = [
            prefix
            for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in checkpoint_state)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "latent_s3_pretrained_pth is not a complete final latent S3 "
                f"checkpoint; missing prefixes={missing_prefixes}: "
                f"{checkpoint_path}"
            )

        current_state = super().state_dict()
        overlay_state = OrderedDict()
        shape_mismatches = []
        for key, value in checkpoint_state.items():
            if key not in current_state:
                continue
            if tuple(value.shape) != tuple(current_state[key].shape):
                shape_mismatches.append(
                    (key, tuple(value.shape), tuple(current_state[key].shape))
                )
                continue
            overlay_state[key] = value
        if shape_mismatches:
            raise RuntimeError(
                "final latent S3 checkpoint has incompatible model shapes: "
                f"{shape_mismatches[:20]}"
            )

        source_root = "segmentor.st3_transport_bridge."
        target_root = "segmentor.latent_s2_post_vlm_bridge."
        shared_subtrees = (
            "latent_input_norm.",
            "post_vlm_blocks.",
            "entry_query_read.",
        )
        mapped_handoff = OrderedDict()
        for key, value in checkpoint_state.items():
            if not key.startswith(source_root):
                continue
            suffix = key[len(source_root) :]
            if not suffix.startswith(shared_subtrees):
                continue
            target_key = target_root + suffix
            if target_key not in current_state:
                raise RuntimeError(
                    "trained S3 handoff has no matching target parameter: "
                    f"{key} -> {target_key}"
                )
            if tuple(value.shape) != tuple(current_state[target_key].shape):
                raise RuntimeError(
                    "trained S3 handoff shape mismatch: "
                    f"{key}{tuple(value.shape)} -> "
                    f"{target_key}{tuple(current_state[target_key].shape)}"
                )
            overlay_state[target_key] = value
            mapped_handoff[target_key] = value

        expected_handoff = {
            key
            for key in current_state
            if key.startswith(tuple(
                target_root + subtree for subtree in shared_subtrees
            ))
        }
        missing_handoff = sorted(expected_handoff - set(mapped_handoff))
        if missing_handoff:
            raise RuntimeError(
                "final latent S3 checkpoint cannot initialize the complete "
                f"shared post-VLM handoff; missing targets={missing_handoff}"
            )
        self.load_state_dict(overlay_state, strict=False)
        self._latent_s3_handoff_loaded = True
        print_log(
            "Load complete final latent S3 from "
            f"{checkpoint_path}; exact overlays={len(overlay_state) - len(mapped_handoff)}, "
            f"remapped frozen handoff tensors={len(mapped_handoff)}; old S3 "
            "st4--st9 transport was intentionally excluded",
            logger="current",
        )

    def _freeze_for_st3_group_alignment(self):
        """Freeze the base model and expose only the group tokenizer."""

        proposal_builder = getattr(
            self.segmentor,
            "st3_proposal_builder",
            None,
        )
        if proposal_builder is None:
            raise RuntimeError(
                "S2-G alignment requires a configured st3 proposal builder"
            )
        self.requires_grad_(False)
        # The decoder reference dtype is BF16, but MMEngine's dynamic
        # GradScaler cannot unscale BF16 parameter gradients on the target
        # PyTorch stack.  Keep the only trainable module in FP32 while the
        # S2-G recipe executes it under BF16 autocast; its parameter gradients
        # and AdamW states then remain FP32 without changing frozen backbones.
        proposal_builder.float()
        proposal_builder.requires_grad_(True)
        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        expected_prefix = "segmentor.st3_proposal_builder."
        unexpected = [
            name for name in trainable if not name.startswith(expected_prefix)
        ]
        if not trainable or unexpected:
            raise RuntimeError(
                "invalid S2-G trainable-parameter contract: "
                f"trainable={trainable}, unexpected={unexpected}"
            )
        self.freeze_llm = True
        self.freeze_visual_encoder = True
        self.freeze_segmentor_encoder = True
        self.freeze_segmentor_connector = True
        print_log(
            "S2-G alignment freezes the base model and trains only "
            f"{expected_prefix}* ({len(trainable)} tensors)",
            logger="current",
        )

    def _freeze_for_st3_latent_alignment(self):
        """Expose only the active visual projection and pre-VLM latent builder.

        The released S1 image segmentor remains the fixed proposal source.
        Post-VLM transport and st4--st9 are intentionally excluded because
        ImgConv alignment has neither SEG tokens nor mask supervision.
        """

        stagewise_enabled = getattr(
            self,
            "_uses_stagewise_latent_trifusion",
            lambda: False,
        )()
        geometry_enabled = getattr(
            self,
            "_uses_geometry_proposal_transport",
            lambda: False,
        )()
        builder_name = (
            "stagewise_latent_builder"
            if stagewise_enabled
            else (
                "geometry_proposal_builder"
                if geometry_enabled
                else "st3_transport_proposal_builder"
            )
        )
        proposal_builder = getattr(self.segmentor, builder_name, None)
        if proposal_builder is None:
            raise RuntimeError(
                "latent S2 alignment requires its configured proposal builder"
            )
        if not stagewise_enabled and not geometry_enabled and not hasattr(self, "visual_projector"):
            raise RuntimeError(
                "latent S2 alignment requires the SigLIP visual projector"
            )

        self.requires_grad_(False)
        # Keep trainable master parameters in FP32. The runner's BF16
        # autocast preserves compatibility with frozen BF16 backbones while
        # avoiding unsupported BF16 GradScaler unscale operations.
        proposal_builder.float()
        proposal_builder.requires_grad_(True)
        if stagewise_enabled:
            if not hasattr(self, "stagewise_visual_fusion") or not hasattr(
                self,
                "stagewise_deepstack_gates",
            ):
                raise RuntimeError(
                    "stagewise S2 requires visual fusion and Phi-3 deepstack gates"
                )
            self.stagewise_visual_fusion.float()
            self.stagewise_visual_fusion.requires_grad_(True)
            self.stagewise_deepstack_gates.requires_grad_(True)
        elif geometry_enabled:
            if not hasattr(self, "geometry_visual_fusion"):
                raise RuntimeError(
                    "geometry S2 requires the aligned SAM/SigLIP visual fusion"
                )
            self.geometry_visual_fusion.float()
            self.geometry_visual_fusion.requires_grad_(True)
        else:
            self.visual_projector.float()
            self.visual_projector.requires_grad_(True)
        # ``requires_grad_(True)`` above intentionally exposes the complete
        # active builder, but the bipartite path does not consume the two
        # inherited gated-affinity heads.  Restore their permanent frozen
        # state so every S2 trainable parameter participates in the LM loss.
        if not stagewise_enabled and not geometry_enabled:
            for unused_affinity_name in ("latent_query", "proposal_key"):
                unused_affinity = getattr(
                    proposal_builder,
                    unused_affinity_name,
                    None,
                )
                if unused_affinity is not None:
                    unused_affinity.requires_grad_(False)
            freeze_unused = getattr(
                proposal_builder, "freeze_unused_affinity_heads", None
            )
            if callable(freeze_unused):
                freeze_unused()

        expected_prefixes = (
            (
                "stagewise_visual_fusion.",
                "stagewise_deepstack_gates",
                "segmentor.stagewise_latent_builder.",
            )
            if stagewise_enabled
            else (
                (
                    "geometry_visual_fusion.",
                    "segmentor.geometry_proposal_builder.",
                )
                if geometry_enabled
                else (
                    "visual_projector.",
                    "segmentor.st3_transport_proposal_builder.",
                )
            )
        )
        trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        unexpected = [
            name
            for name in trainable
            if not name.startswith(expected_prefixes)
        ]
        missing_prefixes = [
            prefix
            for prefix in expected_prefixes
            if not any(name.startswith(prefix) for name in trainable)
        ]
        if unexpected or missing_prefixes:
            raise RuntimeError(
                "invalid latent S2 trainable-parameter contract: "
                f"trainable={trainable}, unexpected={unexpected}, "
                f"missing_prefixes={missing_prefixes}"
            )
        self.freeze_llm = True
        self.freeze_visual_encoder = True
        self.freeze_segmentor_encoder = True
        self.freeze_segmentor_connector = True
        print_log(
            "latent S2 alignment trains only the active visual/latent path "
            f"({len(trainable)} tensors); direct SAM tokens in VLM="
            f"{self.inject_sam_vit_tokens_to_vlm}",
            logger="current",
        )

    def _freeze_for_late_condition_refresh(self):
        """Train only the st6/st9 copies of the learned st3 refresh block."""

        bridge = getattr(
            getattr(self, "segmentor", None),
            "st3_transport_bridge",
            None,
        )
        if bridge is None:
            raise RuntimeError(
                "late condition-refresh finetuning requires the configured "
                "bipartite bridge"
            )
        refreshers = bridge.late_condition_refreshers
        if tuple(bridge.late_condition_refresh_stages) != (6, 9) or (
            tuple(refreshers.keys()) != ("st6", "st9")
        ):
            raise RuntimeError(
                "late condition-refresh modules must target exactly st6/st9"
            )

        if not self._late_condition_refresh_loaded_explicitly:
            source_norm = bridge.latent_input_norm.state_dict()
            source_block = bridge.post_vlm_blocks[0].state_dict()
            for stage, refresher in refreshers.items():
                for key, value in source_norm.items():
                    if not torch.equal(
                        refresher.input_norm.state_dict()[key],
                        value,
                    ):
                        raise RuntimeError(
                            f"{stage} input norm was not initialized from st3"
                        )
                for key, value in source_block.items():
                    if not torch.equal(
                        refresher.condition_refresh.state_dict()[key],
                        value,
                    ):
                        raise RuntimeError(
                            f"{stage} CondRefresh was not initialized from st3"
                        )

        self.requires_grad_(False)
        hook_owners = [self.llm, self.visual_encoder, self.segmentor]
        hook_owners.extend(
            getattr(self, name, None)
            for name in (
                "visual_projector",
                "seg_projector",
                "llm_projector",
                "seg_connector",
            )
        )
        for module in hook_owners:
            disable_input_grads = getattr(
                module,
                "disable_input_require_grads",
                None,
            )
            has_input_grad_hook = hasattr(
                module,
                "_require_grads_hook",
            ) or bool(getattr(module, "_require_grads_hooks", None))
            if callable(disable_input_grads) and has_input_grad_hook:
                disable_input_grads()
        # FP32 master parameters avoid BF16 GradScaler unscale failures while
        # the existing float16 AMP recipe still autocasts their operations.
        refreshers.float()
        refreshers.requires_grad_(True)

        expected_prefix = (
            "segmentor.st3_transport_bridge.late_condition_refreshers."
        )
        trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        unexpected = [
            name for name in trainable
            if not name.startswith(expected_prefix)
        ]
        if not trainable or unexpected or any(
            "gate" in name.lower() for name in trainable
        ):
            raise RuntimeError(
                "invalid st6/st9 incremental trainable contract: "
                f"trainable={trainable}, unexpected={unexpected}"
            )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        print_log(
            "incremental latent refresh trains only gate-free st6/st9 "
            f"modules ({len(trainable)} tensors, "
            f"{trainable_parameters:,} parameters); all inherited weights "
            "are frozen",
            logger="current",
        )

    def _freeze_for_latent_s2_post_vlm_decoder(self):
        """Train only six new stages on top of the complete frozen S3."""

        bridge = getattr(
            getattr(self, "segmentor", None),
            "latent_s2_post_vlm_bridge",
            None,
        )
        if bridge is None:
            raise RuntimeError(
                "post-VLM decoder finetuning requires its configured bridge"
            )
        if len(bridge.transport_stages) != 6:
            raise RuntimeError(
                "the post-VLM bridge must cover exactly st4--st9"
            )
        if not self._latent_s3_handoff_loaded:
            raise RuntimeError(
                "the complete final latent S3 must be loaded before freezing "
                "the post-VLM incremental experiment"
            )

        self.requires_grad_(False)
        hook_owners = [self.llm, self.visual_encoder, self.segmentor]
        hook_owners.extend(
            getattr(self, name, None)
            for name in (
                "visual_projector",
                "seg_projector",
                "llm_projector",
                "seg_connector",
            )
        )
        for module in hook_owners:
            disable_input_grads = getattr(
                module,
                "disable_input_require_grads",
                None,
            )
            has_input_grad_hook = hasattr(
                module,
                "_require_grads_hook",
            ) or bool(getattr(module, "_require_grads_hooks", None))
            if callable(disable_input_grads) and has_input_grad_hook:
                disable_input_grads()

        # Only the six replacement stages use FP32 master parameters. The
        # mapped S3 LN/CondRefresh/QueryRead handoff stays frozen and retains
        # its checkpoint dtype and exact behavior.
        bridge.transport_stages.float()
        bridge.transport_stages.requires_grad_(True)
        self.freeze_llm = True
        self.freeze_visual_encoder = True
        self.freeze_segmentor_encoder = True
        self.freeze_segmentor_connector = True
        expected_prefix = (
            "segmentor.latent_s2_post_vlm_bridge.transport_stages."
        )
        trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        unexpected = [
            name for name in trainable
            if not name.startswith(expected_prefix)
        ]
        if not trainable or unexpected or any(
            "gate" in name.lower() for name in trainable
        ):
            raise RuntimeError(
                "invalid latent-S2 post-VLM trainable contract: "
                f"trainable={trainable}, unexpected={unexpected}"
            )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        print_log(
            "latent-S3 incremental experiment trains only six gate-free "
            f"{bridge.transport_type} st4--st9 stages ({len(trainable)} "
            f"tensors, {trainable_parameters:,} parameters); the complete "
            "final S3 and its LN/CondRefresh/QueryRead/SEG handoff are frozen",
            logger="current",
        )

    def train(self, mode: bool = True):
        """Keep frozen alignment backbones in deterministic evaluation mode."""

        super().train(mode)
        if self.st3_group_alignment_pretrain:
            for child in self.children():
                child.eval()
            self.segmentor.st3_proposal_builder.train(mode)
        elif self.st3_latent_alignment_pretrain:
            for child in self.children():
                child.eval()
            if self._uses_stagewise_latent_trifusion():
                self.stagewise_visual_fusion.train(mode)
                self.segmentor.stagewise_latent_builder.train(mode)
            elif self._uses_geometry_proposal_transport():
                self.geometry_visual_fusion.train(mode)
                self.segmentor.geometry_proposal_builder.train(mode)
            else:
                self.visual_projector.train(mode)
                self.segmentor.st3_transport_proposal_builder.train(mode)
        elif self.late_condition_refresh_finetune:
            for child in self.children():
                child.eval()
            # MaskLATSegmentor gates supervised mask-loss construction on its own
            # ``training`` flag.  Restore only that flag (without recursively
            # putting the frozen SAM/pixel-decoder path back into train mode).
            self.segmentor.training = mode
            # Preserve the original decoder's training/checkpointing behavior
            # after st3 while its parameters remain frozen.
            self.segmentor.decoder.train(mode)
            self.segmentor.st3_transport_bridge.late_condition_refreshers.train(
                mode
            )
        elif self.latent_s2_post_vlm_decoder_finetune:
            for child in self.children():
                child.eval()
            # Loss construction still keys off MaskLATSegmentor.training.  The
            # frozen Mask2Former decoder remains in its original train-time
            # control flow so the new bridge receives all st4--st9 losses.
            self.segmentor.training = mode
            self.segmentor.decoder.train(mode)
            bridge = self.segmentor.latent_s2_post_vlm_bridge
            bridge.eval()
            bridge.transport_stages.train(mode)
        return self

    @staticmethod
    def _finite_diagnostics_enabled():
        return os.environ.get(
            "MASKLAT_FINITE_DIAGNOSTICS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _topology_query_feature_parameter(self):
        decoder = getattr(getattr(self, "segmentor", None), "decoder", None)
        query_features = getattr(decoder, "queries_features", None)
        return (
            getattr(query_features, "weight", None)
            if query_features is not None
            else None
        )

    @staticmethod
    def _finite_tensor_summary(tensor):
        if not isinstance(tensor, torch.Tensor):
            return "not a tensor"
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        nonfinite_count = int((~finite_mask).sum().item())
        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
        finite_values = detached[finite_mask]
        if finite_values.numel() > 0:
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_sum = float(finite_values.float().sum().item())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
            finite_sum = float("nan")
        return (
            f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
            f"device={detached.device}, nonfinite={nonfinite_count}, "
            f"nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
            f"finite_min={finite_min:.6g}, finite_max={finite_max:.6g}, "
            f"finite_sum={finite_sum:.9g}"
        )

    def _audit_topology_query_parameter(self, phase):
        if not self._finite_diagnostics_enabled():
            return
        query_weight = self._topology_query_feature_parameter()
        if query_weight is None:
            return
        if bool(torch.isfinite(query_weight).all()):
            return
        positional_weight = getattr(
            getattr(
                getattr(self.segmentor, "decoder", None),
                "queries_embedder",
                None,
            ),
            "weight",
            None,
        )
        raise FloatingPointError(
            "topology learned Query parameter became non-finite: "
            f"phase={phase}, rank={get_rank()}, "
            f"initial=({self._finite_diag_initial_query_summary}), "
            f"current=({self._finite_tensor_summary(query_weight)}), "
            "positional_query=("
            f"{self._finite_tensor_summary(positional_weight)})"
        )

    def _finish_finite_diagnostics_once(self):
        if not self._finite_diagnostics_enabled():
            return
        enabled_once = os.environ.get(
            "MASKLAT_FINITE_DIAGNOSTICS_ONCE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if enabled_once:
            os.environ["MASKLAT_FINITE_DIAGNOSTICS"] = "0"
            print_log(
                "MaskLAT first-forward finite diagnostics passed; disabling "
                "the optional detailed checks for subsequent iterations.",
                logger="current",
            )

    def _audit_loss_output_finite(self, loss_output):
        """Fail before backward when any reported training loss is non-finite."""

        if not self._finite_diagnostics_enabled():
            return
        if not isinstance(loss_output, dict):
            raise TypeError(
                "finite loss diagnostics require a dictionary output, got "
                f"{type(loss_output).__name__}"
            )
        bad_losses = []
        loss_summaries = []
        for name, value in loss_output.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            summary = self._finite_tensor_summary(value)
            loss_summaries.append(f"{name}=({summary})")
            if not bool(torch.isfinite(value.detach()).all()):
                bad_losses.append(name)
        if bad_losses:
            raise FloatingPointError(
                "training loss became non-finite before backward: "
                f"rank={get_rank()}, bad_losses={bad_losses}; "
                + "; ".join(loss_summaries)
            )

    def _add_special_tokens(self, special_tokens):
        assert all(token in DEFAULT_SPECIAL_TOKENS for token in special_tokens)
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.seg_token_idx = -1
        self.cls_token_idx = -1
        self.pstart_token_idx = -1
        self.pend_token_idx = -1

        if DEFAULT_SEG_TOKEN in special_tokens:
            self.seg_token_idx = self.tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_CLS_TOKEN in special_tokens:
            self.cls_token_idx = self.tokenizer(DEFAULT_CLS_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PSTART_TOKEN in special_tokens:
            self.pstart_token_idx = self.tokenizer(DEFAULT_PSTART_TOKEN, add_special_tokens=False)["input_ids"][0]
        if DEFAULT_PEND_TOKEN in special_tokens:
            self.pend_token_idx = self.tokenizer(DEFAULT_PEND_TOKEN, add_special_tokens=False)["input_ids"][0]

    def _get_index_embeds(self, input_embeds, embed_ids):
        output_embeds = []
        for input_embed, embed_id in zip(input_embeds, embed_ids):
            unique_ids = torch.unique(embed_id[embed_id != -1])
            if len(unique_ids) == 0:
                continue

            embeds = torch.stack([input_embed[embed_id == idx].mean(dim=0) for idx in unique_ids])
            output_embeds.append(embeds)

        return output_embeds if len(output_embeds) > 0 else None

    def _uses_st3_group_vlm_refiner(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(
            getattr(dec_config, "use_st3_group_vlm_refiner", False)
        )

    def _uses_st3_proposal_latent_bridge(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(
            getattr(
                dec_config,
                "use_st3_proposal_latent_bridge",
                False,
            )
        )

    def _uses_st3_bipartite_latent_transport(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(
            getattr(
                dec_config,
                "use_st3_bipartite_latent_transport",
                False,
            )
        )

    def _uses_st123_latent_deepstack(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None), "dec_config", None
        )
        return bool(getattr(dec_config, "use_st123_latent_deepstack", False))

    def _uses_stagewise_latent_trifusion(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(
            getattr(
                dec_config,
                "use_stagewise_latent_trifusion",
                False,
            )
        )

    def _uses_latent_s2_post_vlm_transport(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(getattr(
            dec_config,
            "use_latent_s2_post_vlm_transport",
            False,
        ))

    def _uses_geometry_proposal_transport(self) -> bool:
        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(getattr(
            dec_config,
            "use_proposal_mediated_tripartite_transport",
            False,
        ))

    def _install_stagewise_phi3_deepstack_hooks(self) -> None:
        """Inject z0/z1/z2 after the first three Phi-3 decoder layers."""

        if self._stagewise_deepstack_hook_handles:
            return
        phi3_model = getattr(self.llm, "model", None)
        layers = getattr(phi3_model, "layers", None)
        if layers is None or len(layers) < 3:
            raise RuntimeError(
                "stagewise deepstack could not locate three Phi-3 decoder layers"
            )

        def build_hook(stage_index):
            def inject_stage(module, args, hook_kwargs, output):
                stage_latents = hook_kwargs.get("masklat_stagewise_latents")
                group_ids = hook_kwargs.get("masklat_stagewise_group_ids")
                if stage_latents is None and group_ids is None:
                    stage_latents = self._active_stagewise_generation_latents
                    group_ids = self._active_stagewise_generation_group_ids
                if stage_latents is None and group_ids is None:
                    return output
                if stage_latents is None or group_ids is None:
                    raise ValueError(
                        "stagewise latent injections and group IDs must coexist"
                    )
                if not isinstance(stage_latents, (tuple, list)) or (
                    len(stage_latents) != 3
                ):
                    raise ValueError(
                        "stagewise deepstack requires exactly z0, z1, and z2"
                    )
                hidden_states = output[0] if isinstance(output, tuple) else output
                if hidden_states.ndim != 3 or group_ids.ndim != 2:
                    raise ValueError("deepstack hidden states/group IDs have invalid ranks")
                if hidden_states.shape[1] != group_ids.shape[1]:
                    # Cached generation executes one new token after the full
                    # prefill.  The latent positions are already in its KV cache.
                    if hidden_states.shape[1] == 1:
                        return output
                    raise ValueError(
                        "deepstack hidden sequence and group IDs disagree: "
                        f"{hidden_states.shape[1]} != {group_ids.shape[1]}"
                    )
                latent_table = stage_latents[stage_index]
                if latent_table.ndim != 3 or (
                    latent_table.shape[-1] != hidden_states.shape[-1]
                ):
                    raise ValueError(
                        f"invalid z{stage_index} deepstack table shape "
                        f"{tuple(latent_table.shape)}"
                    )
                if hidden_states.shape[0] % latent_table.shape[0] != 0:
                    raise ValueError("deepstack latent batch cannot expand to LLM batch")
                repeat = hidden_states.shape[0] // latent_table.shape[0]
                if repeat != 1:
                    latent_table = latent_table.repeat_interleave(repeat, dim=0)
                if group_ids.shape[0] != hidden_states.shape[0]:
                    if hidden_states.shape[0] % group_ids.shape[0] != 0:
                        raise ValueError("deepstack group-ID batch cannot expand")
                    group_ids = group_ids.repeat_interleave(
                        hidden_states.shape[0] // group_ids.shape[0],
                        dim=0,
                    )
                group_ids = group_ids.to(device=hidden_states.device, dtype=torch.long)
                valid = group_ids.ge(0)
                if bool(valid.any()) and int(group_ids[valid].max().item()) >= (
                    latent_table.shape[1]
                ):
                    raise ValueError("deepstack group ID escapes the latent table")
                gather_ids = group_ids.clamp_min(0)
                latent_table = latent_table.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                latent_delta = torch.gather(
                    latent_table,
                    1,
                    gather_ids.unsqueeze(-1).expand(
                        -1,
                        -1,
                        latent_table.shape[-1],
                    ),
                )
                gate = torch.sigmoid(
                    self.stagewise_deepstack_gates[stage_index]
                ).to(hidden_states.dtype)
                injected = hidden_states + (
                    gate
                    * valid.unsqueeze(-1).to(hidden_states.dtype)
                    * latent_delta
                )
                if isinstance(output, tuple):
                    return (injected,) + output[1:]
                return injected

            return inject_stage

        for stage_index in range(3):
            handle = layers[stage_index].register_forward_hook(
                build_hook(stage_index),
                with_kwargs=True,
            )
            self._stagewise_deepstack_hook_handles.append(handle)

    @contextmanager
    def _stagewise_deepstack_generation_context(
        self,
        stage_latents: Sequence[torch.Tensor],
        group_ids: torch.Tensor,
    ):
        """Expose z0--z2 to Phi hooks without unsupported generation kwargs."""

        if not isinstance(stage_latents, (tuple, list)) or len(stage_latents) != 3:
            raise ValueError("stagewise generation requires exactly z0, z1, and z2")
        if not isinstance(group_ids, torch.Tensor) or group_ids.ndim != 2:
            raise ValueError("stagewise generation group IDs must be [B,L]")
        if self._active_stagewise_generation_latents is not None or (
            self._active_stagewise_generation_group_ids is not None
        ):
            raise RuntimeError("stagewise generation contexts cannot be nested")
        self._active_stagewise_generation_latents = tuple(stage_latents)
        self._active_stagewise_generation_group_ids = group_ids
        try:
            yield
        finally:
            self._active_stagewise_generation_latents = None
            self._active_stagewise_generation_group_ids = None

    def _uses_st3_latent_path(self) -> bool:
        return (
            self._uses_st3_proposal_latent_bridge()
            or self._uses_st3_bipartite_latent_transport()
            or self._uses_latent_s2_post_vlm_transport()
            or self._uses_stagewise_latent_trifusion()
            or self._uses_geometry_proposal_transport()
        )

    @staticmethod
    def _pack_projected_st3_groups(
        projected_hidden_states: torch.Tensor,
        group_ids: torch.Tensor,
        proposal,
    ) -> torch.Tensor:
        """Restore the exact padded Group-token table after the VLM.

        Every valid packed slot must occur exactly once in its source
        sequence.  Rejecting duplicates is important: silently mean-pooling
        two copies would hide an accidental second image placeholder.
        """

        if projected_hidden_states.ndim != 3 or group_ids.ndim != 2:
            raise ValueError(
                "projected VLM states/group_ids must be [B,L,D] and [B,L]"
            )
        if tuple(projected_hidden_states.shape[:2]) != tuple(group_ids.shape):
            raise ValueError("projected VLM states and group_ids disagree")
        packed_valid = proposal.packed_group_valid_mask
        if packed_valid.shape[0] != projected_hidden_states.shape[0]:
            raise ValueError("proposal/VLM source batches differ")
        batch_size, max_groups = packed_valid.shape
        hidden_dim = projected_hidden_states.shape[-1]
        packed = projected_hidden_states.new_zeros(
            (batch_size, max_groups, hidden_dim)
        )
        for source_index in range(batch_size):
            expected_count = int(packed_valid[source_index].sum().item())
            source_ids = group_ids[source_index]
            positions = torch.nonzero(
                source_ids >= 0,
                as_tuple=False,
            ).flatten()
            ids = source_ids.index_select(0, positions).to(torch.long)
            expected_ids = torch.arange(
                expected_count,
                device=ids.device,
                dtype=torch.long,
            )
            sorted_ids, order = torch.sort(ids)
            if not torch.equal(sorted_ids, expected_ids):
                raise ValueError(
                    "each valid st3 Group token must occur exactly once: "
                    f"source={source_index}, ids={ids.tolist()}, "
                    f"expected={expected_ids.tolist()}"
                )
            source_values = projected_hidden_states[source_index].index_select(
                0,
                positions,
            ).index_select(0, order)
            packed[source_index, :expected_count] = source_values
        return packed

    def _finish_st3_from_source_embeddings(
        self,
        *,
        bundle,
        projected_hidden_states: torch.Tensor,
        group_ids: torch.Tensor,
        condition_sources: Sequence[torch.Tensor],
        segment_sources: Sequence[torch.Tensor],
        task_name: str,
        mask_labels=None,
        class_labels=None,
    ):
        packed_conditions = build_and_pack_seg_conditions(
            task_name,
            condition_sources,
            segment_sources,
            self.bg_embeds.weight,
        )
        scope = packed_conditions.scope
        if scope.num_segments <= 0:
            raise ValueError(
                "st3 Group-in-VLM segmentation requires at least one SEG row"
            )
        seg_states = torch.cat(tuple(segment_sources), dim=0)
        packed_group_states = self._pack_projected_st3_groups(
            projected_hidden_states,
            group_ids,
            bundle.proposal,
        )

        row_mask_labels = None
        row_class_labels = None
        if mask_labels is not None:
            regrouped = regroup_targets_by_seg(
                scope,
                mask_labels,
                class_labels,
            )
            row_mask_labels = list(regrouped.mask_labels)
            row_class_labels = list(regrouped.class_labels)
        elif self.training:
            raise ValueError(
                "st3 Group-in-VLM training requires segmentation masks"
            )

        seg_outputs = self.segmentor.finish_st3_group_execution(
            bundle=bundle,
            packed_group_hidden_states=packed_group_states,
            seg_to_sample=scope.seg_to_sample,
            seg_states=seg_states,
            local_cond_embeddings=packed_conditions.embeddings,
            local_cond_valid_mask=packed_conditions.valid_mask,
            row_mask_labels=row_mask_labels,
            row_class_labels=row_class_labels,
            source_mask_labels=(
                list(mask_labels) if mask_labels is not None else None
            ),
            task_name=task_name,
            return_dict=True,
        )
        return seg_outputs, scope

    def _finish_st3_latent_from_source_embeddings(
        self,
        *,
        bundle,
        projected_hidden_states: torch.Tensor,
        group_ids: torch.Tensor,
        condition_sources: Sequence[torch.Tensor],
        segment_sources: Sequence[torch.Tensor],
        task_name: str,
        mask_labels=None,
        class_labels=None,
    ):
        """Finish the single-pass latent bridge from projected VLM states."""

        packed_conditions = build_and_pack_seg_conditions(
            task_name,
            condition_sources,
            segment_sources,
            self.bg_embeds.weight,
        )
        scope = packed_conditions.scope
        if scope.num_segments <= 0:
            raise ValueError(
                "st3 proposal latent segmentation requires a SEG row"
            )
        seg_states = torch.cat(tuple(segment_sources), dim=0)
        packed_latent_states = self._pack_projected_st3_groups(
            projected_hidden_states,
            group_ids,
            bundle.proposal,
        )

        row_mask_labels = None
        row_class_labels = None
        if mask_labels is not None:
            regrouped = regroup_targets_by_seg(
                scope,
                mask_labels,
                class_labels,
            )
            row_mask_labels = list(regrouped.mask_labels)
            row_class_labels = list(regrouped.class_labels)
        elif self.training:
            raise ValueError(
                "st3 proposal latent training requires segmentation masks"
            )

        seg_outputs = self.segmentor.finish_st3_latent_execution(
            bundle=bundle,
            packed_latent_hidden_states=packed_latent_states,
            seg_to_sample=scope.seg_to_sample,
            seg_states=seg_states,
            local_cond_embeddings=packed_conditions.embeddings,
            local_cond_valid_mask=packed_conditions.valid_mask,
            row_mask_labels=row_mask_labels,
            row_class_labels=row_class_labels,
            task_name=task_name,
            return_dict=True,
        )
        return seg_outputs, scope

    def _run_llm_with_group_attention(
        self,
        data_dict,
        group_ids: torch.Tensor,
    ):
        attention_mask = data_dict.get("attention_mask")
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError(
                "st3 Group-in-VLM teacher forcing requires a 2-D attention mask"
            )
        inputs_embeds = data_dict.get("inputs_embeds")
        mask_dtype = (
            inputs_embeds.dtype
            if isinstance(inputs_embeds, torch.Tensor)
            else self.llm.get_input_embeddings().weight.dtype
        )
        llm_inputs = dict(data_dict)
        llm_inputs["attention_mask"] = build_group_bidirectional_causal_mask(
            attention_mask,
            group_ids,
            dtype=mask_dtype,
        )
        return self.llm(**llm_inputs, output_hidden_states=True)

    @contextmanager
    def _group_attention_generation_context(
        self,
        group_ids: torch.Tensor,
    ):
        """Use a Group-block mask only for generation's full prefill.

        GenerationMixin must retain its ordinary 2-D mask so it can append a
        valid column at each decoding step.  The patched forward converts that
        mask to the 4-D Group-aware form only when the current query is the
        complete sequence.  Cached one-token decoding remains ordinary causal
        attention; the bidirectional Group keys/values are already cached.
        """

        if group_ids.ndim != 2:
            raise ValueError("generation group_ids must be [B,L]")
        original_forward = self.llm.forward
        original_use_cache = self.llm.config.use_cache

        def group_aware_forward(*args, **model_inputs):
            attention_mask = model_inputs.get("attention_mask")
            input_ids = model_inputs.get("input_ids")
            inputs_embeds = model_inputs.get("inputs_embeds")
            query_length = None
            if isinstance(inputs_embeds, torch.Tensor):
                query_length = int(inputs_embeds.shape[1])
                mask_dtype = inputs_embeds.dtype
            elif isinstance(input_ids, torch.Tensor):
                query_length = int(input_ids.shape[1])
                mask_dtype = self.llm.get_input_embeddings().weight.dtype
            else:
                mask_dtype = self.llm.get_input_embeddings().weight.dtype

            if (
                isinstance(attention_mask, torch.Tensor)
                and attention_mask.ndim == 2
                and query_length == int(attention_mask.shape[1])
            ):
                current_batch, current_length = attention_mask.shape
                base_batch, base_length = group_ids.shape
                if current_batch % base_batch != 0:
                    raise ValueError(
                        "generation expanded the Group batch incompatibly: "
                        f"{current_batch} is not a multiple of {base_batch}"
                    )
                expanded_ids = group_ids.repeat_interleave(
                    current_batch // base_batch,
                    dim=0,
                ).to(device=attention_mask.device)
                if current_length < base_length:
                    raise ValueError(
                        "generation prefill is shorter than inserted Group IDs"
                    )
                if current_length > base_length:
                    expanded_ids = torch.cat(
                        (
                            expanded_ids,
                            torch.full(
                                (
                                    current_batch,
                                    current_length - base_length,
                                ),
                                -1,
                                dtype=expanded_ids.dtype,
                                device=expanded_ids.device,
                            ),
                        ),
                        dim=1,
                    )
                model_inputs["attention_mask"] = (
                    build_group_bidirectional_causal_mask(
                        attention_mask,
                        expanded_ids,
                        dtype=mask_dtype,
                    )
                )
            return original_forward(*args, **model_inputs)

        object.__setattr__(self.llm, "forward", group_aware_forward)
        self.llm.config.use_cache = True
        try:
            yield
        finally:
            object.__setattr__(self.llm, "forward", original_forward)
            self.llm.config.use_cache = original_use_cache

    def _uses_topology_v2_source_local_conditions(self):
        """Whether Ref/Rea rows must use the V2 source-local class protocol.

        Version 1 intentionally keeps the historical rank-local condition
        table without a validity mask.  Version 2 keeps that table (and its
        stable global-within-table IDs) but prevents an effective row from
        treating conditions belonging to another source sample as negatives.
        """

        dec_config = getattr(
            getattr(self, "segmentor", None),
            "dec_config",
            None,
        )
        return bool(
            getattr(dec_config, "use_topology_group_decoder", False)
            and int(getattr(dec_config, "topology_group_version", 1)) >= 2
        )

    @staticmethod
    def _topology_v2_global_condition_offsets(cond_lens):
        """Return source offsets into the flattened foreground Cond table."""

        condition_counts = [int(value) for value in cond_lens]
        if any(value <= 0 for value in condition_counts):
            raise ValueError(
                "topology V2 source-local Cond counts must be positive, "
                f"got {condition_counts}"
            )
        return list(accumulate([0] + condition_counts[:-1]))

    @staticmethod
    def _topology_v2_source_local_embed_mask(
        cond_lens,
        row_lens=None,
        *,
        device,
    ):
        """Build source-local condition validity for effective SEG rows.

        Foreground columns and their offsets are defined by ``cond_lens``.
        Rows follow the source-major SEG/target order defined independently by
        ``row_lens``.  Every row may compare against foreground conditions
        from its own source and against the final background column, but never
        against a condition originating from another source sample on the
        rank.
        """

        condition_counts = [int(value) for value in cond_lens]
        if any(value <= 0 for value in condition_counts):
            raise ValueError(
                "topology V2 source-local Cond counts must be positive, "
                f"got {condition_counts}"
            )
        row_counts = (
            condition_counts
            if row_lens is None
            else [int(value) for value in row_lens]
        )
        if len(row_counts) != len(condition_counts) or any(
            value <= 0 for value in row_counts
        ):
            raise ValueError(
                "topology V2 source-local row counts must be positive and "
                "aligned with Cond sources: "
                f"conditions={condition_counts}, rows={row_counts}"
            )
        offsets = list(accumulate([0] + condition_counts[:-1]))
        foreground_count = sum(condition_counts)
        rows = []
        for offset, condition_count, row_count in zip(
            offsets,
            condition_counts,
            row_counts,
        ):
            source_valid = torch.zeros(
                foreground_count + 1,
                dtype=torch.bool,
                device=device,
            )
            source_valid[offset : offset + condition_count] = True
            source_valid[-1] = True
            rows.append(source_valid.unsqueeze(0).expand(row_count, -1))
        return torch.cat(rows, dim=0)

    def _process_embeds(self, cond_embeds, seg_embeds, task_name="genseg"):
        B = len(cond_embeds)
        embed_masks = None
        local_cond_lens = None
        local_row_lens = None
        global_cond_lens = None
        bg_embeds = self.bg_embeds.weight
        if task_name in ["genseg", "vgdseg", "gcgseg", "ovseg", "intseg"]:
            max_cond_len = max([x.shape[0] for x in cond_embeds])
            embed_masks = []
            for i, cond_embed in enumerate(cond_embeds):
                cond_embeds[i] = torch.cat(
                    [cond_embed, bg_embeds.clone().repeat(max_cond_len - cond_embed.shape[0], 1) + -1e9],
                    dim=0,
                )
                embed_masks.append(
                    torch.cat(
                        [
                            torch.ones(cond_embed.shape[0], device=cond_embed.device),
                            torch.zeros(max_cond_len - cond_embed.shape[0], device=cond_embed.device),
                        ]
                    )
                )
            bg_embeds = bg_embeds[None, ...].repeat(B, 1, 1)
            cond_embeds = torch.cat([torch.stack(cond_embeds), bg_embeds], dim=1)
            seg_embeds = torch.stack(seg_embeds) if seg_embeds is not None else None
            embed_masks = torch.cat([torch.stack(embed_masks), torch.ones((B, 1), device=cond_embeds.device)], dim=1)
        elif task_name in ["refseg", "reaseg"]:
            local_cond_lens = [x.shape[0] for x in cond_embeds]
            if seg_embeds is None or len(seg_embeds) != B:
                raise ValueError(
                    "Ref/Rea condition expansion requires one SEG-embedding "
                    f"tensor per source sample: conditions={B}, "
                    f"SEG sources={0 if seg_embeds is None else len(seg_embeds)}"
                )
            local_row_lens = [x.shape[0] for x in seg_embeds]
            uses_v2_protocol = getattr(
                self,
                "_uses_topology_v2_source_local_conditions",
                lambda: False,
            )()
            for sample_index, (cond_embed, seg_embed) in enumerate(
                zip(cond_embeds, seg_embeds)
            ):
                if cond_embed.ndim != 2 or seg_embed.ndim != 2:
                    raise ValueError(
                        "Ref/Rea condition and SEG embeddings must be [N,D]; "
                        f"sample {sample_index} has "
                        f"{tuple(cond_embed.shape)} and {tuple(seg_embed.shape)}"
                    )
                if (
                    not uses_v2_protocol
                    and cond_embed.shape[0] != seg_embed.shape[0]
                ):
                    raise ValueError(
                        "Ref/Rea source-local condition/SEG counts differ: "
                        f"sample {sample_index} has conditions="
                        f"{cond_embed.shape[0]}, SEG tokens={seg_embed.shape[0]}"
                    )
            cond_embeds = torch.cat([torch.cat(cond_embeds), bg_embeds])
            effective_rows = (
                sum(local_row_lens)
                if uses_v2_protocol
                else sum(local_cond_lens)
            )
            cond_embeds = cond_embeds[None, ...].repeat(
                effective_rows,
                1,
                1,
            )
            seg_embeds = torch.cat(seg_embeds).unsqueeze(1) if seg_embeds is not None else None
            if uses_v2_protocol:
                embed_masks = self._topology_v2_source_local_embed_mask(
                    local_cond_lens,
                    local_row_lens,
                    device=cond_embeds.device,
                )
        else:
            raise ValueError(f"Task name {task_name} is not supported in _process_embeds")

        return (
            cond_embeds,
            seg_embeds,
            embed_masks,
            local_cond_lens,
            local_row_lens,
            global_cond_lens,
        )

    def _get_vgd_labels(self, data_samples):
        def _get_attr_from_data_samples(data_samples, attr):
            return getattr(data_samples, attr, None) if data_samples is not None else None

        class_labels = _get_attr_from_data_samples(data_samples, "class_labels")
        sampled_labels = _get_attr_from_data_samples(data_samples, "sampled_labels")
        contiguous_labels = _get_attr_from_data_samples(data_samples, "contiguous_labels")

        if class_labels is not None:
            class_labels = [class_label.cpu().numpy().tolist() for class_label in class_labels]

        if contiguous_labels is not None:
            # convert labels to contiguous labels
            assert class_labels is not None and sampled_labels is not None
            class_labels = [
                [ordered_label.index(sampled_label[label]) for label in class_label]
                for ordered_label, sampled_label, class_label in zip(contiguous_labels, sampled_labels, class_labels)
            ]
            sampled_labels = [
                [ordered_label.index(label) for label in sampled_label]
                for ordered_label, sampled_label in zip(contiguous_labels, sampled_labels)
            ]
        return class_labels, sampled_labels

    def _get_vprompt_feats_and_masks(
        self, vprompt_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
    ):
        sampled_feats = []
        sampled_masks = []
        new_sampled_labels = []

        # Process each batch
        for batch_idx, (
            batch_vprompt_feats,
            batch_vprompt_masks,
            batch_class_labels,
            batch_contiguous_labels,
        ) in enumerate(zip(vprompt_feats, vprompt_masks, class_labels, contiguous_labels)):
            batch_sampled_feats = torch.zeros(
                (len(batch_contiguous_labels), batch_vprompt_feats.shape[1]),
                dtype=batch_vprompt_feats.dtype,
                device=batch_vprompt_feats.device,
            )
            batch_sampled_masks = torch.zeros(
                (len(batch_contiguous_labels), batch_vprompt_masks.shape[1], batch_vprompt_masks.shape[2]),
                dtype=batch_vprompt_masks.dtype,
                device=batch_vprompt_masks.device,
            )
            new_batch_sampled_labels = []

            # Track used labels to avoid duplicate sampling
            used_labels = []
            used_poses = []

            for i, target_label in enumerate(batch_contiguous_labels):
                # Find matching positions across all batches
                pos_matches = [
                    (b_idx, pos)
                    for b_idx, batch_labels in enumerate(class_labels)
                    for pos, label in enumerate(batch_labels)
                    if label == target_label and (b_idx, pos) not in used_poses
                ]
                neg_matches = [
                    (b_idx, pos)
                    for b_idx, batch_labels in enumerate(class_labels)
                    for pos, label in enumerate(batch_labels)
                    if label not in used_labels and (b_idx, pos) not in used_poses and label not in batch_class_labels
                ]

                matches = pos_matches if pos_matches else neg_matches

                if matches:
                    selected_batch, selected_pos = matches[torch.randint(len(matches), (1,)).item()]
                    batch_sampled_feats[i] = vprompt_feats[selected_batch][selected_pos]
                    batch_sampled_masks[i] = vprompt_masks[selected_batch][selected_pos]
                    new_batch_sampled_labels.append(
                        sampled_labels[selected_batch][
                            contiguous_labels[selected_batch].index(class_labels[selected_batch][selected_pos])
                        ]
                    )
                    used_labels.append(class_labels[selected_batch][selected_pos])
                    used_poses.append((selected_batch, selected_pos))
                else:
                    # If no matches found, use default embedding
                    batch_sampled_feats[i] = torch.zeros_like(batch_vprompt_feats[0])
                    batch_sampled_masks[i] = torch.zeros_like(batch_vprompt_masks[0])
                    new_batch_sampled_labels.append(-1)

            sampled_feats.append(batch_sampled_feats)
            sampled_masks.append(batch_sampled_masks)
            new_sampled_labels.append(new_batch_sampled_labels)

        return sampled_feats, sampled_masks, new_sampled_labels

    def _get_attrs_from_data_samples(self, data_samples, attrs, **kwargs):
        if isinstance(attrs, str):
            attrs = [attrs]
        return [getattr(data_samples, attr, None) if data_samples is not None else None for attr in attrs]

    def _get_siglip_patch_size(self):
        """Return the runtime SigLIP patch ``(height,width)``."""

        patch_size = getattr(self.visual_encoder.config, "patch_size", None)
        if patch_size is None and hasattr(self.visual_encoder.config, "vision_config"):
            patch_size = getattr(self.visual_encoder.config.vision_config, "patch_size", None)
        if patch_size is None:
            raise ValueError("SigLIP config does not expose patch_size")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            raise ValueError(f"unsupported SigLIP patch_size: {patch_size!r}")
        if min(patch_size) <= 0:
            raise ValueError(f"SigLIP patch_size must be positive, got {patch_size}")
        return patch_size

    def _prepare_spatial_metadata(self, spatial_transforms, patch_size, device):
        """Stack per-image preprocessing metadata without a silent fallback."""

        if patch_size is None:
            raise ValueError(
                "topology group decoder requires the runtime SigLIP patch size"
            )
        if spatial_transforms is None:
            raise ValueError(
                "topology group decoder requires spatial_transforms from the dataset"
            )
        if any(transform is None for transform in spatial_transforms):
            missing = [i for i, transform in enumerate(spatial_transforms) if transform is None]
            raise ValueError(
                "missing spatial transform metadata for topology samples "
                f"at batch indices {missing}"
            )
        unrecoverable = [
            (i, transform.get("unrecoverable_reason", "unknown transform"))
            for i, transform in enumerate(spatial_transforms)
            if not bool(transform.get("recoverable", False))
        ]
        if unrecoverable:
            raise ValueError(
                "cannot align SAM and SigLIP after an unrecoverable transform: "
                + "; ".join(f"sample {i}: {reason}" for i, reason in unrecoverable)
            )
        keys = (
            "original_size",
            "original_to_sam",
            "sam_input_size",
            "sam_valid_region",
            "original_to_siglip",
            "siglip_input_size",
            "siglip_valid_region",
        )
        metadata = {}
        for key in keys:
            try:
                metadata[key] = torch.stack(
                    [
                        torch.as_tensor(transform[key], dtype=torch.float32, device=device)
                        for transform in spatial_transforms
                    ],
                    dim=0,
                )
            except KeyError as error:
                raise ValueError(
                    f"spatial transform metadata is missing required field {key!r}"
                ) from error
        metadata["siglip_patch_size"] = torch.tensor(
            patch_size,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).repeat(len(spatial_transforms), 1)
        return metadata

    @staticmethod
    def _validate_topology_postprocess_batch(
        seg_outputs,
        image_sizes,
        scaled_sizes,
    ):
        """Reject source/effective-batch truncation before task postprocess."""

        class_logits = getattr(seg_outputs, "class_queries_logits", None)
        mask_logits = getattr(seg_outputs, "masks_queries_logits", None)
        if class_logits is None or mask_logits is None:
            raise ValueError(
                "topology postprocess requires class and mask predictions"
            )
        effective_batch = int(class_logits.shape[0])
        if int(mask_logits.shape[0]) != effective_batch:
            raise ValueError(
                "topology class/mask postprocess batches differ: "
                f"{effective_batch} != {int(mask_logits.shape[0])}"
            )
        image_count = 0 if image_sizes is None else len(image_sizes)
        scaled_count = 0 if scaled_sizes is None else len(scaled_sizes)
        if image_count != effective_batch or scaled_count != effective_batch:
            raise ValueError(
                "topology postprocess metadata must already follow the "
                "effective condition-expanded batch order: "
                f"predictions={effective_batch}, image_sizes={image_count}, "
                f"scaled_sizes={scaled_count}"
            )

    @staticmethod
    def _expand_condition_metadata(values, cond_lens, field_name):
        """Repeat source metadata in the same source-major Cond order."""

        if cond_lens is None:
            return values
        if values is None:
            raise ValueError(
                f"topology condition expansion requires {field_name}"
            )
        condition_counts = [int(value) for value in cond_lens]
        if any(value <= 0 for value in condition_counts):
            raise ValueError(
                f"topology cond_lens must be positive, got {condition_counts}"
            )
        effective_batch = sum(condition_counts)
        if len(values) == effective_batch:
            return values
        if len(values) != len(condition_counts):
            raise ValueError(
                f"cannot expand {field_name}: source values={len(values)}, "
                f"cond_lens={len(condition_counts)}, "
                f"effective_batch={effective_batch}"
            )
        return [
            value
            for value, repeat_count in zip(values, condition_counts)
            for _ in range(repeat_count)
        ]

    @staticmethod
    def _validate_topology_condition_expansion(
        local_cond_lens,
        mask_labels,
        class_labels,
        local_row_lens=None,
    ):
        """Validate source-local target mappings before flattening.

        Effective rows retain the source-local SEG-token/target-row order.
        ``class_labels`` is the authoritative mask-to-condition mapping, so
        permutations and repeated local IDs are valid.  The crucial boundary
        is that a local ID may not escape its source's condition table before
        global offsets are added.
        """

        if class_labels is None:
            raise ValueError(
                "topology cond_lens expansion requires explicit class_labels"
            )
        row_counts = (
            [int(value) for value in local_cond_lens]
            if local_row_lens is None
            else [int(value) for value in local_row_lens]
        )
        if not (
            len(local_cond_lens)
            == len(row_counts)
            == len(mask_labels)
            == len(class_labels)
        ):
            raise ValueError(
                "topology cond_lens/source target batch mismatch: "
                f"cond_lens={len(local_cond_lens)}, "
                f"row_lens={len(row_counts)}, "
                f"masks={len(mask_labels)}, labels={len(class_labels)}"
            )
        for sample_index, (
            cond_len,
            row_len,
            sample_masks,
            sample_labels,
        ) in enumerate(
            zip(
                local_cond_lens,
                row_counts,
                mask_labels,
                class_labels,
            )
        ):
            cond_len = int(cond_len)
            row_len = int(row_len)
            num_masks = int(sample_masks.shape[0])
            sample_labels = torch.as_tensor(sample_labels)
            num_labels = int(sample_labels.numel())
            if cond_len <= 0:
                raise ValueError(
                    "topology cond_lens entries must be positive; "
                    f"sample {sample_index} has {cond_len}"
                )
            if row_len <= 0:
                raise ValueError(
                    "topology row_lens entries must be positive; "
                    f"sample {sample_index} has {row_len}"
                )
            if num_masks != row_len or num_labels != row_len:
                raise ValueError(
                    "topology condition expansion must preserve each "
                    "source sample's order: "
                    f"sample {sample_index} has cond_len={cond_len}, "
                    f"row_len={row_len}, masks={num_masks}, "
                    f"labels={num_labels}"
                )
            if sample_labels.ndim != 1:
                raise ValueError(
                    "topology source-local class_labels must be [N]; "
                    f"sample {sample_index} has shape "
                    f"{tuple(sample_labels.shape)}"
                )
            if sample_labels.is_floating_point():
                if not bool(torch.isfinite(sample_labels).all()):
                    raise ValueError(
                        "topology source-local class_labels must be finite; "
                        f"sample {sample_index} has {sample_labels.tolist()}"
                    )
                if not bool(torch.equal(sample_labels, sample_labels.round())):
                    raise ValueError(
                        "topology source-local class_labels must be "
                        f"integer-valued; sample {sample_index} has "
                        f"{sample_labels.tolist()}"
                    )
            local_labels = sample_labels.to(dtype=torch.long)
            if bool((local_labels < 0).any()) or bool(
                (local_labels >= cond_len).any()
            ):
                raise ValueError(
                    "topology source-local class label escapes its condition "
                    f"table: sample {sample_index}, cond_len={cond_len}, "
                    f"labels={local_labels.tolist()}"
                )

    @staticmethod
    def _topology_v2_exclude_equivalent_condition_negatives(
        embed_masks,
        local_cond_lens,
        mask_labels,
        class_labels,
        local_row_lens=None,
    ):
        """Remove only provably equivalent expressions from negative sets.

        Ref/Rea effective rows are aligned with source target rows.  Within a
        source, two rows with bit-identical GT masks are the same segmentation
        target for classification purposes.  If their local condition IDs
        differ, keeping both columns valid under single-label CE would force
        the equivalent expression to be a negative.  Each expression keeps
        its own row and positive global ID; the other equivalent columns are
        invalidated only for that row.

        We deliberately use exact mask equality.  Approximate geometric
        similarity is not sufficient evidence that two referring expressions
        denote the same GT instance.
        """

        if embed_masks is None:
            raise ValueError(
                "topology V2 Ref/Rea source-local protocol requires "
                "embed_masks"
            )
        condition_counts = [int(value) for value in local_cond_lens]
        if any(value <= 0 for value in condition_counts):
            raise ValueError(
                "topology V2 source-local Cond counts must be positive, "
                f"got {condition_counts}"
            )
        row_counts = (
            condition_counts
            if local_row_lens is None
            else [int(value) for value in local_row_lens]
        )
        if len(row_counts) != len(condition_counts) or any(
            value <= 0 for value in row_counts
        ):
            raise ValueError(
                "topology V2 source-local row counts must be positive and "
                "aligned with Cond sources: "
                f"conditions={condition_counts}, rows={row_counts}"
            )
        effective_batch = sum(row_counts)
        foreground_count = sum(condition_counts)
        expected_shape = (effective_batch, foreground_count + 1)
        if tuple(embed_masks.shape) != expected_shape:
            raise ValueError(
                "topology V2 source-local embed mask has wrong shape: "
                f"{tuple(embed_masks.shape)} != {expected_shape}"
            )
        if not (
            len(condition_counts)
            == len(row_counts)
            == len(mask_labels)
            == len(class_labels)
        ):
            raise ValueError(
                "topology V2 equivalent-expression check requires aligned "
                "source Cond, mask, and class-label batches"
            )

        refined = embed_masks.to(dtype=torch.bool).clone()
        offsets = list(accumulate([0] + condition_counts[:-1]))
        row_offset = 0
        for source_index, (
            condition_count,
            row_count,
            condition_offset,
            source_masks,
            source_labels,
        ) in enumerate(
            zip(
                condition_counts,
                row_counts,
                offsets,
                mask_labels,
                class_labels,
            )
        ):
            labels = torch.as_tensor(source_labels).reshape(-1).to(
                dtype=torch.long
            )
            if (
                int(source_masks.shape[0]) != row_count
                or labels.numel() != row_count
            ):
                raise ValueError(
                    "topology V2 equivalent-expression check requires the "
                    "declared number of effective target rows: "
                    f"source {source_index}, Cond={condition_count}, "
                    f"rows={row_count}, "
                    f"masks={int(source_masks.shape[0])}, "
                    f"labels={labels.numel()}"
                )
            if bool((labels < 0).any()) or bool(
                (labels >= condition_count).any()
            ):
                raise ValueError(
                    "topology V2 equivalent-expression labels escape the "
                    f"source Cond table: source {source_index}, "
                    f"Cond={condition_count}, labels={labels.tolist()}"
                )
            for row_index in range(row_count):
                local_positive = int(labels[row_index].item())
                global_positive = condition_offset + local_positive
                for other_index in range(row_count):
                    other_local = int(labels[other_index].item())
                    if other_local == local_positive:
                        continue
                    if torch.equal(
                        source_masks[row_index],
                        source_masks[other_index],
                    ):
                        refined[
                            row_offset + row_index,
                            condition_offset + other_local,
                        ] = False
                if not bool(
                    refined[row_offset + row_index, global_positive].item()
                ):
                    raise AssertionError(
                        "topology V2 source-local protocol invalidated its "
                        "own positive condition"
                    )
            row_offset += row_count

        if not bool(refined[:, -1].all()):
            raise AssertionError(
                "topology V2 source-local protocol must keep BG valid"
            )
        return refined

    @staticmethod
    def _topology_v2_expand_condition_targets(
        local_cond_lens,
        mask_labels,
        class_labels,
        local_row_lens=None,
    ):
        """Flatten targets and convert source-local IDs to table-global IDs."""

        condition_counts = [int(value) for value in local_cond_lens]
        if any(value <= 0 for value in condition_counts):
            raise ValueError(
                "topology V2 source-local Cond counts must be positive, "
                f"got {condition_counts}"
            )
        row_counts = (
            condition_counts
            if local_row_lens is None
            else [int(value) for value in local_row_lens]
        )
        if len(row_counts) != len(condition_counts) or any(
            value <= 0 for value in row_counts
        ):
            raise ValueError(
                "topology V2 source-local row counts must be positive and "
                "aligned with Cond sources: "
                f"conditions={condition_counts}, rows={row_counts}"
            )
        if not (
            len(condition_counts)
            == len(row_counts)
            == len(mask_labels)
            == len(class_labels)
        ):
            raise ValueError(
                "topology V2 target expansion requires aligned source "
                "Cond, mask, and class-label batches"
            )
        offsets = list(accumulate([0] + condition_counts[:-1]))
        expanded_masks = []
        expanded_labels = []
        for source_index, (
            condition_count,
            row_count,
            offset,
            source_masks,
            source_labels,
        ) in enumerate(
            zip(
                condition_counts,
                row_counts,
                offsets,
                mask_labels,
                class_labels,
            )
        ):
            labels = torch.as_tensor(source_labels).reshape(-1).to(
                dtype=torch.long
            )
            if (
                int(source_masks.shape[0]) != row_count
                or labels.numel() != row_count
            ):
                raise ValueError(
                    "topology V2 target expansion requires the declared "
                    f"number of rows: source {source_index}, "
                    f"Cond={condition_count}, rows={row_count}, "
                    f"masks={int(source_masks.shape[0])}, "
                    f"labels={labels.numel()}"
                )
            if bool((labels < 0).any()) or bool(
                (labels >= condition_count).any()
            ):
                raise ValueError(
                    "topology V2 target labels escape the source Cond table: "
                    f"source {source_index}, Cond={condition_count}, "
                    f"labels={labels.tolist()}"
                )
            expanded_masks.extend(source_masks.split(1))
            expanded_labels.extend((labels + offset).split(1))
        return expanded_masks, expanded_labels, offsets

    def _encode_segmentor_image_embeddings(
        self,
        extra_pixel_values: torch.Tensor,
        *,
        output_attentions: bool = False,
    ):
        """Run the canonical SAM-hidden-state to pixel-decoder input path.

        Both ordinary MaskLAT forward and image-only diagnostics must use this
        boundary.  In the SAM + Mask2Former configuration, the encoder's
        neck output has 256 channels and is *not* a valid substitute for the
        selected internal SAM layers after ``seg_connector``.
        """

        if self.segmentor is None or self.segmentor.encoder is None:
            raise RuntimeError(
                "segmentor image encoding requires a configured visual "
                "segmentor encoder"
            )
        if not isinstance(extra_pixel_values, torch.Tensor) or (
            extra_pixel_values.ndim != 4
        ):
            raise ValueError(
                "extra_pixel_values must be a SAM image tensor [B,C,H,W]"
            )

        seg_visual_outputs = self.segmentor.encoder(
            extra_pixel_values.to(self.segmentor.dtype),
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        seg_image_embeddings = (
            seg_visual_outputs.last_hidden_state
            if hasattr(seg_visual_outputs, "last_hidden_state")
            else seg_visual_outputs.hidden_states[-1].transpose(1, 2)
        )

        if hasattr(self, "seg_connector"):
            hidden_states = getattr(
                seg_visual_outputs,
                "hidden_states",
                None,
            )
            if hidden_states is None:
                raise RuntimeError(
                    "seg_connector requires SAM encoder hidden_states"
                )
            hidden_count = len(hidden_states)
            invalid_layers = [
                layer
                for layer in self.seg_select_layers
                if not -hidden_count <= int(layer) < hidden_count
            ]
            if invalid_layers:
                raise IndexError(
                    "seg_select_layers escape the SAM hidden-state table: "
                    f"layers={invalid_layers}, hidden_states={hidden_count}"
                )
            selected_hidden_states = [
                hidden_states[layer] for layer in self.seg_select_layers
            ]
            seg_image_embeddings = self.seg_connector(
                selected_hidden_states
            )
        elif self.segmentor.pixel_decoder is not None and hasattr(
            seg_visual_outputs,
            "feature_maps",
        ):
            seg_image_embeddings = seg_visual_outputs.feature_maps

        return seg_visual_outputs, seg_image_embeddings

    @torch.no_grad()
    def forward_image_only_stage_masks(
        self,
        extra_pixel_values: torch.Tensor,
        *,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Return st0--st9 masks through MaskLAT's canonical visual path.

        This diagnostic deliberately stops before VLM, SEG, Cond, Query
        classification, and topology logic.  Unlike calling ``MaskLATSegmentor``
        with raw SAM pixels, it preserves the trained
        ``seg_select_layers -> seg_connector -> pixel_decoder`` chain.
        """

        if self.training:
            raise RuntimeError(
                "image-only stage-mask diagnostics require model.eval()"
            )
        if self.segmentor is None:
            raise RuntimeError(
                "image-only stage-mask diagnostics require a segmentor"
            )
        if not self.extract_seg_embeds:
            raise RuntimeError(
                "image-only stage-mask diagnostics require "
                "extract_seg_embeds=True"
            )

        seg_visual_outputs, seg_image_embeddings = (
            self._encode_segmentor_image_embeddings(
                extra_pixel_values,
                output_attentions=output_attentions,
            )
        )
        if (
            self.segmentor.pixel_decoder is not None
            and not hasattr(self, "seg_connector")
            and not hasattr(seg_visual_outputs, "feature_maps")
        ):
            raise RuntimeError(
                "SAM + Mask2Former image-only diagnostics require the "
                "trained seg_connector; the 256-channel SAM neck output "
                "cannot replace its multi-scale features"
            )
        return self.segmentor.forward_image_only_stage_masks(
            image_embeddings=seg_image_embeddings,
            output_attentions=output_attentions,
        )

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        self._audit_topology_query_parameter("forward_entry_before_encoders")

        if data_samples is not None:
            data_samples = data_sample_to_device(data_samples, device=get_device())

        extra_data_dict = {}
        siglip_spatial_features = None
        siglip_valid_mask = None
        siglip_patch_size = None
        sam_spatial_features = None
        geometry_sam_spatial_features = None
        stagewise_siglip_spatial_features = None
        stagewise_sam_spatial_features = None
        geometry_fused_features_64 = None
        geometry_fused_valid_mask_64 = None
        siglip_base_tokens = None
        spatial_metadata = None
        seg_image_embeddings = None
        siglip_vlm_token_count = None
        sam_vlm_token_count = 0
        st3_group_enabled = getattr(
            self,
            "_uses_st3_group_vlm_refiner",
            lambda: False,
        )()
        st3_latent_enabled = getattr(
            self,
            "_uses_st3_latent_path",
            lambda: False,
        )()
        st3_vlm_enabled = st3_group_enabled or st3_latent_enabled
        st3_task_names = None
        st3_task_name = None
        if st3_vlm_enabled:
            st3_task_names = self._get_attrs_from_data_samples(
                data_samples,
                "task_names",
                **kwargs,
            )[0]
            st3_task_names = (
                st3_task_names if st3_task_names is not None else ["genseg"]
            )
            if (
                len(set(st3_task_names)) != 1
                or st3_task_names[0] not in DEFAULT_TASKS
            ):
                raise ValueError(
                    "st3 VLM paths require one homogeneous task per batch, "
                    f"got {st3_task_names}"
                )
            st3_task_name = st3_task_names[0]
        st3_latent_batch_enabled = st3_latent_enabled and (
            st3_task_name != "imgconv"
            or self.st3_latent_alignment_pretrain
            or self.use_st3_latent_for_imgconv
        )
        skip_st3_imgconv_latent = (
            st3_latent_enabled and not st3_latent_batch_enabled
        )
        st3_vlm_enabled = st3_group_enabled or st3_latent_batch_enabled
        stagewise_enabled = getattr(
            self,
            "_uses_stagewise_latent_trifusion",
            lambda: False,
        )()
        geometry_enabled = getattr(
            self,
            "_uses_geometry_proposal_transport",
            lambda: False,
        )()
        spatial_features_required = bool(
            getattr(
                getattr(self.segmentor, "dec_config", None),
                "use_topology_group_decoder",
                False,
            )
            or st3_vlm_enabled
        )
        if "pixel_values" in data_dict and self.visual_encoder is not None:
            visual_input = data_dict["pixel_values"].to(self.visual_encoder.dtype)
            # S2 freezes SigLIP parameters, but the checkpointed trainable
            # projector still needs an input connected to autograd.  The
            # encoder's input-grad hook supplies that connection.
            visual_outputs = self.visual_encoder(
                visual_input,
                output_hidden_states=True,
            )
            visual_hidden_states = visual_outputs.hidden_states
            selected_visual_tokens = visual_hidden_states[self.visual_select_layer][
                :, self.visual_select_indx :
            ]
            stagewise_visual_tokens = None
            if stagewise_enabled:
                hidden_count = len(visual_hidden_states)
                invalid_layers = [
                    index
                    for index in self.stagewise_visual_select_layers
                    if not -hidden_count <= index < hidden_count
                ]
                if invalid_layers:
                    raise IndexError(
                        "stagewise SigLIP layers escape hidden-state table: "
                        f"{invalid_layers}, hidden_states={hidden_count}"
                    )
                stagewise_visual_tokens = tuple(
                    visual_hidden_states[index][:, self.visual_select_indx :]
                    for index in self.stagewise_visual_select_layers
                )
            del visual_outputs, visual_hidden_states
            if spatial_features_required:
                if self.visual_select_indx != 0:
                    raise ValueError(
                        "spatial Group mode cannot discard SigLIP patch token 0 "
                        f"(visual_select_indx={self.visual_select_indx})"
                    )
                patch_height, patch_width = self._get_siglip_patch_size()
                input_height, input_width = visual_input.shape[-2:]
                patch_grid_height = input_height // patch_height
                patch_grid_width = input_width // patch_width
                expected_tokens = patch_grid_height * patch_grid_width
                if selected_visual_tokens.shape[1] != expected_tokens:
                    raise ValueError(
                        "SigLIP patch-token/grid mismatch: "
                        f"tokens={selected_visual_tokens.shape[1]}, "
                        f"input={(input_height, input_width)}, "
                        f"patch={(patch_height, patch_width)}, "
                        f"grid={(patch_grid_height, patch_grid_width)}"
                    )
                siglip_spatial_features = selected_visual_tokens.reshape(
                    selected_visual_tokens.shape[0],
                    patch_grid_height,
                    patch_grid_width,
                    selected_visual_tokens.shape[-1],
                ).permute(0, 3, 1, 2).contiguous()
                extra_data_dict["siglip_spatial_features"] = siglip_spatial_features
                if stagewise_enabled:
                    invalid_stage_tokens = [
                        stage_index
                        for stage_index, tokens in enumerate(
                            stagewise_visual_tokens
                        )
                        if tokens.shape[1] != expected_tokens
                    ]
                    if invalid_stage_tokens:
                        raise ValueError(
                            "stagewise SigLIP token/grid mismatch at stages "
                            f"{invalid_stage_tokens}"
                        )
                    stagewise_siglip_spatial_features = tuple(
                        tokens.reshape(
                            tokens.shape[0],
                            patch_grid_height,
                            patch_grid_width,
                            tokens.shape[-1],
                        ).permute(0, 3, 1, 2).contiguous()
                        for tokens in stagewise_visual_tokens
                    )
                siglip_valid_mask = torch.ones(
                    siglip_spatial_features.shape[0],
                    1,
                    patch_grid_height,
                    patch_grid_width,
                    dtype=torch.bool,
                    device=siglip_spatial_features.device,
                )
                siglip_patch_size = (patch_height, patch_width)
                extra_data_dict["siglip_valid_mask"] = siglip_valid_mask
                extra_data_dict["siglip_patch_size"] = siglip_patch_size
            if not stagewise_enabled and not geometry_enabled:
                siglip_base_tokens = self.visual_projector(
                    selected_visual_tokens
                )
                data_dict["pixel_values"] = siglip_base_tokens.to(self.llm.dtype)
                siglip_vlm_token_count = int(siglip_base_tokens.shape[1])
                self._audit_topology_query_parameter(
                    "after_siglip_and_visual_projector"
                )

        skip_segmentor_for_imgconv = (
            skip_st3_imgconv_latent
            and not self.inject_sam_vit_tokens_to_vlm
        )
        if skip_segmentor_for_imgconv:
            # ImgConv has no mask supervision.  In the opt-out S3 protocol it
            # keeps the ordinary SigLIP/text LM path, while SAM, the proposal
            # builder and all latent transport are uniformly absent on every
            # rank processing this homogeneous source batch.
            data_dict["extra_pixel_values"] = None
        elif "extra_pixel_values" in data_dict and self.segmentor is not None:
            if self.extract_seg_embeds:
                segmentor_context = (
                    torch.no_grad()
                    if getattr(self, "st3_latent_alignment_pretrain", False)
                    else torch.set_grad_enabled(torch.is_grad_enabled())
                )
                with segmentor_context:
                    seg_visual_outputs, seg_image_embeddings = (
                        self._encode_segmentor_image_embeddings(
                            data_dict["extra_pixel_values"],
                            output_attentions=False,
                        )
                    )
                if st3_vlm_enabled:
                    sam_spatial_features = getattr(
                        seg_visual_outputs,
                        "last_hidden_state",
                        None,
                    )
                    if (
                        not isinstance(sam_spatial_features, torch.Tensor)
                        or sam_spatial_features.ndim != 4
                    ):
                        raise ValueError(
                            "st3 Group-in-VLM requires the SAM encoder to "
                            "expose last_hidden_state [B,C,H,W]"
                        )
                    if stagewise_enabled:
                        sam_hidden_states = getattr(
                            seg_visual_outputs,
                            "hidden_states",
                            None,
                        )
                        if sam_hidden_states is None:
                            raise ValueError(
                                "stagewise fusion requires SAM hidden states"
                            )
                        hidden_count = len(sam_hidden_states)
                        invalid_layers = [
                            index
                            for index in self.stagewise_sam_select_layers
                            if not -hidden_count <= index < hidden_count
                        ]
                        if invalid_layers:
                            raise IndexError(
                                "stagewise SAM layers escape hidden-state table: "
                                f"{invalid_layers}, hidden_states={hidden_count}"
                            )
                        stagewise_sam_spatial_features = tuple(
                            sam_hidden_states[index].permute(0, 3, 1, 2).contiguous()
                            for index in self.stagewise_sam_select_layers
                        )
                    elif geometry_enabled:
                        sam_hidden_states = getattr(
                            seg_visual_outputs,
                            "hidden_states",
                            None,
                        )
                        if sam_hidden_states is None:
                            raise ValueError(
                                "geometry fusion requires SAM ViT hidden states"
                            )
                        geometry_sam_spatial_features = (
                            sam_hidden_states[-1]
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                extra_pixel_values = None
                if (
                    hasattr(self, "seg_projector")
                    and self.inject_sam_vit_tokens_to_vlm
                ):
                    extra_pixel_values = self.seg_projector(seg_visual_outputs.hidden_states[self.visual_select_layer])
                    extra_pixel_values = extra_pixel_values.to(self.llm.dtype)
                    sam_vlm_token_count = int(extra_pixel_values.shape[1])

                # here, extra_pixel_values is seg_projector output
                data_dict["extra_pixel_values"] = extra_pixel_values
                # Preserve the SigLIP spatial tensors captured from the single
                # visual-encoder pass above.  Replacing this dictionary here
                # would make topology mode fail later with missing spatial
                # evidence whenever SAM embeddings are pre-extracted.
                extra_data_dict.update(
                    {
                        "extra_pixel_values": None,
                        "seg_image_embeddings": seg_image_embeddings,
                    }
                )
                if stagewise_enabled:
                    if (
                        stagewise_siglip_spatial_features is None
                        or stagewise_sam_spatial_features is None
                        or siglip_patch_size is None
                    ):
                        raise ValueError(
                            "stagewise fusion is missing SigLIP/SAM feature levels"
                        )
                    spatial_transforms = self._get_attrs_from_data_samples(
                        data_samples,
                        "spatial_transforms",
                        **kwargs,
                    )[0]
                    spatial_metadata = self._prepare_spatial_metadata(
                        spatial_transforms,
                        siglip_patch_size,
                        siglip_spatial_features.device,
                    )
                    fused_pixel_values = self.stagewise_visual_fusion(
                        siglip_spatial_features=(
                            stagewise_siglip_spatial_features[-1]
                        ),
                        sam_spatial_features=(
                            stagewise_sam_spatial_features[-1]
                        ),
                        spatial_metadata=spatial_metadata,
                        siglip_valid_mask=siglip_valid_mask,
                    )
                    data_dict["pixel_values"] = fused_pixel_values.to(
                        self.llm.dtype
                    )
                    siglip_vlm_token_count = int(fused_pixel_values.shape[1])
                    self._audit_topology_query_parameter(
                        "after_stagewise_sam_siglip_fusion"
                    )
                elif geometry_enabled:
                    if (
                        siglip_spatial_features is None
                        or geometry_sam_spatial_features is None
                        or siglip_patch_size is None
                    ):
                        raise ValueError(
                            "geometry fusion is missing SigLIP/SAM spatial features"
                        )
                    spatial_transforms = self._get_attrs_from_data_samples(
                        data_samples,
                        "spatial_transforms",
                        **kwargs,
                    )[0]
                    spatial_metadata = self._prepare_spatial_metadata(
                        spatial_transforms,
                        siglip_patch_size,
                        siglip_spatial_features.device,
                    )
                    geometry_fusion = self.geometry_visual_fusion(
                        siglip_spatial_features=siglip_spatial_features,
                        sam_spatial_features=geometry_sam_spatial_features,
                        spatial_metadata=spatial_metadata,
                        siglip_valid_mask=siglip_valid_mask,
                    )
                    geometry_fused_features_64 = (
                        geometry_fusion.fused_features_64
                    )
                    geometry_fused_valid_mask_64 = (
                        geometry_fusion.fused_valid_mask_64
                    )
                    data_dict["pixel_values"] = geometry_fusion.vlm_tokens.to(
                        self.llm.dtype
                    )
                    siglip_vlm_token_count = int(
                        geometry_fusion.vlm_tokens.shape[1]
                    )
                    self._audit_topology_query_parameter(
                        "after_geometry_aligned_sam_siglip_fusion"
                    )
                del seg_visual_outputs
                self._audit_topology_query_parameter(
                    "after_segmentor_encoder_and_connector"
                )
            else:
                # here, extra_pixel_values is image_processor output
                extra_data_dict.update(
                    {
                        "extra_pixel_values": data_dict["extra_pixel_values"].to(self.segmentor.dtype),
                        "seg_image_embeddings": None,
                    }
                )
                data_dict["extra_pixel_values"] = None
        else:
            data_dict["extra_pixel_values"] = None

        if st3_vlm_enabled:
            task_name = st3_task_name
            if self.st3_group_alignment_pretrain and task_name != "imgconv":
                raise ValueError(
                    "S2-G alignment accepts only homogeneous imgconv batches, "
                    f"got {task_name!r}"
                )
            if self.st3_latent_alignment_pretrain and task_name != "imgconv":
                raise ValueError(
                    "latent S2 alignment accepts only homogeneous imgconv "
                    f"batches, got {task_name!r}"
                )
            build_group_tokens = (
                task_name != "imgconv"
                or self.st3_group_alignment_pretrain
                or self.st3_latent_alignment_pretrain
                # Ordinary S3 ImgConv contributes latent carriers only when
                # the experiment explicitly keeps that protocol enabled.
                or st3_latent_batch_enabled
            )
            if build_group_tokens:
                if (
                    siglip_spatial_features is None
                    or siglip_valid_mask is None
                    or siglip_patch_size is None
                    or sam_spatial_features is None
                    or seg_image_embeddings is None
                ):
                    raise ValueError(
                        "st3 Group-in-VLM requires the existing SigLIP and SAM "
                        "encoder outputs"
                    )
                if spatial_metadata is None:
                    spatial_transforms = self._get_attrs_from_data_samples(
                        data_samples,
                        "spatial_transforms",
                        **kwargs,
                    )[0]
                    spatial_metadata = self._prepare_spatial_metadata(
                        spatial_transforms,
                        siglip_patch_size,
                        siglip_spatial_features.device,
                    )
                if st3_group_enabled:
                    st3_bundle = self.segmentor.prepare_st3_group_execution(
                        image_embeddings=seg_image_embeddings,
                        sam_spatial_features=sam_spatial_features,
                        siglip_spatial_features=siglip_spatial_features,
                        siglip_valid_mask=siglip_valid_mask,
                        spatial_metadata=spatial_metadata,
                        output_attentions=False,
                    )
                else:
                    st3_bundle = self.segmentor.prepare_st3_latent_execution(
                        image_embeddings=seg_image_embeddings,
                        sam_spatial_features=sam_spatial_features,
                        siglip_spatial_features=siglip_spatial_features,
                        siglip_valid_mask=siglip_valid_mask,
                        spatial_metadata=spatial_metadata,
                        stagewise_sam_spatial_features=(
                            stagewise_sam_spatial_features
                        ),
                        stagewise_siglip_spatial_features=(
                            stagewise_siglip_spatial_features
                        ),
                        geometry_fused_features_64=(
                            geometry_fused_features_64
                        ),
                        geometry_fused_valid_mask_64=(
                            geometry_fused_valid_mask_64
                        ),
                        output_attentions=False,
                        freeze_proposal_source=(
                            getattr(
                                self,
                                "st3_latent_alignment_pretrain",
                                False,
                            )
                            or self.latent_s2_post_vlm_decoder_finetune
                        ),
                    )
                group_valid_mask = (
                    st3_bundle.proposal.packed_group_valid_mask
                )
                if task_name == "imgconv" and not (
                    self.st3_group_alignment_pretrain
                    or self.st3_latent_alignment_pretrain
                ):
                    image_files = self._get_attrs_from_data_samples(
                        data_samples,
                        "image_files",
                        **kwargs,
                    )[0]
                    group_valid_mask = _mask_text_only_imgconv_groups(
                        group_valid_mask,
                        image_files,
                    )
                data_dict["group_values"] = (
                    st3_bundle.proposal.packed_group_tokens
                )
                data_dict["group_valid_mask"] = group_valid_mask
                if stagewise_enabled:
                    stage_vlm_tokens = getattr(
                        st3_bundle.proposal,
                        "stage_vlm_tokens",
                        None,
                    )
                    if stage_vlm_tokens is None or len(stage_vlm_tokens) != 4:
                        raise RuntimeError(
                            "stagewise proposal must expose z0--z3 VLM tokens"
                        )
                    extra_data_dict["stagewise_deepstack_latents"] = tuple(
                        stage_vlm_tokens[:3]
                    )
                if self._uses_st123_latent_deepstack():
                    stage_vlm_tokens = getattr(
                        st3_bundle.proposal, "stage_vlm_tokens", None
                    )
                    if stage_vlm_tokens is None or len(stage_vlm_tokens) != 3:
                        raise RuntimeError(
                            "st123 proposal must expose exactly E1, E2, E3"
                        )
                    extra_data_dict["st123_deepstack_latents"] = tuple(
                        stage_vlm_tokens
                    )
                if self.st3_group_alignment_pretrain:
                    extra_data_dict["st3_group_alignment_only"] = True
                elif self.st3_latent_alignment_pretrain:
                    extra_data_dict["st3_latent_alignment_only"] = True
                elif st3_latent_batch_enabled:
                    if task_name == "imgconv":
                        extra_data_dict["st3_latent_lm_only"] = True
                    else:
                        extra_data_dict["st3_latent_bundle"] = st3_bundle
                else:
                    extra_data_dict["st3_group_bundle"] = st3_bundle

                if self.st3_latent_alignment_pretrain or (
                    st3_latent_batch_enabled and task_name == "imgconv"
                ):
                    # The 64 carriers are already held by group_values.  An
                    # LM-only batch never consumes decoder context, stage-3
                    # state, or spatial evidence after this point.  Dropping
                    # those references before the LLM materially lowers the
                    # first-batch memory peak while autograd keeps precisely
                    # the tensors required by the latent-builder backward.
                    for unused_key in (
                        "extra_pixel_values",
                        "seg_image_embeddings",
                        "siglip_spatial_features",
                        "siglip_valid_mask",
                        "siglip_patch_size",
                    ):
                        extra_data_dict.pop(unused_key, None)
                    st3_bundle = None
                    sam_spatial_features = None
                    geometry_sam_spatial_features = None
                    seg_image_embeddings = None
                    siglip_spatial_features = None
                    siglip_valid_mask = None
                    stagewise_siglip_spatial_features = None
                    stagewise_sam_spatial_features = None
                    geometry_fused_features_64 = None
                    geometry_fused_valid_mask_64 = None
                    siglip_base_tokens = None
                    selected_visual_tokens = None

        if data_dict.get("vprompt_masks", None) is not None and hasattr(self, "vision_sampler"):
            vprompt_masks = data_dict.pop("vprompt_masks")
            class_labels, contiguous_labels = self._get_vgd_labels(data_samples)
            sampled_labels = self._get_attrs_from_data_samples(data_samples, ["sampled_labels"])[0]
            sampler_features = data_dict.get(self.sampler_input_feat)
            if sampler_features is None:
                raise RuntimeError(
                    "vision sampler input is unavailable: "
                    f"sampler_input_feat={self.sampler_input_feat!r}, "
                    "inject_sam_vit_tokens_to_vlm="
                    f"{self.inject_sam_vit_tokens_to_vlm}. When direct SAM "
                    "VLM tokens are disabled, configure the sampler to use "
                    "projected SigLIP pixel_values."
                )
            sampled_feats = self.vision_sampler(sampler_features, vprompt_masks)
            assert all(
                sampled_feat is not None for sampled_feat in sampled_feats
            ), f"{sampler_features}, {vprompt_masks}"
            vprompt_feats, vprompt_masks, new_sampled_labels = self._get_vprompt_feats_and_masks(
                sampled_feats, vprompt_masks, class_labels, contiguous_labels, sampled_labels
            )
            data_dict["vprompt_feats"] = vprompt_feats
            kwargs["vprompt_masks"] = vprompt_masks
            kwargs["sampled_labels"] = sampled_labels

        if self.llm is not None:
            if siglip_vlm_token_count is not None:
                direct_sam_tokens = data_dict.get("extra_pixel_values")
                inject_sam_tokens = getattr(
                    self,
                    "inject_sam_vit_tokens_to_vlm",
                    True,
                )
                if not inject_sam_tokens and (
                    direct_sam_tokens is not None
                ):
                    raise RuntimeError(
                        "SAM-ViT tokens reached multimodal preparation while "
                        "inject_sam_vit_tokens_to_vlm=False"
                    )
                expected_num_image_tokens = siglip_vlm_token_count
                if inject_sam_tokens:
                    expected_num_image_tokens += sam_vlm_token_count
                data_dict["expected_num_image_tokens"] = (
                    expected_num_image_tokens
                )
            data_dict = prepare_inputs_labels_for_multimodal(llm=self.llm, **data_dict)
            self._audit_topology_query_parameter(
                "after_multimodal_input_preparation"
            )

        data_dict.update(extra_data_dict)

        if mode == "loss":
            result = self.compute_loss(data_dict, data_samples, **kwargs)
            self._audit_loss_output_finite(result)
        elif mode == "predict":
            result = self.predict(data_dict, data_samples, **kwargs)
        elif mode == "tensor":
            result = self._forward(data_dict, data_samples, **kwargs)
        else:
            raise NotImplementedError
        self._finish_finite_diagnostics_once()
        return result

    def _forward(
        self,
        data_dict,
        data_samples=None,
        **kwargs,
    ):
        return_raw_seg_outputs = bool(
            kwargs.pop("return_raw_seg_outputs", False)
        )
        return_variant_diagnostic_stages = bool(
            kwargs.pop("return_variant_diagnostic_stages", False)
        )
        if data_dict.get("inputs_embeds", None) is not None:
            data_dict["input_ids"] = None

        cond_ids = data_dict.pop("cond_ids", None)
        seg_ids = data_dict.pop("seg_ids", None)
        group_ids = data_dict.pop("group_ids", None)
        st3_group_bundle = data_dict.pop("st3_group_bundle", None)
        st3_latent_bundle = data_dict.pop("st3_latent_bundle", None)
        st123_deepstack_latents = data_dict.pop("st123_deepstack_latents", None)
        if st123_deepstack_latents is not None and not (
            self._uses_st123_latent_deepstack()
        ):
            raise RuntimeError("st123 tensors reached a non-st123 model")
        stagewise_deepstack_latents = data_dict.pop(
            "stagewise_deepstack_latents",
            None,
        )
        if stagewise_deepstack_latents is not None and not (
            self._uses_stagewise_latent_trifusion()
        ):
            raise RuntimeError(
                "stagewise deepstack tensors reached a non-stagewise model"
            )
        if st3_group_bundle is not None and st3_latent_bundle is not None:
            raise RuntimeError("hard-Group and latent st3 bundles cannot coexist")
        st3_group_alignment_only = data_dict.pop(
            "st3_group_alignment_only",
            False,
        )
        if not isinstance(st3_group_alignment_only, bool):
            raise TypeError("st3_group_alignment_only must be boolean")
        st3_latent_alignment_only = data_dict.pop(
            "st3_latent_alignment_only",
            False,
        )
        if not isinstance(st3_latent_alignment_only, bool):
            raise TypeError("st3_latent_alignment_only must be boolean")
        if st3_group_alignment_only and st3_latent_alignment_only:
            raise RuntimeError("hard-Group and latent alignment markers coexist")
        st3_latent_lm_only = data_dict.pop("st3_latent_lm_only", False)
        if not isinstance(st3_latent_lm_only, bool):
            raise TypeError("st3_latent_lm_only must be boolean")
        if st3_latent_lm_only and (
            st3_group_bundle is not None
            or st3_latent_bundle is not None
            or st3_group_alignment_only
            or st3_latent_alignment_only
        ):
            raise RuntimeError("latent LM-only marker cannot coexist with st3 bundles")
        extra_pixel_values = data_dict.pop("extra_pixel_values", None)
        seg_image_embeddings = data_dict.pop("seg_image_embeddings", None)
        siglip_spatial_features = data_dict.pop("siglip_spatial_features", None)
        siglip_valid_mask = data_dict.pop("siglip_valid_mask", None)
        siglip_patch_size = data_dict.pop("siglip_patch_size", None)
        task_names, image_size, scaled_size, mask_labels, class_labels = self._get_attrs_from_data_samples(
            data_samples,
            [
                "task_names",
                "image_sizes",
                "scaled_sizes",
                "mask_labels",
                "class_labels",
            ],
            **kwargs,
        )
        task_names = task_names if task_names is not None else ["genseg"]
        assert (
            len(set(task_names)) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        seg_embeds = None
        cond_embeds = None
        embed_masks = None
        llm_outputs = None
        seg_outputs = None
        raw_seg_outputs = None
        local_cond_lens = None
        local_row_lens = None
        global_cond_lens = None
        spatial_metadata = None

        if self.llm is not None:
            if st3_group_bundle is not None or st3_group_alignment_only:
                if group_ids is None:
                    raise ValueError(
                        "st3 Group-in-VLM bundle is missing sequence group_ids"
                    )
                llm_outputs = self._run_llm_with_group_attention(
                    data_dict,
                    group_ids,
                )
            elif (
                st3_latent_bundle is not None
                or st3_latent_alignment_only
                or st3_latent_lm_only
            ):
                if group_ids is None:
                    raise ValueError(
                        "st3 proposal latent bundle is missing latent token ids"
                    )
                # The 64 carriers have already undergone explicit self-
                # attention before insertion.  The VLM therefore keeps its
                # ordinary causal backend; text/SEG read the latent prefix.
                latent_llm_inputs = dict(data_dict)
                if self._uses_stagewise_latent_trifusion():
                    if stagewise_deepstack_latents is None:
                        raise ValueError(
                            "stagewise latent execution is missing z0--z2"
                        )
                    latent_llm_inputs.update(
                        masklat_stagewise_latents=stagewise_deepstack_latents,
                        masklat_stagewise_group_ids=group_ids,
                    )
                if self._uses_st123_latent_deepstack():
                    if st123_deepstack_latents is None:
                        raise ValueError("st123 execution is missing E1, E2, E3")
                    latent_llm_inputs.update(
                        masklat_st123_latents=st123_deepstack_latents,
                        masklat_st123_group_ids=group_ids,
                    )
                llm_outputs = self.llm(
                    **latent_llm_inputs,
                    output_hidden_states=True,
                )
            else:
                if group_ids is not None:
                    raise ValueError(
                        "group_ids were produced without an st3 execution bundle"
                    )
                llm_outputs = self.llm(
                    **data_dict,
                    output_hidden_states=True,
                )
            self._audit_topology_query_parameter("after_llm_forward")

        if st3_group_alignment_only:
            if not self.st3_group_alignment_pretrain:
                raise RuntimeError(
                    "an S2-G alignment batch reached a model that was not "
                    "configured for st3_group_alignment_pretrain"
                )
            if st3_group_bundle is not None:
                raise RuntimeError(
                    "S2-G alignment must not enter the segmentation finish path"
                )
            if task_names[0] != "imgconv":
                raise RuntimeError(
                    "S2-G alignment may return an LM-only loss only for imgconv"
                )
            if llm_outputs is None:
                raise RuntimeError("S2-G alignment requires an LLM output")
            result = (llm_outputs, None)
            return (
                (*result, None)
                if return_raw_seg_outputs
                else result
            )

        if st3_latent_alignment_only:
            if not self.st3_latent_alignment_pretrain:
                raise RuntimeError(
                    "a latent S2 alignment batch reached a model that was not "
                    "configured for st3_latent_alignment_pretrain"
                )
            if st3_latent_bundle is not None:
                raise RuntimeError(
                    "latent S2 alignment must not enter the post-VLM "
                    "segmentation transport path"
                )
            if task_names[0] != "imgconv":
                raise RuntimeError(
                    "latent S2 alignment may return an LM-only loss only for "
                    "imgconv"
                )
            if llm_outputs is None:
                raise RuntimeError("latent S2 alignment requires an LLM output")
            result = (llm_outputs, None)
            return (
                (*result, None)
                if return_raw_seg_outputs
                else result
            )

        if st3_latent_lm_only:
            if not self._uses_st3_latent_path():
                raise RuntimeError(
                    "latent LM-only batch reached a model without a latent path"
                )
            if task_names[0] != "imgconv":
                raise RuntimeError(
                    "latent LM-only execution is restricted to imgconv"
                )
            if llm_outputs is None:
                raise RuntimeError("latent LM-only execution requires an LLM output")
            result = (llm_outputs, None)
            return (*result, None) if return_raw_seg_outputs else result

        if self.segmentor is None or self.segmentor.decoder is None:
            result = (llm_outputs, None)
            return (*result, None) if return_raw_seg_outputs else result

        if llm_outputs is not None:
            llm_hidden_states = llm_outputs.hidden_states
            llm_last_hidden_state = llm_hidden_states[-1]
            llm_embeds = self.llm_projector(llm_last_hidden_state)
            if st3_group_bundle is not None:
                if return_variant_diagnostic_stages:
                    raise ValueError(
                        "base-model variant diagnostics cannot be mixed "
                        "with the st3 Group-in-VLM training path"
                    )
                if cond_ids is None or seg_ids is None:
                    raise ValueError(
                        "st3 Group-in-VLM requires both Cond and SEG token IDs"
                    )
                condition_sources = extract_indexed_embeddings(
                    llm_embeds,
                    cond_ids,
                    field_name="st3 Cond",
                )
                segment_sources = extract_indexed_embeddings(
                    llm_embeds,
                    seg_ids,
                    field_name="st3 SEG",
                )
                seg_outputs, st3_scope = (
                    self._finish_st3_from_source_embeddings(
                        bundle=st3_group_bundle,
                        projected_hidden_states=llm_embeds,
                        group_ids=group_ids,
                        condition_sources=condition_sources,
                        segment_sources=segment_sources,
                        task_name=task_names[0],
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                    )
                )
                raw_seg_outputs = seg_outputs
                if kwargs.pop("do_postprocess", False):
                    image_size = self._expand_condition_metadata(
                        image_size,
                        st3_scope.segment_counts,
                        "image_sizes",
                    )
                    scaled_size = self._expand_condition_metadata(
                        scaled_size,
                        st3_scope.segment_counts,
                        "scaled_sizes",
                    )
                    self._validate_topology_postprocess_batch(
                        seg_outputs,
                        image_size,
                        scaled_size,
                    )
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )
                result = (llm_outputs, seg_outputs)
                return (
                    (*result, raw_seg_outputs)
                    if return_raw_seg_outputs
                    else result
                )
            if st3_latent_bundle is not None:
                if return_variant_diagnostic_stages:
                    raise ValueError(
                        "base-model diagnostics cannot be mixed with the "
                        "st3 proposal latent path"
                    )
                if cond_ids is None or seg_ids is None:
                    raise ValueError(
                        "the proposal latent path requires Cond and SEG ids"
                    )
                condition_sources = extract_indexed_embeddings(
                    llm_embeds,
                    cond_ids,
                    field_name="st3 latent Cond",
                )
                segment_sources = extract_indexed_embeddings(
                    llm_embeds,
                    seg_ids,
                    field_name="st3 latent SEG",
                )
                seg_outputs, st3_scope = (
                    self._finish_st3_latent_from_source_embeddings(
                        bundle=st3_latent_bundle,
                        projected_hidden_states=llm_embeds,
                        group_ids=group_ids,
                        condition_sources=condition_sources,
                        segment_sources=segment_sources,
                        task_name=task_names[0],
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                    )
                )
                raw_seg_outputs = seg_outputs
                if kwargs.pop("do_postprocess", False):
                    image_size = self._expand_condition_metadata(
                        image_size,
                        st3_scope.segment_counts,
                        "image_sizes",
                    )
                    scaled_size = self._expand_condition_metadata(
                        scaled_size,
                        st3_scope.segment_counts,
                        "scaled_sizes",
                    )
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )
                result = (llm_outputs, seg_outputs)
                return (
                    (*result, raw_seg_outputs)
                    if return_raw_seg_outputs
                    else result
                )
            if cond_ids is not None:
                cond_embeds = self._get_index_embeds(llm_embeds, cond_ids)
            if seg_ids is not None:
                seg_embeds = self._get_index_embeds(llm_embeds, seg_ids)
            if cond_embeds is not None and seg_embeds is not None:
                (
                    cond_embeds,
                    seg_embeds,
                    embed_masks,
                    local_cond_lens,
                    local_row_lens,
                    global_cond_lens,
                ) = self._process_embeds(
                    cond_embeds,
                    seg_embeds,
                    task_names[0],
                )

        topology_v2_source_local = getattr(
            self,
            "_uses_topology_v2_source_local_conditions",
            lambda: False,
        )()
        if (
            getattr(self.segmentor.dec_config, "use_topology_group_decoder", False)
            and local_cond_lens is not None
            and mask_labels is not None
        ):
            self._validate_topology_condition_expansion(
                local_cond_lens,
                mask_labels,
                class_labels,
                local_row_lens=(
                    local_row_lens
                    if topology_v2_source_local
                    else local_cond_lens
                ),
            )
            if topology_v2_source_local:
                embed_masks = (
                    self._topology_v2_exclude_equivalent_condition_negatives(
                        embed_masks,
                        local_cond_lens,
                        mask_labels,
                        class_labels,
                        local_row_lens=local_row_lens,
                    )
                )

        if (local_cond_lens or global_cond_lens) is not None and mask_labels is not None:
            if topology_v2_source_local:
                if global_cond_lens is not None:
                    raise ValueError(
                        "topology V2 Ref/Rea source-local protocol must not "
                        "gather other ranks' Cond tables"
                    )
                (
                    mask_labels,
                    class_labels,
                    _,
                ) = self._topology_v2_expand_condition_targets(
                    local_cond_lens,
                    mask_labels,
                    class_labels,
                    local_row_lens=local_row_lens,
                )
            else:
                cur_rank = get_rank()
                mask_labels = list(chain(*[mask_label.split(1) for mask_label in mask_labels]))
                if global_cond_lens is not None:
                    label_offsets = (
                        list(accumulate([sum(torch.cat(global_cond_lens[:cur_rank])).item()] + local_cond_lens[:-1]))
                        if cur_rank > 0
                        else list(accumulate([0] + local_cond_lens[:-1]))
                    )
                else:
                    label_offsets = list(accumulate([0] + local_cond_lens[:-1]))

                class_labels = list(
                    chain(
                        *[
                            (class_label + label_offset).split(1)
                            for label_offset, class_label in zip(label_offsets, class_labels)
                        ]
                    )
                )

        if seg_embeds is not None or llm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                if getattr(self.segmentor.dec_config, "use_topology_group_decoder", False):
                    if siglip_spatial_features is None:
                        raise ValueError(
                            "topology group decoder requires SigLIP spatial "
                            "features from the existing visual-encoder pass"
                        )
                    spatial_transforms = self._get_attrs_from_data_samples(
                        data_samples, "spatial_transforms", **kwargs
                    )[0]
                    spatial_metadata = self._prepare_spatial_metadata(
                        spatial_transforms,
                        siglip_patch_size,
                        siglip_spatial_features.device,
                    )
                try:
                    seg_outputs = self.segmentor(
                        pixel_values=extra_pixel_values,
                        image_embeddings=seg_image_embeddings,
                        cond_embeddings=cond_embeds,
                        seg_embeddings=seg_embeds,
                        embed_masks=embed_masks,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                        cond_lens=(
                            local_row_lens
                            if topology_v2_source_local
                            else local_cond_lens
                        ),
                        task_name=task_names[0],
                        siglip_spatial_features=siglip_spatial_features,
                        siglip_valid_mask=siglip_valid_mask,
                        spatial_metadata=spatial_metadata,
                        return_variant_diagnostic_stages=(
                            return_variant_diagnostic_stages
                        ),
                        return_dict=True,
                    )
                except FloatingPointError as error:
                    image_files = (
                        getattr(data_samples, "image_files", None)
                        if data_samples is not None
                        else None
                    )
                    image_preview = (
                        list(image_files[:4])
                        if isinstance(image_files, (list, tuple))
                        else image_files
                    )
                    raise FloatingPointError(
                        "MaskLAT segmentation forward produced a non-finite "
                        f"tensor: rank={get_rank()}, task={task_names[0]!r}, "
                        f"images={image_preview!r}, "
                        f"seg_shape={None if seg_embeds is None else tuple(seg_embeds.shape)}, "
                        f"cond_shape={None if cond_embeds is None else tuple(cond_embeds.shape)}, "
                        f"local_cond_lens={local_cond_lens}, "
                        f"local_row_lens={local_row_lens}; {error}"
                    ) from error
                raw_seg_outputs = seg_outputs
                if kwargs.pop("do_postprocess", False):
                    if getattr(
                        self.segmentor.dec_config,
                        "use_topology_group_decoder",
                        False,
                    ):
                        image_size = self._expand_condition_metadata(
                            image_size,
                            (
                                local_row_lens
                                if topology_v2_source_local
                                else local_cond_lens
                            ),
                            "image_sizes",
                        )
                        scaled_size = self._expand_condition_metadata(
                            scaled_size,
                            (
                                local_row_lens
                                if topology_v2_source_local
                                else local_cond_lens
                            ),
                            "scaled_sizes",
                        )
                        self._validate_topology_postprocess_batch(
                            seg_outputs,
                            image_size,
                            scaled_size,
                        )
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )

        result = (llm_outputs, seg_outputs)
        return (
            (*result, raw_seg_outputs)
            if return_raw_seg_outputs
            else result
        )

    @torch.no_grad()
    def predict(self, data_dict, data_samples=None, **kwargs):
        return_raw_seg_outputs = bool(
            kwargs.pop("return_raw_seg_outputs", False)
        )
        return_variant_diagnostic_stages = bool(
            kwargs.pop("return_variant_diagnostic_stages", False)
        )
        st3_group_bundle = data_dict.pop("st3_group_bundle", None)
        st3_latent_bundle = data_dict.pop("st3_latent_bundle", None)
        st123_deepstack_latents = data_dict.pop("st123_deepstack_latents", None)
        if st123_deepstack_latents is not None and not (
            self._uses_st123_latent_deepstack()
        ):
            raise RuntimeError("st123 tensors reached a non-st123 model")
        stagewise_deepstack_latents = data_dict.pop(
            "stagewise_deepstack_latents",
            None,
        )
        st3_latent_alignment_only = data_dict.pop(
            "st3_latent_alignment_only",
            False,
        )
        if not isinstance(st3_latent_alignment_only, bool):
            raise TypeError("st3_latent_alignment_only must be boolean")
        st3_latent_lm_only = data_dict.pop("st3_latent_lm_only", False)
        if not isinstance(st3_latent_lm_only, bool):
            raise TypeError("st3_latent_lm_only must be boolean")
        if st3_group_bundle is not None and st3_latent_bundle is not None:
            raise RuntimeError("hard-Group and latent st3 bundles cannot coexist")
        if st3_latent_lm_only and (
            st3_group_bundle is not None or st3_latent_bundle is not None
        ):
            raise RuntimeError("latent LM-only marker cannot coexist with st3 bundles")
        st3_any_bundle = (
            st3_group_bundle
            if st3_group_bundle is not None
            else st3_latent_bundle
        )
        group_ids = data_dict.pop("group_ids", None)

        if data_dict.get("inputs_embeds", None) is not None:
            data_dict["input_ids"] = None

        if data_dict.get("labels", None) is not None:
            data_dict["labels"] = None

        if data_dict.get("position_ids", None) is not None:
            data_dict["position_ids"] = None

        if (
            st3_any_bundle is None
            and not st3_latent_alignment_only
            and not st3_latent_lm_only
            and data_dict.get("attention_mask", None) is not None
        ):
            data_dict["attention_mask"] = None

        seg_ids = data_dict.pop("seg_ids", None)
        extra_pixel_values = data_dict.pop("extra_pixel_values", None)
        seg_image_embeddings = data_dict.pop("seg_image_embeddings", None)
        siglip_spatial_features = data_dict.pop("siglip_spatial_features", None)
        siglip_valid_mask = data_dict.pop("siglip_valid_mask", None)
        siglip_patch_size = data_dict.pop("siglip_patch_size", None)
        input_cond_ids = data_dict.pop("cond_ids", None)
        task_names, image_size, scaled_size = self._get_attrs_from_data_samples(
            data_samples,
            ["task_names", "image_sizes", "scaled_sizes"],
            **kwargs,
        )
        task_names = task_names if task_names is not None else ["genseg"]
        assert (
            len(task_names) == 1 and task_names[0] in DEFAULT_TASKS
        ), f"Task name {task_names} is not in {DEFAULT_TASKS}"

        generation_config = kwargs.pop("generation_config", None)
        stopping_criteria = kwargs.pop("stopping_criteria", None)

        seg_embeds = None
        cond_embeds = None
        llm_outputs = None
        seg_outputs = None
        raw_seg_outputs = None
        local_cond_lens = None
        local_row_lens = None
        topology_v2_source_local = getattr(
            self,
            "_uses_topology_v2_source_local_conditions",
            lambda: False,
        )()

        if self.llm is not None:
            if st3_group_bundle is not None:
                if group_ids is None:
                    raise ValueError(
                        "st3 Group-in-VLM prediction is missing group_ids"
                    )
                num_beams = int(
                    getattr(
                        generation_config,
                        "num_beams",
                        getattr(self.llm.generation_config, "num_beams", 1),
                    )
                )
                if num_beams != 1:
                    raise ValueError(
                        "st3 Group-in-VLM prediction currently requires "
                        "num_beams=1 so generated SEG rows preserve source order"
                    )
                with self._group_attention_generation_context(group_ids):
                    llm_outputs = self.llm.generate(
                        **data_dict,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        generation_config=generation_config,
                        stopping_criteria=stopping_criteria,
                    )
            elif (
                st3_latent_bundle is not None
                or st3_latent_alignment_only
                or st3_latent_lm_only
            ):
                if group_ids is None:
                    raise ValueError(
                        "st3 proposal latent prediction is missing latent ids"
                    )
                if st3_latent_bundle is not None:
                    num_beams = int(
                        getattr(
                            generation_config,
                            "num_beams",
                            getattr(self.llm.generation_config, "num_beams", 1),
                        )
                    )
                    if num_beams != 1:
                        raise ValueError(
                            "st3 latent segmentation prediction requires "
                            "num_beams=1 so the post-VLM latent table remains "
                            "one-to-one with its source image"
                        )
                if self._uses_st123_latent_deepstack():
                    if st123_deepstack_latents is None:
                        raise ValueError("st123 generation is missing E1, E2, E3")
                    with self._st123_deepstack_generation_context(
                        st123_deepstack_latents, group_ids
                    ):
                        llm_outputs = self.llm.generate(
                            **data_dict,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            generation_config=generation_config,
                            stopping_criteria=stopping_criteria,
                        )
                elif self._uses_stagewise_latent_trifusion():
                    if stagewise_deepstack_latents is None:
                        raise ValueError(
                            "stagewise generation is missing z0--z2"
                        )
                    with self._stagewise_deepstack_generation_context(
                        stagewise_deepstack_latents,
                        group_ids,
                    ):
                        llm_outputs = self.llm.generate(
                            **data_dict,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            generation_config=generation_config,
                            stopping_criteria=stopping_criteria,
                        )
                else:
                    llm_outputs = self.llm.generate(
                        **data_dict,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        generation_config=generation_config,
                        stopping_criteria=stopping_criteria,
                    )
            else:
                if group_ids is not None:
                    raise ValueError(
                        "group_ids were produced without an st3 execution bundle"
                    )
                llm_outputs = self.llm.generate(
                    **data_dict,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                )

        if st3_latent_alignment_only:
            if not self.st3_latent_alignment_pretrain:
                raise RuntimeError(
                    "latent alignment prediction reached the wrong model mode"
                )
            result = (llm_outputs, None)
            return (*result, None) if return_raw_seg_outputs else result

        if st3_latent_lm_only:
            if not self._uses_st3_latent_path():
                raise RuntimeError(
                    "latent LM-only prediction requires an enabled latent path"
                )
            if task_names[0] != "imgconv":
                raise RuntimeError(
                    "latent LM-only prediction is restricted to imgconv"
                )
            result = (llm_outputs, None)
            return (*result, None) if return_raw_seg_outputs else result

        if self.segmentor is None or self.segmentor.decoder is None:
            result = (llm_outputs, None)
            return (*result, None) if return_raw_seg_outputs else result

        if llm_outputs is not None:
            llm_output_ids = llm_outputs.sequences
            llm_hidden_states = llm_outputs.hidden_states
            input_hidden_states = llm_hidden_states[0][-1]
            llm_last_hidden_state = torch.cat([x[-1] for x in llm_hidden_states], dim=1)
            llm_input_embeds = self.llm_projector(input_hidden_states)
            llm_output_embeds = self.llm_projector(llm_last_hidden_state)

            st3_condition_sources = None
            st3_segment_sources = None
            if st3_any_bundle is not None:
                if input_cond_ids is not None:
                    st3_condition_sources = extract_indexed_embeddings(
                        llm_input_embeds,
                        input_cond_ids,
                        field_name="st3 prediction input Cond",
                    )

            L = input_hidden_states.shape[1]
            if input_cond_ids is not None:
                if st3_any_bundle is None:
                    cond_embeds = self._get_index_embeds(
                        llm_input_embeds,
                        input_cond_ids,
                    )

            # update cond_embeds if there is pstart and pend token in the output
            pstart_idx = (llm_output_ids[..., :-1] == self.pstart_token_idx).nonzero()[:, 1]
            pend_idx = (llm_output_ids[..., :-1] == self.pend_token_idx).nonzero()[:, 1]
            cls_idx = (llm_output_ids[..., :-1] == self.cls_token_idx).nonzero()[:, 1]
            if len(pstart_idx) > 0 or len(cls_idx) > 0:
                output_cond_ids = torch.full(
                    llm_last_hidden_state.shape[:2], -1, dtype=torch.long, device=input_hidden_states.device
                )
                shift = llm_input_embeds.shape[1]
                if self.cond_type in ["phrase", "all"]:
                    for i, (pstart, pend) in enumerate(zip(pstart_idx, pend_idx)):
                        output_cond_ids[:, shift + pstart : shift + pend + 1] = i
                if self.cond_type in ["cls", "all"]:
                    for i, ci in enumerate(cls_idx):
                        output_cond_ids[:, shift + ci] = i

                if st3_any_bundle is not None:
                    st3_condition_sources = extract_indexed_embeddings(
                        llm_output_embeds,
                        output_cond_ids,
                        field_name="st3 prediction generated Cond",
                    )
                else:
                    cond_embeds = self._get_index_embeds(
                        llm_output_embeds,
                        output_cond_ids,
                    )

            # update seg_ids if there is seg token in the output
            seg_idx = (llm_output_ids[..., :-1] == self.seg_token_idx).nonzero()[:, 1]
            if len(seg_idx) > 0:
                # fmt: off
                B = (seg_image_embeddings.shape[0] if isinstance(seg_image_embeddings, torch.Tensor) 
                    else seg_image_embeddings[0].shape[0]) if self.extract_seg_embeds else extra_pixel_values.shape[0]
                assert B == 1, "Only support batch size 1 for prediction"
                # fmt: on
                seg_ids = torch.full_like(
                    llm_output_ids[..., :-1], -1, dtype=torch.long, device=input_hidden_states.device
                )
                for i, idx in enumerate(seg_idx):
                    seg_ids[:, idx] = i
                seg_ids = torch.cat(
                    [torch.full((B, L), -1, dtype=torch.long, device=input_hidden_states.device), seg_ids], dim=-1
                )
                if st3_any_bundle is not None:
                    st3_segment_sources = extract_indexed_embeddings(
                        llm_output_embeds,
                        seg_ids,
                        field_name="st3 prediction SEG",
                    )
                else:
                    seg_embeds = self._get_index_embeds(
                        llm_output_embeds,
                        seg_ids,
                    )

            if st3_any_bundle is not None:
                if st3_condition_sources is None or st3_segment_sources is None:
                    raise ValueError(
                        "st3 Group-in-VLM generation did not produce the "
                        "required Cond/SEG embedding contract"
                    )
                if st3_group_bundle is not None:
                    seg_outputs, st3_scope = (
                        self._finish_st3_from_source_embeddings(
                            bundle=st3_group_bundle,
                            projected_hidden_states=llm_input_embeds,
                            group_ids=group_ids,
                            condition_sources=st3_condition_sources,
                            segment_sources=st3_segment_sources,
                            task_name=task_names[0],
                        )
                    )
                else:
                    seg_outputs, st3_scope = (
                        self._finish_st3_latent_from_source_embeddings(
                            bundle=st3_latent_bundle,
                            projected_hidden_states=llm_input_embeds,
                            group_ids=group_ids,
                            condition_sources=st3_condition_sources,
                            segment_sources=st3_segment_sources,
                            task_name=task_names[0],
                        )
                    )
                raw_seg_outputs = seg_outputs
                if kwargs.pop("do_postprocess", True):
                    image_size = self._expand_condition_metadata(
                        image_size,
                        st3_scope.segment_counts,
                        "image_sizes",
                    )
                    scaled_size = self._expand_condition_metadata(
                        scaled_size,
                        st3_scope.segment_counts,
                        "scaled_sizes",
                    )
                    self._validate_topology_postprocess_batch(
                        seg_outputs,
                        image_size,
                        scaled_size,
                    )
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )
                result = (llm_outputs, seg_outputs)
                return (
                    (*result, raw_seg_outputs)
                    if return_raw_seg_outputs
                    else result
                )

            if cond_embeds is not None and seg_embeds is not None:
                (
                    cond_embeds,
                    seg_embeds,
                    embed_masks,
                    local_cond_lens,
                    local_row_lens,
                    _,
                ) = self._process_embeds(
                    cond_embeds,
                    seg_embeds,
                    task_names[0],
                )

        if (cond_embeds is not None and seg_embeds is not None) or llm_outputs is None:
            if seg_embeds is not None and seg_embeds.shape[1] != 1:
                seg_outputs = None
            else:
                spatial_metadata = None
                if getattr(self.segmentor.dec_config, "use_topology_group_decoder", False):
                    if siglip_spatial_features is None:
                        raise ValueError(
                            "topology group decoder requires SigLIP spatial "
                            "features from the existing visual-encoder pass"
                        )
                    spatial_transforms = self._get_attrs_from_data_samples(
                        data_samples, "spatial_transforms", **kwargs
                    )[0]
                    spatial_metadata = self._prepare_spatial_metadata(
                        spatial_transforms,
                        siglip_patch_size,
                        siglip_spatial_features.device,
                    )
                seg_outputs = self.segmentor(
                    pixel_values=extra_pixel_values,
                    image_embeddings=seg_image_embeddings,
                    cond_embeddings=cond_embeds,
                    seg_embeddings=seg_embeds,
                    embed_masks=embed_masks,
                    cond_lens=(
                        local_row_lens
                        if topology_v2_source_local
                        else local_cond_lens
                    ),
                    task_name=task_names[0],
                    siglip_spatial_features=siglip_spatial_features,
                    siglip_valid_mask=siglip_valid_mask,
                    spatial_metadata=spatial_metadata,
                    return_variant_diagnostic_stages=(
                        return_variant_diagnostic_stages
                    ),
                    return_dict=True,
                )
                raw_seg_outputs = seg_outputs
                if kwargs.pop("do_postprocess", True):
                    if getattr(
                        self.segmentor.dec_config,
                        "use_topology_group_decoder",
                        False,
                    ):
                        image_size = self._expand_condition_metadata(
                            image_size,
                            (
                                local_row_lens
                                if topology_v2_source_local
                                else local_cond_lens
                            ),
                            "image_sizes",
                        )
                        scaled_size = self._expand_condition_metadata(
                            scaled_size,
                            (
                                local_row_lens
                                if topology_v2_source_local
                                else local_cond_lens
                            ),
                            "scaled_sizes",
                        )
                        self._validate_topology_postprocess_batch(
                            seg_outputs,
                            image_size,
                            scaled_size,
                        )
                    seg_outputs = self.postprocess_fn(
                        seg_outputs,
                        image_sizes=image_size,
                        scaled_sizes=scaled_size,
                        **kwargs,
                    )
        result = (llm_outputs, seg_outputs)
        return (
            (*result, raw_seg_outputs)
            if return_raw_seg_outputs
            else result
        )

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        llm_outputs, seg_outputs = self._forward(data_dict, data_samples, **kwargs)
        loss, loss_llm, loss_seg = 0.0, 0.0, 0.0
        if llm_outputs is not None and seg_outputs is None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss = loss_llm
            loss_dict = {"loss": loss, "loss_llm": loss_llm}
        elif llm_outputs is None and seg_outputs is not None:
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_seg
            loss_dict = {"loss": loss, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        elif llm_outputs is not None and seg_outputs is not None:
            loss_llm = llm_outputs.loss * self.llm_loss_weight
            loss_seg = seg_outputs.loss * self.seg_loss_weight
            loss_seg_dict = {k: v * self.seg_loss_weight for k, v in seg_outputs.loss_dict.items()}
            loss = loss_llm + loss_seg
            loss_dict = {"loss": loss, "loss_llm": loss_llm, "loss_seg": loss_seg}
            loss_dict.update(loss_seg_dict)
        else:
            raise ValueError("llm_outputs and seg_outputs are both None")

        if (
            seg_outputs is not None
            and self.segmentor is not None
            and getattr(
                self.segmentor.dec_config,
                "use_topology_group_decoder",
                False,
            )
            and getattr(
                self.segmentor.dec_config,
                "group_log_diagnostics",
                False,
            )
        ):
            loss_dict.update(self._group_training_log_scalars(seg_outputs))
        return loss_dict

    @torch.no_grad()
    def _group_training_log_scalars(self, seg_outputs):
        """Return detached stage statistics for the runner's log interval.

        These keys deliberately do not contain ``"loss"`` so MMEngine does
        not add them to the optimization objective.  The ordinary runner log
        processor emits them at its configured interval; nothing is printed
        from model forward.
        """

        stages = getattr(seg_outputs, "group_stage_outputs", None)
        if stages is None:
            return {}
        core = getattr(self.segmentor, "topology_group_core", None)
        if core is None:
            raise RuntimeError(
                "topology outputs exist but topology_group_core is missing"
            )
        scalars = {}
        for expected_stage, stage in enumerate(stages):
            if stage.stage_index != expected_stage:
                raise AssertionError(
                    "group log stages must be ordered st0..st9; "
                    f"found st{stage.stage_index} at position {expected_stage}"
                )
            valid = stage.group_valid_mask.to(torch.bool)
            valid_sizes = stage.group_sizes[valid].float()
            if valid_sizes.numel() == 0:
                raise RuntimeError(f"st{expected_stage} contains no valid group")
            prefix = f"diag_group_st{expected_stage}"
            scalars[f"{prefix}_avg_groups"] = (
                valid.sum(dim=1).float().mean().detach()
            )
            scalars[f"{prefix}_avg_group_size"] = valid_sizes.mean().detach()
            scalars[f"{prefix}_singleton_ratio"] = (
                valid_sizes.eq(1).float().mean().detach()
            )
            scalars[f"{prefix}_largest_group"] = valid_sizes.max().detach()
            scalars[f"{prefix}_query_norm"] = (
                stage.query_states.float().norm(dim=-1).mean().detach()
            )
            scalars[f"{prefix}_group_feature_norm"] = (
                stage.group_features.float().norm(dim=-1)[valid].mean().detach()
            )
            scalars[f"{prefix}_alpha_region"] = (
                core.region_pooler.alpha_region[expected_stage].float().detach()
            )
            if stage.feedback_delta_norm is not None:
                scalars[f"{prefix}_feedback_delta_norm"] = (
                    stage.feedback_delta_norm.float().detach()
                )
                scalars[f"{prefix}_alpha_fb"] = (
                    core.feedback.alpha_feedback[expected_stage].float().detach()
                )
        return scalars

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.st3_group_alignment_pretrain:
            prefix = "segmentor.st3_proposal_builder."
            proposal_state = OrderedDict(
                (key, value)
                for key, value in state_dict.items()
                if key.startswith(prefix)
            )
            if not proposal_state:
                raise RuntimeError(
                    "S2-G checkpoint would contain no st3 proposal parameters"
                )
            return proposal_state
        if self.st3_latent_alignment_pretrain:
            prefixes = (
                (
                    "stagewise_visual_fusion.siglip_norm.",
                    "stagewise_visual_fusion.sam_norm.",
                    "stagewise_visual_fusion.fusion_merger.",
                    "stagewise_deepstack_gates",
                    "segmentor.stagewise_latent_builder.",
                )
                if self._uses_stagewise_latent_trifusion()
                else (
                    (
                        "geometry_visual_fusion.",
                        "segmentor.geometry_proposal_builder.",
                        "segmentor.geometry_proposal_builder."
                        "geometry_area_encoder.",
                        "segmentor.geometry_proposal_builder."
                        "prefix_fused_latent_stages.",
                        "segmentor.geometry_proposal_builder."
                        "prefix_transport_stages.",
                    )
                    if self._uses_geometry_proposal_transport()
                    else (
                        "visual_projector.",
                        "segmentor.st3_transport_proposal_builder.",
                    )
                )
            )
            alignment_state = OrderedDict(
                (key, value)
                for key, value in state_dict.items()
                if key.startswith(prefixes)
            )
            missing_prefixes = [
                prefix
                for prefix in prefixes
                if not any(key.startswith(prefix) for key in alignment_state)
            ]
            if missing_prefixes:
                raise RuntimeError(
                    "latent S2 checkpoint is incomplete; missing prefixes="
                    f"{missing_prefixes}"
                )
            if self._uses_st123_latent_deepstack():
                self._validate_st123_s2_checkpoint(alignment_state)
            return alignment_state
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.visual_encoder is not None:
            if self.use_visual_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.visual_encoder, state_dict=state_dict))
            elif not self.freeze_visual_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "visual_encoder." in k})
        # Step 2. segmentor
        if self.segmentor is not None:
            if self.use_segmentor_encoder_lora:
                to_return.update(get_peft_model_state_dict(self.segmentor.encoder, state_dict=state_dict))
            elif not self.freeze_segmentor_encoder:
                to_return.update({k: v for k, v in state_dict.items() if "segmentor.encoder" in k})

            # segmentor other parts except encoder
            to_return.update(
                {k: v for k, v in state_dict.items() if "segmentor" in k and "segmentor.encoder" not in k}
            )
        # Step 3. LLM
        if self.llm is not None:
            if self.use_llm_lora:
                to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
            elif not self.freeze_llm:
                to_return.update({k: v for k, v in state_dict.items() if "llm." in k})
        # Step 4. Projector
        if not self._uses_stagewise_latent_trifusion() and not self._uses_geometry_proposal_transport():
            to_return.update({k: v for k, v in state_dict.items() if "visual_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "stagewise_visual_fusion." in k})
        to_return.update({k: v for k, v in state_dict.items() if k == "stagewise_deepstack_gates"})
        to_return.update({k: v for k, v in state_dict.items() if "geometry_visual_fusion." in k})
        to_return.update({k: v for k, v in state_dict.items() if "seg_projector." in k})
        to_return.update({k: v for k, v in state_dict.items() if "llm_projector." in k})
        # Step 5. seg_connector
        to_return.update({k: v for k, v in state_dict.items() if "seg_connector." in k})
        # Step 6. other embeds
        to_return.update({k: v for k, v in state_dict.items() if "bg_embeds." in k})
        to_return.update({k: v for k, v in state_dict.items() if "vgd_embeds." in k})
        # Step 7. vision_sampler
        to_return.update({k: v for k, v in state_dict.items() if "vision_sampler." in k})
        return to_return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def _prepare_segmentor_for_lora(self, lora_config, use_activation_checkpointing=True):
        if self.segmentor is None:
            return
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.segmentor.encoder)
            lora_config.target_modules = modules
        self.segmentor = get_peft_model(self.segmentor.encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        if self._uses_stagewise_latent_trifusion() or getattr(
            self, "_uses_st123_latent_deepstack", lambda: False
        )():
            # z0--z3 share one recurrent graph while z0--z2 enter Phi-3
            # through layer hooks and z3 remains in the token stream.
            # Reentrant checkpoint backward cannot safely traverse that
            # shared graph from both paths.  Keep this override strictly
            # local to the stagewise experiment; applying it globally changes
            # the established backward path of the original latent model.
            # st123 likewise shares its graph between E1 in the token stream
            # and E2/E3 at the inputs of the second/third Phi layers.
            checkpointing_kwargs = {"use_reentrant": False}
            if self.llm is not None:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpointing_kwargs
                )
            if self.visual_encoder is not None:
                self.visual_encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpointing_kwargs
                )
                self.visual_projector.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpointing_kwargs
                )
            if self.segmentor is not None:
                self.segmentor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=checkpointing_kwargs
                )
                if hasattr(self, "seg_projector"):
                    self.seg_projector.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=checkpointing_kwargs
                    )
                if hasattr(self, "llm_projector"):
                    self.llm_projector.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=checkpointing_kwargs
                    )
                if hasattr(self, "seg_connector"):
                    self.seg_connector.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=checkpointing_kwargs
                    )
            return

        # Preserve the checkpoint policy used by the successful original
        # latent runs: the language/vision/projector modules retain their
        # Transformers defaults, while Mask2Former alone is explicitly
        # non-reentrant for its split decoder execution.
        if self.llm is not None:
            self.llm.gradient_checkpointing_enable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_enable()
            self.visual_projector.gradient_checkpointing_enable()
        if self.segmentor is not None:
            self.segmentor.gradient_checkpointing_enable(
                {"use_reentrant": False}
            )
            if hasattr(self, "seg_projector"):
                self.seg_projector.gradient_checkpointing_enable()
            if hasattr(self, "llm_projector"):
                self.llm_projector.gradient_checkpointing_enable()
            if hasattr(self, "seg_connector"):
                self.seg_connector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        if self.llm is not None:
            self.llm.gradient_checkpointing_disable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_disable()
            self.visual_projector.gradient_checkpointing_disable()
        if self.segmentor is not None:
            self.segmentor.gradient_checkpointing_disable()
            if hasattr(self, "seg_projector"):
                self.seg_projector.gradient_checkpointing_disable()
            if hasattr(self, "llm_projector"):
                self.llm_projector.gradient_checkpointing_disable()
            if hasattr(self, "seg_connector"):
                self.seg_connector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings):
        orig_rope_scaling = getattr(llm_cfg, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1}

        orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # hardcode for internlm2
        llm_cfg.attn_implementation = "flash_attention_2"
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = (
            "LlamaConfig",
            "GemmaConfig",
            "MistralConfig",
            "MixtralConfig",
            "Qwen2Config",
            "Qwen2MoeConfig",
            "Qwen3Config",
            "Qwen3MoEConfig",
            "Starcoder2Config",
            "Starcoder2Config",
            "Phi3Config",
        )
        SUPPORT_FLASH_ATTN2 = (
            "InternLM2Config",
            "LlamaConfig",
            "GemmaConfig",
            "MistralConfig",
            "MixtralConfig",
            "Qwen2Config",
            "Qwen2MoeConfig",
            "Qwen3Config",
            "Qwen3MoEConfig",
            "Starcoder2Config",
            "Starcoder2Config",
            "Phi3Config",
        )

        torch_dtype = (
            torch.bfloat16
            if (get_torch_device().is_available() and get_torch_device().is_bf16_supported())
            else torch.float16
        )

        if getattr(cfg, "attn_implementation", None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == "flash_attention_2":
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = "flash_attention_2"
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = "sdpa"

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(cfg, "quantization_config")):
            return cfg

        torch_dtype = (
            torch.bfloat16
            if (get_torch_device().is_available() and get_torch_device().is_bf16_supported())
            else torch.float16
        )

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        # Keep AutoConfig on the same loading path as the actual model.  The
        # Phi-3 recipes deliberately use Transformers' native implementation
        # (trust_remote_code=False); forcing True here makes eight ranks race
        # through the dynamic-module cache and can import a stale/partial
        # configuration_phi3.py with no Phi3Config attribute.
        trust_remote_code = bool(
            getattr(cfg, "trust_remote_code", False)
        )
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if cfg_or_mod is None:
            return None

        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def to_hf(
        self,
        cfg,
        save_dir,
        fp32=False,
        save_pretrained_kwargs={},
        save_format="xtuner",
        **kwargs,
    ):
        assert save_format == "xtuner", "Only support xtuner format for now"
        self.to_xtuner(cfg, save_dir, fp32, save_pretrained_kwargs)

    def to_xtuner(self, cfg, save_dir, fp32=False, save_pretrained_kwargs={}):
        if self._uses_st3_latent_path():
            raise NotImplementedError(
                "The legacy XTuner/HuggingFace export cannot represent the "
                "Mask2Former proposal path and its 64-token latent prefix. "
                "Run latent-model inference with the MaskLAT config and the "
                "native .pth checkpoint instead of exporting it."
            )
        # Only save the model weights of: LLM, Visual Encoder, Segment Encoder, Visual Projector, Segmentor Projector
        # LLM
        if self.llm is not None:
            self.llm.config.use_cache = True
            if not fp32:
                print_log("Convert LLM to float16", "current")
                self.llm.half()
            if self.use_llm_lora:
                llm_path = osp.join(save_dir, "llm_adapter")
                print_log(f"Saving LLM adapter to {llm_path}", "current")
                self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
            elif not self.freeze_llm:
                llm_path = osp.join(save_dir, "llm")
                print_log(f"Saving LLM tokenizer to {llm_path}", "current")
                tokenizer = BUILDER.build(cfg.tokenizer)
                tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
                print_log(f"Saving LLM to {llm_path}", "current")
                self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
            self.llm.config.use_cache = False

        # Visual Encoder
        if self.visual_encoder is not None:
            if self.use_visual_encoder_lora:
                visual_encoder_path = osp.join(save_dir, "visual_encoder_adapter")
                print_log(f"Saving visual_encoder adapter to {visual_encoder_path}", "current")
                self.visual_encoder.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)
            elif not self.freeze_visual_encoder:
                visual_encoder_path = osp.join(save_dir, "visual_encoder")
                print_log(
                    "Saving visual_encoder image_processor to" f"{visual_encoder_path}",
                    "current",
                )
                image_processor = BUILDER.build(cfg.image_processor)
                image_processor.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving visual_encoder to {visual_encoder_path}", "current")
                self.visual_encoder.save_pretrained(visual_encoder_path, **save_pretrained_kwargs)

            # Visual Projector
            visual_projector_path = osp.join(save_dir, "visual_projector")
            print_log(f"Saving visual_projector to {visual_projector_path}", "current")
            self.visual_projector.save_pretrained(visual_projector_path, **save_pretrained_kwargs)

        # Segmentor Encoder
        if self.segmentor is not None:
            # TODO: add segmentor_encoder_adapter
            if self.use_segmentor_encoder_lora:
                segmentor_encoder_path = osp.join(save_dir, "segmentor_encoder_adapter")
                print_log(f"Saving segmentor_encoder adapter to {segmentor_encoder_path}", "current")
                self.segmentor.encoder.save_pretrained(segmentor_encoder_path, **save_pretrained_kwargs)
            elif not self.freeze_segmentor_encoder:
                segmentor_encoder_path = osp.join(save_dir, "segmentor_encoder")
                print_log(f"Saving segmentor image_processor to {segmentor_encoder_path}", "current")
                extra_image_processor = BUILDER.build(cfg.extra_image_processor)
                extra_image_processor.save_pretrained(segmentor_encoder_path, **save_pretrained_kwargs)
                print_log(f"Saving segmentor_encoder to {segmentor_encoder_path}", "current")
                state_dict = {
                    k.replace("segmentor.encoder.", "vision_encoder."): v
                    for k, v in self.state_dict().items()
                    if "segmentor.encoder" in k
                }
                self.segmentor.save_pretrained(segmentor_encoder_path, state_dict=state_dict, **save_pretrained_kwargs)

            # Segmentor Projector
            if hasattr(self, "seg_projector"):
                seg_projector_path = osp.join(save_dir, "segmentor_projector")
                print_log(f"Saving segmentor_projector to {seg_projector_path}", "current")
                self.seg_projector.save_pretrained(seg_projector_path, **save_pretrained_kwargs)
