# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mask2Former model configuration"""

import math
from typing import Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging
from transformers.utils.backbone_utils import verify_backbone_config_arguments

logger = logging.get_logger(__name__)


class Mask2FormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [facebook/mask2former-swin-small-coco-instance](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `SwinConfig()`):
            The configuration of the backbone model. If unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        feature_size (`int`, *optional*, defaults to 256):
            The features (channels) of the resulting feature maps.
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension of feedforward network for deformable detr encoder used as part of pixel decoder.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in the deformable detr encoder used as part of pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feature dimension in feedforward network for transformer decoder.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to use pre-LayerNorm or not for transformer decoder.
        enforce_input_projection (`bool`, *optional*, defaults to `False`):
            Whether to add an input projection 1x1 convolution even if the input channels and hidden dim are identical
            in the Transformer decoder.
        common_stride (`int`, *optional*, defaults to 4):
            Parameter used for determining number of FPN levels used as part of pixel decoder.
        ignore_value (`int`, *optional*, defaults to 255):
            Category id to be ignored during training.
        num_queries (`int`, *optional*, defaults to 100):
            Number of queries for the decoder.
        no_object_weight (`int`, *optional*, defaults to 0.1):
            The weight to apply to the null (no object) class.
        class_weight (`int`, *optional*, defaults to 2.0):
            The weight for the cross entropy loss.
        mask_weight (`int`, *optional*, defaults to 5.0):
            The weight for the mask loss.
        dice_weight (`int`, *optional*, defaults to 5.0):
            The weight for the dice loss.
        train_num_points (`str` or `function`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        use_auxiliary_loss (`boolean``, *optional*, defaults to `True`):
            If `True` [`Mask2FormerForUniversalSegmentationOutput`] will contain the auxiliary losses computed using
            the logits from each decoder's stage.
        feature_strides (`List[int]`, *optional*, defaults to `[4, 8, 16, 32]`):
            Feature strides corresponding to features generated from backbone network.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco-instance style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "mask2former"
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        use_backbone: bool = True,
        backbone_config: Optional[Dict] = None,
        image_size: int = 1024,
        feature_channels: List[int] = [384, 768, 1536, 3072],
        num_feature_levels: int = 3,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        activation_function: str = "relu",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_projection: bool = False,
        common_stride: int = 4,
        ignore_value: int = 255,
        num_queries: int = 100,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_strides: List[int] = [4, 8, 16, 32],
        output_auxiliary_logits: bool = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        use_timm_backbone: bool = False,
        backbone_kwargs: Optional[Dict] = None,
        use_sample_point: bool = True,
        use_nolabel_cls_loss: bool = True,
        loss_cls_type: str = "ce_loss",  # [focal_loss, ce_loss]
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_topology_group_decoder: bool = False,
        topology_group_version: int = 1,
        group_mask_threshold: float = 0.5,
        group_iou_threshold: float = 0.7,
        topology_diagnostic_query_mask_threshold: float = 0.5,
        use_sam_region_evidence: bool = True,
        use_siglip_region_evidence: bool = True,
        require_spatial_alignment: bool = True,
        siglip_hidden_dim: Optional[int] = None,
        region_init_scale: float = 1e-2,
        feedback_init_scale: float = 1e-3,
        topology_bias_init: float = 1.0,
        group_shape_temperature: float = 0.1,
        group_quality_temperature: float = 0.1,
        quality_iou_query_chunk_size: int = 8,
        quality_reg_weight: float = 1.0,
        quality_rank_weight: float = 1.0,
        unmatched_coverage_weight: float = 1.0,
        member_selector_weight: float = 1.0,
        member_selector_tie_epsilon: float = 1.0e-4,
        tournament_pair_chunk_size: int = 64,
        tournament_train_max_opponents_per_member: int = 4,
        topology_relation_dim: int = 64,
        topology_relation_num_heads: int = 4,
        topology_relation_logit_bound: float = 4.0,
        topology_relation_condition_chunk_size: int = 16,
        group_loss_stage_weights: Optional[List[float]] = None,
        group_log_diagnostics: bool = True,
        group_return_raw_query_outputs: bool = True,
        topology_group_lr_mult: float = 1.0,
        topology_require_expected_shapes: bool = True,
        topology_invalid_mask_logit: float = -20.0,
        use_st3_group_vlm_refiner: bool = False,
        st3_group_mask_threshold: float = 0.5,
        st3_group_iou_threshold: float = 0.7,
        st3_group_num_heads: int = 8,
        st3_group_fourier_features: int = 64,
        st3_group_loss_stage_weights: Optional[List[float]] = None,
        st3_group_member_selector_weight: float = 1.0,
        st3_group_require_expected_shapes: bool = True,
        use_st3_proposal_latent_bridge: bool = False,
        st3_latent_num_tokens: int = 64,
        st3_latent_num_heads: int = 8,
        st3_latent_mask_threshold: float = 0.5,
        st3_latent_fourier_features: int = 64,
        st3_latent_use_sam_region: bool = True,
        st3_latent_use_siglip_region: bool = True,
        st3_latent_use_xywh: bool = True,
        st3_latent_pre_vlm_depth: int = 2,
        st3_latent_post_vlm_depth: int = 2,
        st3_latent_persistent_mixer_depth: int = 6,
        st3_latent_gate_init_bias: float = -2.0,
        st3_latent_require_expected_shapes: bool = True,
        use_st3_bipartite_latent_transport: bool = False,
        st3_transport_pre_vlm_depth: int = 2,
        st3_transport_post_vlm_depth: int = 1,
        st3_transport_decoder_depth: int = 6,
        st3_transport_output_init_std: float = 1.0e-3,
        st3_transport_require_expected_shapes: bool = True,
        st3_transport_late_condition_refresh_stages: Optional[List[int]] = None,
        use_stagewise_latent_trifusion: bool = False,
        stagewise_trifusion_decoder_depth: int = 6,
        stagewise_trifusion_gate_init_bias: float = -2.0,
        stagewise_trifusion_keep_background_fixed: bool = True,
        stagewise_trifusion_require_expected_shapes: bool = True,
        use_latent_s2_post_vlm_transport: bool = False,
        latent_s2_post_vlm_transport_type: str = "direct",
        latent_s2_post_vlm_transport_depth: int = 6,
        latent_s2_post_vlm_output_init_std: float = 1.0e-3,
        latent_s2_direct_endpoint_ffn_depth: int = 2,
        latent_s2_post_vlm_require_expected_shapes: bool = True,
        use_geometry_aligned_visual_fusion: bool = False,
        use_proposal_mediated_tripartite_transport: bool = False,
        geometry_fusion_dim: int = 256,
        geometry_target_size: int = 64,
        geometry_output_size: int = 32,
        geometry_prefix_transport_depth: int = 3,
        geometry_transport_decoder_depth: int = 6,
        geometry_transport_output_init_std: float = 1.0e-3,
        geometry_transport_require_expected_shapes: bool = True,
        proposal_split_stage: int = 3,
        proposal_latent_position: str = "before_text",
        share_bidirectional_relation_logits: bool = True,
        use_st123_latent_deepstack: bool = False,
        st3_transport_enable_latent_writeback: bool = True,
        use_st123_latent_cascade: bool = False,
        **kwargs,
    ):
        if use_backbone:
            if backbone_config is None and backbone is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
                backbone_config = CONFIG_MAPPING["swin"](
                    image_size=224,
                    num_channels=3,
                    patch_size=4,
                    embed_dim=96,
                    depths=[2, 2, 18, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    drop_path_rate=0.3,
                    use_absolute_embeddings=False,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.pop("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)

            verify_backbone_config_arguments(
                use_timm_backbone=use_timm_backbone,
                use_pretrained_backbone=use_pretrained_backbone,
                backbone=backbone,
                backbone_config=backbone_config,
                backbone_kwargs=backbone_kwargs,
            )
            # verify that the backbone is supported
            if backbone_config is not None and backbone_config.model_type not in self.backbones_supported:
                logger.warning_once(
                    f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with Mask2Former. "
                    f"Supported model types: {','.join(self.backbones_supported)}"
                )
        else:
            backbone_config = None
            logger.info("`use_backbone` is `False`. The config will not be initialized with a backbone.")
        self.use_backbone = use_backbone
        self.backbone_config = backbone_config
        self.image_size = image_size
        self.feature_channels = feature_channels
        self.num_feature_levels = num_feature_levels
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_projection = enforce_input_projection
        self.common_stride = common_stride
        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.use_auxiliary_loss = use_auxiliary_loss
        self.feature_strides = feature_strides
        self.output_auxiliary_logits = output_auxiliary_logits
        self.num_hidden_layers = decoder_layers
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs

        self.use_sample_point = use_sample_point
        self.use_nolabel_cls_loss = use_nolabel_cls_loss
        self.loss_cls_type = loss_cls_type
        self.alpha = alpha
        self.gamma = gamma

        # Mask-topology grouped hierarchical decoder.  Defaults preserve the
        # unmodified Query-level Mask2Former path.
        self.use_topology_group_decoder = use_topology_group_decoder
        if int(topology_group_version) not in (1, 2):
            raise ValueError(
                "topology_group_version must be either 1 (legacy) or 2 "
                f"(dynamic bank), got {topology_group_version}"
            )
        self.topology_group_version = int(topology_group_version)
        self.group_mask_threshold = group_mask_threshold
        self.group_iou_threshold = group_iou_threshold
        if not 0.0 <= float(topology_diagnostic_query_mask_threshold) <= 1.0:
            raise ValueError(
                "topology_diagnostic_query_mask_threshold must be in [0,1], "
                f"got {topology_diagnostic_query_mask_threshold}"
            )
        self.topology_diagnostic_query_mask_threshold = float(
            topology_diagnostic_query_mask_threshold
        )
        self.use_sam_region_evidence = use_sam_region_evidence
        self.use_siglip_region_evidence = use_siglip_region_evidence
        self.require_spatial_alignment = require_spatial_alignment
        self.siglip_hidden_dim = siglip_hidden_dim
        self.region_init_scale = region_init_scale
        self.feedback_init_scale = feedback_init_scale
        self.topology_bias_init = topology_bias_init
        self.group_shape_temperature = group_shape_temperature
        self.group_quality_temperature = group_quality_temperature
        if int(quality_iou_query_chunk_size) <= 0:
            raise ValueError(
                "quality_iou_query_chunk_size must be positive, got "
                f"{quality_iou_query_chunk_size}"
            )
        self.quality_iou_query_chunk_size = int(
            quality_iou_query_chunk_size
        )
        self.quality_reg_weight = quality_reg_weight
        self.quality_rank_weight = quality_rank_weight
        self.unmatched_coverage_weight = unmatched_coverage_weight
        if float(member_selector_weight) < 0:
            raise ValueError("member_selector_weight must be non-negative")
        if float(member_selector_tie_epsilon) < 0:
            raise ValueError(
                "member_selector_tie_epsilon must be non-negative"
            )
        if int(tournament_pair_chunk_size) <= 0:
            raise ValueError("tournament_pair_chunk_size must be positive")
        if int(tournament_train_max_opponents_per_member) < 0:
            raise ValueError(
                "tournament_train_max_opponents_per_member must be "
                "non-negative"
            )
        if (
            int(tournament_train_max_opponents_per_member) > 0
            and int(tournament_train_max_opponents_per_member) % 2 != 0
        ):
            raise ValueError(
                "tournament_train_max_opponents_per_member must be even; "
                "use 0 only for an explicit exhaustive-training ablation"
            )
        if int(topology_relation_dim) <= 0:
            raise ValueError("topology_relation_dim must be positive")
        if int(topology_relation_num_heads) <= 0:
            raise ValueError(
                "topology_relation_num_heads must be positive"
            )
        if int(topology_relation_dim) % int(
            topology_relation_num_heads
        ) != 0:
            raise ValueError(
                "topology_relation_dim must be divisible by "
                "topology_relation_num_heads"
            )
        if float(topology_relation_logit_bound) <= 0:
            raise ValueError(
                "topology_relation_logit_bound must be positive"
            )
        if int(topology_relation_condition_chunk_size) <= 0:
            raise ValueError(
                "topology_relation_condition_chunk_size must be positive"
            )
        self.member_selector_weight = float(member_selector_weight)
        self.member_selector_tie_epsilon = float(
            member_selector_tie_epsilon
        )
        self.tournament_pair_chunk_size = int(
            tournament_pair_chunk_size
        )
        self.tournament_train_max_opponents_per_member = int(
            tournament_train_max_opponents_per_member
        )
        self.topology_relation_dim = int(topology_relation_dim)
        self.topology_relation_num_heads = int(
            topology_relation_num_heads
        )
        self.topology_relation_logit_bound = float(
            topology_relation_logit_bound
        )
        self.topology_relation_condition_chunk_size = int(
            topology_relation_condition_chunk_size
        )
        self.group_loss_stage_weights = (
            [1.0] * decoder_layers
            if group_loss_stage_weights is None
            else list(group_loss_stage_weights)
        )
        if len(self.group_loss_stage_weights) != decoder_layers:
            raise ValueError(
                "group_loss_stage_weights must have one value per prediction "
                f"stage ({decoder_layers}), got {len(self.group_loss_stage_weights)}"
            )
        self.group_log_diagnostics = group_log_diagnostics
        self.group_return_raw_query_outputs = group_return_raw_query_outputs
        self.topology_group_lr_mult = topology_group_lr_mult
        self.topology_require_expected_shapes = topology_require_expected_shapes
        self.topology_invalid_mask_logit = topology_invalid_mask_logit

        # Independent st3 Group-in-VLM path.  It does not share execution
        # semantics with the older per-stage topology V1/V2 decoder.
        self.use_st3_group_vlm_refiner = bool(use_st3_group_vlm_refiner)
        if self.use_st3_group_vlm_refiner and self.use_topology_group_decoder:
            raise ValueError(
                "use_st3_group_vlm_refiner and use_topology_group_decoder "
                "are mutually exclusive"
            )
        for name, value in (
            ("st3_group_mask_threshold", st3_group_mask_threshold),
            ("st3_group_iou_threshold", st3_group_iou_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if int(st3_group_num_heads) <= 0 or hidden_dim % int(st3_group_num_heads) != 0:
            raise ValueError(
                "st3_group_num_heads must be positive and divide hidden_dim"
            )
        if int(st3_group_fourier_features) <= 0:
            raise ValueError("st3_group_fourier_features must be positive")
        if float(st3_group_member_selector_weight) < 0:
            raise ValueError("st3_group_member_selector_weight must be non-negative")
        self.st3_group_mask_threshold = float(st3_group_mask_threshold)
        self.st3_group_iou_threshold = float(st3_group_iou_threshold)
        self.st3_group_num_heads = int(st3_group_num_heads)
        self.st3_group_fourier_features = int(st3_group_fourier_features)
        self.st3_group_loss_stage_weights = (
            [1.0] * (decoder_layers - 3)
            if st3_group_loss_stage_weights is None
            else [float(value) for value in st3_group_loss_stage_weights]
        )
        if len(self.st3_group_loss_stage_weights) != decoder_layers - 3:
            raise ValueError(
                "st3_group_loss_stage_weights must cover st3--st9 "
                f"({decoder_layers - 3} entries), got "
                f"{len(self.st3_group_loss_stage_weights)}"
            )
        if any(value < 0 for value in self.st3_group_loss_stage_weights):
            raise ValueError("st3_group_loss_stage_weights must be non-negative")
        self.st3_group_member_selector_weight = float(
            st3_group_member_selector_weight
        )
        self.st3_group_require_expected_shapes = bool(
            st3_group_require_expected_shapes
        )
        self.use_st3_proposal_latent_bridge = bool(
            use_st3_proposal_latent_bridge
        )
        self.use_st3_bipartite_latent_transport = bool(
            use_st3_bipartite_latent_transport
        )
        # HF from_pretrained/from_dict only applies keyword overrides to
        # attributes that already exist on the config constructed from JSON.
        # Register both switches even for old pretrained configs, otherwise
        # writeback=False/cascade=True overrides are silently left unused.
        for name, value in (
            ("st3_transport_enable_latent_writeback", st3_transport_enable_latent_writeback),
            ("use_st123_latent_cascade", use_st123_latent_cascade),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
            setattr(self, name, value)
        self.use_st123_latent_deepstack = bool(use_st123_latent_deepstack)
        if self.use_st123_latent_deepstack and not (
            self.use_st3_bipartite_latent_transport
        ):
            raise ValueError(
                "st123 latent deepstack requires bipartite latent transport"
            )
        self.use_stagewise_latent_trifusion = bool(
            use_stagewise_latent_trifusion
        )
        self.use_latent_s2_post_vlm_transport = bool(
            use_latent_s2_post_vlm_transport
        )
        self.use_geometry_aligned_visual_fusion = bool(
            use_geometry_aligned_visual_fusion
        )
        self.use_proposal_mediated_tripartite_transport = bool(
            use_proposal_mediated_tripartite_transport
        )
        if (
            self.use_geometry_aligned_visual_fusion
            != self.use_proposal_mediated_tripartite_transport
        ):
            raise ValueError(
                "geometry-aligned visual fusion and proposal-mediated "
                "tripartite transport must be enabled together"
            )
        enabled_experiments = sum(
            int(enabled)
            for enabled in (
                self.use_topology_group_decoder,
                self.use_st3_group_vlm_refiner,
                self.use_st3_proposal_latent_bridge,
                self.use_st3_bipartite_latent_transport,
                self.use_stagewise_latent_trifusion,
                self.use_latent_s2_post_vlm_transport,
                self.use_proposal_mediated_tripartite_transport,
            )
        )
        if enabled_experiments > 1:
            raise ValueError(
                "topology decoding, st3 Group-in-VLM, the gated proposal "
                "latent bridge, bipartite latent transport, stagewise "
                "latent tri-fusion, latent-S2 post-VLM transport, and "
                "geometry proposal transport are mutually exclusive"
            )
        if int(st3_latent_num_tokens) <= 0:
            raise ValueError("st3_latent_num_tokens must be positive")
        if (
            int(st3_latent_num_heads) <= 0
            or hidden_dim % int(st3_latent_num_heads) != 0
        ):
            raise ValueError(
                "st3_latent_num_heads must be positive and divide hidden_dim"
            )
        if not 0.0 <= float(st3_latent_mask_threshold) <= 1.0:
            raise ValueError("st3_latent_mask_threshold must be in [0, 1]")
        if int(st3_latent_fourier_features) <= 0:
            raise ValueError("st3_latent_fourier_features must be positive")
        for name, value in (
            ("st3_latent_use_sam_region", st3_latent_use_sam_region),
            ("st3_latent_use_siglip_region", st3_latent_use_siglip_region),
            ("st3_latent_use_xywh", st3_latent_use_xywh),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        for name, value in (
            ("st3_latent_pre_vlm_depth", st3_latent_pre_vlm_depth),
            ("st3_latent_post_vlm_depth", st3_latent_post_vlm_depth),
            (
                "st3_latent_persistent_mixer_depth",
                st3_latent_persistent_mixer_depth,
            ),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(st3_latent_gate_init_bias)):
            raise ValueError("st3_latent_gate_init_bias must be finite")
        self.st3_latent_num_tokens = int(st3_latent_num_tokens)
        self.st3_latent_num_heads = int(st3_latent_num_heads)
        self.st3_latent_mask_threshold = float(st3_latent_mask_threshold)
        self.st3_latent_fourier_features = int(
            st3_latent_fourier_features
        )
        self.st3_latent_use_sam_region = st3_latent_use_sam_region
        self.st3_latent_use_siglip_region = st3_latent_use_siglip_region
        self.st3_latent_use_xywh = st3_latent_use_xywh
        self.st3_latent_pre_vlm_depth = int(st3_latent_pre_vlm_depth)
        self.st3_latent_post_vlm_depth = int(st3_latent_post_vlm_depth)
        self.st3_latent_persistent_mixer_depth = int(
            st3_latent_persistent_mixer_depth
        )
        self.st3_latent_gate_init_bias = float(st3_latent_gate_init_bias)
        self.st3_latent_require_expected_shapes = bool(
            st3_latent_require_expected_shapes
        )
        for name, value in (
            ("st3_transport_pre_vlm_depth", st3_transport_pre_vlm_depth),
            ("st3_transport_post_vlm_depth", st3_transport_post_vlm_depth),
            ("st3_transport_decoder_depth", st3_transport_decoder_depth),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(float(st3_transport_output_init_std))
            or float(st3_transport_output_init_std) <= 0.0
        ):
            raise ValueError(
                "st3_transport_output_init_std must be finite and positive"
            )
        self.st3_transport_pre_vlm_depth = int(
            st3_transport_pre_vlm_depth
        )
        self.st3_transport_post_vlm_depth = int(
            st3_transport_post_vlm_depth
        )
        self.st3_transport_decoder_depth = int(
            st3_transport_decoder_depth
        )
        self.st3_transport_output_init_std = float(
            st3_transport_output_init_std
        )
        self.st3_transport_require_expected_shapes = bool(
            st3_transport_require_expected_shapes
        )
        late_refresh_stages = tuple(
            int(stage)
            for stage in (
                st3_transport_late_condition_refresh_stages or ()
            )
        )
        if len(set(late_refresh_stages)) != len(late_refresh_stages):
            raise ValueError(
                "st3_transport_late_condition_refresh_stages must be unique"
            )
        if any(stage < 4 or stage > 9 for stage in late_refresh_stages):
            raise ValueError(
                "late condition refresh stages must be within st4--st9"
            )
        if late_refresh_stages and not self.use_st3_bipartite_latent_transport:
            raise ValueError(
                "late condition refresh requires bipartite latent transport"
            )
        if late_refresh_stages and self.use_st123_latent_deepstack:
            raise ValueError(
                "st123 latent deepstack cannot enable late condition refresh"
            )
        self.st3_transport_late_condition_refresh_stages = (
            late_refresh_stages
        )
        if int(stagewise_trifusion_decoder_depth) <= 0:
            raise ValueError(
                "stagewise_trifusion_decoder_depth must be positive"
            )
        if not math.isfinite(float(stagewise_trifusion_gate_init_bias)):
            raise ValueError(
                "stagewise_trifusion_gate_init_bias must be finite"
            )
        self.stagewise_trifusion_decoder_depth = int(
            stagewise_trifusion_decoder_depth
        )
        self.stagewise_trifusion_gate_init_bias = float(
            stagewise_trifusion_gate_init_bias
        )
        self.stagewise_trifusion_keep_background_fixed = bool(
            stagewise_trifusion_keep_background_fixed
        )
        self.stagewise_trifusion_require_expected_shapes = bool(
            stagewise_trifusion_require_expected_shapes
        )
        if str(latent_s2_post_vlm_transport_type) not in (
            "direct",
            "tripartite",
        ):
            raise ValueError(
                "latent_s2_post_vlm_transport_type must be direct or tripartite"
            )
        if int(latent_s2_post_vlm_transport_depth) <= 0:
            raise ValueError(
                "latent_s2_post_vlm_transport_depth must be positive"
            )
        if (
            not math.isfinite(float(latent_s2_post_vlm_output_init_std))
            or float(latent_s2_post_vlm_output_init_std) <= 0.0
        ):
            raise ValueError(
                "latent_s2_post_vlm_output_init_std must be finite and positive"
            )
        if int(latent_s2_direct_endpoint_ffn_depth) < 0:
            raise ValueError(
                "latent_s2_direct_endpoint_ffn_depth must be non-negative"
            )
        self.latent_s2_post_vlm_transport_type = str(
            latent_s2_post_vlm_transport_type
        )
        self.latent_s2_post_vlm_transport_depth = int(
            latent_s2_post_vlm_transport_depth
        )
        self.latent_s2_post_vlm_output_init_std = float(
            latent_s2_post_vlm_output_init_std
        )
        self.latent_s2_direct_endpoint_ffn_depth = int(
            latent_s2_direct_endpoint_ffn_depth
        )
        self.latent_s2_post_vlm_require_expected_shapes = bool(
            latent_s2_post_vlm_require_expected_shapes
        )
        if int(geometry_fusion_dim) <= 0:
            raise ValueError("geometry_fusion_dim must be positive")
        if int(geometry_target_size) != 64 or int(geometry_output_size) != 32:
            raise ValueError(
                "the first geometry experiment fixes the fusion/output grids "
                "to 64x64 and 32x32"
            )
        if int(geometry_target_size) != 2 * int(geometry_output_size):
            raise ValueError("geometry grids must support 2x pixel-unshuffle")
        if int(geometry_prefix_transport_depth) != 3:
            raise ValueError(
                "geometry prefix transport must cover exactly st1--st3"
            )
        if int(geometry_transport_decoder_depth) <= 0:
            raise ValueError("geometry_transport_decoder_depth must be positive")
        if (
            not math.isfinite(float(geometry_transport_output_init_std))
            or float(geometry_transport_output_init_std) <= 0.0
        ):
            raise ValueError(
                "geometry_transport_output_init_std must be finite and positive"
            )
        if int(proposal_split_stage) != 3:
            raise ValueError(
                "the geometry experiment requires its VLM handoff after st3"
            )
        if str(proposal_latent_position) != "before_text":
            raise ValueError(
                "the first proposal experiment places latent tokens before text"
            )
        if not bool(share_bidirectional_relation_logits):
            raise ValueError(
                "proposal-mediated transport requires shared bidirectional relation logits"
            )
        self.geometry_fusion_dim = int(geometry_fusion_dim)
        self.geometry_target_size = int(geometry_target_size)
        self.geometry_output_size = int(geometry_output_size)
        self.geometry_prefix_transport_depth = int(
            geometry_prefix_transport_depth
        )
        self.geometry_transport_decoder_depth = int(
            geometry_transport_decoder_depth
        )
        self.geometry_transport_output_init_std = float(
            geometry_transport_output_init_std
        )
        self.geometry_transport_require_expected_shapes = bool(
            geometry_transport_require_expected_shapes
        )
        self.proposal_split_stage = int(proposal_split_stage)
        self.proposal_latent_position = str(proposal_latent_position)
        self.share_bidirectional_relation_logits = bool(
            share_bidirectional_relation_logits
        )

        super().__init__(**kwargs)

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`Mask2FormerConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.

        Returns:
            [`Mask2FormerConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


__all__ = ["Mask2FormerConfig"]
