import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel, get_parameter_dtype
from transformers.models.swin import SwinBackbone

from masklat.utils.logging import print_log

from .mask2former import (
    BipartiteLatentTransportBridge,
    FixedGroupVLMRefiner,
    FusedProposalBuilder,
    LatentS2PostVLMBridge,
    PosteriorLatentQueryBridge,
    ProposalMediatedTripartiteBridge,
    St3ProposalLatentBuilder,
    St3BipartiteLatentBuilder,
    St123LatentDeepstackBuilder,
    StagewiseLatentTriFusionBridge,
    StagewiseProposalLatentBuilder,
    St3ProposalBuilder,
    Mask2FormerLoss,
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerModel,
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    Mask2FormerPixelDecoderEncoderOnly,
    Mask2FormerPixelLevelModule,
    Mask2FormerTransformerModule,
    build_group_local_self_attention_mask,
    unpack_packed_groups,
)
from .mask2former.group_criterion import (
    HierarchicalCriterionDiagnostics,
    HierarchicalGroupCriterion,
)
from .mask2former.st123_latent_cascade import St123LatentCascadeBuilder
from .mask2former.spatial_validity import (
    masked_normalized_resize,
    valid_mask_from_normalized_boxes,
    validate_normalized_valid_boxes,
)
from .mask2former.topology_group_decoder import (
    GroupConditionClassifier,
    GroupPredictionAdapter,
    TopologyGroupDecoderCore,
    TopologyGroupStageOutput,
)
from .mask2former.topology_condition_relation import (
    TopologyConditionRelationBank,
)
from .sam import SamMaskDecoder, SamModel, SamVisionEncoder


@dataclass
class MaskLATSegmentorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    raw_query_class_logits: Optional[torch.FloatTensor] = None
    raw_query_mask_logits: Optional[torch.FloatTensor] = None
    group_valid_mask: Optional[torch.BoolTensor] = None
    query_to_group: Optional[torch.LongTensor] = None
    group_features: Optional[torch.FloatTensor] = None
    group_class_logits: Optional[torch.FloatTensor] = None
    member_attention_weights: Optional[torch.FloatTensor] = None
    member_quality_logits: Optional[torch.FloatTensor] = None
    group_representative_query_indices: Optional[torch.LongTensor] = None
    group_counts: Optional[torch.LongTensor] = None
    condition_valid_mask: Optional[torch.BoolTensor] = None
    group_stage_outputs: Optional[Tuple[TopologyGroupStageOutput, ...]] = None
    group_criterion_diagnostics: Optional[HierarchicalCriterionDiagnostics] = None
    diagnostic_stage_class_logits: Optional[
        Tuple[torch.FloatTensor, ...]
    ] = None
    diagnostic_stage_mask_logits: Optional[
        Tuple[torch.FloatTensor, ...]
    ] = None


@dataclass
class St3GroupExecutionBundle:
    """Source-image state held while Group tokens pass through the VLM."""

    decoder_context: Any
    stage3_state: Any
    proposal: Any


@dataclass
class St3LatentExecutionBundle:
    """Source-image state held while 64 proposal latents cross the VLM."""

    decoder_context: Any
    stage3_state: Any
    proposal: Any


class MaskLATSegmentor(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, encoder: Literal[SamModel, Mask2FormerModel], decoder=None, torch_dtype=torch.float32, reinit_decoder=False, drop_decoder=False, close_cls=False, open_cls=False):  # type: ignore
        PreTrainedModel.__init__(self, encoder.config)
        self.topology_group_core = None
        self.group_classifier = None
        self.group_prediction_adapter = None
        self.group_criterion = None
        self.topology_condition_relation = None
        self.st3_proposal_builder = None
        self.st3_vlm_refiner = None
        self.st3_group_classifier = None
        self.st3_group_prediction_adapter = None
        self.st3_group_criterion = None
        self.st3_latent_proposal_builder = None
        self.st3_latent_bridge = None
        self.st3_transport_proposal_builder = None
        self.st3_transport_bridge = None
        self.latent_s2_post_vlm_bridge = None
        self.stagewise_latent_builder = None
        self.stagewise_trifusion_bridge = None
        self.geometry_proposal_builder = None
        self.geometry_transport_bridge = None

        if isinstance(encoder, SamModel) and decoder is None:
            self.enc_config = encoder.config.vision_config
            self.dec_config = encoder.config.mask_decoder_config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.prompt_encoder = encoder.prompt_encoder
            self.pixel_decoder = None
            self.decoder = encoder.mask_decoder
        elif isinstance(encoder, SamModel) and isinstance(decoder, Mask2FormerModel):
            self.enc_config = encoder.config.vision_config
            self.dec_config = decoder.config
            self.prompt_enc_config = encoder.config.prompt_encoder_config

            self.shared_image_embedding = encoder.shared_image_embedding
            self.encoder = encoder.vision_encoder
            self.pixel_decoder = decoder.pixel_level_module.decoder
            self.decoder = decoder.transformer_module
        elif isinstance(encoder, Mask2FormerModel) and decoder is None:
            self.enc_config = encoder.config
            self.dec_config = copy.deepcopy(encoder.config)
            self.enc_config.hidden_size = encoder.config.backbone_config.hidden_size

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module.encoder
            self.pixel_decoder = encoder.pixel_level_module.decoder
            self.decoder = encoder.transformer_module
        elif isinstance(encoder, Mask2FormerModel) and isinstance(decoder, SamModel):
            # TODO: check if this is correct
            self.enc_config = encoder.config
            self.dec_config = decoder.config

            self.shared_image_embedding = None
            self.encoder = encoder.pixel_level_module
            self.pixel_decoder = encoder.pixel_level_module.decoder
            self.decoder = decoder.mask_decoder
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(encoder)} and {type(decoder)}")

        if drop_decoder:
            self.encoder = self.encoder.to(torch_dtype)
            self.shared_image_embedding = None
            self.prompt_encoder = None
            self.pixel_decoder = None
            self.decoder = None

            return

        if reinit_decoder:
            print_log(f"Reinitializing decoder of {self.decoder.__class__.__name__}.", logger="current")
            # means decoder and pixel_decoder are not from pretained model, so we need to initialize the weights
            self.decoder.apply(self._init_weights)
            if self.pixel_decoder is not None:
                print_log(
                    f"Reinitializing pixel_decoder of {self.pixel_decoder.__class__.__name__}.", logger="current"
                )
                self.pixel_decoder.apply(self._init_weights)

        self.weight_dict: Dict[str, float] = {
            "loss_cls": self.decoder.config.class_weight,
            "loss_mask": self.decoder.config.mask_weight,
            "loss_dice": self.decoder.config.dice_weight,
        }
        self.criterion = Mask2FormerLoss(config=self.decoder.config, weight_dict=self.weight_dict)

        if close_cls:
            self.class_predictor = nn.Linear(self.decoder.config.hidden_dim, self.decoder.config.num_labels + 1).to(
                torch_dtype
            )
        if open_cls:
            self.logit_scale = nn.Parameter(torch.ones([1], dtype=torch_dtype) * np.log(1 / 0.07), requires_grad=True)

        self.close_cls = close_cls
        self.open_cls = open_cls
        self.use_cls = open_cls or close_cls

        self.encoder = self.encoder.to(torch_dtype)
        self.decoder = self.decoder.to(torch_dtype)
        self.pixel_decoder = self.pixel_decoder.to(torch_dtype) if self.pixel_decoder is not None else None
        self.shared_image_embedding = (
            self.shared_image_embedding.to(torch_dtype).requires_grad_(False)
            if self.shared_image_embedding is not None
            else None
        )
        self.criterion = self.criterion.to(torch_dtype)
        if getattr(self.dec_config, "use_topology_group_decoder", False):
            configured_siglip_dim = getattr(
                self.dec_config,
                "siglip_hidden_dim",
                None,
            )
            if configured_siglip_dim is not None:
                self.configure_topology_group_decoder(
                    siglip_hidden_dim=int(configured_siglip_dim)
                )

    @property
    def config_class(self):
        return self.enc_config.__class__

    def configure_topology_group_decoder(self, siglip_hidden_dim: int) -> None:
        """Build all trainable group modules from runtime model dimensions.

        This method is called before S1/S2 checkpoint loading, so the new
        parameters are part of the normal ``state_dict`` and are reported as
        missing (and freshly initialized) when loading baseline checkpoints.
        Repeated calls with the same actual SigLIP width are idempotent.
        """

        if not getattr(self.dec_config, "use_topology_group_decoder", False):
            return
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "topology group decoding requires a Mask2Former transformer decoder"
            )
        siglip_hidden_dim = int(siglip_hidden_dim)
        topology_group_version = int(
            getattr(self.dec_config, "topology_group_version", 1)
        )
        if siglip_hidden_dim <= 0:
            raise ValueError(
                f"siglip_hidden_dim must be positive, got {siglip_hidden_dim}"
            )
        existing = self.topology_group_core
        if existing is not None:
            existing_width = (
                existing.region_pooler.siglip_region_projector.in_features
            )
            existing_version = int(
                getattr(existing, "topology_group_version", 1)
            )
            if (
                existing_width == siglip_hidden_dim
                and existing_version == topology_group_version
            ):
                if (
                    topology_group_version == 2
                    and self.topology_condition_relation is None
                ):
                    reference_parameter = next(existing.parameters())
                    self.topology_condition_relation = (
                        TopologyConditionRelationBank(
                            hidden_dim=int(self.decoder.config.hidden_dim),
                            relation_dim=int(
                                getattr(
                                    self.dec_config,
                                    "topology_relation_dim",
                                    64,
                                )
                            ),
                            num_heads=int(
                                getattr(
                                    self.dec_config,
                                    "topology_relation_num_heads",
                                    4,
                                )
                            ),
                            relation_logit_bound=float(
                                getattr(
                                    self.dec_config,
                                    "topology_relation_logit_bound",
                                    4.0,
                                )
                            ),
                            condition_chunk_size=int(
                                getattr(
                                    self.dec_config,
                                    "topology_relation_condition_chunk_size",
                                    16,
                                )
                            ),
                        ).to(
                            device=reference_parameter.device,
                            dtype=reference_parameter.dtype,
                        )
                    )
                self.dec_config.siglip_hidden_dim = siglip_hidden_dim
                return

        hidden_dim = int(self.decoder.config.hidden_dim)
        sam_feature_dim = int(self.decoder.config.mask_feature_size)
        num_queries = int(self.decoder.config.num_queries)
        num_stages = len(self.decoder.decoder.layers) + 1
        core = TopologyGroupDecoderCore(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_hidden_dim,
            num_stages=num_stages,
            num_queries=num_queries,
            group_mask_threshold=self.decoder.config.group_mask_threshold,
            group_iou_threshold=self.decoder.config.group_iou_threshold,
            use_sam_region_evidence=self.decoder.config.use_sam_region_evidence,
            use_siglip_region_evidence=self.decoder.config.use_siglip_region_evidence,
            require_spatial_alignment=self.decoder.config.require_spatial_alignment,
            region_init_scale=self.decoder.config.region_init_scale,
            feedback_init_scale=self.decoder.config.feedback_init_scale,
            topology_bias_init=self.decoder.config.topology_bias_init,
            require_expected_shapes=self.decoder.config.topology_require_expected_shapes,
            topology_group_version=topology_group_version,
            tournament_pair_chunk_size=int(
                getattr(
                    self.decoder.config,
                    "tournament_pair_chunk_size",
                    64,
                )
            ),
            tournament_train_max_opponents_per_member=int(
                getattr(
                    self.decoder.config,
                    "tournament_train_max_opponents_per_member",
                    4,
                )
            ),
        )
        group_criterion = HierarchicalGroupCriterion(
            class_weight=self.decoder.config.class_weight,
            mask_weight=self.decoder.config.mask_weight,
            dice_weight=self.decoder.config.dice_weight,
            no_object_weight=self.decoder.config.no_object_weight,
            loss_cls_type=self.decoder.config.loss_cls_type,
            alpha=self.decoder.config.alpha,
            gamma=self.decoder.config.gamma,
            train_num_points=self.decoder.config.train_num_points,
            oversample_ratio=self.decoder.config.oversample_ratio,
            importance_sample_ratio=self.decoder.config.importance_sample_ratio,
            use_sample_point=self.decoder.config.use_sample_point,
            group_shape_temperature=self.decoder.config.group_shape_temperature,
            group_quality_temperature=self.decoder.config.group_quality_temperature,
            quality_iou_query_chunk_size=(
                self.decoder.config.quality_iou_query_chunk_size
            ),
            quality_reg_weight=self.decoder.config.quality_reg_weight,
            quality_rank_weight=self.decoder.config.quality_rank_weight,
            unmatched_coverage_weight=self.decoder.config.unmatched_coverage_weight,
            group_loss_stage_weights=self.decoder.config.group_loss_stage_weights,
            topology_group_version=topology_group_version,
            member_selector_weight=float(
                getattr(
                    self.decoder.config,
                    "member_selector_weight",
                    1.0,
                )
            ),
            member_selector_tie_epsilon=float(
                getattr(
                    self.decoder.config,
                    "member_selector_tie_epsilon",
                    1.0e-4,
                )
            ),
        )
        try:
            reference_parameter = next(self.decoder.parameters())
            module_device = reference_parameter.device
            module_dtype = reference_parameter.dtype
        except StopIteration:  # pragma: no cover - transformer always has parameters.
            module_device = torch.device("cpu")
            module_dtype = self.dtype

        self.topology_group_core = core.to(
            device=module_device,
            dtype=module_dtype,
        )
        self.group_classifier = GroupConditionClassifier(
            float32_open_vocab=(topology_group_version == 2),
        ).to(device=module_device)
        self.group_prediction_adapter = GroupPredictionAdapter(
            invalid_mask_logit=self.decoder.config.topology_invalid_mask_logit
        ).to(device=module_device)
        self.group_criterion = group_criterion.to(device=module_device)
        if topology_group_version == 2:
            self.topology_condition_relation = TopologyConditionRelationBank(
                hidden_dim=hidden_dim,
                relation_dim=int(
                    getattr(self.dec_config, "topology_relation_dim", 64)
                ),
                num_heads=int(
                    getattr(
                        self.dec_config,
                        "topology_relation_num_heads",
                        4,
                    )
                ),
                relation_logit_bound=float(
                    getattr(
                        self.dec_config,
                        "topology_relation_logit_bound",
                        4.0,
                    )
                ),
                condition_chunk_size=int(
                    getattr(
                        self.dec_config,
                        "topology_relation_condition_chunk_size",
                        16,
                    )
                ),
            ).to(device=module_device, dtype=module_dtype)
        else:
            self.topology_condition_relation = None

    def configure_st3_group_vlm_refiner(
        self,
        *,
        siglip_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Build the independent st3 Group-in-VLM modules exactly once."""

        if not getattr(self.dec_config, "use_st3_group_vlm_refiner", False):
            return
        if getattr(self.dec_config, "use_topology_group_decoder", False):
            raise RuntimeError(
                "st3 Group-in-VLM and legacy topology decoding are mutually exclusive"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "st3 Group-in-VLM requires a Mask2Former transformer decoder"
            )
        if getattr(
            self.dec_config,
            "st3_group_require_expected_shapes",
            True,
        ):
            expected = {
                "num_queries": 200,
                "prediction_stages": 10,
                "decoder_layers": 9,
            }
            actual = {
                "num_queries": int(self.decoder.config.num_queries),
                "prediction_stages": int(
                    self.decoder.config.decoder_layers
                ),
                "decoder_layers": len(self.decoder.decoder.layers),
            }
            if actual != expected:
                raise ValueError(
                    "the paper st3 Group-in-VLM contract requires "
                    f"{expected}, got {actual}"
                )
        siglip_hidden_dim = int(siglip_hidden_dim)
        llm_hidden_dim = int(llm_hidden_dim)
        if siglip_hidden_dim <= 0 or llm_hidden_dim <= 0:
            raise ValueError("SigLIP and LLM hidden dimensions must be positive")
        existing = self.st3_proposal_builder
        if existing is not None:
            actual_siglip = int(existing.siglip_region_projector.in_features)
            actual_llm = int(existing.tokenizer.token_mlp[-1].out_features)
            if (actual_siglip, actual_llm) != (
                siglip_hidden_dim,
                llm_hidden_dim,
            ):
                raise ValueError(
                    "st3 Group-in-VLM was already configured for different "
                    f"widths: {(actual_siglip, actual_llm)} != "
                    f"{(siglip_hidden_dim, llm_hidden_dim)}"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        sam_feature_dim = int(self.decoder.config.mask_feature_size)
        num_heads = int(self.decoder.config.st3_group_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        self.st3_proposal_builder = St3ProposalBuilder(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_group_mask_threshold
            ),
            iou_threshold=float(
                self.decoder.config.st3_group_iou_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_group_fourier_features
            ),
        ).to(**common_kwargs)
        self.st3_vlm_refiner = FixedGroupVLMRefiner(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        ).to(**common_kwargs)
        self.st3_group_classifier = GroupConditionClassifier(
            float32_open_vocab=True
        ).to(**common_kwargs)
        self.st3_group_prediction_adapter = GroupPredictionAdapter(
            invalid_mask_logit=float(
                self.decoder.config.topology_invalid_mask_logit
            )
        ).to(**common_kwargs)
        self.st3_group_criterion = HierarchicalGroupCriterion(
            class_weight=self.decoder.config.class_weight,
            mask_weight=self.decoder.config.mask_weight,
            dice_weight=self.decoder.config.dice_weight,
            no_object_weight=self.decoder.config.no_object_weight,
            loss_cls_type=self.decoder.config.loss_cls_type,
            alpha=self.decoder.config.alpha,
            gamma=self.decoder.config.gamma,
            train_num_points=self.decoder.config.train_num_points,
            oversample_ratio=self.decoder.config.oversample_ratio,
            importance_sample_ratio=(
                self.decoder.config.importance_sample_ratio
            ),
            use_sample_point=self.decoder.config.use_sample_point,
            group_shape_temperature=(
                self.decoder.config.group_shape_temperature
            ),
            group_quality_temperature=(
                self.decoder.config.group_quality_temperature
            ),
            quality_iou_query_chunk_size=(
                self.decoder.config.quality_iou_query_chunk_size
            ),
            quality_reg_weight=0.0,
            quality_rank_weight=0.0,
            unmatched_coverage_weight=0.0,
            group_loss_stage_weights=(
                self.decoder.config.st3_group_loss_stage_weights
            ),
            topology_group_version=2,
            member_selector_weight=float(
                self.decoder.config.st3_group_member_selector_weight
            ),
            member_selector_tie_epsilon=float(
                self.decoder.config.member_selector_tie_epsilon
            ),
        ).to(**common_kwargs)
        self.dec_config.siglip_hidden_dim = siglip_hidden_dim

    def configure_st3_proposal_latent_bridge(
        self,
        *,
        siglip_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Build the single-pass 200->64->200 proposal bridge once."""

        if not getattr(
            self.dec_config,
            "use_st3_proposal_latent_bridge",
            False,
        ):
            return
        if (
            getattr(self.dec_config, "use_topology_group_decoder", False)
            or getattr(
                self.dec_config,
                "use_st3_group_vlm_refiner",
                False,
            )
            or getattr(
                self.dec_config,
                "use_st3_bipartite_latent_transport",
                False,
            )
        ):
            raise RuntimeError(
                "the st3 proposal latent bridge is mutually exclusive with "
                "all hard/topology Group decoder paths"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "the st3 proposal latent bridge requires Mask2Former"
            )
        if getattr(
            self.dec_config,
            "st3_latent_require_expected_shapes",
            True,
        ):
            expected = {
                "num_queries": 200,
                "prediction_stages": 10,
                "decoder_layers": 9,
            }
            actual = {
                "num_queries": int(self.decoder.config.num_queries),
                "prediction_stages": int(
                    self.decoder.config.decoder_layers
                ),
                "decoder_layers": len(self.decoder.decoder.layers),
            }
            if actual != expected:
                raise ValueError(
                    "the proposal-latent paper contract requires "
                    f"{expected}, got {actual}"
                )
            expected_mixer_depth = len(self.decoder.decoder.layers) - 3
            actual_mixer_depth = int(
                self.decoder.config.st3_latent_persistent_mixer_depth
            )
            if actual_mixer_depth != expected_mixer_depth:
                raise ValueError(
                    "the persistent latent path requires one prediction-before "
                    "Mixer for every st4--st9 layer: "
                    f"{actual_mixer_depth} != {expected_mixer_depth}"
                )
        siglip_hidden_dim = int(siglip_hidden_dim)
        llm_hidden_dim = int(llm_hidden_dim)
        if siglip_hidden_dim <= 0 or llm_hidden_dim <= 0:
            raise ValueError("SigLIP/LLM hidden dimensions must be positive")
        existing = self.st3_latent_proposal_builder
        if existing is not None:
            actual = (
                int(existing.siglip_projector.in_features),
                int(existing.llm_hidden_dim),
            )
            expected = (siglip_hidden_dim, llm_hidden_dim)
            if actual != expected:
                raise ValueError(
                    "the proposal latent bridge was already configured for "
                    f"different widths: {actual} != {expected}"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        sam_feature_dim = int(self.decoder.config.mask_feature_size)
        num_heads = int(self.decoder.config.st3_latent_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        self.st3_latent_proposal_builder = St3ProposalLatentBuilder(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_latents=int(self.decoder.config.st3_latent_num_tokens),
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_latent_mask_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_latent_fourier_features
            ),
            pre_vlm_depth=int(
                self.decoder.config.st3_latent_pre_vlm_depth
            ),
            gate_init_bias=float(
                self.decoder.config.st3_latent_gate_init_bias
            ),
        ).to(**common_kwargs)
        self.st3_latent_bridge = PosteriorLatentQueryBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            post_vlm_depth=int(
                self.decoder.config.st3_latent_post_vlm_depth
            ),
            persistent_mixer_depth=int(
                self.decoder.config.st3_latent_persistent_mixer_depth
            ),
            gate_init_bias=float(
                self.decoder.config.st3_latent_gate_init_bias
            ),
        ).to(**common_kwargs)
        self.dec_config.siglip_hidden_dim = siglip_hidden_dim

    def configure_st3_bipartite_latent_transport(
        self,
        *,
        siglip_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Build the independent gate-free 200<->64 transport path once."""

        cascade_enabled = getattr(self.dec_config, "use_st123_latent_cascade", False)
        if not isinstance(cascade_enabled, bool):
            raise TypeError("use_st123_latent_cascade must be a bool")
        if cascade_enabled and not getattr(
            self.dec_config, "use_st3_bipartite_latent_transport", False
        ):
            raise ValueError("st123 latent cascade requires bipartite transport")
        if not getattr(
            self.dec_config,
            "use_st3_bipartite_latent_transport",
            False,
        ):
            return
        if any(
            bool(getattr(self.dec_config, name, False))
            for name in (
                "use_topology_group_decoder",
                "use_st3_group_vlm_refiner",
                "use_st3_proposal_latent_bridge",
            )
        ):
            raise RuntimeError(
                "bipartite latent transport is mutually exclusive with all "
                "topology, hard-Group, and gated latent paths"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "bipartite latent transport requires Mask2Former"
            )
        if getattr(
            self.dec_config,
            "st3_transport_require_expected_shapes",
            True,
        ):
            expected = {
                "num_queries": 200,
                "prediction_stages": 10,
                "decoder_layers": 9,
            }
            actual = {
                "num_queries": int(self.decoder.config.num_queries),
                "prediction_stages": int(
                    self.decoder.config.decoder_layers
                ),
                "decoder_layers": len(self.decoder.decoder.layers),
            }
            if actual != expected:
                raise ValueError(
                    "the bipartite latent paper contract requires "
                    f"{expected}, got {actual}"
                )
            expected_depth = len(self.decoder.decoder.layers) - 3
            actual_depth = int(
                self.decoder.config.st3_transport_decoder_depth
            )
            if actual_depth != expected_depth:
                raise ValueError(
                    "bipartite transport requires one module for every "
                    f"st4--st9 layer: {actual_depth} != {expected_depth}"
                )

        siglip_hidden_dim = int(siglip_hidden_dim)
        llm_hidden_dim = int(llm_hidden_dim)
        if siglip_hidden_dim <= 0 or llm_hidden_dim <= 0:
            raise ValueError("SigLIP/LLM hidden dimensions must be positive")
        existing = self.st3_transport_proposal_builder
        st123_enabled = bool(getattr(
            self.dec_config, "use_st123_latent_deepstack", False
        ))
        if cascade_enabled and st123_enabled:
            raise ValueError("st123 latent cascade and Phi deepstack are mutually exclusive")
        builder_type = (
            St123LatentCascadeBuilder if cascade_enabled else (
                St123LatentDeepstackBuilder if st123_enabled else St3BipartiteLatentBuilder
            )
        )
        enable_latent_writeback = getattr(
            self.dec_config, "st3_transport_enable_latent_writeback", True
        )
        if not isinstance(enable_latent_writeback, bool):
            raise TypeError("st3_transport_enable_latent_writeback must be a bool")
        late_refresh_stages = tuple(getattr(
            self.dec_config, "st3_transport_late_condition_refresh_stages", ()
        ))
        if not enable_latent_writeback and late_refresh_stages:
            raise ValueError("read-only latent transport cannot enable late refresh")
        if (st123_enabled or cascade_enabled) and tuple(getattr(
            self.dec_config, "st3_transport_late_condition_refresh_stages", ()
        )):
            mode = "cascade" if cascade_enabled else "deepstack"
            raise ValueError(f"st123 latent {mode} cannot enable late refresh")
        if existing is not None:
            if st123_enabled != isinstance(existing, St123LatentDeepstackBuilder):
                raise ValueError("cannot change the configured st123 latent mode")
            if not isinstance(existing, builder_type):
                raise ValueError("cannot change the configured st123 latent builder mode")
            if (
                self.st3_transport_bridge.enable_latent_writeback
                != enable_latent_writeback
            ):
                raise ValueError("cannot change the configured latent writeback mode")
            actual = (
                int(existing.siglip_projector.in_features),
                int(existing.llm_hidden_dim),
            )
            expected = (siglip_hidden_dim, llm_hidden_dim)
            if actual != expected:
                raise ValueError(
                    "bipartite latent transport was already configured for "
                    f"different widths: {actual} != {expected}"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        sam_feature_dim = int(self.decoder.config.mask_feature_size)
        num_heads = int(self.decoder.config.st3_latent_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        self.st3_transport_proposal_builder = builder_type(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_latents=int(self.decoder.config.st3_latent_num_tokens),
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_latent_mask_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_latent_fourier_features
            ),
            use_sam_region=bool(getattr(
                self.decoder.config,
                "st3_latent_use_sam_region",
                True,
            )),
            use_siglip_region=bool(getattr(
                self.decoder.config,
                "st3_latent_use_siglip_region",
                True,
            )),
            use_xywh=bool(getattr(
                self.decoder.config,
                "st3_latent_use_xywh",
                True,
            )),
            pre_vlm_depth=int(
                self.decoder.config.st3_transport_pre_vlm_depth
            ),
        ).to(**common_kwargs)
        self.st3_transport_bridge = BipartiteLatentTransportBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            post_vlm_depth=int(
                self.decoder.config.st3_transport_post_vlm_depth
            ),
            transport_depth=int(
                self.decoder.config.st3_transport_decoder_depth
            ),
            output_init_std=float(
                self.decoder.config.st3_transport_output_init_std
            ),
            enable_latent_writeback=enable_latent_writeback,
            late_condition_refresh_stages=tuple(
                getattr(
                    self.decoder.config,
                    "st3_transport_late_condition_refresh_stages",
                    (),
                )
            ),
        ).to(**common_kwargs)
        self.dec_config.siglip_hidden_dim = siglip_hidden_dim

    def configure_latent_s2_post_vlm_transport(
        self,
        *,
        siglip_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Reuse latent S2 and replace only its post-VLM st4--st9 backend."""

        if not bool(getattr(
            self.dec_config,
            "use_latent_s2_post_vlm_transport",
            False,
        )):
            return
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
            if bool(getattr(self.dec_config, name, False))
        ]
        if enabled:
            raise RuntimeError(
                "latent-S2 post-VLM transport is mutually exclusive with "
                f"{enabled}"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "latent-S2 post-VLM transport requires Mask2Former"
            )

        expected = {
            "num_queries": 200,
            "prediction_stages": 10,
            "decoder_layers": 9,
        }
        actual = {
            "num_queries": int(self.decoder.config.num_queries),
            "prediction_stages": int(self.decoder.config.decoder_layers),
            "decoder_layers": len(self.decoder.decoder.layers),
        }
        if bool(getattr(
            self.dec_config,
            "latent_s2_post_vlm_require_expected_shapes",
            True,
        )) and actual != expected:
            raise ValueError(
                "latent-S2 post-VLM experiment requires "
                f"{expected}, got {actual}"
            )
        transport_depth = int(
            self.dec_config.latent_s2_post_vlm_transport_depth
        )
        expected_depth = len(self.decoder.decoder.layers) - 3
        if transport_depth != expected_depth:
            raise ValueError(
                "post-VLM transport needs one module per st4--st9 layer: "
                f"{transport_depth} != {expected_depth}"
            )

        siglip_hidden_dim = int(siglip_hidden_dim)
        llm_hidden_dim = int(llm_hidden_dim)
        if min(siglip_hidden_dim, llm_hidden_dim) <= 0:
            raise ValueError("SigLIP/LLM hidden dimensions must be positive")
        transport_type = str(
            self.dec_config.latent_s2_post_vlm_transport_type
        )
        if self.st3_transport_proposal_builder is not None:
            actual_widths = (
                int(self.st3_transport_proposal_builder.siglip_projector.in_features),
                int(self.st3_transport_proposal_builder.llm_hidden_dim),
            )
            expected_widths = (siglip_hidden_dim, llm_hidden_dim)
            if actual_widths != expected_widths:
                raise ValueError(
                    "latent S2 builder was configured for different widths: "
                    f"{actual_widths} != {expected_widths}"
                )
            if self.latent_s2_post_vlm_bridge is None or (
                self.latent_s2_post_vlm_bridge.transport_type
                != transport_type
            ):
                raise ValueError(
                    "latent-S2 post-VLM bridge was configured inconsistently"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        sam_feature_dim = int(self.decoder.config.mask_feature_size)
        num_heads = int(self.decoder.config.st3_latent_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        # Keep this exact module name and construction contract so the trained
        # Latent S2 checkpoint overlays without remapping any parameter key.
        self.st3_transport_proposal_builder = St3BipartiteLatentBuilder(
            hidden_dim=hidden_dim,
            sam_feature_dim=sam_feature_dim,
            siglip_feature_dim=siglip_hidden_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_latents=int(self.decoder.config.st3_latent_num_tokens),
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_latent_mask_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_latent_fourier_features
            ),
            pre_vlm_depth=int(
                self.decoder.config.st3_transport_pre_vlm_depth
            ),
        ).to(**common_kwargs)
        self.latent_s2_post_vlm_bridge = LatentS2PostVLMBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transport_type=transport_type,
            post_vlm_depth=int(
                self.decoder.config.st3_transport_post_vlm_depth
            ),
            transport_depth=transport_depth,
            output_init_std=float(
                self.decoder.config.latent_s2_post_vlm_output_init_std
            ),
            direct_endpoint_ffn_depth=int(getattr(
                self.decoder.config,
                "latent_s2_direct_endpoint_ffn_depth",
                2,
            )),
        ).to(**common_kwargs)
        self.dec_config.siglip_hidden_dim = siglip_hidden_dim

    def configure_stagewise_latent_trifusion(
        self,
        *,
        siglip_hidden_dim: int,
        sam_hidden_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Build the opt-in st0--st3 recurrent latent and st4--st9 tri-fusion."""

        if not getattr(
            self.dec_config,
            "use_stagewise_latent_trifusion",
            False,
        ):
            return
        incompatible = (
            "use_topology_group_decoder",
            "use_st3_group_vlm_refiner",
            "use_st3_proposal_latent_bridge",
            "use_st3_bipartite_latent_transport",
        )
        enabled = [
            name for name in incompatible
            if bool(getattr(self.dec_config, name, False))
        ]
        if enabled:
            raise RuntimeError(
                "stagewise latent trifusion is mutually exclusive with "
                f"{enabled}"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError("stagewise latent trifusion requires Mask2Former")

        expected = {
            "num_queries": 200,
            "prediction_stages": 10,
            "decoder_layers": 9,
        }
        actual = {
            "num_queries": int(self.decoder.config.num_queries),
            "prediction_stages": int(self.decoder.config.decoder_layers),
            "decoder_layers": len(self.decoder.decoder.layers),
        }
        if bool(getattr(
            self.dec_config,
            "stagewise_trifusion_require_expected_shapes",
            True,
        )) and actual != expected:
            raise ValueError(
                "stagewise latent trifusion requires "
                f"{expected}, got {actual}"
            )
        transport_depth = int(
            self.dec_config.stagewise_trifusion_decoder_depth
        )
        expected_depth = len(self.decoder.decoder.layers) - 3
        if transport_depth != expected_depth:
            raise ValueError(
                "stagewise trifusion needs one module per st4--st9 layer: "
                f"{transport_depth} != {expected_depth}"
            )

        widths = (
            int(siglip_hidden_dim),
            int(sam_hidden_dim),
            int(llm_hidden_dim),
        )
        if min(widths) <= 0:
            raise ValueError("SigLIP/SAM/LLM hidden dimensions must be positive")
        if self.stagewise_latent_builder is not None:
            existing = self.stagewise_latent_builder
            actual_widths = (
                int(existing.siglip_feature_dim),
                int(existing.sam_feature_dim),
                int(existing.llm_hidden_dim),
            )
            if actual_widths != widths:
                raise ValueError(
                    "stagewise latent builder was already configured for "
                    f"{actual_widths}, expected {widths}"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        num_heads = int(self.decoder.config.st3_latent_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        self.stagewise_latent_builder = StagewiseProposalLatentBuilder(
            hidden_dim=hidden_dim,
            sam_feature_dim=int(sam_hidden_dim),
            siglip_feature_dim=int(siglip_hidden_dim),
            llm_hidden_dim=int(llm_hidden_dim),
            num_latents=int(self.decoder.config.st3_latent_num_tokens),
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_latent_mask_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_latent_fourier_features
            ),
        ).to(**common_kwargs)
        self.stagewise_trifusion_bridge = StagewiseLatentTriFusionBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transport_depth=transport_depth,
            gate_init_bias=float(
                self.decoder.config.stagewise_trifusion_gate_init_bias
            ),
            keep_background_fixed=bool(
                self.decoder.config.stagewise_trifusion_keep_background_fixed
            ),
        ).to(**common_kwargs)
        self.dec_config.siglip_hidden_dim = int(siglip_hidden_dim)
        self.dec_config.stagewise_sam_hidden_dim = int(sam_hidden_dim)

    def configure_geometry_proposal_transport(
        self,
        *,
        fused_feature_dim: int,
        llm_hidden_dim: int,
    ) -> None:
        """Build the isolated st0--st3 Q/Z and st4--st9 tripartite path."""

        if not bool(getattr(
            self.dec_config,
            "use_proposal_mediated_tripartite_transport",
            False,
        )):
            return
        incompatible = (
            "use_topology_group_decoder",
            "use_st3_group_vlm_refiner",
            "use_st3_proposal_latent_bridge",
            "use_st3_bipartite_latent_transport",
            "use_stagewise_latent_trifusion",
        )
        enabled = [
            name for name in incompatible
            if bool(getattr(self.dec_config, name, False))
        ]
        if enabled:
            raise RuntimeError(
                "geometry proposal transport is mutually exclusive with "
                f"{enabled}"
            )
        if not bool(getattr(
            self.dec_config,
            "use_geometry_aligned_visual_fusion",
            False,
        )):
            raise RuntimeError(
                "geometry proposal transport requires geometry-aligned visual fusion"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError("geometry proposal transport requires Mask2Former")

        expected = {
            "num_queries": 200,
            "prediction_stages": 10,
            "decoder_layers": 9,
        }
        actual = {
            "num_queries": int(self.decoder.config.num_queries),
            "prediction_stages": int(self.decoder.config.decoder_layers),
            "decoder_layers": len(self.decoder.decoder.layers),
        }
        if bool(getattr(
            self.dec_config,
            "geometry_transport_require_expected_shapes",
            True,
        )) and actual != expected:
            raise ValueError(
                "geometry proposal transport requires "
                f"{expected}, got {actual}"
            )
        transport_depth = int(
            self.dec_config.geometry_transport_decoder_depth
        )
        expected_depth = len(self.decoder.decoder.layers) - 3
        if transport_depth != expected_depth:
            raise ValueError(
                "geometry transport needs one module per st4--st9 layer: "
                f"{transport_depth} != {expected_depth}"
            )
        prefix_transport_depth = int(
            self.dec_config.geometry_prefix_transport_depth
        )
        if prefix_transport_depth != 3:
            raise ValueError(
                "geometry prefix transport needs one module for each of "
                "st1--st3"
            )

        fused_feature_dim = int(fused_feature_dim)
        llm_hidden_dim = int(llm_hidden_dim)
        if min(fused_feature_dim, llm_hidden_dim) <= 0:
            raise ValueError("fused/LLM hidden dimensions must be positive")
        if self.geometry_proposal_builder is not None:
            actual_widths = (
                int(self.geometry_proposal_builder.fused_feature_dim),
                int(self.geometry_proposal_builder.llm_hidden_dim),
            )
            expected_widths = (fused_feature_dim, llm_hidden_dim)
            if actual_widths != expected_widths:
                raise ValueError(
                    "geometry proposal builder was already configured for "
                    f"{actual_widths}, expected {expected_widths}"
                )
            if len(
                self.geometry_proposal_builder.prefix_transport_stages
            ) != prefix_transport_depth:
                raise ValueError(
                    "configured geometry builder has the wrong st1--st3 "
                    "transport depth"
                )
            if self.geometry_transport_bridge is None or len(
                self.geometry_transport_bridge.transport_stages
            ) != transport_depth:
                raise ValueError(
                    "configured geometry bridge has the wrong st4--st9 "
                    "transport depth"
                )
            return

        hidden_dim = int(self.decoder.config.hidden_dim)
        num_heads = int(self.decoder.config.st3_latent_num_heads)
        reference = next(self.decoder.parameters())
        common_kwargs = dict(device=reference.device, dtype=reference.dtype)
        self.geometry_proposal_builder = FusedProposalBuilder(
            hidden_dim=hidden_dim,
            fused_feature_dim=fused_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_latents=int(self.decoder.config.st3_latent_num_tokens),
            num_heads=num_heads,
            mask_threshold=float(
                self.decoder.config.st3_latent_mask_threshold
            ),
            fourier_features=int(
                self.decoder.config.st3_latent_fourier_features
            ),
            pre_vlm_depth=int(
                self.decoder.config.st3_transport_pre_vlm_depth
            ),
            prefix_transport_depth=prefix_transport_depth,
            output_init_std=float(
                self.decoder.config.geometry_transport_output_init_std
            ),
        ).to(**common_kwargs)
        self.geometry_transport_bridge = ProposalMediatedTripartiteBridge(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transport_depth=transport_depth,
            output_init_std=float(
                self.decoder.config.geometry_transport_output_init_std
            ),
        ).to(**common_kwargs)
        self.dec_config.geometry_fused_feature_dim = fused_feature_dim

    def enable_input_require_grads(self):
        self.disable_input_require_grads()

        def make_inputs_require_grad(module, input, output):
            if isinstance(output, Tensor):
                output.requires_grad_(True)
            elif isinstance(output, tuple):
                output[0].requires_grad_(True)

        hook = self.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad
        )
        self._require_grads_hook = hook
        self._require_grads_hooks = [hook]

    def disable_input_require_grads(self):
        """Remove input-gradient hooks across old and new Transformers."""

        hooks = list(getattr(self, "_require_grads_hooks", ()) or ())
        legacy_hook = getattr(self, "_require_grads_hook", None)
        if legacy_hook is not None and all(
            legacy_hook is not hook for hook in hooks
        ):
            hooks.append(legacy_hook)
        seen = set()
        for hook in hooks:
            if id(hook) not in seen:
                hook.remove()
                seen.add(id(hook))
        self._require_grads_hooks = []
        if hasattr(self, "_require_grads_hook"):
            del self._require_grads_hook

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.encoder, "patch_embed"):
            return self.encoder.patch_embed
        elif hasattr(self.encoder, "embeddings"):
            return self.encoder.embeddings.patch_embeddings
        else:
            raise ValueError(f"Unsupported encoder: {type(self.encoder)}")

    def _init_weights(self, module: nn.Module):
        xavier_std = self.dec_config.init_xavier_std if hasattr(self.dec_config, "init_xavier_std") else 1.0
        std = self.dec_config.init_std if hasattr(self.dec_config, "init_std") else 0.02

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    @torch.no_grad()
    def get_image_wide_positional_embeddings(self):
        size = self.prompt_enc_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []  # type: ignore

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )

        return auxiliary_logits

    def get_class_prediction(
        self,
        query_embeddings,
        cond_embeddings,
        embed_masks=None,
        finite_mask_logits=False,
    ):
        if cond_embeddings is None:
            return self.class_predictor(query_embeddings)

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        cond_embeddings = F.normalize(cond_embeddings, dim=-1)
        cls_pred = self.logit_scale.exp() * torch.einsum("bqd,bcd->bqc", query_embeddings, cond_embeddings)
        cls_pred = torch.clamp(cls_pred, min=-500, max=500)

        if embed_masks is not None:
            if embed_masks.ndim == 2:
                embed_masks = embed_masks[:, None, :]
            embed_masks = embed_masks.to(torch.bool)
            invalid_logit = -1e9
            if finite_mask_logits:
                invalid_logit = -float(
                    min(1.0e4, torch.finfo(cls_pred.dtype).max / 2.0)
                )
            cls_pred = cls_pred.masked_fill(~embed_masks, invalid_logit)

        return cls_pred

    def _classify_topology_groups(
        self,
        group_stage_outputs: Tuple[TopologyGroupStageOutput, ...],
        cond_embeddings: Optional[Tensor],
        embed_masks: Optional[Tensor],
    ) -> Tuple[TopologyGroupStageOutput, ...]:
        """Attach condition/BG logits to every stage's sparse group table."""

        if self.group_classifier is None:
            raise RuntimeError("topology group classifier has not been configured")
        if not self.use_cls:
            raise RuntimeError(
                "topology group decoding requires open- or closed-set classification"
            )

        topology_group_version = int(
            getattr(self.dec_config, "topology_group_version", 1)
        )
        # V1, and the closed-set path for which there is no immutable C0,
        # retain the original independent per-stage classifier exactly.
        if topology_group_version != 2 or cond_embeddings is None:
            classified = []
            for expected_stage, stage_output in enumerate(group_stage_outputs):
                if stage_output.stage_index != expected_stage:
                    raise AssertionError(
                        "topology stages must be ordered st0..st9; "
                        f"position {expected_stage} stores "
                        f"st{stage_output.stage_index}"
                    )
                batch_size = stage_output.group_features.shape[0]
                if (
                    cond_embeddings is not None
                    and cond_embeddings.shape[0] != batch_size
                ):
                    raise AssertionError(
                        "expanded condition batch must equal topology query "
                        f"batch: {cond_embeddings.shape[0]} != {batch_size}"
                    )
                if embed_masks is not None and embed_masks.shape[0] != batch_size:
                    raise AssertionError(
                        "condition-padding mask batch must equal topology "
                        f"query batch: {embed_masks.shape[0]} != {batch_size}"
                    )
                stage_output.group_class_logits = self.group_classifier(
                    group_features=stage_output.group_features,
                    group_valid_mask=stage_output.group_valid_mask,
                    cond_embeddings=cond_embeddings,
                    embed_masks=embed_masks,
                    logit_scale=getattr(self, "logit_scale", None),
                    class_predictor=getattr(self, "class_predictor", None),
                )
                classified.append(stage_output)
            return tuple(classified)

        if self.topology_condition_relation is None:
            raise RuntimeError(
                "topology V2 condition-relation Bank has not been configured"
            )

        anchors = []
        for expected_stage, stage_output in enumerate(group_stage_outputs):
            if stage_output.stage_index != expected_stage:
                raise AssertionError(
                    "topology stages must be ordered st0..st9; "
                    f"position {expected_stage} stores st{stage_output.stage_index}"
                )
            batch_size = stage_output.group_features.shape[0]
            if cond_embeddings is not None and cond_embeddings.shape[0] != batch_size:
                raise AssertionError(
                    "expanded condition batch must equal topology query batch: "
                    f"{cond_embeddings.shape[0]} != {batch_size}"
                )
            if embed_masks is not None and embed_masks.shape[0] != batch_size:
                raise AssertionError(
                    "condition-padding mask batch must equal topology query batch: "
                    f"{embed_masks.shape[0]} != {batch_size}"
                )
            anchors.append(
                self.group_classifier(
                    group_features=stage_output.group_features,
                    group_valid_mask=stage_output.group_valid_mask,
                    cond_embeddings=cond_embeddings,
                    embed_masks=embed_masks,
                    logit_scale=getattr(self, "logit_scale", None),
                    class_predictor=getattr(self, "class_predictor", None),
                )
            )
        relation_output = self.topology_condition_relation(
            stage_outputs=group_stage_outputs,
            cond_embeddings=cond_embeddings,
            anchor_logits=tuple(anchors),
            embed_masks=embed_masks,
        )
        classified = []
        for stage_output, stage_logits in zip(
            group_stage_outputs,
            relation_output.stage_logits,
        ):
            stage_output.group_class_logits = stage_logits
            classified.append(stage_output)
        return tuple(classified)

    @staticmethod
    def _assert_topology_target_batches(
        effective_batch_size: int,
        mask_labels: Optional[List[Tensor]],
        class_labels: Optional[List[Tensor]],
    ) -> None:
        """Verify cond_lens expansion kept prediction and target order aligned."""

        if mask_labels is not None and len(mask_labels) != effective_batch_size:
            raise AssertionError(
                "expanded mask-label batch must equal SAM/query/SigLIP batch: "
                f"{len(mask_labels)} != {effective_batch_size}"
            )
        if class_labels is not None and len(class_labels) != effective_batch_size:
            raise AssertionError(
                "expanded class-label batch must equal SAM/query/SigLIP batch: "
                f"{len(class_labels)} != {effective_batch_size}"
            )

    def postprocess_masks_preds(
        self,
        masks_preds,
        sam_valid_boxes_normalized: Optional[Tensor] = None,
    ):
        """Upscale masks while excluding SAM padding from interpolation.

        SAM-valid resizing is a spatial operation shared by ordinary Query,
        hard-Group, and proposal-latent predictions.  It must therefore not
        depend on a Group prediction adapter being configured.
        """

        new_masks_preds = []
        for masks_pred in masks_preds:
            output_size = (
                self.enc_config.image_size,
                self.enc_config.image_size,
            )
            if sam_valid_boxes_normalized is None:
                masks_pred = F.interpolate(
                    masks_pred,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                boxes = validate_normalized_valid_boxes(
                    sam_valid_boxes_normalized,
                    batch_size=masks_pred.shape[0],
                    device=masks_pred.device,
                )
                source_valid = valid_mask_from_normalized_boxes(
                    boxes,
                    masks_pred.shape[-2:],
                )
                target_valid = valid_mask_from_normalized_boxes(
                    boxes,
                    output_size,
                )
                resized, _ = masked_normalized_resize(
                    masks_pred,
                    source_valid,
                    output_size,
                    target_valid_mask=target_valid,
                )
                invalid_logit = float(
                    getattr(
                        getattr(self, "dec_config", None),
                        "topology_invalid_mask_logit",
                        -20.0,
                    )
                )
                if not float("-inf") < invalid_logit < float("inf"):
                    raise ValueError(
                        "topology_invalid_mask_logit must be finite, got "
                        f"{invalid_logit}"
                    )
                masks_pred = torch.where(
                    target_valid,
                    resized,
                    torch.full_like(resized, invalid_logit),
                ).to(dtype=masks_pred.dtype)
            new_masks_preds.append(masks_pred)

        return new_masks_preds

    def prepare_st3_group_execution(
        self,
        *,
        image_embeddings: Tuple[Tensor],
        sam_spatial_features: Optional[Tensor],
        siglip_spatial_features: Tensor,
        siglip_valid_mask: Tensor,
        spatial_metadata: Dict[str, Tensor],
        output_attentions: bool = False,
    ) -> St3GroupExecutionBundle:
        """Run image-only Mask2Former st0--st3 and create fixed Groups."""

        if not getattr(self.dec_config, "use_st3_group_vlm_refiner", False):
            raise RuntimeError("the st3 Group-in-VLM path is not enabled")
        if self.st3_proposal_builder is None or self.st3_vlm_refiner is None:
            raise RuntimeError(
                "st3 Group-in-VLM modules have not been configured by MaskLATModel"
            )
        if image_embeddings is None:
            raise ValueError(
                "the staged path requires pre-extracted SAM image embeddings"
            )
        if sam_spatial_features is None or sam_spatial_features.ndim != 4:
            raise ValueError(
                "the staged path requires raw SAM spatial features [B,C,H,W]"
            )
        if siglip_spatial_features is None or spatial_metadata is None:
            raise ValueError(
                "the staged path requires SigLIP spatial features and transforms"
            )
        pixel_decoder_outputs = self.pixel_decoder(
            image_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        context = self.decoder.prepare_decoder_context(
            pixel_decoder_outputs.multi_scale_features,
            pixel_decoder_outputs.mask_features,
            siglip_spatial_features=siglip_spatial_features,
            siglip_valid_mask=siglip_valid_mask,
            spatial_metadata=spatial_metadata,
        )
        stage3_state = self.decoder.decoder.forward_to_stage3(
            context,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        proposal = self.st3_proposal_builder(
            query_states=stage3_state.normalized_query_states.transpose(0, 1),
            mask_logits=stage3_state.mask_logits,
            sam_mask_features=sam_spatial_features,
            siglip_spatial_features=siglip_spatial_features,
            siglip_valid_mask=siglip_valid_mask,
            spatial_metadata=spatial_metadata,
        )
        return St3GroupExecutionBundle(
            decoder_context=context,
            stage3_state=stage3_state,
            proposal=proposal,
        )

    def prepare_st3_latent_execution(
        self,
        *,
        image_embeddings: Tuple[Tensor],
        sam_spatial_features: Optional[Tensor],
        siglip_spatial_features: Tensor,
        siglip_valid_mask: Tensor,
        spatial_metadata: Dict[str, Tensor],
        stagewise_sam_spatial_features: Optional[Sequence[Tensor]] = None,
        stagewise_siglip_spatial_features: Optional[Sequence[Tensor]] = None,
        geometry_fused_features_64: Optional[Tensor] = None,
        geometry_fused_valid_mask_64: Optional[Tensor] = None,
        output_attentions: bool = False,
        freeze_proposal_source: bool = False,
    ) -> St3LatentExecutionBundle:
        """Run image-only st0--st3 and construct exactly 64 soft latents."""

        cascade_enabled = getattr(self.dec_config, "use_st123_latent_cascade", False)
        if not isinstance(cascade_enabled, bool):
            raise TypeError("use_st123_latent_cascade must be a bool")
        if cascade_enabled and bool(getattr(
            self.dec_config, "use_st123_latent_deepstack", False
        )):
            raise ValueError("st123 latent cascade and Phi deepstack are mutually exclusive")
        gated_enabled = bool(
            getattr(
                self.dec_config,
                "use_st3_proposal_latent_bridge",
                False,
            )
        )
        transport_enabled = bool(
            getattr(
                self.dec_config,
                "use_st3_bipartite_latent_transport",
                False,
            )
        )
        if cascade_enabled and not transport_enabled:
            raise ValueError("st123 latent cascade requires bipartite transport")
        post_vlm_enabled = bool(getattr(
            self.dec_config,
            "use_latent_s2_post_vlm_transport",
            False,
        ))
        stagewise_enabled = bool(
            getattr(
                self.dec_config,
                "use_stagewise_latent_trifusion",
                False,
            )
        )
        geometry_enabled = bool(
            getattr(
                self.dec_config,
                "use_proposal_mediated_tripartite_transport",
                False,
            )
        )
        if sum((
            gated_enabled,
            transport_enabled,
            post_vlm_enabled,
            stagewise_enabled,
            geometry_enabled,
        )) != 1:
            raise RuntimeError(
                "exactly one gated, bipartite, latent-S2-post-VLM, stagewise, "
                "or geometry latent path must be enabled"
            )
        if gated_enabled:
            proposal_builder = self.st3_latent_proposal_builder
            bridge = self.st3_latent_bridge
        elif transport_enabled:
            proposal_builder = self.st3_transport_proposal_builder
            bridge = self.st3_transport_bridge
        elif post_vlm_enabled:
            proposal_builder = self.st3_transport_proposal_builder
            bridge = self.latent_s2_post_vlm_bridge
        elif stagewise_enabled:
            proposal_builder = self.stagewise_latent_builder
            bridge = self.stagewise_trifusion_bridge
        else:
            proposal_builder = self.geometry_proposal_builder
            bridge = self.geometry_transport_bridge
        if proposal_builder is None or bridge is None:
            raise RuntimeError("the selected proposal latent modules are not configured")
        if image_embeddings is None:
            raise ValueError("the staged path requires SAM image embeddings")
        if sam_spatial_features is None or sam_spatial_features.ndim != 4:
            raise ValueError("raw SAM spatial features must be [B,C,H,W]")
        if siglip_spatial_features is None or spatial_metadata is None:
            raise ValueError(
                "SigLIP spatial features and transforms are required"
            )
        if not isinstance(freeze_proposal_source, bool):
            raise TypeError("freeze_proposal_source must be boolean")
        source_grad_enabled = torch.is_grad_enabled() and not freeze_proposal_source
        with torch.set_grad_enabled(source_grad_enabled):
            pixel_decoder_outputs = self.pixel_decoder(
                image_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
            context = self.decoder.prepare_decoder_context(
                pixel_decoder_outputs.multi_scale_features,
                pixel_decoder_outputs.mask_features,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
            )

        if geometry_enabled:
            if geometry_fused_features_64 is None or (
                geometry_fused_valid_mask_64 is None
            ):
                raise ValueError(
                    "geometry latent execution requires fused 64x64 features and validity"
                )
            initial_proposals: List[Any] = []

            def initialize_geometry_latents(
                st0_query_states: Tensor,
                st0_mask_logits: Tensor,
            ) -> Tensor:
                proposal_state = proposal_builder(
                    query_states=st0_query_states,
                    mask_logits=st0_mask_logits,
                    fused_features_64=geometry_fused_features_64,
                    fused_valid_mask_64=geometry_fused_valid_mask_64,
                    spatial_metadata=spatial_metadata,
                )
                initial_proposals.append(proposal_state)
                return proposal_state.latent_features

            def refine_geometry_prefix_latents(
                stage_index: int,
                query_states: Tensor,
                latent_states: Tensor,
                preceding_mask_logits: Tensor,
                fused_features_64: Tensor,
            ) -> Tensor:
                # stage_index 0/1/2 denotes formal st1/st2/st3.  The mask is
                # therefore M0/M1/M2, while query_states has already passed
                # that decoder layer's original visual cross-attention.
                return proposal_builder.refine_prefix_latents(
                    stage_index=stage_index,
                    query_states=query_states,
                    mask_logits=preceding_mask_logits,
                    latent_states=latent_states,
                    fused_features_64=fused_features_64,
                    fused_valid_mask_64=geometry_fused_valid_mask_64,
                    spatial_metadata=spatial_metadata,
                )

            stage3_state, stage3_latents = (
                self.decoder.decoder.forward_to_stage3_with_latent_transport(
                    context,
                    latent_initializer=initialize_geometry_latents,
                    transport_modules=proposal_builder.prefix_transport_stages,
                    prefix_fused_refiner=refine_geometry_prefix_latents,
                    prefix_fused_features=geometry_fused_features_64,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                )
            )
            if len(initial_proposals) != 1:
                raise RuntimeError(
                    "geometry st0 latent initializer must run exactly once"
                )
            proposal = replace(
                initial_proposals[0],
                latent_features=stage3_latents,
                packed_group_tokens=proposal_builder.project_latents_for_vlm(
                    stage3_latents
                ),
            )
        else:
            with torch.set_grad_enabled(source_grad_enabled):
                stage3_state = self.decoder.decoder.forward_to_stage3(
                    context,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                )
        if stagewise_enabled:
            if stagewise_sam_spatial_features is None or (
                stagewise_siglip_spatial_features is None
            ):
                raise ValueError(
                    "stagewise latent execution requires four SAM and SigLIP feature maps"
                )
            proposal = proposal_builder(
                query_states=tuple(
                    query.transpose(0, 1)
                    for query in stage3_state.intermediate_hidden_states
                ),
                mask_logits=tuple(stage3_state.masks_queries_logits),
                sam_spatial_features=tuple(stagewise_sam_spatial_features),
                siglip_spatial_features=tuple(
                    stagewise_siglip_spatial_features
                ),
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
            )
        elif cascade_enabled or bool(getattr(
            self.dec_config, "use_st123_latent_deepstack", False
        )):
            if not transport_enabled:
                raise RuntimeError("st123 latent deepstack requires bipartite transport")
            if len(stage3_state.intermediate_hidden_states) != 4 or (
                len(stage3_state.masks_queries_logits) != 4
            ):
                raise RuntimeError("st123 requires the complete st0--st3 prefix")
            proposal = proposal_builder(
                query_states=tuple(
                    query.transpose(0, 1)
                    for query in stage3_state.intermediate_hidden_states[1:4]
                ),
                mask_logits=tuple(stage3_state.masks_queries_logits[1:4]),
                sam_mask_features=sam_spatial_features,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
            )
        elif not geometry_enabled:
            proposal = proposal_builder(
                query_states=stage3_state.normalized_query_states.transpose(0, 1),
                mask_logits=stage3_state.mask_logits,
                sam_mask_features=sam_spatial_features,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
            )
        expected_latents = int(self.dec_config.st3_latent_num_tokens)
        if (
            proposal.packed_group_tokens.shape[1] != expected_latents
            or proposal.packed_group_valid_mask.shape[1] != expected_latents
            or not bool(proposal.packed_group_valid_mask.all().item())
        ):
            raise RuntimeError(
                "the pre-VLM latent builder must emit exactly the configured "
                f"{expected_latents} valid carriers"
            )
        return St3LatentExecutionBundle(
            decoder_context=context,
            stage3_state=stage3_state,
            proposal=proposal,
        )

    @torch.no_grad()
    def forward_image_only_stage_masks(
        self,
        pixel_values: Optional[Tensor] = None,
        image_embeddings: Optional[Tuple[Tensor, ...]] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, ...]:
        """Return raw image-only mask logits for prediction stages st0--st9.

        This is an evaluation-only diagnostic boundary around the pixel
        decoder, learned Queries, and original Mask2Former decoder.  A Swin
        segmentor may supply ``pixel_values`` directly.  SAM + Mask2Former
        callers must supply the connector-produced ``image_embeddings`` via
        :meth:`MaskLATModel.forward_image_only_stage_masks`; the 256-channel SAM
        neck output is not interchangeable with those multi-scale features.
        The method deliberately supplies no SEG embedding, condition table,
        SigLIP evidence, or topology module, and it never invokes Query
        classification or any VLM component.

        The returned tuple contains the ten raw mask-logit tensors in formal
        stage order.  Every tensor is ``[B, Q, Hmask, Wmask]`` and remains in
        decoder-mask coordinates; dataset-specific restoration to the
        original image canvas belongs to the diagnostic evaluator.
        """

        if self.training:
            raise RuntimeError(
                "image-only stage-mask diagnostics require model.eval()"
            )
        if not isinstance(self.decoder, Mask2FormerTransformerModule):
            raise TypeError(
                "image-only stage-mask diagnostics require a Mask2Former "
                "decoder"
            )
        if (pixel_values is None) == (image_embeddings is None):
            raise ValueError(
                "provide exactly one of pixel_values or image_embeddings"
            )

        if image_embeddings is None:
            if isinstance(self.encoder, SamVisionEncoder):
                raise ValueError(
                    "raw SAM pixels must enter image-only diagnostics through "
                    "MaskLATModel.forward_image_only_stage_masks so the trained "
                    "seg_select_layers and seg_connector are preserved"
                )
            elif isinstance(self.encoder, SwinBackbone):
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_embeddings = tuple(encoder_outputs.feature_maps)
            else:
                raise TypeError(
                    "unsupported visual encoder for image-only stage-mask "
                    f"diagnostics: {type(self.encoder)}"
                )

        pixel_decoder_outputs = self.pixel_decoder(
            image_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_outputs = self.decoder(
            multi_scale_features=(
                pixel_decoder_outputs.multi_scale_features
            ),
            mask_features=pixel_decoder_outputs.mask_features,
            seg_embeddings=None,
            cond_lens=None,
            siglip_spatial_features=None,
            siglip_valid_mask=None,
            spatial_metadata=None,
            topology_group_core=None,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        stage_masks = tuple(decoder_outputs.masks_queries_logits)
        expected_stage_count = len(self.decoder.decoder.layers) + 1
        if expected_stage_count != 10:
            raise RuntimeError(
                "image-only diagnostic is defined for formal stages st0--st9; "
                "the configured Mask2Former decoder exposes "
                f"{expected_stage_count} stages"
            )
        if len(stage_masks) != expected_stage_count:
            raise RuntimeError(
                "Mask2Former stage-mask output count disagrees with its "
                f"decoder depth: {len(stage_masks)} != "
                f"{expected_stage_count}"
            )

        expected_batch = int(pixel_decoder_outputs.mask_features.shape[0])
        expected_queries = int(self.decoder.config.num_queries)
        for stage_index, mask_logits in enumerate(stage_masks):
            if mask_logits.ndim != 4 or tuple(mask_logits.shape[:2]) != (
                expected_batch,
                expected_queries,
            ):
                raise RuntimeError(
                    "invalid image-only mask-logit shape at stage "
                    f"{stage_index}: {tuple(mask_logits.shape)}; expected "
                    f"[{expected_batch}, {expected_queries}, Hmask, Wmask]"
                )
        finite_by_stage = torch.stack(
            [torch.isfinite(mask_logits).all() for mask_logits in stage_masks]
        )
        if not bool(finite_by_stage.all().item()):
            invalid_stages = [
                stage_index
                for stage_index, finite in enumerate(
                    finite_by_stage.detach().cpu().tolist()
                )
                if not finite
            ]
            raise FloatingPointError(
                "image-only stage-mask logits contain NaN or Inf at stages "
                f"{invalid_stages}"
            )
        return stage_masks

    @staticmethod
    def _gather_st3_proposal_field(value: Tensor, row_source_indices: Tensor) -> Tensor:
        return value.index_select(
            0,
            row_source_indices.to(device=value.device, dtype=torch.long),
        )

    def finish_st3_latent_execution(
        self,
        *,
        bundle: St3LatentExecutionBundle,
        packed_latent_hidden_states: Tensor,
        seg_to_sample: Tensor,
        seg_states: Tensor,
        local_cond_embeddings: Tensor,
        local_cond_valid_mask: Tensor,
        row_mask_labels: Optional[List[Tensor]] = None,
        row_class_labels: Optional[List[Tensor]] = None,
        task_name: Optional[str] = None,
        return_dict: bool = True,
    ) -> MaskLATSegmentorOutput:
        """Condition the 64 carriers and resume the global st4--st9 decoder."""

        if not isinstance(bundle, St3LatentExecutionBundle):
            raise TypeError("bundle must be a St3LatentExecutionBundle")
        gated_enabled = bool(
            getattr(
                self.dec_config,
                "use_st3_proposal_latent_bridge",
                False,
            )
        )
        transport_enabled = bool(
            getattr(
                self.dec_config,
                "use_st3_bipartite_latent_transport",
                False,
            )
        )
        post_vlm_enabled = bool(getattr(
            self.dec_config,
            "use_latent_s2_post_vlm_transport",
            False,
        ))
        stagewise_enabled = bool(
            getattr(
                self.dec_config,
                "use_stagewise_latent_trifusion",
                False,
            )
        )
        geometry_enabled = bool(
            getattr(
                self.dec_config,
                "use_proposal_mediated_tripartite_transport",
                False,
            )
        )
        if sum((
            gated_enabled,
            transport_enabled,
            post_vlm_enabled,
            stagewise_enabled,
            geometry_enabled,
        )) != 1:
            raise RuntimeError(
                "exactly one gated, bipartite, latent-S2-post-VLM, stagewise, "
                "or geometry latent path must be enabled"
            )
        active_bridge = (
            self.st3_latent_bridge
            if gated_enabled
            else (
                self.st3_transport_bridge
                if transport_enabled
                else (
                    self.latent_s2_post_vlm_bridge
                    if post_vlm_enabled
                    else (
                        self.stagewise_trifusion_bridge
                        if stagewise_enabled
                        else self.geometry_transport_bridge
                    )
                )
            )
        )
        if active_bridge is None:
            raise RuntimeError("the selected st3 latent bridge is not configured")
        proposal = bundle.proposal
        expected_tokens = proposal.packed_group_tokens.shape[:2]
        if packed_latent_hidden_states.shape[:2] != expected_tokens:
            raise ValueError(
                "VLM latent hidden states do not match inserted latent tokens"
            )
        hidden_dim = int(self.decoder.config.hidden_dim)
        if (
            packed_latent_hidden_states.ndim != 3
            or packed_latent_hidden_states.shape[-1] != hidden_dim
        ):
            raise ValueError(
                "projected VLM latent states must be [B,L,Ddecoder]"
            )
        if seg_to_sample.ndim != 1 or seg_states.ndim != 2:
            raise ValueError("seg_to_sample/seg_states must be [Nseg] and [Nseg,D]")
        if seg_states.shape != (seg_to_sample.numel(), hidden_dim):
            raise ValueError("SEG state shape disagrees with Nseg/Ddecoder")
        if local_cond_embeddings.ndim != 3 or local_cond_valid_mask.shape != (
            local_cond_embeddings.shape[0],
            local_cond_embeddings.shape[1],
        ):
            raise ValueError("local Cond tables must be [Nseg,C,D] and [Nseg,C]")
        if local_cond_embeddings.shape[0] != seg_states.shape[0] or (
            local_cond_embeddings.shape[-1] != hidden_dim
        ):
            raise ValueError("local Cond rows/width disagree with SEG states")
        if local_cond_valid_mask.dtype != torch.bool:
            local_cond_valid_mask = local_cond_valid_mask.to(torch.bool)
        if not bool(local_cond_valid_mask[:, -1].all()):
            raise ValueError("the final local condition column must be BG")

        row_indices = seg_to_sample.to(
            device=bundle.decoder_context.pixel_embeddings.device,
            dtype=torch.long,
        )
        context = self.decoder.expand_decoder_context(
            bundle.decoder_context,
            row_indices,
        )
        state = self.decoder.decoder.expand_stage3_state(
            bundle.stage3_state,
            row_indices,
        )

        def gather(value: Tensor) -> Tensor:
            return self._gather_st3_proposal_field(value, row_indices)

        proposal_latents = gather(proposal.latent_features)
        vlm_latents = gather(packed_latent_hidden_states)
        raw_query_states = state.raw_query_states.transpose(0, 1)
        if gated_enabled:
            affinity_logits = gather(proposal.affinity_logits)
            resumed_queries, persistent_latents, _ = active_bridge(
                query_states=raw_query_states,
                proposal_latent_states=proposal_latents,
                vlm_latent_states=vlm_latents,
                affinity_logits=affinity_logits,
                seg_states=seg_states,
                local_conditions=local_cond_embeddings,
                local_condition_valid_mask=local_cond_valid_mask,
            )

            def pre_prediction_refiner(
                stage_index: int,
                raw_query_states: Tensor,
                _context: Any,
            ) -> Tensor:
                """Persist gated Q/Z through st4--st9 before mask heads."""

                nonlocal persistent_latents
                raw_query_states, persistent_latents = (
                    active_bridge.mix_stage(
                        stage_index=stage_index,
                        query_states=raw_query_states,
                        latent_states=persistent_latents,
                    )
                )
                return raw_query_states

            decoder_outputs = self.decoder.decoder.forward_from_stage3(
                context,
                state,
                resumed_query_states=resumed_queries,
                query_self_attention_mask=None,
                pre_prediction_refiner=pre_prediction_refiner,
                stage_refiner=None,
                return_dict=True,
            )
        elif transport_enabled:
            resumed_queries, persistent_latents, _ = active_bridge(
                query_states=raw_query_states,
                proposal_latent_states=proposal_latents,
                vlm_latent_states=vlm_latents,
                seg_states=seg_states,
                local_conditions=local_cond_embeddings,
                local_condition_valid_mask=local_cond_valid_mask,
            )
            condition_refreshers = active_bridge.late_condition_refreshers
            refresh_enabled = len(condition_refreshers) > 0
            decoder_outputs = self.decoder.decoder.forward_from_stage3(
                context,
                state,
                resumed_query_states=resumed_queries,
                query_self_attention_mask=None,
                pre_prediction_refiner=None,
                stage_refiner=None,
                transport_latent_states=persistent_latents,
                transport_modules=active_bridge.transport_stages,
                transport_cached_vlm_latent_states=(
                    vlm_latents if refresh_enabled else None
                ),
                transport_local_condition_states=(
                    local_cond_embeddings if refresh_enabled else None
                ),
                transport_local_condition_valid_mask=(
                    local_cond_valid_mask if refresh_enabled else None
                ),
                transport_condition_refreshers=(
                    condition_refreshers if refresh_enabled else None
                ),
                return_dict=True,
            )
        else:
            (
                resumed_queries,
                persistent_latents,
                persistent_conditions,
                condition_anchor,
                condition_update_mask,
            ) = active_bridge.initialize(
                query_states=raw_query_states,
                proposal_latent_states=proposal_latents,
                vlm_latent_states=vlm_latents,
                seg_states=seg_states,
                local_conditions=local_cond_embeddings,
                local_condition_valid_mask=local_cond_valid_mask,
            )
            decoder_outputs = self.decoder.decoder.forward_from_stage3(
                context,
                state,
                resumed_query_states=resumed_queries,
                query_self_attention_mask=None,
                pre_prediction_refiner=None,
                stage_refiner=None,
                trifusion_latent_states=persistent_latents,
                trifusion_condition_states=persistent_conditions,
                trifusion_condition_anchor=condition_anchor,
                trifusion_condition_valid_mask=local_cond_valid_mask,
                trifusion_condition_update_mask=condition_update_mask,
                trifusion_modules=active_bridge.transport_stages,
                return_dict=True,
            )
        stage_queries = tuple(decoder_outputs.intermediate_hidden_states)
        stage_masks = tuple(decoder_outputs.masks_queries_logits)
        if len(stage_queries) != 10 or len(stage_masks) != 10:
            raise RuntimeError(
                "proposal latent execution must retain st0--st9 outputs, got "
                f"queries={len(stage_queries)}, masks={len(stage_masks)}"
            )
        if post_vlm_enabled or stagewise_enabled or geometry_enabled:
            updated_conditions = decoder_outputs.stage_condition_states
            if updated_conditions is None or len(updated_conditions) != 6:
                raise RuntimeError(
                    "tripartite transport must return Cond states for st4--st9"
                )
            stage_conditions = (
                (local_cond_embeddings,) * 4 + tuple(updated_conditions)
            )
        else:
            stage_conditions = (local_cond_embeddings,) * len(stage_queries)
        stage_classes = tuple(
            self.get_class_prediction(
                query.transpose(0, 1),
                condition_states,
                local_cond_valid_mask,
                finite_mask_logits=True,
            )
            for query, condition_states in zip(
                stage_queries,
                stage_conditions,
            )
        )
        final_classes = stage_classes[-1]
        final_raw_masks = stage_masks[-1]
        auxiliary_logits = self.get_auxiliary_logits(
            stage_classes,
            stage_masks,
        )

        loss_dict = None
        loss = None
        if self.training and row_mask_labels is not None:
            if row_class_labels is None:
                raise ValueError("latent-bridge training requires class labels")
            loss_dict = self.get_loss_dict(
                masks_queries_logits=final_raw_masks,
                class_queries_logits=final_classes,
                mask_labels=row_mask_labels,
                class_labels=row_class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)
            final_masks = final_raw_masks
        else:
            sam_valid_boxes = gather(
                proposal.sam_valid_boxes_normalized
            )
            final_masks = self.postprocess_masks_preds(
                (final_raw_masks,),
                sam_valid_boxes_normalized=sam_valid_boxes,
            )[-1]

        output = MaskLATSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=final_classes,
            masks_queries_logits=final_masks,
            auxiliary_logits=(
                auxiliary_logits
                if self.decoder.config.output_auxiliary_logits
                else None
            ),
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            raw_query_class_logits=final_classes,
            raw_query_mask_logits=final_raw_masks,
        )
        if not return_dict:
            return tuple(value for value in output.values() if value is not None)
        return output

    def finish_st3_group_execution(
        self,
        *,
        bundle: St3GroupExecutionBundle,
        packed_group_hidden_states: Tensor,
        seg_to_sample: Tensor,
        seg_states: Tensor,
        local_cond_embeddings: Tensor,
        local_cond_valid_mask: Tensor,
        row_mask_labels: Optional[List[Tensor]] = None,
        row_class_labels: Optional[List[Tensor]] = None,
        source_mask_labels: Optional[List[Tensor]] = None,
        task_name: Optional[str] = None,
        return_dict: bool = True,
    ) -> MaskLATSegmentorOutput:
        """Resume st4--st9 with fixed Groups and SEG-local Cond interaction."""

        if not isinstance(bundle, St3GroupExecutionBundle):
            raise TypeError("bundle must be a St3GroupExecutionBundle")
        required_modules = (
            self.st3_vlm_refiner,
            self.st3_group_classifier,
            self.st3_group_prediction_adapter,
            self.st3_group_criterion,
        )
        if any(module is None for module in required_modules):
            raise RuntimeError("st3 Group-in-VLM modules are incomplete")
        proposal = bundle.proposal
        if packed_group_hidden_states.shape[:2] != (
            proposal.packed_group_tokens.shape[0],
            proposal.packed_group_tokens.shape[1],
        ):
            raise ValueError(
                "VLM Group hidden states do not match the inserted Group table"
            )
        expected_hidden = int(self.decoder.config.hidden_dim)
        if packed_group_hidden_states.ndim != 3 or (
            packed_group_hidden_states.shape[-1] != expected_hidden
        ):
            raise ValueError(
                "projected VLM Group states must be [B,G,Ddecoder], got "
                f"{tuple(packed_group_hidden_states.shape)} with "
                f"Ddecoder={expected_hidden}"
            )
        if seg_to_sample.ndim != 1 or seg_states.ndim != 2:
            raise ValueError("seg_to_sample/seg_states must be [Nseg] and [Nseg,D]")
        if seg_states.shape[0] != seg_to_sample.numel():
            raise ValueError("SEG state count does not match seg_to_sample")
        if local_cond_embeddings.ndim != 3 or local_cond_valid_mask.shape != (
            local_cond_embeddings.shape[0],
            local_cond_embeddings.shape[1],
        ):
            raise ValueError("local Cond tables must be [Nseg,C,D] and [Nseg,C]")
        if local_cond_embeddings.shape[0] != seg_states.shape[0]:
            raise ValueError("local Cond rows do not match SEG rows")
        if seg_states.shape[-1] != expected_hidden or (
            local_cond_embeddings.shape[-1] != expected_hidden
        ):
            raise ValueError(
                "SEG and local Cond states must use decoder hidden width "
                f"{expected_hidden}, got {seg_states.shape[-1]} and "
                f"{local_cond_embeddings.shape[-1]}"
            )
        if not (
            packed_group_hidden_states.device
            == seg_states.device
            == local_cond_embeddings.device
        ):
            raise ValueError(
                "projected Group, SEG, and local Cond states must share a device"
            )
        if not (
            packed_group_hidden_states.dtype
            == seg_states.dtype
            == local_cond_embeddings.dtype
        ):
            raise ValueError(
                "projected Group, SEG, and local Cond states must share a dtype"
            )
        if not bool(local_cond_valid_mask[:, -1].all()):
            raise ValueError("the final local condition column must be BG")
        if bool(local_cond_valid_mask[:, :-1].sum(dim=1).eq(0).any()):
            raise ValueError("every SEG row must have a foreground Cond")

        row_indices = seg_to_sample.to(
            device=bundle.decoder_context.pixel_embeddings.device,
            dtype=torch.long,
        )
        context = self.decoder.expand_decoder_context(
            bundle.decoder_context,
            row_indices,
        )
        state = self.decoder.decoder.expand_stage3_state(
            bundle.stage3_state,
            row_indices,
        )
        physical_vlm_groups = unpack_packed_groups(
            packed_group_hidden_states,
            proposal.packed_group_valid_mask,
            proposal.packed_group_root_indices,
            num_queries=int(self.decoder.config.num_queries),
        )

        def gather(value: Tensor) -> Tensor:
            return self._gather_st3_proposal_field(value, row_indices)

        query_to_group = gather(proposal.query_to_group)
        group_valid_mask = gather(proposal.group_valid_mask)
        group_sizes = gather(proposal.group_sizes)
        pairwise_iou = gather(proposal.pairwise_iou)
        topology_centrality = gather(proposal.topology_centrality)
        sam_region_evidence = gather(proposal.sam_region_evidence)
        siglip_region_evidence = gather(proposal.siglip_region_evidence)
        query_mask_on_siglip = gather(proposal.query_mask_on_siglip)
        resolved_siglip_valid = gather(proposal.siglip_valid_mask)
        sam_valid_boxes = gather(proposal.sam_valid_boxes_normalized)
        proposal_group_states = gather(proposal.group_features)
        group_states = gather(physical_vlm_groups)
        raw_query_states = state.raw_query_states.transpose(0, 1)
        raw_query_states, group_states = self.st3_vlm_refiner.initialize_after_vlm(
            query_states=raw_query_states,
            group_states=group_states,
            proposal_group_states=proposal_group_states,
            query_to_group=query_to_group,
            group_valid_mask=group_valid_mask,
            seg_states=seg_states,
            local_conditions=local_cond_embeddings,
            local_condition_valid_mask=local_cond_valid_mask.to(torch.bool),
        )

        group_stage_outputs = []

        def make_stage(
            decoder_stage_index: int,
            query_states: Tensor,
            query_masks: Tensor,
            member_attention_weights: Tensor,
            *,
            final: bool,
        ) -> TopologyGroupStageOutput:
            group_logits = self.st3_group_classifier(
                group_features=group_states,
                group_valid_mask=group_valid_mask,
                cond_embeddings=local_cond_embeddings,
                embed_masks=local_cond_valid_mask,
                logit_scale=self.logit_scale if self.open_cls else None,
                class_predictor=(
                    self.class_predictor if self.close_cls else None
                ),
            )
            quality = (
                self.st3_vlm_refiner.quality_logits(
                    query_states,
                    group_states,
                    query_to_group,
                )
                if final
                else None
            )
            return TopologyGroupStageOutput(
                stage_index=decoder_stage_index,
                query_states=query_states,
                query_mask_logits=query_masks,
                query_to_group=query_to_group,
                group_valid_mask=group_valid_mask,
                group_sizes=group_sizes,
                pairwise_iou=pairwise_iou,
                topology_centrality=topology_centrality,
                sam_region_evidence=sam_region_evidence,
                siglip_region_evidence=siglip_region_evidence,
                member_evidence=query_states,
                group_features=group_states,
                member_attention_weights=member_attention_weights,
                query_mask_on_siglip=query_mask_on_siglip,
                siglip_valid_mask=resolved_siglip_valid,
                sam_valid_boxes_normalized=sam_valid_boxes,
                group_class_logits=group_logits,
                member_quality_logits=quality,
            )

        initial_member_weights = gather(proposal.member_attention_weights)
        group_stage_outputs.append(
            make_stage(
                3,
                state.normalized_query_states.transpose(0, 1),
                state.mask_logits,
                initial_member_weights,
                final=False,
            )
        )

        def stage_refiner(
            stage_index: int,
            raw_queries: Tensor,
            normalized_queries: Tensor,
            mask_logits: Tensor,
            _context: Any,
        ) -> Tensor:
            nonlocal group_states
            updated_queries, group_states, member_weights = (
                self.st3_vlm_refiner.update_after_stage(
                    query_states=normalized_queries,
                    group_states=group_states,
                    query_to_group=query_to_group,
                    group_valid_mask=group_valid_mask,
                    local_conditions=local_cond_embeddings,
                    local_condition_valid_mask=local_cond_valid_mask.to(
                        torch.bool
                    ),
                    apply_query_feedback=stage_index < 9,
                )
            )
            group_stage_outputs.append(
                make_stage(
                    stage_index,
                    normalized_queries,
                    mask_logits,
                    member_weights,
                    final=stage_index == 9,
                )
            )
            # Feedback is a residual update of the raw Decoder stream.  The
            # refiner consumed the normalized feature, so transfer only its
            # learned delta onto the raw state.
            return raw_queries + (updated_queries - normalized_queries)

        query_self_mask = build_group_local_self_attention_mask(
            query_to_group,
            dtype=raw_query_states.dtype,
            num_heads=int(self.decoder.config.num_attention_heads),
        )
        decoder_outputs = self.decoder.decoder.forward_from_stage3(
            context,
            state,
            resumed_query_states=raw_query_states,
            query_self_attention_mask=query_self_mask,
            stage_refiner=stage_refiner,
            return_dict=True,
        )
        if len(group_stage_outputs) != 7:
            raise RuntimeError(
                "st3 Group-in-VLM must produce stages st3--st9, got "
                f"{len(group_stage_outputs)}"
            )
        final_stage = group_stage_outputs[-1]
        class_queries_logits, masks_queries_logits, representatives = (
            self.st3_group_prediction_adapter(
                group_class_logits=final_stage.group_class_logits,
                final_query_masks=final_stage.query_mask_logits,
                query_to_group=query_to_group,
                group_valid_mask=group_valid_mask,
                member_quality_logits=final_stage.member_quality_logits,
                embed_masks=local_cond_valid_mask,
            )
        )

        loss_dict = None
        loss = None
        diagnostics = None
        if self.training and row_mask_labels is not None:
            if row_class_labels is None:
                raise ValueError("Group training requires row-local class labels")
            loss_dict, diagnostics = self.st3_group_criterion(
                tuple(group_stage_outputs),
                (row_mask_labels, row_class_labels),
                embed_masks=local_cond_valid_mask,
                task_name=task_name,
            )
            if source_mask_labels is not None:
                prefix_masks = bundle.stage3_state.masks_queries_logits[:3]
                prefix_aux = [
                    {
                        "masks_queries_logits": masks,
                        "class_queries_logits": None,
                    }
                    for masks in prefix_masks[:-1]
                ]
                proposal_losses = self.get_loss_dict(
                    masks_queries_logits=prefix_masks[-1],
                    class_queries_logits=None,
                    mask_labels=source_mask_labels,
                    class_labels=None,
                    auxiliary_predictions=prefix_aux,
                )
                loss_dict.update(
                    {
                        f"loss_proposal_{key.removeprefix('loss_')}": value
                        for key, value in proposal_losses.items()
                    }
                )
            loss = self.get_loss(loss_dict)
        else:
            masks_queries_logits = self.postprocess_masks_preds(
                (masks_queries_logits,),
                sam_valid_boxes_normalized=sam_valid_boxes,
            )[-1]

        output = MaskLATSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxiliary_logits=None,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            raw_query_class_logits=None,
            raw_query_mask_logits=final_stage.query_mask_logits,
            group_valid_mask=group_valid_mask,
            query_to_group=query_to_group,
            group_features=final_stage.group_features,
            group_class_logits=final_stage.group_class_logits,
            member_attention_weights=final_stage.member_attention_weights,
            member_quality_logits=final_stage.member_quality_logits,
            group_representative_query_indices=representatives,
            group_counts=group_valid_mask.sum(dim=1),
            condition_valid_mask=local_cond_valid_mask,
            group_stage_outputs=tuple(group_stage_outputs),
            group_criterion_diagnostics=diagnostics,
        )
        if not return_dict:
            return tuple(value for value in output.values() if value is not None)
        return output

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        image_embeddings: Optional[Tuple[Tensor]] = None,
        seg_embeddings: Optional[Tensor] = None,
        cond_embeddings: Optional[Tensor] = None,
        embed_masks: Optional[Tensor] = None,
        cond_lens: Optional[List] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
        output_auxiliary_logits: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        task_name: Optional[str] = None,
        siglip_spatial_features: Optional[Tensor] = None,
        siglip_valid_mask: Optional[Tensor] = None,
        spatial_metadata: Optional[Dict[str, Tensor]] = None,
        return_variant_diagnostic_stages: bool = False,
        **kwargs,
    ) -> MaskLATSegmentorOutput:
        topology_enabled = bool(
            getattr(self.dec_config, "use_topology_group_decoder", False)
        )
        if return_variant_diagnostic_stages:
            if self.training:
                raise RuntimeError(
                    "variant Query diagnostics are evaluation-only"
                )
            if topology_enabled:
                raise RuntimeError(
                    "variant Query diagnostics require original topology-off "
                    "MaskLAT"
                )
        if topology_enabled and self.topology_group_core is None:
            raise RuntimeError(
                "topology group modules are not configured; the actual "
                "SigLIP hidden dimension must be provided before forward"
            )

        # sam_enc + sam_dec
        if isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, SamMaskDecoder):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeddings
                image_embeddings = encoder_outputs.last_hidden_state

            image_positional_embeddings = self.get_image_wide_positional_embeddings()
            batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
            image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
            seg_embeddings = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=None,
                input_embeds=seg_embeddings,
            )
            decoder_outputs = self.decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                attention_similarity=None,
                cond_lens=cond_lens,
                target_embedding=None,
                output_attentions=output_attentions,
            )

        # sam_enc + mask2former_dec
        elif isinstance(self.encoder, SamVisionEncoder) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                # TODO: multi-scale image_embeddings
                image_embeddings = [encoder_outputs.last_hidden_state] * 4

            pixel_decoder_outputs = self.pixel_decoder(
                image_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeddings=seg_embeddings,
                cond_lens=cond_lens,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
                topology_group_core=(
                    self.topology_group_core if topology_enabled else None
                ),
                output_attentions=output_attentions,
            )
        # mask2former_enc(swin) + mask2former_dec
        elif isinstance(self.encoder, SwinBackbone) and isinstance(self.decoder, Mask2FormerTransformerModule):
            if image_embeddings is None:
                encoder_outputs = self.encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                image_embeddings = encoder_outputs.feature_maps
            pixel_decoder_outputs = self.pixel_decoder(
                image_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = self.decoder(
                multi_scale_features=pixel_decoder_outputs.multi_scale_features,
                mask_features=pixel_decoder_outputs.mask_features,
                seg_embeddings=seg_embeddings,
                cond_lens=cond_lens,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
                topology_group_core=(
                    self.topology_group_core if topology_enabled else None
                ),
                output_attentions=output_attentions,
            )
        else:
            raise ValueError(f"Unsupported encoder and decoder type: {type(self.encoder)} and {type(self.decoder)}")

        loss, loss_dict, auxiliary_logits = None, None, None
        raw_class_queries_logits = ()

        for decoder_output in decoder_outputs.intermediate_hidden_states:
            # Raw Query classification is retained only as a diagnostic in
            # topology mode; it remains the formal baseline output otherwise.
            class_prediction = (
                self.get_class_prediction(
                    decoder_output.transpose(0, 1),
                    cond_embeddings,
                    embed_masks,
                    finite_mask_logits=topology_enabled,
                )
                if self.use_cls
                else None
            )
            raw_class_queries_logits += (class_prediction,)

        raw_masks_queries_logits = decoder_outputs.masks_queries_logits
        auxiliary_logits = self.get_auxiliary_logits(
            raw_class_queries_logits,
            raw_masks_queries_logits,
        )

        group_valid_mask = None
        query_to_group = None
        group_features = None
        group_class_logits = None
        member_attention_weights = None
        member_quality_logits = None
        representative_indices = None
        group_counts = None
        condition_valid_mask = None
        group_stage_outputs = None
        group_criterion_diagnostics = None
        diagnostic_stage_class_logits = None
        diagnostic_stage_mask_logits = None

        if topology_enabled:
            group_stage_outputs = decoder_outputs.group_stage_outputs
            if group_stage_outputs is None:
                raise RuntimeError(
                    "Mask2Former topology mode did not return stage group outputs"
                )
            group_stage_outputs = tuple(group_stage_outputs)
            expected_stages = self.topology_group_core.num_stages
            if len(group_stage_outputs) != expected_stages:
                raise AssertionError(
                    "topology mode must return exactly one group output per "
                    f"prediction stage: {len(group_stage_outputs)} != {expected_stages}"
                )
            if len(raw_masks_queries_logits) != expected_stages:
                raise AssertionError(
                    "raw Query mask stages and topology stages differ: "
                    f"{len(raw_masks_queries_logits)} != {expected_stages}"
                )
            if len(raw_class_queries_logits) != expected_stages:
                raise AssertionError(
                    "raw Query class stages and topology stages differ: "
                    f"{len(raw_class_queries_logits)} != {expected_stages}"
                )
            group_stage_outputs = self._classify_topology_groups(
                group_stage_outputs,
                cond_embeddings,
                embed_masks,
            )
            final_stage = group_stage_outputs[-1]
            if final_stage.member_quality_logits is None:
                raise RuntimeError("st9 did not produce member quality logits")
            effective_batch_size = final_stage.query_mask_logits.shape[0]
            self._assert_topology_target_batches(
                effective_batch_size,
                mask_labels,
                class_labels,
            )
            (
                class_queries_logits,
                masks_queries_logits,
                representative_indices,
            ) = self.group_prediction_adapter(
                group_class_logits=final_stage.group_class_logits,
                final_query_masks=final_stage.query_mask_logits,
                query_to_group=final_stage.query_to_group,
                group_valid_mask=final_stage.group_valid_mask,
                member_quality_logits=final_stage.member_quality_logits,
                embed_masks=embed_masks,
            )
            group_valid_mask = final_stage.group_valid_mask
            query_to_group = final_stage.query_to_group
            group_features = final_stage.group_features
            group_class_logits = final_stage.group_class_logits
            member_attention_weights = final_stage.member_attention_weights
            member_quality_logits = final_stage.member_quality_logits
            group_counts = group_valid_mask.sum(dim=1)
            condition_valid_mask = (
                embed_masks.to(dtype=torch.bool)
                if embed_masks is not None
                else torch.ones(
                    group_class_logits.shape[0],
                    group_class_logits.shape[-1],
                    dtype=torch.bool,
                    device=group_class_logits.device,
                )
            )

            if mask_labels is not None and self.training:
                if self.group_criterion is None:
                    raise RuntimeError(
                        "topology group criterion has not been configured"
                    )
                targets: Any = (
                    (mask_labels, class_labels)
                    if class_labels is not None
                    else mask_labels
                )
                (
                    loss_dict,
                    group_criterion_diagnostics,
                ) = self.group_criterion(
                    group_stage_outputs,
                    targets,
                    embed_masks=embed_masks,
                    task_name=task_name,
                )
                loss = self.get_loss(loss_dict)
            else:
                masks_queries_logits = self.postprocess_masks_preds(
                    (masks_queries_logits,),
                    sam_valid_boxes_normalized=(
                        final_stage.sam_valid_boxes_normalized
                    ),
                )[-1]
        else:
            class_queries_logits = raw_class_queries_logits[-1]
            if mask_labels is not None and self.training:
                loss_dict = self.get_loss_dict(
                    masks_queries_logits=raw_masks_queries_logits[-1],
                    class_queries_logits=class_queries_logits,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    auxiliary_predictions=auxiliary_logits,
                )
                loss = self.get_loss(loss_dict)
                masks_queries_logits = raw_masks_queries_logits[-1]
            else:
                processed_masks = (
                    self.postprocess_masks_preds(raw_masks_queries_logits)
                    if raw_masks_queries_logits[-1] is not None
                    else None
                )
                masks_queries_logits = (
                    None if processed_masks is None else processed_masks[-1]
                )
                if return_variant_diagnostic_stages:
                    diagnostic_stage_class_logits = tuple(
                        raw_class_queries_logits
                    )
                    diagnostic_stage_mask_logits = tuple(
                        raw_masks_queries_logits
                    )

        output_auxiliary_logits = (
            self.decoder.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = MaskLATSegmentorOutput(
            loss=loss,
            loss_dict=loss_dict,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxiliary_logits=auxiliary_logits,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            raw_query_class_logits=(
                raw_class_queries_logits[-1]
                if topology_enabled
                and self.decoder.config.group_return_raw_query_outputs
                else None
            ),
            raw_query_mask_logits=(
                raw_masks_queries_logits[-1]
                if topology_enabled
                and self.decoder.config.group_return_raw_query_outputs
                else None
            ),
            group_valid_mask=group_valid_mask,
            query_to_group=query_to_group,
            group_features=group_features,
            group_class_logits=group_class_logits,
            member_attention_weights=member_attention_weights,
            member_quality_logits=member_quality_logits,
            group_representative_query_indices=representative_indices,
            group_counts=group_counts,
            condition_valid_mask=condition_valid_mask,
            group_stage_outputs=group_stage_outputs,
            group_criterion_diagnostics=group_criterion_diagnostics,
            diagnostic_stage_class_logits=diagnostic_stage_class_logits,
            diagnostic_stage_mask_logits=diagnostic_stage_mask_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output
