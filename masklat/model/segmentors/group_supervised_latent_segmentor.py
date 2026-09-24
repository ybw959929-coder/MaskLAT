"""Segmentor with grouped latent supervision."""

from __future__ import annotations

import torch

from .masklat_segmentor import MaskLATSegmentor, MaskLATSegmentorOutput, St3LatentExecutionBundle
from .mask2former.group_supervised_latent_builder import GroupSupervisedLatentBuilder
from .mask2former.group_supervised_latent_transport import GroupSupervisedLatentTransportBridge
from .mask2former.latent_mask_group_loss import MaskGroupAuxiliaryLoss
from .mask2former.spatial_validity import valid_mask_from_normalized_boxes
from .mask2former.st3_bipartite_latent_transport import QueryLatentRead


class GroupSupervisedLatentSegmentor(MaskLATSegmentor):
    proposal_builder_type = GroupSupervisedLatentBuilder
    transport_bridge_type = GroupSupervisedLatentTransportBridge
    incompatible_builder_flags = (
        "use_st123_latent_cascade", "use_st123_latent_deepstack",
        "use_stagewise_latent_trifusion", "use_latent_s2_post_vlm_transport",
        "use_proposal_mediated_tripartite_transport",
    )

    def __init__(self, *args, group_loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_loss_config = dict(group_loss_config or {})
        self.mask_group_auxiliary = MaskGroupAuxiliaryLoss(**self.group_loss_config)
        self._group_supervision_records = None

    def configure_st3_bipartite_latent_transport(self, *, siglip_hidden_dim, llm_hidden_dim):
        incompatible = self.incompatible_builder_flags
        if any(bool(getattr(self.dec_config, key, False)) for key in incompatible):
            raise ValueError("the grouped path requires the ST3 builder")
        if tuple(getattr(self.dec_config, "st3_transport_late_condition_refresh_stages", ())):
            raise ValueError("late condition refresh must be disabled")
        if int(self.dec_config.st3_latent_num_tokens) != 64:
            raise ValueError("the latent capacity must be 64")
        if isinstance(self.st3_transport_proposal_builder, self.proposal_builder_type):
            super().configure_st3_bipartite_latent_transport(
                siglip_hidden_dim=siglip_hidden_dim, llm_hidden_dim=llm_hidden_dim
            )
            return
        super().configure_st3_bipartite_latent_transport(
            siglip_hidden_dim=siglip_hidden_dim, llm_hidden_dim=llm_hidden_dim
        )
        old_builder = self.st3_transport_proposal_builder
        old_bridge = self.st3_transport_bridge
        if old_builder is None or old_bridge is None:
            raise ValueError("group supervision requires bipartite transport")
        reference = next(old_builder.parameters())
        device_dtype = dict(device=reference.device, dtype=reference.dtype)
        config = self.decoder.config
        builder = self.proposal_builder_type(
            hidden_dim=int(config.hidden_dim),
            sam_feature_dim=int(config.mask_feature_size),
            siglip_feature_dim=int(siglip_hidden_dim),
            llm_hidden_dim=int(llm_hidden_dim),
            num_latents=64,
            num_heads=int(config.st3_latent_num_heads),
            mask_threshold=float(config.st3_latent_mask_threshold),
            fourier_features=int(config.st3_latent_fourier_features),
            use_sam_region=bool(getattr(config, "st3_latent_use_sam_region", True)),
            use_siglip_region=bool(getattr(config, "st3_latent_use_siglip_region", True)),
            use_xywh=bool(getattr(config, "st3_latent_use_xywh", True)),
            pre_vlm_depth=int(config.st3_transport_pre_vlm_depth),
            group_loss_config=self.group_loss_config,
        ).to(**device_dtype)
        builder_state = builder.load_state_dict(old_builder.state_dict(), strict=False)
        if builder_state.unexpected_keys or any(
            not key.endswith("group_supervision_version")
            for key in builder_state.missing_keys
        ):
            raise RuntimeError(f"unexpected new-builder initialization: {builder_state}")
        bridge = self.transport_bridge_type(
            hidden_dim=int(config.hidden_dim),
            num_heads=int(config.st3_latent_num_heads),
            post_vlm_depth=int(config.st3_transport_post_vlm_depth),
            transport_depth=int(config.st3_transport_decoder_depth),
            output_init_std=float(config.st3_transport_output_init_std),
        ).to(**device_dtype)
        old_bridge_state = old_bridge.state_dict()
        independent_transport = bool(getattr(
            bridge, "initializes_independent_transport", False,
        ))
        if independent_transport:
            old_bridge_state = {
                key: value for key, value in old_bridge_state.items()
                if not key.startswith("transport_stages.")
            }
        incompatible_state = bridge.load_state_dict(old_bridge_state, strict=False)
        allowed_missing = (
            ("transport_stages.", "final_condition_reader.")
            if independent_transport else
            ("transport_stages.5.", "final_condition_reader.")
        )
        if incompatible_state.unexpected_keys or any(
            not key.startswith(allowed_missing) for key in incompatible_state.missing_keys
        ):
            raise RuntimeError(f"unexpected new-transport initialization: {incompatible_state}")
        self.st3_transport_proposal_builder = builder
        self.st3_transport_bridge = bridge

    def _record_group_supervision(self, stage_index, result):
        if result is None or self._group_supervision_records is None:
            return
        if stage_index in self._group_supervision_records:
            raise RuntimeError(f"group supervision for st{stage_index} was collected twice")
        self._group_supervision_records[stage_index] = result

    def prepare_st3_latent_execution(self, **kwargs):
        bundle = super().prepare_st3_latent_execution(**kwargs)
        self._record_group_supervision(3, bundle.proposal.group_supervision)
        return bundle

    def finish_st3_latent_execution(
        self, *, bundle, packed_latent_hidden_states, seg_to_sample, seg_states,
        local_cond_embeddings, local_cond_valid_mask, row_mask_labels=None,
        row_class_labels=None, task_name=None, return_dict=True,
    ):
        if not isinstance(bundle, St3LatentExecutionBundle):
            raise TypeError("bundle must be a St3LatentExecutionBundle")
        proposal = bundle.proposal
        if packed_latent_hidden_states.ndim != 3 or (
            packed_latent_hidden_states.shape[:2] != proposal.packed_group_tokens.shape[:2]
            or packed_latent_hidden_states.shape[-1] != self.decoder.config.hidden_dim
        ):
            raise ValueError("projected VLM latent table must match the ST3 [B,64,D] table")
        if seg_to_sample.ndim != 1:
            raise ValueError("seg_to_sample must be [Nseg]")
        if seg_states.shape != (seg_to_sample.numel(), self.decoder.config.hidden_dim):
            raise ValueError("SEG states must be [Nseg,D]")
        if local_cond_embeddings.ndim != 3 or local_cond_valid_mask.shape != local_cond_embeddings.shape[:2]:
            raise ValueError("local Cond table and validity mask disagree")
        local_cond_valid_mask = local_cond_valid_mask.bool()
        if not bool(local_cond_valid_mask[:, -1].all()):
            raise ValueError("last local condition must be valid background")
        indices = seg_to_sample.to(
            device=bundle.decoder_context.pixel_embeddings.device, dtype=torch.long
        )
        context = self.decoder.expand_decoder_context(bundle.decoder_context, indices)
        state = self.decoder.decoder.expand_stage3_state(bundle.stage3_state, indices)

        def gather(tensor):
            return self._gather_st3_proposal_field(tensor, indices)

        bridge = self.st3_transport_bridge
        resumed_queries, persistent_latents, _ = bridge(
            query_states=state.raw_query_states.transpose(0, 1),
            proposal_latent_states=gather(proposal.latent_features),
            vlm_latent_states=gather(packed_latent_hidden_states),
            seg_states=seg_states,
            local_conditions=local_cond_embeddings,
            local_condition_valid_mask=local_cond_valid_mask,
        )
        decoder_outputs = self.decoder.decoder.forward_from_stage3(
            context, state,
            resumed_query_states=resumed_queries,
            transport_latent_states=persistent_latents,
            transport_modules=bridge.transport_stages,
            return_dict=True,
        )
        queries = tuple(decoder_outputs.intermediate_hidden_states)
        masks = tuple(decoder_outputs.masks_queries_logits)
        if len(queries) != 10 or len(masks) != 10:
            raise RuntimeError("group-supervised decoder must retain st0--st9 predictions")
        final_latents = decoder_outputs.transport_final_latent_states
        if final_latents is None:
            raise RuntimeError("decoder did not return its updated ST9 latent table")
        final_conditions = bridge.readout_conditions(
            local_cond_embeddings, final_latents, local_cond_valid_mask
        )
        # Only the FINAL classification reads updated Cond. All original
        # intermediate classification/mask/Dice heads and targets are retained.
        classes = tuple(
            self.get_class_prediction(
                query.transpose(0, 1),
                final_conditions if stage == 9 else local_cond_embeddings,
                local_cond_valid_mask,
                finite_mask_logits=True,
            )
            for stage, query in enumerate(queries)
        )
        auxiliary = self.get_auxiliary_logits(classes, masks)
        supervised = self.training and row_mask_labels is not None
        if supervised and row_class_labels is None:
            raise ValueError("segmentation supervision requires class labels")
        loss_dict = None
        loss = None
        if supervised:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks[-1], class_queries_logits=classes[-1],
                mask_labels=row_mask_labels, class_labels=row_class_labels,
                auxiliary_predictions=auxiliary,
            )
            loss = self.get_loss(loss_dict)

        if supervised and torch.is_grad_enabled() and self._group_supervision_records is not None:
            relations = decoder_outputs.transport_relation_logits
            if relations is None or len(relations) != 6:
                raise RuntimeError("six differentiable shared relations are required")
            boxes = gather(proposal.sam_valid_boxes_normalized)
            for stage, relation in enumerate(relations, start=4):
                transport_stage = bridge.transport_stages[stage - 4]
                group_weights = getattr(
                    transport_stage, "group_supervision_weights", None,
                )
                if callable(group_weights):
                    read_weights = group_weights(relation)
                else:
                    _, read_weights = QueryLatentRead.routing_weights(relation)
                result = self.mask_group_auxiliary(
                    read_weights, masks[stage],
                    spatial_valid_mask=valid_mask_from_normalized_boxes(
                        boxes, masks[stage].shape[-2:]
                    ),
                )
                self._record_group_supervision(stage, result)

        final_masks = masks[-1] if supervised else self.postprocess_masks_preds(
            (masks[-1],),
            sam_valid_boxes_normalized=gather(proposal.sam_valid_boxes_normalized),
        )[-1]
        output = MaskLATSegmentorOutput(
            loss=loss, loss_dict=loss_dict,
            class_queries_logits=classes[-1], masks_queries_logits=final_masks,
            auxiliary_logits=(auxiliary if self.decoder.config.output_auxiliary_logits else None),
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            raw_query_class_logits=classes[-1], raw_query_mask_logits=masks[-1],
        )
        return output if return_dict else tuple(value for value in output.values() if value is not None)


__all__ = ["GroupSupervisedLatentSegmentor"]
