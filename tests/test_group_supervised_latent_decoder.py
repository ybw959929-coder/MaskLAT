"""Execute production decoder ordering/checkpoint paths with tiny CPU tensors.

AST extraction only avoids unrelated SAM/CUDA/transformers import dependencies;
the attention, mask predictor, decoder layers and split execution methods are
the actual production implementations, not copies of their algorithm.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from test_group_supervised_latent_transport import legacy, transport


def _production_classes():
    path = (Path(__file__).resolve().parents[1] /
            "masklat/model/segmentors/mask2former/modeling_mask2former.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "Mask2FormerAttention", "Mask2FormerMaskedAttentionDecoderLayer",
        "Mask2FormerMaskedAttentionDecoder", "Mask2FormerDecoderContext",
        "Mask2FormerStage3State", "Mask2FormerMaskedAttentionDecoderOutput",
        "Mask2FormerMaskPredictor", "Mask2FormerMLPPredictionHead",
        "Mask2FormerPredictionBlock",
    }
    selected = [node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        if node.name == "Mask2FormerMaskedAttentionDecoder":
            node.body = [method for method in node.body
                         if isinstance(method, ast.FunctionDef) and method.name in {
                             "__init__", "_validate_split_context",
                             "forward_to_stage3", "forward_from_stage3",
                         }]
    namespace = {
        "__name__": __name__, "torch": torch, "nn": nn, "Tensor": torch.Tensor,
        "dataclass": dataclass, "BaseModelOutputWithCrossAttentions": object,
        "ACT2FN": {"relu": torch.nn.functional.relu},
        "_finite_diagnostics_enabled": lambda: False,
        "is_torchdynamo_compiling": lambda: False,
        "is_torch_greater_or_equal_than_2_1": True,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=future + selected, type_ignores=[],
    )), str(path), "exec"), namespace)
    return namespace


production = _production_classes()
Decoder = production["Mask2FormerMaskedAttentionDecoder"]
Context = production["Mask2FormerDecoderContext"]


class GroupSupervisedDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _case(self, *, delayed=True, pre_norm=False, checkpoint_mode=None,
              training=True, dtype=torch.float32):
        torch.manual_seed(7001)
        config = SimpleNamespace(
            hidden_dim=8, pre_norm=pre_norm, num_attention_heads=2,
            dropout=0.0, activation_function="relu", dim_feedforward=16,
            mask_feature_size=8, num_feature_levels=3, decoder_layers=10,
            use_return_dict=True,
        )
        decoder = Decoder(config).to(dtype).train(training)
        if checkpoint_mode is not None:
            decoder.gradient_checkpointing = True
            decoder._gradient_checkpointing_func = partial(
                checkpoint, use_reentrant=checkpoint_mode,
            )
        bridge_class = (transport.GroupSupervisedLatentTransportBridge if delayed
                        else legacy.BipartiteLatentTransportBridge)
        bridge = bridge_class(8, 2).to(dtype).train(training)
        visual = tuple(torch.randn(4, 2, 8, dtype=dtype, requires_grad=True)
                       for _ in range(3))
        context = Context(
            initial_query_states=torch.randn(5, 2, 8, dtype=dtype, requires_grad=True),
            query_position_embeddings=torch.randn(5, 2, 8, dtype=dtype),
            encoder_hidden_states=visual,
            positional_embeddings=tuple(torch.zeros_like(v) for v in visual),
            pixel_embeddings=torch.randn(2, 8, 2, 2, dtype=dtype, requires_grad=True),
            feature_size_list=((2, 2),) * 3,
        )
        state = decoder.forward_to_stage3(
            context, output_attentions=True, output_hidden_states=True,
        )
        cond = torch.randn(2, 4, 8, dtype=dtype, requires_grad=True)
        valid = torch.tensor([[True, True, False, True], [True] * 4])
        query, latent, _ = bridge(
            query_states=state.normalized_query_states.transpose(0, 1),
            proposal_latent_states=torch.randn(2, 3, 8, dtype=dtype, requires_grad=True),
            vlm_latent_states=torch.randn(2, 3, 8, dtype=dtype, requires_grad=True),
            seg_states=torch.randn(2, 8, dtype=dtype, requires_grad=True),
            local_conditions=cond, local_condition_valid_mask=valid,
        )
        kwargs = dict(
            resumed_query_states=query, transport_latent_states=latent,
            transport_modules=bridge.transport_stages,
        )
        return decoder, bridge, context, state, kwargs, cond, valid

    def _trace(self, decoder, bridge):
        events, reads, writes, mask_queries = [], {}, {}, {}
        handles = []
        mask_counter = [4]

        def on_mask(module, arguments):
            stage = mask_counter[0]
            mask_counter[0] += 1
            events.append((stage, "mask"))
            mask_queries[stage] = arguments[0].transpose(0, 1)

        handles.append(decoder.mask_predictor.register_forward_pre_hook(on_mask))
        for index, stage_module in enumerate(bridge.transport_stages, start=4):
            layer = decoder.layers[index - 1]
            for module, event in ((layer.cross_attn, "visual"),
                                  (layer.self_attn, "self"), (layer.fc1, "ffn")):
                handles.append(module.register_forward_pre_hook(
                    lambda module, arguments, index=index, event=event:
                    events.append((index, event))
                ))
            original_read = stage_module.update_queries
            original_write = stage_module.update_latents

            def read(query, latent, index=index, original_read=original_read):
                events.append((index, "read"))
                result = original_read(query, latent)
                reads[index] = result[1]
                return result

            def write(query, latent, relation, index=index, original_write=original_write):
                events.append((index, "write"))
                result = original_write(query, latent, relation)
                writes[index] = (query, relation, result)
                return result

            stage_module.update_queries = read
            stage_module.update_latents = write
        return events, reads, writes, mask_queries, handles

    def test_real_stage4_to9_order_same_relation_and_final_query(self):
        for pre_norm in (True, False):
            with self.subTest(pre_norm=pre_norm):
                decoder, bridge, context, state, kwargs, _, _ = self._case(pre_norm=pre_norm)
                events, reads, writes, mask_queries, handles = self._trace(decoder, bridge)
                output = decoder.forward_from_stage3(context, state, **kwargs)
                for handle in handles:
                    handle.remove()
                self.assertEqual(events, [
                    (stage, event) for stage in range(4, 10)
                    for event in ("visual", "read", "self", "ffn", "mask", "write")
                ])
                self.assertEqual(len(output.transport_relation_logits), 6)
                for stage in range(4, 10):
                    query, relation, _ = writes[stage]
                    self.assertIs(relation, reads[stage])
                    self.assertIs(relation, output.transport_relation_logits[stage - 4])
                    torch.testing.assert_close(query, mask_queries[stage], rtol=0, atol=0)
                    torch.testing.assert_close(
                        query, output.intermediate_hidden_states[stage].transpose(0, 1),
                        rtol=0, atol=0,
                    )
                self.assertIs(output.transport_final_latent_states, writes[9][2])
                self.assertEqual(len(output.masks_queries_logits), 10)
                self.assertEqual(len(output.attentions), 9)
                self.assertEqual(output.attentions[-1].shape, (2, 2, 5, 5))

    def test_real_legacy_order_and_output_layout_unchanged(self):
        decoder, bridge, context, state, kwargs, _, _ = self._case(delayed=False)
        events, _, _, _, handles = self._trace(decoder, bridge)
        output = decoder.forward_from_stage3(context, state, **kwargs)
        for handle in handles:
            handle.remove()
        self.assertEqual(events, [
            (stage, event) for stage in range(4, 10)
            for event in ("visual", "read", "self", "write", "ffn", "mask")
        ])
        self.assertIsNone(output.transport_final_latent_states)
        self.assertIsNone(output.transport_relation_logits)
        self.assertEqual(output.attentions[-1].shape, (2, 2, 5, 5))
        fresh = self._case(delayed=False)
        decoder, _, context, state, kwargs, _, _ = fresh
        as_tuple = decoder.forward_from_stage3(context, state, return_dict=False, **kwargs)
        self.assertEqual(len(as_tuple), 5)
        torch.testing.assert_close(as_tuple[0], output.last_hidden_state, rtol=0, atol=0)
        self.assertEqual(len(as_tuple[1]), 10)
        self.assertEqual(len(as_tuple[2]), 9)
        self.assertEqual(len(as_tuple[3]), 10)
        self.assertEqual(len(as_tuple[4]), 10)

    def test_layer_tuple_layout_delayed_vs_legacy_attention_offsets(self):
        for delayed in (True, False):
            decoder, bridge, context, state, kwargs, _, _ = self._case(delayed=delayed)
            layer = decoder.layers[3]
            z = kwargs["transport_latent_states"]
            outputs = layer.forward_with_bipartite_transport(
                kwargs["resumed_query_states"].transpose(0, 1), z,
                bridge.transport_stages[0], level_index=0,
                position_embeddings=context.positional_embeddings,
                query_position_embeddings=context.query_position_embeddings,
                encoder_hidden_states=context.encoder_hidden_states,
                # The outer decoder clears fully-masked query rows before
                # calling the layer. This layout-only direct call uses no mask.
                encoder_attention_mask=None,
                output_attentions=True,
            )
            self.assertEqual(len(outputs), 5 if delayed else 4)
            if delayed:
                self.assertIs(outputs[1], z)
                self.assertEqual(outputs[2].shape, (2, 5, 3))
            self.assertEqual(outputs[3 if delayed else 2].shape, (2, 2, 5, 5))

    def _gradient_run(self, pre_norm, checkpoint_mode):
        decoder, bridge, context, state, kwargs, cond, valid = self._case(
            pre_norm=pre_norm, checkpoint_mode=checkpoint_mode,
        )
        output = decoder.forward_from_stage3(context, state, **kwargs)
        final_cond = bridge.readout_conditions(
            cond, output.transport_final_latent_states, valid,
        )
        query = output.intermediate_hidden_states[-1].transpose(0, 1)
        classification = torch.einsum("bqd,bcd->bqc", query, final_cond)
        loss = classification.square().mean()
        loss = loss + sum(mask.square().mean() for mask in output.masks_queries_logits)
        # Exercise the explicit R graph returned by checkpoint, as actual group
        # supervision will do, independently of its indirect writeback route.
        loss = loss + sum(-relation.log_softmax(dim=1)[:, 0].mean()
                          for relation in output.transport_relation_logits)
        loss.backward()
        gradients = {}
        for root_name, model in (("decoder", decoder), ("bridge", bridge)):
            for name, parameter in model.named_parameters():
                name = f"{root_name}.{name}"
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(parameter.grad.isfinite().all()), name)
                gradients[name] = parameter.grad.detach().clone()
        for level, visual in enumerate(context.encoder_hidden_states):
            self.assertIsNotNone(visual.grad, f"visual {level}")
            gradients[f"visual.{level}"] = visual.grad.detach().clone()
        return (output.last_hidden_state.detach(),
                output.transport_final_latent_states.detach(), gradients)

    def test_checkpoint_off_reentrant_and_nonreentrant_outputs_and_all_gradients(self):
        for pre_norm in (False, True):
            expected_q, expected_z, expected_grads = self._gradient_run(pre_norm, None)
            for reentrant in (True, False):
                with self.subTest(pre_norm=pre_norm, reentrant=reentrant):
                    actual_q, actual_z, actual_grads = self._gradient_run(pre_norm, reentrant)
                    torch.testing.assert_close(actual_q, expected_q)
                    torch.testing.assert_close(actual_z, expected_z)
                    self.assertEqual(actual_grads.keys(), expected_grads.keys())
                    for name in actual_grads:
                        torch.testing.assert_close(actual_grads[name], expected_grads[name],
                                                   rtol=2e-4, atol=1e-6, msg=name)

    def test_eval_or_no_grad_keeps_final_z_without_retaining_relation_tuple(self):
        for training in (False, True):
            decoder, _, context, state, kwargs, _, _ = self._case(training=training)
            with torch.no_grad():
                output = decoder.forward_from_stage3(context, state, **kwargs)
            self.assertEqual(output.transport_final_latent_states.shape, (2, 3, 8))
            self.assertIsNone(output.transport_relation_logits)
        decoder, _, context, state, kwargs, _, _ = self._case(training=False)
        output = decoder.forward_from_stage3(context, state, **kwargs)
        self.assertIsNone(output.transport_relation_logits)

    def test_delayed_protocol_rejects_mixed_stage_modes_before_execution(self):
        decoder, bridge, context, state, kwargs, _, _ = self._case()
        bridge.transport_stages[2] = legacy.BipartiteLatentTransportStage(8, 2)
        with self.assertRaisesRegex(ValueError, "every st4--st9"):
            decoder.forward_from_stage3(context, state, **kwargs)

    def test_transport_rejects_legacy_prediction_refiners(self):
        decoder, _, context, state, kwargs, _, _ = self._case()
        for argument in ("pre_prediction_refiner", "stage_refiner"):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                decoder.forward_from_stage3(context, state, **kwargs,
                                           **{argument: lambda *args: None})


if __name__ == "__main__":
    unittest.main()
