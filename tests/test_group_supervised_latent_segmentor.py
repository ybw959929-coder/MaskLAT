"""CPU integration of real grouped segmentor, decoder, bridge and group loss.

Only the heavyweight SAM/base-model constructor and criterion are stubs. The
new segmentor class, legacy class/aux/loss forwarding and spatial postprocess,
context expansion, decoder attention/mask prediction, and grouping all execute
production source. A criterion spy checks the unmodified ten-stage contract.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

import test_group_supervised_latent_decoder as decoder_tests
from test_group_supervised_latent_decoder import Decoder, Context, production
from test_group_supervised_latent_transport import legacy, transport


_ROOT = Path(__file__).resolve().parents[1] / "masklat/model/segmentors"
builder_module = importlib.import_module(
    f"{transport.__package__}.group_supervised_latent_builder"
)
group_module = importlib.import_module(f"{transport.__package__}.latent_mask_group_loss")
spatial = importlib.import_module(f"{transport.__package__}.spatial_validity")


class _Output(SimpleNamespace):
    def values(self):
        return self.__dict__.values()


class _CriterionSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        result = {
            "loss_mask": kwargs["masks_queries_logits"].square().mean(),
            "loss_class": kwargs["class_queries_logits"].square().mean(),
        }
        for stage, prediction in enumerate(kwargs["auxiliary_predictions"]):
            result[f"loss_mask_{stage}"] = prediction["masks_queries_logits"].square().mean()
            result[f"loss_class_{stage}"] = prediction["class_queries_logits"].square().mean()
        return result


def _classes():
    base_path = _ROOT / "masklat_segmentor.py"
    model_path = _ROOT / "mask2former/modeling_mask2former.py"
    new_path = _ROOT / "group_supervised_latent_segmentor.py"
    base_tree = ast.parse(base_path.read_text(encoding="utf-8"))
    model_tree = ast.parse(model_path.read_text(encoding="utf-8"))
    new_tree = ast.parse(new_path.read_text(encoding="utf-8"))
    base = next(node for node in base_tree.body
                if isinstance(node, ast.ClassDef) and node.name == "MaskLATSegmentor")
    keep = {"configure_st3_bipartite_latent_transport", "get_class_prediction",
            "get_auxiliary_logits", "get_loss_dict", "get_loss",
            "postprocess_masks_preds", "_gather_st3_proposal_field"}
    base_methods = [method for method in base.body
                    if isinstance(method, ast.FunctionDef) and method.name in keep]
    base_stub = ast.parse('''
class MaskLATSegmentor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.dec_config = decoder.config
        self.enc_config = SimpleNamespace(image_size=6)
        self.st3_transport_proposal_builder = None
        self.st3_transport_bridge = None
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.weight_dict = {"loss_mask": 1.0, "loss_class": 1.0}
        self.criterion = _CriterionSpy()
    def prepare_st3_latent_execution(self, **kwargs):
        return kwargs["test_bundle"]
''').body[0]
    base_stub.body.extend(base_methods)
    decoder_node = next(node for node in model_tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "Mask2FormerMaskedAttentionDecoder")
    expand_state = next(method for method in decoder_node.body
                        if isinstance(method, ast.FunctionDef) and method.name == "expand_stage3_state")
    transformer_node = next(node for node in model_tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "Mask2FormerTransformerModule")
    expand_context = next(method for method in transformer_node.body
                          if isinstance(method, ast.FunctionDef) and method.name == "expand_decoder_context")
    transformer_stub = ast.parse('''
class Mask2FormerTransformerModule(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.config = decoder.config
''').body[0]
    transformer_stub.body.append(expand_context)
    bundle = next(node for node in base_tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "St3LatentExecutionBundle")
    new_class = next(node for node in new_tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        "__name__": __name__, "torch": torch, "nn": nn, "F": F,
        "dataclass": dataclass, "SimpleNamespace": SimpleNamespace,
        "Tensor": torch.Tensor, "_CriterionSpy": _CriterionSpy,
        "MaskLATSegmentorOutput": _Output,
        "Mask2FormerDecoderContext": Context,
        "Mask2FormerStage3State": production["Mask2FormerStage3State"],
        "St3BipartiteLatentBuilder": legacy.St3BipartiteLatentBuilder,
        "BipartiteLatentTransportBridge": legacy.BipartiteLatentTransportBridge,
        "St123LatentCascadeBuilder": type("UnusedCascade", (), {}),
        "St123LatentDeepstackBuilder": type("UnusedDeepstack", (), {}),
        "GroupSupervisedLatentBuilder": builder_module.GroupSupervisedLatentBuilder,
        "GroupSupervisedLatentTransportBridge": transport.GroupSupervisedLatentTransportBridge,
        "MaskGroupAuxiliaryLoss": group_module.MaskGroupAuxiliaryLoss,
        "QueryLatentRead": legacy.QueryLatentRead,
        "valid_mask_from_normalized_boxes": spatial.valid_mask_from_normalized_boxes,
        "validate_normalized_valid_boxes": spatial.validate_normalized_valid_boxes,
        "masked_normalized_resize": spatial.masked_normalized_resize,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=future + [expand_state, transformer_stub, base_stub, bundle, new_class],
        type_ignores=[],
    )), str(new_path), "exec"), namespace)
    return namespace


classes = _classes()
Segmentor = classes["GroupSupervisedLatentSegmentor"]
Bundle = classes["St3LatentExecutionBundle"]


class GroupSupervisedSegmentorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _case(self, training=True, repeats=(1, 0, 1)):
        # Reuse production decoder construction, not its already constructed
        # bridge: exercise the actual new segmentor configure path below.
        helper = decoder_tests.GroupSupervisedDecoderTest()
        decoder, _, context, state, _, _, _ = helper._case(training=training)
        decoder.expand_stage3_state = classes["expand_stage3_state"].__get__(decoder)
        for key, value in dict(
            use_st3_bipartite_latent_transport=True,
            st3_transport_require_expected_shapes=False,
            st3_latent_num_tokens=64, st3_latent_num_heads=2,
            st3_latent_mask_threshold=0.5, st3_latent_fourier_features=4,
            st3_transport_pre_vlm_depth=2, st3_transport_post_vlm_depth=1,
            st3_transport_decoder_depth=6, st3_transport_output_init_std=1e-3,
            output_auxiliary_logits=True,
        ).items():
            setattr(decoder.config, key, value)
        wrapper = classes["Mask2FormerTransformerModule"](decoder)
        segmentor = Segmentor(decoder=wrapper, group_loss_config={"min_mask_area": 1})
        segmentor.configure_st3_bipartite_latent_transport(siglip_hidden_dim=5, llm_hidden_dim=12)
        segmentor.train(training)
        stage3_record = segmentor.mask_group_auxiliary(
            torch.randn(2, 64, 5, requires_grad=True).softmax(-1), state.mask_logits,
        ) if training else None
        proposal = SimpleNamespace(
            latent_features=torch.randn(2, 64, 8, requires_grad=True),
            packed_group_tokens=torch.randn(2, 64, 12),
            sam_valid_boxes_normalized=torch.tensor([[0., 0., 1., 1.], [0., 0., 0.5, 1.]]),
            group_supervision=stage3_record,
        )
        bundle = Bundle(decoder_context=context, stage3_state=state, proposal=proposal)
        segmentor._group_supervision_records = {} if training else None
        segmentor.prepare_st3_latent_execution(test_bundle=bundle)
        nseg = len(repeats)
        cond = torch.randn(nseg, 4, 8, requires_grad=True)
        valid = torch.tensor([[True, True, False, True]] * nseg)
        masks = [torch.ones(1, 2, 2) for _ in repeats]
        labels = [torch.zeros(1, dtype=torch.long) for _ in repeats]
        kwargs = dict(
            bundle=bundle,
            packed_latent_hidden_states=torch.randn(2, 64, 8, requires_grad=True),
            seg_to_sample=torch.tensor(repeats),
            seg_states=torch.randn(nseg, 8, requires_grad=True),
            local_cond_embeddings=cond,
            local_cond_valid_mask=valid,
            row_mask_labels=masks if training else None,
            row_class_labels=labels if training else None,
        )
        captured = {"class_calls": [], "mask_calls": []}
        original_decoder = decoder.forward_from_stage3
        original_class = segmentor.get_class_prediction
        original_postprocess = segmentor.postprocess_masks_preds

        def decoder_call(*args, **kwargs):
            captured["context"] = args[0]
            captured["state"] = args[1]
            output = original_decoder(*args, **kwargs)
            captured["decoder_output"] = output
            return output

        def class_call(query, cond, valid, **kwargs):
            captured["class_calls"].append((query, cond, valid))
            return original_class(query, cond, valid, **kwargs)

        def postprocess_call(masks, **kwargs):
            captured["mask_calls"].append((masks, kwargs))
            return original_postprocess(masks, **kwargs)

        decoder.forward_from_stage3 = decoder_call
        segmentor.get_class_prediction = class_call
        segmentor.postprocess_masks_preds = postprocess_call
        return segmentor, kwargs, captured

    def test_only_final_classification_uses_readout_cond_raw_m9_unchanged(self):
        segmentor, kwargs, captured = self._case()
        output = segmentor.finish_st3_latent_execution(**kwargs)
        calls = captured["class_calls"]
        self.assertEqual(len(calls), 10)
        original_cond = kwargs["local_cond_embeddings"]
        for _, cond, _ in calls[:9]:
            self.assertIs(cond, original_cond)
        final_cond = calls[9][1]
        self.assertIsNot(final_cond, original_cond)
        inactive = ~kwargs["local_cond_valid_mask"].clone()
        inactive[:, -1] = True
        self.assertTrue(torch.equal(final_cond[inactive], original_cond[inactive]))
        raw_m9 = captured["decoder_output"].masks_queries_logits[-1]
        self.assertIs(output.raw_query_mask_logits, raw_m9)
        self.assertIs(output.masks_queries_logits, raw_m9)
        self.assertEqual(captured["mask_calls"], [])
        self.assertEqual(len(segmentor.criterion.calls), 1)
        payload = segmentor.criterion.calls[0]
        self.assertIs(payload["masks_queries_logits"], raw_m9)
        self.assertIs(payload["class_queries_logits"], output.raw_query_class_logits)
        self.assertIs(payload["mask_labels"], kwargs["row_mask_labels"])
        self.assertIs(payload["class_labels"], kwargs["row_class_labels"])
        self.assertEqual(len(payload["auxiliary_predictions"]), 9)
        for stage, auxiliary in enumerate(payload["auxiliary_predictions"]):
            self.assertIs(auxiliary["masks_queries_logits"], captured["decoder_output"].masks_queries_logits[stage])
            self.assertEqual(auxiliary["class_queries_logits"].shape, (3, 5, 4))
        self.assertEqual(set(segmentor._group_supervision_records), set(range(3, 10)))
        self.assertEqual(len(output.loss_dict), 20)
        self.assertFalse(any("group" in key for key in output.loss_dict))
        torch.testing.assert_close(output.loss, sum(output.loss_dict.values()), rtol=0, atol=0)

    def test_per_seg_expansion_and_valid_regions_follow_source_image_indices(self):
        segmentor, kwargs, captured = self._case(repeats=(1, 0, 1))
        observed = []
        handle = segmentor.mask_group_auxiliary.register_forward_pre_hook(
            lambda module, args, call_kwargs: observed.append((args, call_kwargs)),
            with_kwargs=True,
        )
        segmentor.finish_st3_latent_execution(**kwargs)
        handle.remove()
        indices = kwargs["seg_to_sample"]
        original_context = kwargs["bundle"].decoder_context
        torch.testing.assert_close(
            captured["context"].pixel_embeddings,
            original_context.pixel_embeddings.index_select(0, indices), rtol=0, atol=0,
        )
        torch.testing.assert_close(
            captured["state"].mask_logits,
            kwargs["bundle"].stage3_state.mask_logits.index_select(0, indices), rtol=0, atol=0,
        )
        boxes = kwargs["bundle"].proposal.sam_valid_boxes_normalized.index_select(0, indices)
        expected_valid = spatial.valid_mask_from_normalized_boxes(boxes, (2, 2))
        self.assertEqual(len(observed), 6)
        for stage, (args, call_kwargs) in enumerate(observed, start=4):
            self.assertIs(args[1], captured["decoder_output"].masks_queries_logits[stage])
            self.assertEqual(args[0].shape, (3, 64, 5))
            torch.testing.assert_close(call_kwargs["spatial_valid_mask"], expected_valid, rtol=0, atol=0)
        self.assertTrue(torch.equal(expected_valid[0], expected_valid[2]))
        self.assertFalse(torch.equal(expected_valid[0], expected_valid[1]))

    def test_final_classification_alone_backpropagates_all_six_writebacks(self):
        segmentor, kwargs, captured = self._case()
        output = segmentor.finish_st3_latent_execution(**kwargs)
        # Exclude mask and group losses: test the claimed final readout route.
        valid = kwargs["local_cond_valid_mask"][:, None, :].expand_as(output.class_queries_logits)
        output.class_queries_logits[valid].square().mean().backward()
        for index, stage in enumerate(segmentor.st3_transport_bridge.transport_stages, start=4):
            for name in ("query_value.weight", "latent_output.weight"):
                parameter = dict(stage.named_parameters())[name]
                self.assertIsNotNone(parameter.grad, f"st{index} {name}")
                self.assertGreater(float(parameter.grad.abs().sum()), 0, f"st{index} {name}")
        self.assertGreater(float(kwargs["packed_latent_hidden_states"].grad.abs().sum()), 0)

    def test_eval_postprocesses_only_m9_and_never_collects_training_auxiliary(self):
        segmentor, kwargs, captured = self._case(training=False)
        with torch.no_grad():
            output = segmentor.finish_st3_latent_execution(**kwargs)
        self.assertIsNone(output.loss)
        self.assertIsNone(output.loss_dict)
        self.assertIsNone(segmentor._group_supervision_records)
        self.assertEqual(segmentor.criterion.calls, [])
        self.assertEqual(len(captured["mask_calls"]), 1)
        self.assertIs(captured["mask_calls"][0][0][0], output.raw_query_mask_logits)
        self.assertEqual(output.raw_query_mask_logits.shape, (3, 5, 2, 2))
        self.assertEqual(output.masks_queries_logits.shape, (3, 5, 6, 6))
        expected_boxes = kwargs["bundle"].proposal.sam_valid_boxes_normalized.index_select(0, kwargs["seg_to_sample"])
        torch.testing.assert_close(
            captured["mask_calls"][0][1]["sam_valid_boxes_normalized"], expected_boxes,
            rtol=0, atol=0,
        )
        # Valid boxes are (top, left, bottom, right), so image 1's bottom
        # half is padding in this fixture.
        self.assertTrue(torch.equal(output.masks_queries_logits[0, :, 3:, :], torch.full((5, 3, 6), -20.)))
        self.assertIsNone(captured["decoder_output"].transport_relation_logits)

    def test_configuration_keeps_weight_paths_and_is_idempotent(self):
        segmentor, _, _ = self._case()
        before = {key: value.detach().clone() for key, value in segmentor.state_dict().items()}
        builder_id = id(segmentor.st3_transport_proposal_builder)
        bridge_id = id(segmentor.st3_transport_bridge)
        segmentor.configure_st3_bipartite_latent_transport(siglip_hidden_dim=5, llm_hidden_dim=12)
        self.assertEqual(id(segmentor.st3_transport_proposal_builder), builder_id)
        self.assertEqual(id(segmentor.st3_transport_bridge), bridge_id)
        after = segmentor.state_dict()
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertIn("st3_transport_proposal_builder.group_supervision_version", after)
        self.assertIn("st3_transport_bridge.transport_stages.5.latent_output.weight", after)
        self.assertIn("st3_transport_bridge.final_condition_reader.cross_attention.out_proj.weight", after)
        with self.assertRaisesRegex(ValueError, "different widths"):
            segmentor.configure_st3_bipartite_latent_transport(siglip_hidden_dim=6, llm_hidden_dim=12)

    def test_duplicate_auxiliary_record_rejected_and_class_labels_required(self):
        segmentor, kwargs, _ = self._case()
        with self.assertRaisesRegex(RuntimeError, "collected twice"):
            segmentor.prepare_st3_latent_execution(test_bundle=kwargs["bundle"])
        kwargs["row_class_labels"] = None
        with self.assertRaisesRegex(ValueError, "requires class labels"):
            segmentor.finish_st3_latent_execution(**kwargs)

    def test_configure_occurs_before_original_s1_s2_loads(self):
        source = (_ROOT.parent / "masklat.py").read_text(encoding="utf-8")
        configure = source.index("self.segmentor.configure_st3_bipartite_latent_transport(")
        self.assertLess(configure, source.index("if s1_pretrained_pth is not None:"))
        self.assertLess(configure, source.index("if s2_pretrained_pth is not None:"))


if __name__ == "__main__":
    unittest.main()
