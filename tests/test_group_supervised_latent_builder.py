"""Real CPU builder integration, without importing optional training stacks."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


_DIRECTORY = Path(__file__).resolve().parents[1] / "masklat/model/segmentors/mask2former"
_PACKAGE = "_group_supervised_builder_test"


def load_builder_module():
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_DIRECTORY)]
    sys.modules[_PACKAGE] = package
    for name in (
        "spatial_validity", "mask_topology_grouping", "topology_group_decoder",
        "st3_group_vlm_refiner", "st3_proposal_latent_bridge",
        "st3_bipartite_latent_transport", "latent_mask_group_loss",
        "group_supervised_latent_builder",
    ):
        qualified = f"{_PACKAGE}.{name}"
        specification = importlib.util.spec_from_file_location(qualified, _DIRECTORY / f"{name}.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[qualified] = module
        specification.loader.exec_module(module)
    return module


module = load_builder_module()
GroupBuilder = module.GroupSupervisedLatentBuilder
LegacyBuilder = module.St3BipartiteLatentBuilder


def kwargs():
    return dict(
        hidden_dim=8, sam_feature_dim=3, siglip_feature_dim=5,
        llm_hidden_dim=12, num_latents=4, num_heads=2,
        fourier_features=4, pre_vlm_depth=2,
    )


def builder_inputs(padded=False):
    masks = torch.full((1, 4, 4, 4), -4.)
    masks[0, :2, :, :2] = 4.
    masks[0, 2:, :, 2:] = 4.
    identity = torch.eye(3).unsqueeze(0)
    metadata = {
        "original_size": torch.tensor([[4., 4.]]),
        "original_to_sam": identity,
        "sam_input_size": torch.tensor([[4., 4.]]),
        "sam_valid_region": torch.tensor([[0., 0., 4., 2. if padded else 4.]]),
        "original_to_siglip": identity,
        "siglip_input_size": torch.tensor([[4., 4.]]),
        "siglip_valid_region": torch.tensor([[0., 0., 4., 4.]]),
        "siglip_patch_size": torch.tensor([[2., 2.]]),
    }
    return dict(
        query_states=torch.randn(1, 4, 8, requires_grad=True),
        mask_logits=masks.requires_grad_(),
        sam_mask_features=torch.randn(1, 3, 4, 4),
        siglip_spatial_features=torch.randn(1, 5, 2, 2),
        spatial_metadata=metadata,
    )


class GroupBuilderIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(439)

    def assert_hooks_removed(self, builder):
        for block in builder.pre_vlm_blocks:
            self.assertEqual(len(block.cross_attention._forward_pre_hooks), 0)
            self.assertEqual(len(block.cross_attention._forward_hooks), 0)

    def test_last_real_attention_is_supervised_and_backpropagates(self):
        builder = GroupBuilder(**kwargs())
        inputs = builder_inputs()
        first = builder.pre_vlm_blocks[0].cross_attention
        last = builder.pre_vlm_blocks[-1].cross_attention
        with (
            mock.patch.object(first, "forward", wraps=first.forward) as first_call,
            mock.patch.object(last, "forward", wraps=last.forward) as last_call,
        ):
            output = builder(**inputs)
        self.assertFalse(first_call.call_args.kwargs["need_weights"])
        self.assertTrue(last_call.call_args.kwargs["need_weights"])
        self.assertTrue(last_call.call_args.kwargs["average_attn_weights"])
        self.assertEqual(output.group_supervision["selected_groups"], [[[0, 1], [2, 3]]])
        self.assertTrue(output.group_supervision["loss"].requires_grad)
        output.group_supervision["loss"].backward()
        self.assertIsNotNone(last.in_proj_weight.grad)
        self.assertGreater(last.in_proj_weight.grad.abs().sum().item(), 0.)
        self.assertIsNotNone(inputs["query_states"].grad)
        self.assertIsNone(builder.latent_query.weight.grad)
        self.assertIsNone(builder.proposal_key.weight.grad)
        self.assert_hooks_removed(builder)

    def test_grouping_stops_mask_target_gradient_but_proposal_path_can_use_masks(self):
        builder = GroupBuilder(**kwargs())
        inputs = builder_inputs()
        attention = torch.full((1, 4, 4), .25, requires_grad=True)
        pure_loss = builder.mask_group_auxiliary(attention, inputs["mask_logits"])["loss"]
        pure_loss.backward()
        self.assertIsNone(inputs["mask_logits"].grad)
        result = builder(**inputs)
        result.packed_group_tokens.square().mean().backward()
        self.assertIsNotNone(inputs["mask_logits"].grad)
        self.assertTrue(torch.isfinite(inputs["mask_logits"].grad).all())
        self.assertGreater(inputs["mask_logits"].grad.abs().sum().item(), 0.)

    def test_no_grad_eval_or_disabled_forward_does_not_install_hooks_or_group(self):
        for eval_mode, no_grad, enabled in ((True, False, True), (False, True, True), (False, False, False)):
            with self.subTest(eval_mode=eval_mode, no_grad=no_grad, enabled=enabled):
                builder = GroupBuilder(**kwargs())
                builder.train(not eval_mode)
                builder.group_supervision_enabled = enabled
                last = builder.pre_vlm_blocks[-1].cross_attention
                with (
                    mock.patch.object(last, "register_forward_pre_hook", side_effect=AssertionError("unexpected hook")),
                    mock.patch.object(builder.mask_group_auxiliary, "forward", side_effect=AssertionError("unexpected grouping")),
                    torch.set_grad_enabled(not no_grad),
                ):
                    output = builder(**builder_inputs())
                self.assertIsNone(output.group_supervision)
                self.assert_hooks_removed(builder)

    def test_forward_exception_removes_scoped_hooks(self):
        builder = GroupBuilder(**kwargs())
        attention = builder.pre_vlm_blocks[-1].cross_attention
        with mock.patch.object(attention, "forward", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                builder(**builder_inputs())
        self.assert_hooks_removed(builder)
        output = builder(**builder_inputs())
        self.assertIsNotNone(output.group_supervision)

    def test_group_loss_exception_also_leaves_no_hooks(self):
        builder = GroupBuilder(**kwargs())
        with mock.patch.object(builder.mask_group_auxiliary, "forward", side_effect=RuntimeError("loss injected")):
            with self.assertRaisesRegex(RuntimeError, "loss injected"):
                builder(**builder_inputs())
        self.assert_hooks_removed(builder)

    def test_only_checkpoint_difference_is_version_marker(self):
        legacy = LegacyBuilder(**kwargs())
        grouped = GroupBuilder(**kwargs())
        self.assertEqual(set(grouped.state_dict()) - set(legacy.state_dict()), {"group_supervision_version"})
        self.assertEqual(set(legacy.state_dict()) - set(grouped.state_dict()), set())
        incompatibility = grouped.load_state_dict(legacy.state_dict(), strict=False)
        self.assertEqual(incompatibility.missing_keys, ["group_supervision_version"])
        self.assertEqual(incompatibility.unexpected_keys, [])
        self.assertEqual(grouped.state_dict()["group_supervision_version"].item(), 1)
        clone = GroupBuilder(**kwargs())
        clone.load_state_dict(grouped.state_dict(), strict=True)
        legacy.eval()
        grouped.eval()
        inputs = builder_inputs()
        with torch.no_grad():
            old_output, new_output = legacy(**inputs), grouped(**inputs)
        torch.testing.assert_close(old_output.packed_group_tokens, new_output.packed_group_tokens)
        torch.testing.assert_close(old_output.latent_features, new_output.latent_features)

    def test_valid_geometry_and_padding_exclude_padding_only_queries(self):
        builder = GroupBuilder(**kwargs())
        result = builder(**builder_inputs(padded=True))
        self.assertEqual(result.group_supervision["selected_groups"], [[[0, 1]]])
        self.assertEqual(result.group_supervision["num_valid_queries"].tolist(), [2])
        self.assertEqual(result.query_geometry_valid_mask.tolist(), [[True, True, False, False]])

    def test_no_attention_graph_is_stored_on_builder_after_forward(self):
        builder = GroupBuilder(**kwargs())
        first = builder(**builder_inputs())
        second = builder(**builder_inputs())
        self.assertIsNot(first.group_supervision, second.group_supervision)
        self.assert_hooks_removed(builder)
        self.assertFalse(any("captur" in key or "cache" in key for key in vars(builder)))


if __name__ == "__main__":
    unittest.main()
