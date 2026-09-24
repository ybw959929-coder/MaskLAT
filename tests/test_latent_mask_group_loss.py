"""CPU tests of detached grouping and differentiable latent-read supervision."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import torch


_PATH = (
    Path(__file__).resolve().parents[1]
    / "masklat/model/segmentors/mask2former/latent_mask_group_loss.py"
)
_SPEC = importlib.util.spec_from_file_location("_latent_mask_group_loss_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
MaskGroupAuxiliaryLoss = _MODULE.MaskGroupAuxiliaryLoss


def mask_logits(foreground):
    binary = torch.as_tensor(foreground, dtype=torch.bool)
    return torch.where(binary, 10.0, -10.0)


def two_groups():
    return mask_logits([
        [[[1, 1, 0, 0], [1, 1, 0, 0]],
         [[1, 1, 0, 0], [1, 1, 0, 0]],
         [[0, 0, 1, 1], [0, 0, 1, 1]],
         [[0, 0, 1, 1], [0, 0, 1, 1]]]
    ])


class MaskGroupLossTest(unittest.TestCase):
    def test_perfect_uniform_group_reads_have_zero_loss(self):
        attention = torch.tensor([[[0., 0., .5, .5], [.5, .5, 0., 0.]]], requires_grad=True)
        result = MaskGroupAuxiliaryLoss()(attention, two_groups())
        self.assertEqual(result["selected_groups"], [[[0, 1], [2, 3]]])
        self.assertAlmostEqual(result["loss"].item(), 0.0, places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(attention.grad).all())

    def test_uniform_collapsed_reads_receive_different_group_gradients(self):
        attention = torch.full((1, 2, 4), .25, requires_grad=True)
        logits = two_groups().requires_grad_()
        result = MaskGroupAuxiliaryLoss()(attention, logits)
        self.assertAlmostEqual(result["loss"].item(), math.log(2), places=6)
        result["loss"].backward()
        self.assertIsNone(logits.grad)
        self.assertTrue(torch.allclose(attention.grad[0, 0], -attention.grad[0, 1]))
        self.assertLess(attention.grad[0, 0, 0].item(), 0)
        self.assertGreater(attention.grad[0, 0, 2].item(), 0)

    def test_complete_link_does_not_merge_overlap_chain(self):
        binary = torch.zeros(1, 3, 1, 10, dtype=torch.bool)
        binary[0, 0, 0, :8] = True
        binary[0, 1, 0, 1:9] = True
        binary[0, 2, 0, 2:10] = True
        result = MaskGroupAuxiliaryLoss().build_mask_groups(mask_logits(binary), 64)
        self.assertEqual(result["selected_groups"], [[[0, 1], [2]]])

    def test_top_groups_are_selected_by_size_then_min_query_id(self):
        # IDs 1/4/5 form the largest group, IDs 0/3 form the next largest.
        binary = torch.zeros(1, 6, 3, 4, dtype=torch.bool)
        for query in (0, 3):
            binary[0, query, 0] = True
        for query in (1, 4, 5):
            binary[0, query, 1] = True
        binary[0, 2, 2] = True
        result = MaskGroupAuxiliaryLoss().build_mask_groups(mask_logits(binary), 2)
        self.assertEqual(result["selected_groups"], [[[1, 4, 5], [0, 3]]])
        self.assertEqual(result["num_groups"].tolist(), [3])
        self.assertEqual(result["num_selected_groups"].tolist(), [2])
        self.assertEqual(result["selected_query_count"].tolist(), [5])
        self.assertAlmostEqual(result["selected_query_fraction"].item(), 5 / 6, places=6)
        tied = MaskGroupAuxiliaryLoss().build_mask_groups(two_groups(), 1)
        self.assertEqual(tied["selected_groups"], [[[0, 1]]])

    def test_more_than_64_groups_truncate_without_dropping_queries(self):
        binary = torch.eye(70, dtype=torch.bool).reshape(1, 70, 7, 10)
        attention = torch.full((1, 64, 70), 1 / 70., requires_grad=True)
        result = MaskGroupAuxiliaryLoss(min_mask_area=1)(attention, mask_logits(binary))
        self.assertEqual(result["num_groups"].item(), 70)
        self.assertEqual(result["num_selected_groups"].item(), 64)
        self.assertEqual(result["selected_groups"][0], [[q] for q in range(64)])
        self.assertAlmostEqual(result["loss"].item(), math.log(70), places=5)
        result["loss"].backward()
        self.assertEqual(attention.shape, (1, 64, 70))
        self.assertTrue((attention.grad[:, :, 64:] > 0).all())

    def test_fewer_groups_do_not_assign_background_to_unused_latents(self):
        logits = two_groups()[:, :2]
        attention = torch.full((1, 64, 2), .5, requires_grad=True)
        result = MaskGroupAuxiliaryLoss()(attention, logits)
        self.assertEqual(result["num_selected_groups"].item(), 1)
        self.assertAlmostEqual(result["loss"].item(), 0.0, places=6)
        result["loss"].backward()
        self.assertEqual(torch.count_nonzero(attention.grad[0, 1:]).item(), 0)

    def test_excluded_queries_remain_in_probability_denominator(self):
        attention = torch.tensor([[[.25, .25, .25, .25]]], requires_grad=True)
        result = MaskGroupAuxiliaryLoss()(
            attention, two_groups(), query_valid_mask=torch.tensor([[True, True, False, False]])
        )
        self.assertEqual(result["selected_groups"], [[[0, 1]]])
        self.assertAlmostEqual(result["loss"].item(), math.log(2), places=6)

    def test_padding_only_and_tiny_masks_are_excluded(self):
        binary = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
        binary[0, 0, :2] = True  # Padding only.
        binary[0, 1, 2, 0] = True  # One valid pixel, too small.
        binary[0, 2, 2:] = True  # Eight valid pixels.
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        valid[:, :, 2:] = True
        result = MaskGroupAuxiliaryLoss().build_mask_groups(mask_logits(binary), 64, valid)
        self.assertEqual(result["selected_groups"], [[[2]]])
        self.assertEqual(result["num_valid_queries"].tolist(), [1])

    def test_validity_can_have_different_resolution(self):
        logits = torch.full((1, 2, 8, 8), 10.)
        valid = torch.tensor([[[False, False], [True, True]]])
        loss = MaskGroupAuxiliaryLoss(spatial_size=4)
        result = loss.build_mask_groups(logits, 64, valid)
        self.assertEqual(result["selected_groups"], [[[0, 1]]])

    def test_no_groups_returns_graph_connected_zero_without_mask_grad(self):
        attention = torch.full((2, 64, 3), 1 / 3., requires_grad=True)
        logits = torch.full((2, 3, 4, 4), -10., requires_grad=True)
        result = MaskGroupAuxiliaryLoss()(attention, logits)
        self.assertEqual(result["valid_sample_mask"].tolist(), [False, False])
        self.assertEqual(result["per_sample_loss"].tolist(), [0., 0.])
        result["per_sample_loss"].sum().backward()
        self.assertTrue(torch.equal(attention.grad, torch.zeros_like(attention)))
        self.assertIsNone(logits.grad)

    def test_batch_loss_averages_only_samples_with_groups(self):
        attention = torch.full((2, 2, 4), .25, requires_grad=True)
        logits = torch.cat((two_groups(), torch.full_like(two_groups(), -10.)))
        result = MaskGroupAuxiliaryLoss()(attention, logits)
        self.assertEqual(result["valid_sample_mask"].tolist(), [True, False])
        self.assertAlmostEqual(result["loss"].item(), math.log(2), places=6)
        self.assertAlmostEqual(result["per_sample_loss"][0].item(), math.log(2), places=6)
        self.assertEqual(result["per_sample_loss"][1].item(), 0.)

    def test_full_attention_rows_are_normalized_not_selected_subset(self):
        attention = torch.full((1, 2, 4), .125, requires_grad=True)
        result = MaskGroupAuxiliaryLoss()(attention, two_groups())
        self.assertAlmostEqual(result["loss"].item(), math.log(2), places=6)

    def test_cpu_autocast_keeps_group_costs_float32(self):
        attention = torch.tensor([[[.11, .13, .35, .41], [.29, .31, .19, .21]]], requires_grad=True)
        module = MaskGroupAuxiliaryLoss()
        expected = module(attention, two_groups())["loss"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module(attention, two_groups())["loss"]
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-7, rtol=0.))

    def test_empty_batch_and_zero_queries_are_supported(self):
        for attention, logits in (
            (torch.empty(0, 64, 4, requires_grad=True), torch.empty(0, 4, 8, 8)),
            (torch.empty(2, 64, 0, requires_grad=True), torch.empty(2, 0, 8, 8)),
        ):
            result = MaskGroupAuxiliaryLoss()(attention, logits)
            self.assertEqual(result["loss"].item(), 0.)
            result["loss"].backward()
            self.assertIsNotNone(attention.grad)

    def test_invalid_input_fails_loudly(self):
        module = MaskGroupAuxiliaryLoss()
        valid_attention = torch.full((1, 2, 4), .25)
        for replacement in (float("nan"), float("inf"), -1.):
            attention = valid_attention.clone()
            attention[0, 0, 0] = replacement
            with self.assertRaises(ValueError):
                module(attention, two_groups())
        with self.assertRaisesRegex(ValueError, "positive probability mass"):
            module(torch.zeros_like(valid_attention), two_groups())
        logits = two_groups()
        logits[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "mask_logits contains"):
            module(valid_attention, logits)
        with self.assertRaisesRegex(ValueError, "boolean dtype"):
            module(valid_attention, two_groups(), query_valid_mask=torch.ones(1, 4))
        with self.assertRaisesRegex(ValueError, "batch/query dimensions"):
            module(valid_attention[:, :, :3], two_groups())
        with self.assertRaisesRegex(ValueError, "zero/one"):
            module(valid_attention, two_groups(), spatial_valid_mask=torch.full((1, 1, 2, 4), .5))


if __name__ == "__main__":
    unittest.main()
