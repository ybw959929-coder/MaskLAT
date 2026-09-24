"""Execute the new model class against tiny stand-ins for heavy runtimes.

The production class is compiled unchanged from its AST; only imports/base
training machinery are substituted.  These test loss accounting and lifecycle,
not Phi/SAM correctness or a multi-GPU training run.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
import math
from pathlib import Path
import unittest

import torch
from torch import nn


class FakeMessageHub:
    iteration = None
    missing = False

    @classmethod
    def get_current_instance(cls):
        return cls

    @classmethod
    def get_info(cls, key):
        if cls.missing:
            raise KeyError(key)
        return cls.iteration


class FakeSegmentor(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self._group_supervision_records = None


class FakeBase(nn.Module):
    def __init__(self, *, segmentor, **kwargs):
        super().__init__()
        config = dict(segmentor)
        self.segmentor = config.pop("type")(**config)
        self.st3_latent_alignment_pretrain = kwargs.get("st3_latent_alignment_pretrain", False)
        self.base_loss = nn.Parameter(torch.tensor(2.))
        self.compute_calls = 0

    def _adapt_s2_latent_token_count(self, checkpoint_state):
        return dict(checkpoint_state)

    def forward(self, data_dict, data_samples=None, mode="loss", **kwargs):
        if data_dict.get("fail_forward"):
            raise RuntimeError("injected base failure")
        if mode != "loss":
            return {"prediction": 7}
        return self.compute_loss(data_dict, data_samples, **kwargs)

    def compute_loss(self, data_dict, data_samples=None, **kwargs):
        self.compute_calls += 1
        if self.segmentor._group_supervision_records is not None:
            self.segmentor._group_supervision_records.update(data_dict.get("records", {}))
        return {"loss": self.base_loss * 1., "loss_mask": self.base_loss * 3.}

    def parse_losses(self, losses):
        # MMEngine BaseModel semantics: reduce each value, then sum EVERY
        # reduced entry whose key contains 'loss'.  Existing MaskLAT may expose
        # both its declared loss and components; changing this here would
        # change the baseline objective rather than merely adding an auxiliary.
        logs = OrderedDict()
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                logs[name] = value.mean()
            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                logs[name] = sum(item.mean() for item in value)
            else:
                raise TypeError(f"{name} must be a tensor or list of tensors")
        total = sum(value for name, value in logs.items() if "loss" in name)
        logs["loss"] = total
        return total, logs


def load_model_class():
    path = Path(__file__).resolve().parents[1] / "masklat/model/group_supervised_latent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GroupSupervisedLatentMaskLATModel")
    namespace = dict(
        MaskLATModel=FakeBase, GroupSupervisedLatentSegmentor=FakeSegmentor,
        torch=torch, MessageHub=FakeMessageHub, deepcopy=deepcopy,
        math=math, OrderedDict=OrderedDict,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[definition.name]


Model = load_model_class()


def record(value, active=True):
    value = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), requires_grad=True)
    return dict(
        loss=value,
        valid_sample_mask=torch.tensor([active]),
        num_groups=torch.tensor([2 if active else 0]),
        num_selected_groups=torch.tensor([2 if active else 0]),
    )


def make_model(s2=False, **kwargs):
    return Model(segmentor={"sentinel": 19}, st3_latent_alignment_pretrain=s2, **kwargs)


class GroupModelContractTest(unittest.TestCase):
    def setUp(self):
        FakeMessageHub.iteration = None
        FakeMessageHub.missing = False

    def test_new_config_is_copied_and_legacy_dict_not_mutated(self):
        original = {"type": "OriginalSegmentor", "sentinel": [1, 2]}
        model = Model(segmentor=original)
        self.assertEqual(original, {"type": "OriginalSegmentor", "sentinel": [1, 2]})
        self.assertIsInstance(model.segmentor, FakeSegmentor)
        self.assertEqual(model.segmentor.config["group_loss_config"], dict(mask_threshold=.5, group_iou_threshold=.7, spatial_size=64, min_mask_area=4))

    def test_s2_auxiliary_is_added_once_and_logs_are_detached(self):
        model = make_model(s2=True, group_loss_weight=.1, group_loss_warmup_steps=0)
        auxiliary = torch.tensor(3., requires_grad=True)
        result = model({"records": {3: record(auxiliary)}})
        self.assertEqual(model.compute_calls, 1)
        self.assertAlmostEqual(result["loss"].item(), 2.3, places=6)
        self.assertEqual(result["aux_group_active_stages"].item(), 1)
        for name in ("aux_group_raw", "aux_group_weighted", "grp_st3_kl", "grp_st3_groups"):
            self.assertFalse(result[name].requires_grad)
        result["loss"].backward()
        self.assertAlmostEqual(auxiliary.grad.item(), .1, places=6)
        self.assertEqual(model.base_loss.grad.item(), 1.)
        self.assertIsNone(model.segmentor._group_supervision_records)

    def test_diagnostic_switch_disconnects_only_auxiliary_backward(self):
        model = make_model(
            s2=True,
            group_loss_weight=.1,
            group_loss_warmup_steps=0,
            group_loss_backprop_enabled=False,
        )
        auxiliary = torch.tensor(3., requires_grad=True)
        result = model({"records": {3: record(auxiliary)}})
        self.assertEqual(result["loss"].item(), 2.)
        self.assertEqual(result["aux_group_raw"].item(), 3.)
        self.assertAlmostEqual(result["aux_group_weight"].item(), .1, places=6)
        self.assertEqual(result["aux_group_weighted"].item(), 0.)
        self.assertEqual(result["aux_group_backprop_enabled"].item(), 0.)
        result["loss"].backward()
        self.assertIsNone(auxiliary.grad)
        self.assertEqual(model.base_loss.grad.item(), 1.)

    def test_auxiliary_backward_switch_requires_boolean(self):
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            make_model(group_loss_backprop_enabled=0)

    def test_s3_normalizes_by_active_stages_not_raw_sum(self):
        model = make_model(group_loss_weight=.1, group_loss_warmup_steps=0)
        first, second = torch.tensor(3., requires_grad=True), torch.tensor(5., requires_grad=True)
        records = {stage: record(torch.tensor(0., requires_grad=True) * 0., active=False) for stage in range(3, 10)}
        records[3], records[9] = record(first), record(second)
        result = model({"records": records})
        self.assertAlmostEqual(result["loss"].item(), 2.4, places=6)
        self.assertEqual(result["aux_group_raw"].item(), 4.)
        self.assertEqual(result["aux_group_active_stages"].item(), 2)
        result["loss"].backward()
        self.assertAlmostEqual(first.grad.item(), .05, places=6)
        self.assertAlmostEqual(second.grad.item(), .05, places=6)

    def test_all_empty_stage_losses_remain_graph_connected_zero(self):
        model = make_model(group_loss_warmup_steps=0)
        sources = [torch.tensor(1., requires_grad=True) for _ in range(7)]
        records = {stage: record(source * 0., active=False) for stage, source in zip(range(3, 10), sources)}
        result = model({"records": records})
        self.assertEqual(result["loss"].item(), 2.)
        self.assertEqual(result["aux_group_active_stages"].item(), 0)
        result["loss"].backward()
        self.assertTrue(all(source.grad is not None and source.grad.item() == 0. for source in sources))

    def test_imgconv_without_decoder_records_keeps_original_loss(self):
        model = make_model(group_loss_warmup_steps=0)
        result = model({"task": "ImgConv", "records": {}})
        self.assertEqual(result["loss"].item(), 2.)
        self.assertEqual(result["aux_group_weighted"].item(), 0.)
        self.assertEqual(result["aux_group_active_stages"].item(), 0)
        self.assertIsNone(model.segmentor._group_supervision_records)

    def test_s2_requires_exactly_st3_and_s3_rejects_incomplete_segmentation(self):
        cases = (
            (make_model(s2=True), {}),
            (make_model(s2=True), {3: record(1), 4: record(1)}),
            (make_model(), {3: record(1)}),
        )
        for model, records in cases:
            with self.subTest(s2=model.st3_latent_alignment_pretrain, stages=list(records)):
                with self.assertRaisesRegex(RuntimeError, "S2 must|segmentation S3 must"):
                    model({"records": records})
                self.assertIsNone(model.segmentor._group_supervision_records)

    def test_forward_exception_and_prediction_do_not_leave_graph_scope(self):
        model = make_model()
        with self.assertRaisesRegex(RuntimeError, "injected base"):
            model({"fail_forward": True})
        self.assertIsNone(model.segmentor._group_supervision_records)
        self.assertEqual(model({}, mode="predict"), {"prediction": 7})
        self.assertIsNone(model.segmentor._group_supervision_records)
        model.segmentor._group_supervision_records = {}
        with self.assertRaisesRegex(RuntimeError, "nested"):
            model({})

    def test_direct_compute_loss_without_scope_is_rejected(self):
        model = make_model(s2=True)
        with self.assertRaisesRegex(RuntimeError, "call forward"):
            model.compute_loss({"records": {3: record(1)}})

    def test_warmup_uses_runner_iteration_including_resumed_step(self):
        model = make_model(s2=True, group_loss_weight=.1, group_loss_warmup_steps=500)
        for iteration, expected in ((0, .0002), (249, .05), (499, .1), (8500, .1)):
            FakeMessageHub.iteration = iteration
            self.assertAlmostEqual(model._current_group_weight(), expected, places=9)
        self.assertEqual(model._group_fallback_iteration, 0)

    def test_warmup_fallback_advances_only_compute_calls(self):
        model = make_model(s2=True, group_loss_weight=.1, group_loss_warmup_steps=2)
        FakeMessageHub.missing = True
        self.assertAlmostEqual(model._current_group_weight(), .05)
        first = model({"records": {3: record(1)}})
        second = model({"records": {3: record(1)}})
        self.assertAlmostEqual(first["aux_group_weight"].item(), .05)
        self.assertAlmostEqual(second["aux_group_weight"].item(), .1)
        self.assertEqual(model._group_fallback_iteration, 2)

    def test_inherited_parse_preserves_baseline_objective_and_adds_auxiliary_once(self):
        model = make_model(s2=True, group_loss_weight=.1, group_loss_warmup_steps=0)
        self.assertIs(Model.parse_losses, FakeBase.parse_losses)
        baseline_result = FakeBase.compute_loss(model, {})
        baseline, _ = model.parse_losses(baseline_result)
        self.assertEqual(baseline.item(), 8.)  # 2 total + 6 existing mask term.
        baseline_gradient = torch.autograd.grad(baseline, model.base_loss)[0]
        auxiliary = torch.tensor(3., requires_grad=True)
        result = model({"records": {3: record(auxiliary)}})
        optimized, logs = model.parse_losses(result)
        self.assertAlmostEqual(optimized.item() - baseline.item(), .3, places=5)
        self.assertAlmostEqual(optimized.item() - baseline.item(), result["aux_group_weighted"].item(), places=5)
        self.assertEqual(logs["loss_mask"].item(), 6.)
        self.assertTrue(all("loss" not in name for name in result if name.startswith(("aux_group", "grp_st"))))
        optimized.backward()
        self.assertEqual(model.base_loss.grad.item(), baseline_gradient.item())
        self.assertEqual(model.base_loss.grad.item(), 4.)
        self.assertAlmostEqual(auxiliary.grad.item(), .1, places=6)

        # The inherited parser still includes every original component, even
        # list-valued components, rather than treating them as log-only values.
        total = torch.tensor([2., 4.], requires_grad=True)
        component = torch.tensor(100., requires_grad=True)
        reduced, logs = model.parse_losses({"loss": total, "loss_mask": component, "loss_dice": [component, component]})
        self.assertEqual(reduced.item(), 303.)
        self.assertEqual(logs["loss_dice"].item(), 200.)
        reduced.backward()
        torch.testing.assert_close(total.grad, torch.full((2,), .5))
        self.assertEqual(component.grad.item(), 3.)

    def test_old_s2_checkpoint_is_rejected_and_versioned_handoff_keeps_state(self):
        model = make_model()
        key = "segmentor.st3_transport_proposal_builder.group_supervision_version"
        for state in ({}, {key: torch.tensor(0)}, {key: torch.tensor(2)}):
            with self.assertRaisesRegex(ValueError, "grouped-builder parameters"):
                model._adapt_s2_latent_token_count(state)
        state = {key: torch.tensor(1), "other": torch.tensor(3.)}
        result = model._adapt_s2_latent_token_count(state)
        self.assertEqual(set(result), set(state))

    def test_legacy_s3_and_incompatible_freeze_policies_are_rejected(self):
        for kwargs in (
            {"latent_s3_pretrained_pth": "old.bin"},
            {"late_condition_refresh_finetune": True},
            {"latent_s2_post_vlm_decoder_finetune": True},
            {"group_loss_weight": -1.},
            {"group_loss_weight": float("nan")},
            {"group_loss_warmup_steps": -1},
            {"group_loss_warmup_steps": True},
            {"group_loss_warmup_steps": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_model(**kwargs)


if __name__ == "__main__":
    unittest.main()
