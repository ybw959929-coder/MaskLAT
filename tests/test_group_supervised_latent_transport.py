"""CPU contracts for the opt-in group-supervised transport architecture."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


_ROOT = Path(__file__).resolve().parents[1] / "masklat/model/segmentors/mask2former"
_PACKAGE = "_masklat_group_supervised_transport_test"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{_PACKAGE}.group_supervised_latent_transport",
    _ROOT / "group_supervised_latent_transport.py",
)
transport = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = transport
spec.loader.exec_module(transport)
legacy = sys.modules[f"{_PACKAGE}.st3_bipartite_latent_transport"]


class GroupSupervisedTransportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)

    def test_all_six_stages_delay_writeback_and_no_late_refresh(self):
        bridge = transport.GroupSupervisedLatentTransportBridge(8, 2)
        self.assertEqual(len(bridge.transport_stages), 6)
        self.assertTrue(all(s.enable_latent_writeback for s in bridge.transport_stages))
        self.assertTrue(all(s.writeback_after_prediction for s in bridge.transport_stages))
        self.assertEqual(len(bridge.post_vlm_blocks), 1)
        self.assertEqual(bridge.late_condition_refresh_stages, ())
        self.assertEqual(len(bridge.late_condition_refreshers), 0)
        self.assertIsNone(bridge.condition_refresher(6))
        self.assertIsNone(bridge.condition_refresher(9))

    def test_old_bridge_remains_last_stage_read_only(self):
        old = legacy.BipartiteLatentTransportBridge(8, 2)
        self.assertFalse(old.transport_stages[-1].enable_latent_writeback)
        self.assertFalse(hasattr(old.transport_stages[0], "writeback_after_prediction"))
        self.assertFalse(hasattr(old, "final_condition_reader"))

    def test_only_new_state_keys_are_st9_writeback_and_final_reader(self):
        old = legacy.BipartiteLatentTransportBridge(8, 2)
        new = transport.GroupSupervisedLatentTransportBridge(8, 2)
        result = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(
            key.startswith("transport_stages.5.") or
            key.startswith("final_condition_reader.")
            for key in result.missing_keys
        ))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)

    def test_invalid_experiment_combinations_fail_early(self):
        for kwargs in (
            {"late_condition_refresh_stages": (6, 9)},
            {"enable_latent_writeback": False},
            {"transport_depth": 5},
            {"post_vlm_depth": 2},
        ):
            with self.assertRaises(ValueError):
                transport.GroupSupervisedLatentTransportBridge(8, 2, **kwargs)

    def test_final_reader_preserves_bg_padding_and_has_no_cond_mixing(self):
        reader = transport.FinalConditionLatentRead(8, 2).eval()
        cond = torch.randn(2, 5, 8)
        latent = torch.randn(2, 4, 8)
        valid = torch.tensor([
            [True, False, True, False, True],
            [True, True, True, True, True],
        ])
        result = reader(cond, latent, valid)
        inactive = ~valid.clone()
        inactive[:, -1] = True
        self.assertTrue(torch.equal(result[inactive], cond[inactive]))
        self.assertFalse(torch.equal(result[~inactive], cond[~inactive]))
        changed = cond.clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 100
        changed_result = reader(changed, latent, valid)
        torch.testing.assert_close(result[:, 0], changed_result[:, 0], rtol=0, atol=0)
        changed_latent = latent + torch.randn_like(latent)
        self.assertFalse(torch.equal(reader(cond, changed_latent, valid)[:, 0], result[:, 0]))

    def test_final_reader_queries_cond_and_uses_latent_as_memory(self):
        reader = transport.FinalConditionLatentRead(8, 2)
        cond = torch.randn(2, 5, 8, requires_grad=True)
        latent = torch.randn(2, 7, 8, requires_grad=True)
        valid = torch.ones(2, 5, dtype=torch.bool)
        with patch.object(
            reader.cross_attention, "forward", wraps=reader.cross_attention.forward,
        ) as attention:
            output = reader(cond, latent, valid)
        arguments = attention.call_args.kwargs
        self.assertEqual(arguments["query"].shape, (2, 5, 8))
        self.assertEqual(arguments["key"].shape, (2, 7, 8))
        torch.testing.assert_close(arguments["key"], reader.latent_norm(latent))
        torch.testing.assert_close(arguments["value"], reader.latent_norm(latent))
        output[:, :-1].square().sum().backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0)
        self.assertGreater(float(cond.grad[:, :-1].abs().sum()), 0)
        for name, parameter in reader.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_bg_only_has_identity_output_and_zero_but_present_parameter_grads(self):
        reader = transport.FinalConditionLatentRead(8, 2)
        cond = torch.randn(2, 1, 8, requires_grad=True)
        latent = torch.randn(2, 4, 8, requires_grad=True)
        output = reader(cond, latent, torch.ones(2, 1, dtype=torch.bool))
        self.assertTrue(torch.equal(output, cond))
        output.square().sum().backward()
        for name, parameter in reader.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertEqual(float(parameter.grad.abs().sum()), 0, name)

    def test_writeback_uses_original_relation_and_final_query_values(self):
        stage = transport.GroupSupervisedLatentTransportStage(8, 2)
        query_early = torch.randn(2, 7, 8, requires_grad=True)
        query_final = torch.randn(2, 7, 8, requires_grad=True)
        latent = torch.randn(2, 4, 8, requires_grad=True)
        _, relation = stage.update_queries(query_early, latent)
        relation.retain_grad()
        observed = []
        handle = stage.latent_output.register_forward_pre_hook(
            lambda module, args: observed.append(args[0])
        )
        with patch.object(stage.query_read, "relation_logits", side_effect=AssertionError("R recomputed")):
            output = stage.update_latents(query_final, latent, relation)
        handle.remove()
        weights = relation.softmax(dim=1).transpose(1, 2)
        values = stage.query_value(stage.query_value_norm(query_final))
        torch.testing.assert_close(observed[0], weights @ values)
        output.square().sum().backward()
        self.assertGreater(float(relation.grad.abs().sum()), 0)
        self.assertGreater(float(query_early.grad.abs().sum()), 0)
        self.assertGreater(float(query_final.grad.abs().sum()), 0)

    def test_final_classification_gradient_reaches_st9_writeback(self):
        bridge = transport.GroupSupervisedLatentTransportBridge(8, 2)
        query = torch.randn(2, 7, 8, requires_grad=True)
        latent = torch.randn(2, 4, 8, requires_grad=True)
        cond = torch.randn(2, 5, 8, requires_grad=True)
        valid = torch.ones(2, 5, dtype=torch.bool)
        stage9 = bridge.transport_stages[-1]
        query_read, relation = stage9.update_queries(query, latent)
        final_query = torch.tanh(query_read)
        final_latent = stage9.update_latents(final_query, latent, relation)
        final_cond = bridge.readout_conditions(cond, final_latent, valid)
        classification = torch.einsum("bqd,bcd->bqc", final_query, final_cond)
        classification[:, :, :-1].square().mean().backward()
        for name, parameter in stage9.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)


if __name__ == "__main__":
    unittest.main()
