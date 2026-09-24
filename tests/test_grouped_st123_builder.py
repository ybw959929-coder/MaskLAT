from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1] / "masklat" / "model" / "segmentors"
PACKAGE = "_masklat_grouped_st123_test"


def _load_builder():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "mask2former")]
    sys.modules[PACKAGE] = package
    names = (
        "spatial_validity",
        "mask_topology_grouping",
        "topology_group_decoder",
        "st3_group_vlm_refiner",
        "st3_proposal_latent_bridge",
        "st3_bipartite_latent_transport",
        "st123_latent_cascade",
        "latent_mask_group_loss",
        "group_supervised_latent_builder",
        "grouped_st123_builder",
    )
    loaded = {}
    for name in names:
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE}.{name}", ROOT / "mask2former" / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["grouped_st123_builder"].GroupedST123Builder


Builder = _load_builder()


def _inputs():
    batch = 2
    identity = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    size = torch.tensor([[4.0, 4.0]]).repeat(batch, 1)
    region = torch.tensor([[0.0, 0.0, 4.0, 4.0]]).repeat(batch, 1)
    return dict(
        query_states=tuple(
            torch.randn(batch, 7, 8, requires_grad=True) for _ in range(3)
        ),
        mask_logits=tuple(
            torch.randn(batch, 7, 4, 4, requires_grad=True) for _ in range(3)
        ),
        sam_mask_features=torch.randn(batch, 3, 4, 4, requires_grad=True),
        siglip_spatial_features=torch.randn(batch, 5, 2, 2, requires_grad=True),
        spatial_metadata=dict(
            original_size=size,
            original_to_sam=identity,
            sam_input_size=size,
            sam_valid_region=region,
            original_to_siglip=identity,
            siglip_input_size=size,
            siglip_valid_region=region,
            siglip_patch_size=torch.tensor([[2.0, 2.0]]).repeat(batch, 1),
        ),
    )


class GroupedST123BuilderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        self.builder = Builder(
            hidden_dim=8,
            sam_feature_dim=3,
            siglip_feature_dim=5,
            llm_hidden_dim=12,
            num_latents=64,
            num_heads=2,
            fourier_features=4,
            pre_vlm_depth=2,
            group_loss_config=dict(spatial_size=4, min_mask_area=1),
        )

    def test_recurrent_prefix_and_final_projection(self):
        calls = []
        outputs = []
        handles = []
        for stage in self.builder.stages.values():
            handles.append(
                stage.register_forward_pre_hook(
                    lambda module, args, kwargs: calls.append(kwargs.copy()),
                    with_kwargs=True,
                )
            )
            handles.append(
                stage.register_forward_hook(
                    lambda module, args, output: outputs.append(output)
                )
            )
        try:
            result = self.builder(**_inputs())
        finally:
            for handle in handles:
                handle.remove()

        self.assertIsNone(calls[0]["latent_seed"])
        self.assertIs(calls[1]["latent_seed"], outputs[0].latent_features)
        self.assertIs(calls[2]["latent_seed"], outputs[1].latent_features)
        self.assertIsInstance(self.builder.stages["st1"].vlm_projector, nn.Identity)
        self.assertIsInstance(self.builder.stages["st2"].vlm_projector, nn.Identity)
        self.assertNotIsInstance(self.builder.stages["st3"].vlm_projector, nn.Identity)
        self.assertEqual(set(result.stage_group_supervision), {1, 2, 3})
        self.assertIs(result.group_supervision, result.stage_group_supervision[3])

    def test_joint_gradient_reaches_all_prefix_stages(self):
        result = self.builder(**_inputs())
        loss = result.packed_group_tokens.square().mean()
        loss = loss + sum(
            result.stage_group_supervision[index]["loss"]
            for index in (1, 2, 3)
        )
        loss.backward()
        for stage in self.builder.stages.values():
            parameter = stage.pre_vlm_blocks[-1].cross_attention.in_proj_weight
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
