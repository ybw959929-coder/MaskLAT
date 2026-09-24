"""CPU contracts for st1--st3 recurrence with only final Z3 entering the VLM."""

from __future__ import annotations

import ast
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1] / "masklat" / "model" / "segmentors"
PACKAGE = "_masklat_st123_cascade_test"


def _load_modules():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "mask2former")]
    sys.modules[PACKAGE] = package
    loaded = {}
    for name in (
        "spatial_validity", "mask_topology_grouping", "topology_group_decoder",
        "st3_group_vlm_refiner", "st3_proposal_latent_bridge",
        "st3_bipartite_latent_transport", "st123_latent_deepstack",
        "st123_latent_cascade",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE}.{name}", ROOT / "mask2former" / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


MODULES = _load_modules()
Builder = MODULES["st123_latent_cascade"].St123LatentCascadeBuilder
Baseline = MODULES["st3_bipartite_latent_transport"].St3BipartiteLatentBuilder
Deepstack = MODULES["st123_latent_deepstack"].St123LatentDeepstackBuilder
Bridge = MODULES["st3_bipartite_latent_transport"].BipartiteLatentTransportBridge
Proposal = MODULES["st3_proposal_latent_bridge"].St3LatentProposal


def _kwargs():
    return dict(
        hidden_dim=8, sam_feature_dim=3, siglip_feature_dim=5,
        llm_hidden_dim=12, num_latents=64, num_heads=2,
        fourier_features=4, pre_vlm_depth=2,
    )


def _inputs(dtype=torch.float32, requires_grad=False):
    batch = 2
    identity = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    size = torch.tensor([[4.0, 4.0]]).repeat(batch, 1)
    region = torch.tensor([[0.0, 0.0, 4.0, 4.0]]).repeat(batch, 1)
    return dict(
        query_states=tuple(torch.randn(batch, 7, 8, dtype=dtype, requires_grad=requires_grad) for _ in range(3)),
        mask_logits=tuple(torch.randn(batch, 7, 4, 4, dtype=dtype, requires_grad=requires_grad) for _ in range(3)),
        sam_mask_features=torch.randn(batch, 3, 4, 4, dtype=dtype, requires_grad=requires_grad),
        siglip_spatial_features=torch.randn(batch, 5, 2, 2, dtype=dtype, requires_grad=requires_grad),
        spatial_metadata=dict(
            original_size=size, original_to_sam=identity,
            sam_input_size=size, sam_valid_region=region,
            original_to_siglip=identity, siglip_input_size=size,
            siglip_valid_region=region,
            siglip_patch_size=torch.tensor([[2.0, 2.0]]).repeat(batch, 1),
        ),
    )


class _TinyDecoder(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.reference = nn.Parameter(torch.zeros(1, dtype=dtype))


def _segmentor(dtype=torch.float32, **overrides):
    config = types.SimpleNamespace(
        use_st3_bipartite_latent_transport=True,
        st3_transport_require_expected_shapes=False,
        hidden_dim=8, mask_feature_size=3, st3_latent_num_heads=2,
        st3_latent_num_tokens=64, st3_latent_mask_threshold=0.5,
        st3_latent_fourier_features=4, st3_transport_pre_vlm_depth=2,
        st3_transport_post_vlm_depth=1, st3_transport_decoder_depth=6,
        st3_transport_output_init_std=1e-3,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return types.SimpleNamespace(
        dec_config=config,
        decoder=_TinyDecoder(config, dtype),
        st3_transport_proposal_builder=None,
        st3_transport_bridge=None,
    )


def _method(name):
    """Run real integration methods with only heavyweight decoder types stubbed."""
    tree = ast.parse((ROOT / "masklat_segmentor.py").read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), method], type_ignores=[])
    namespace = {
        "torch": torch,
        "St3LatentExecutionBundle": types.SimpleNamespace,
        "Mask2FormerTransformerModule": _TinyDecoder,
        "St3BipartiteLatentBuilder": Baseline,
        "St123LatentDeepstackBuilder": Deepstack,
        "St123LatentCascadeBuilder": Builder,
        "BipartiteLatentTransportBridge": Bridge,
    }
    exec(compile(ast.fix_missing_locations(module), "masklat_segmentor.py", "exec"), namespace)
    return namespace[name]


CONFIGURE = _method("configure_st3_bipartite_latent_transport")
PREPARE = _method("prepare_st3_latent_execution")


class St123LatentCascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(2903)

    def test_independent_stages_have_one_seed_and_only_final_projection(self):
        builder = Builder(**_kwargs())
        state = builder.state_dict()
        self.assertEqual(tuple(builder.stages), ("st1", "st2", "st3"))
        self.assertEqual([key for key in state if key.endswith("learned_latents")], ["stages.st1.learned_latents"])
        self.assertIsNone(builder.stages["st2"].learned_latents)
        self.assertIsNone(builder.stages["st3"].learned_latents)
        self.assertFalse(any("latent_query" in key or "proposal_key" in key for key in state))
        self.assertIsInstance(builder.stages["st1"].vlm_projector, nn.Identity)
        self.assertIsInstance(builder.stages["st2"].vlm_projector, nn.Identity)
        projection_keys = [key for key in state if ".vlm_projector." in key]
        self.assertTrue(projection_keys)
        self.assertTrue(all(key.startswith("stages.st3.") for key in projection_keys))
        self.assertEqual(builder.llm_hidden_dim, 12)
        self.assertIs(builder.siglip_projector, builder.stages["st1"].siglip_projector)
        self.assertFalse(any(key.startswith("siglip_projector.") for key in state))
        for name in ("sam_projector", "siglip_projector", "geometry_encoder", "pre_vlm_blocks"):
            self.assertEqual(len({id(getattr(stage, name)) for stage in builder.stages.values()}), 3)

    def test_recurrence_passes_exact_latent_objects_and_returns_only_final_proposal(self):
        builder, inputs = Builder(**_kwargs()), _inputs()
        calls, outputs, handles = [], [], []
        for stage in builder.stages.values():
            handles.append(stage.register_forward_pre_hook(
                lambda module, args, kwargs: calls.append(kwargs.copy()), with_kwargs=True
            ))
            handles.append(stage.register_forward_hook(lambda module, args, output: outputs.append(output)))
        try:
            result = builder(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertIsNone(calls[0]["latent_seed"])
        self.assertIs(calls[1]["latent_seed"], outputs[0].latent_features)
        self.assertIs(calls[2]["latent_seed"], outputs[1].latent_features)
        self.assertTrue(all(call["compute_affinity"] is False for call in calls))
        self.assertIs(result, outputs[2])
        self.assertIs(type(result), Proposal)
        self.assertFalse(hasattr(result, "stage_vlm_tokens"))
        self.assertFalse(hasattr(result, "stage_latent_features"))
        self.assertEqual(result.packed_group_tokens.shape, (2, 64, 12))
        self.assertEqual(result.latent_features.shape, (2, 64, 8))
        self.assertEqual(result.packed_group_valid_mask.shape, (2, 64))
        self.assertTrue(result.packed_group_valid_mask.all())
        torch.testing.assert_close(result.packed_group_tokens, builder.stages["st3"].vlm_projector(result.latent_features))

    def test_final_only_loss_reaches_every_trainable_stage_parameter_and_sources(self):
        builder, inputs = Builder(**_kwargs()), _inputs(requires_grad=True)
        builder.requires_grad_(True)
        builder.freeze_unused_affinity_heads()
        result = builder(**inputs)
        (result.packed_group_tokens * torch.randn_like(result.packed_group_tokens)).sum().backward()
        for name, parameter in builder.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for stage in builder.stages.values():
            self.assertGreater(stage.pre_vlm_blocks[0].cross_attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(builder.stages["st1"].learned_latents.grad.abs().sum().item(), 0)
        for source in (*inputs["query_states"], inputs["sam_mask_features"], inputs["siglip_spatial_features"]):
            self.assertIsNotNone(source.grad)
            self.assertGreater(source.grad.abs().sum().item(), 0)

    def test_manual_three_stage_computation_matches_final_fields(self):
        builder, inputs = Builder(**_kwargs()), _inputs()
        result = builder(**inputs)
        common = {key: value for key, value in inputs.items() if key not in ("query_states", "mask_logits")}
        seed = None
        for index in range(3):
            manual = builder.stages[f"st{index + 1}"](
                **common, query_states=inputs["query_states"][index],
                mask_logits=inputs["mask_logits"][index],
                latent_seed=seed, compute_affinity=False,
            )
            seed = manual.latent_features
        for key, value in vars(result).items():
            torch.testing.assert_close(value, getattr(manual, key), rtol=0, atol=0)

    def test_bf16_forward_has_only_64_final_tokens(self):
        builder = Builder(**_kwargs()).to(torch.bfloat16)
        with torch.no_grad():
            result = builder(**_inputs(torch.bfloat16))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in builder.parameters()))
        self.assertEqual(result.packed_group_tokens.shape, (2, 64, 12))
        self.assertEqual(result.packed_group_tokens.dtype, torch.bfloat16)
        self.assertEqual(result.latent_features.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result.packed_group_tokens).all())
        self.assertTrue(torch.isfinite(result.latent_features).all())

    def test_strict_checkpoint_roundtrip_preserves_all_registered_weights(self):
        builder = Builder(**_kwargs())
        buffer = io.BytesIO()
        torch.save(builder.state_dict(), buffer)
        buffer.seek(0)
        restored = Builder(**_kwargs())
        report = restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(report.missing_keys, [])
        self.assertEqual(report.unexpected_keys, [])
        inputs = _inputs()
        before, after = builder(**inputs), restored(**inputs)
        torch.testing.assert_close(before.packed_group_tokens, after.packed_group_tokens, rtol=0, atol=0)
        torch.testing.assert_close(before.latent_features, after.latent_features, rtol=0, atol=0)

    def test_nonreentrant_checkpoint_preserves_shared_prefix_graph(self):
        builder, inputs = Builder(**_kwargs()), _inputs(requires_grad=True)
        raw_queries = inputs["query_states"][0]
        q1 = checkpoint(lambda value: value.sin(), raw_queries, use_reentrant=False)
        q2 = checkpoint(lambda value: value.tanh(), q1, use_reentrant=False)
        q3 = checkpoint(lambda value: value.cos(), q2, use_reentrant=False)
        result = builder(**dict(inputs, query_states=(q1, q2, q3)))
        result.packed_group_tokens.square().mean().backward()
        self.assertGreater(raw_queries.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in builder.parameters()))

    def test_stage_sequence_validation_and_no_extra_seed(self):
        builder, inputs = Builder(**_kwargs()), _inputs()
        for key in ("query_states", "mask_logits"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                builder(**dict(inputs, **{key: inputs[key][0]}))
            with self.subTest(short=key), self.assertRaisesRegex(ValueError, "exactly formal"):
                builder(**dict(inputs, **{key: inputs[key][:2]}))
        common = {key: value for key, value in inputs.items() if key not in ("query_states", "mask_logits")}
        with self.assertRaisesRegex(ValueError, "requires a latent_seed"):
            builder.stages["st2"](**common, query_states=inputs["query_states"][1], mask_logits=inputs["mask_logits"][1])

    def test_baseline_and_deepstack_behavior_remain_distinct_and_unchanged(self):
        inputs = _inputs()
        baseline = Baseline(**_kwargs())
        baseline_result = baseline(**dict(inputs, query_states=inputs["query_states"][2], mask_logits=inputs["mask_logits"][2]))
        self.assertIs(type(baseline_result), Proposal)
        self.assertIn("learned_latents", baseline.state_dict())
        self.assertTrue(any(key.startswith("vlm_projector.") for key in baseline.state_dict()))
        deepstack = Deepstack(**_kwargs())
        result = deepstack(**inputs)
        self.assertEqual(len(result.stage_vlm_tokens), 3)
        self.assertIs(result.packed_group_tokens, result.stage_vlm_tokens[0])
        self.assertTrue(all(any(key.startswith(f"stages.st{index}.vlm_projector.") for key in deepstack.state_dict()) for index in (1, 2, 3)))

    def test_real_configure_cascade_and_original_bridge_dtype(self):
        segmentor = _segmentor(torch.bfloat16, use_st123_latent_cascade=True)
        CONFIGURE(segmentor, siglip_hidden_dim=5, llm_hidden_dim=12)
        builder, bridge = segmentor.st3_transport_proposal_builder, segmentor.st3_transport_bridge
        self.assertIsInstance(builder, Builder)
        self.assertEqual(builder.llm_hidden_dim, 12)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in builder.parameters()))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in bridge.parameters()))
        self.assertEqual([stage.enable_latent_writeback for stage in bridge.transport_stages], [True] * 5 + [False])
        CONFIGURE(segmentor, siglip_hidden_dim=5, llm_hidden_dim=12)
        self.assertIs(segmentor.st3_transport_proposal_builder, builder)

    def test_disabled_cascade_preserves_constructor_rng_state_keys_and_values(self):
        for deepstack in (False, True):
            with self.subTest(deepstack=deepstack):
                config_kwargs = {"use_st123_latent_deepstack": deepstack}
                implicit = _segmentor(**config_kwargs)
                explicit = _segmentor(use_st123_latent_cascade=False, **config_kwargs)
                torch.manual_seed(77)
                CONFIGURE(implicit, siglip_hidden_dim=5, llm_hidden_dim=12)
                implicit_rng = torch.random.get_rng_state().clone()
                torch.manual_seed(77)
                CONFIGURE(explicit, siglip_hidden_dim=5, llm_hidden_dim=12)
                torch.testing.assert_close(torch.random.get_rng_state(), implicit_rng, rtol=0, atol=0)
                expected_type = Deepstack if deepstack else Baseline
                self.assertIs(type(implicit.st3_transport_proposal_builder), expected_type)
                for module_name in ("st3_transport_proposal_builder", "st3_transport_bridge"):
                    a = getattr(implicit, module_name).state_dict()
                    b = getattr(explicit, module_name).state_dict()
                    self.assertEqual(list(a), list(b))
                    for key, value in a.items():
                        torch.testing.assert_close(value, b[key], rtol=0, atol=0)

    def test_real_configure_rejects_wrong_flag_combinations_and_mode_changes(self):
        for invalid in (None, 0, 1, "false"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "must be a bool"):
                CONFIGURE(_segmentor(use_st123_latent_cascade=invalid), siglip_hidden_dim=5, llm_hidden_dim=12)
        for options, error in (
            (dict(use_st123_latent_cascade=True, use_st123_latent_deepstack=True), "mutually exclusive"),
            (dict(use_st123_latent_cascade=True, use_st3_bipartite_latent_transport=False), "requires bipartite"),
            (dict(use_st123_latent_cascade=True, st3_transport_late_condition_refresh_stages=(6, 9)), "late refresh"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, error):
                CONFIGURE(_segmentor(**options), siglip_hidden_dim=5, llm_hidden_dim=12)
        for first, second in (("baseline", "cascade"), ("cascade", "baseline"), ("deepstack", "cascade"), ("cascade", "deepstack")):
            segmentor = _segmentor(use_st123_latent_cascade=first == "cascade", use_st123_latent_deepstack=first == "deepstack")
            CONFIGURE(segmentor, siglip_hidden_dim=5, llm_hidden_dim=12)
            segmentor.dec_config.use_st123_latent_cascade = second == "cascade"
            segmentor.dec_config.use_st123_latent_deepstack = second == "deepstack"
            with self.subTest(first=first, second=second), self.assertRaisesRegex(ValueError, "cannot change"):
                CONFIGURE(segmentor, siglip_hidden_dim=5, llm_hidden_dim=12)

    def test_real_prepare_selects_st1_st2_st3_keeps_full_source_gradients_and_only_z3(self):
        inputs = _inputs(requires_grad=True)
        modes, captured = [], {}
        builder = Builder(**_kwargs())
        state = types.SimpleNamespace(
            intermediate_hidden_states=(torch.zeros_like(inputs["query_states"][0]).transpose(0, 1),)
            + tuple(q.transpose(0, 1) for q in inputs["query_states"]),
            masks_queries_logits=(torch.zeros_like(inputs["mask_logits"][0]),) + inputs["mask_logits"],
            normalized_query_states=inputs["query_states"][-1].transpose(0, 1),
            mask_logits=inputs["mask_logits"][-1],
        )

        def pixel_decoder(*args, **kwargs):
            modes.append(torch.is_grad_enabled())
            return types.SimpleNamespace(multi_scale_features=(), mask_features=None)

        def prefix(*args, **kwargs):
            modes.append(torch.is_grad_enabled())
            return state

        def record_builder(**kwargs):
            captured.update(kwargs)
            return builder(**kwargs)

        segmentor = types.SimpleNamespace(
            dec_config=types.SimpleNamespace(use_st3_bipartite_latent_transport=True, use_st123_latent_cascade=True, st3_latent_num_tokens=64),
            st3_transport_proposal_builder=record_builder, st3_transport_bridge=object(),
            pixel_decoder=pixel_decoder,
            decoder=types.SimpleNamespace(prepare_decoder_context=lambda *a, **k: object(), decoder=types.SimpleNamespace(forward_to_stage3=prefix)),
        )
        bundle = PREPARE(
            segmentor, image_embeddings=(), sam_spatial_features=inputs["sam_mask_features"],
            siglip_spatial_features=inputs["siglip_spatial_features"], siglip_valid_mask=None,
            spatial_metadata=inputs["spatial_metadata"], freeze_proposal_source=False,
        )
        self.assertEqual(modes, [True, True])
        self.assertIs(bundle.stage3_state, state)
        self.assertFalse(hasattr(bundle.proposal, "stage_vlm_tokens"))
        self.assertEqual(bundle.proposal.packed_group_tokens.shape, (2, 64, 12))
        for index in range(3):
            torch.testing.assert_close(captured["query_states"][index], inputs["query_states"][index], rtol=0, atol=0)
            self.assertIs(captured["mask_logits"][index], inputs["mask_logits"][index])
        bundle.proposal.packed_group_tokens.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in builder.parameters()))
        self.assertTrue(all(q.grad is not None for q in inputs["query_states"]))

    def test_real_prepare_rejects_invalid_modes_before_source_execution(self):
        inputs = _inputs()
        for options, error in (
            (dict(use_st123_latent_cascade="true"), TypeError),
            (dict(use_st123_latent_cascade=True, use_st123_latent_deepstack=True), ValueError),
            (dict(use_st123_latent_cascade=True, use_st3_bipartite_latent_transport=False), ValueError),
        ):
            segmentor = _segmentor(**options)
            with self.subTest(options=options), self.assertRaises(error):
                PREPARE(segmentor, image_embeddings=(), sam_spatial_features=inputs["sam_mask_features"], siglip_spatial_features=inputs["siglip_spatial_features"], siglip_valid_mask=None, spatial_metadata=inputs["spatial_metadata"])


if __name__ == "__main__":
    unittest.main()
