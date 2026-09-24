"""Read-only, full-model checkpoint validation for the latent spatial probe.

Do not use ``MaskLATModel.state_dict()`` here: its training export deliberately
filters frozen branches.  Likewise, check *before* ``load_state_dict`` because
some bridge load hooks initialize absent refreshers from another branch.
This module imports only PyTorch and the Python standard library.
"""

from collections.abc import Mapping

import torch
from torch import nn


_BUILDER_ROOT = "segmentor.st3_transport_proposal_builder"
_BUILDER_CLASS = (
    "masklat.model.segmentors.mask2former.st3_bipartite_latent_transport",
    "St3BipartiteLatentBuilder",
)
_CASCADE_CLASS = (
    "masklat.model.segmentors.mask2former.st123_latent_cascade",
    "St123LatentCascadeBuilder",
)


def _known_type(module, identity):
    cls = type(module)
    return (cls.__module__, cls.__name__) == identity


def _frozen_stateful(module):
    """A missing active module is not excused merely by a parent freeze flag."""
    if not isinstance(module, nn.Module):
        return False
    parameters = tuple(module.parameters())
    return bool(parameters) and all(not value.requires_grad for value in parameters)


def _inactive_prefixes(model):
    allowed = []
    segmentor = getattr(model, "segmentor", None)
    config = getattr(segmentor, "dec_config", None)
    bipartite = getattr(config, "use_st3_bipartite_latent_transport", None) is True
    # This probe only understands the known gate-free bipartite family.  Do
    # not transfer its inactive-head exceptions to a gated/other architecture.
    mutually_exclusive = (
        "use_st3_proposal_latent_bridge", "use_latent_s2_post_vlm_transport",
        "use_stagewise_latent_trifusion", "use_proposal_mediated_tripartite_transport",
        "use_topology_group_decoder", "use_st3_group_vlm_refiner",
        "use_st123_latent_deepstack",
    )
    if not bipartite or any(getattr(config, flag, False) for flag in mutually_exclusive):
        return ()

    if (getattr(model, "inject_sam_vit_tokens_to_vlm", None) is False
            and _frozen_stateful(getattr(model, "seg_projector", None))):
        allowed.append("seg_projector.")

    builder = getattr(segmentor, "st3_transport_proposal_builder", None)
    builders = []
    if (_known_type(builder, _BUILDER_CLASS)
            and not getattr(config, "use_st123_latent_cascade", False)):
        builders.append((_BUILDER_ROOT, builder))
    elif (_known_type(builder, _CASCADE_CLASS)
          and getattr(config, "use_st123_latent_cascade", None) is True):
        stages = getattr(builder, "stages", None)
        if isinstance(stages, nn.ModuleDict) and tuple(stages) == ("st1", "st2", "st3"):
            for name, stage in stages.items():
                if _known_type(stage, _BUILDER_CLASS):
                    builders.append((f"{_BUILDER_ROOT}.stages.{name}", stage))

    for path, builder in builders:
        for head_name in ("latent_query", "proposal_key"):
            if _frozen_stateful(getattr(builder, head_name, None)):
                allowed.append(f"{path}.{head_name}.")
    return tuple(allowed)


def _validated_contract(contract, model_keys):
    """Normalize the same topology contract used by the formal evaluator."""
    if contract is None:
        return (), (), True
    if not isinstance(contract, Mapping):
        raise TypeError("checkpoint_load_contract must be a mapping")

    def prefixes(name):
        value = tuple(contract.get(name, ()))
        if any(not isinstance(prefix, str) or not prefix for prefix in value):
            raise ValueError(f"{name} must contain nonempty string prefixes")
        unknown = [prefix for prefix in value if not any(
            key.startswith(prefix) for key in model_keys
        )]
        if unknown:
            raise ValueError(f"{name} contains prefixes absent from the model: {unknown}")
        return value

    required = prefixes("required_matched_prefixes")
    allowed = prefixes("allowed_missing_model_prefixes")
    reject_unexpected = contract.get("reject_unexpected_checkpoint_keys", True)
    if not isinstance(reject_unexpected, bool):
        raise TypeError("reject_unexpected_checkpoint_keys must be boolean")
    return required, allowed, reject_unexpected


def validate_full_checkpoint(model, state_dict, checkpoint_load_contract=None):
    """Check a full matching S3 state, without loading or changing any tensor.

    Floating checkpoint tensors may use a different floating dtype: native
    ``load_state_dict(assign=False)`` copies them into the established model
    dtype.  Persistent integer/bool buffers must have exactly matching dtype.
    Frozen active backbones remain required; this is not a delta loader.

    Returns a JSON-serializable report. Raises before any load hook can seed
    missing active weights, including late condition refreshers.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must be an unwrapped state-dict mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("checkpoint state-dict keys must be strings")
    current = nn.Module.state_dict(model)
    if not current or not state_dict:
        raise ValueError("full S3 validation requires nonempty model and checkpoint state")

    model_keys, checkpoint_keys = set(current), set(state_dict)
    required, configured_allowed, reject_unexpected = _validated_contract(
        checkpoint_load_contract, model_keys,
    )
    allowed = tuple(dict.fromkeys((*_inactive_prefixes(model), *configured_allowed)))
    missing = sorted(model_keys - checkpoint_keys)
    permitted_missing = [key for key in missing if key.startswith(allowed)]
    forbidden_missing = [key for key in missing if not key.startswith(allowed)]
    unexpected = sorted(checkpoint_keys - model_keys)
    malformed = []
    shape_mismatches = []
    dtype_mismatches = []
    matched = sorted(model_keys & checkpoint_keys)
    absent_required = [
        prefix for prefix in required
        if not any(key.startswith(prefix) for key in matched)
    ]
    for key in matched:
        target, source = current[key], state_dict[key]
        if (not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor)
                or target.is_meta or source.is_meta
                or target.layout != torch.strided or source.layout != torch.strided):
            malformed.append(key)
            continue
        if tuple(target.shape) != tuple(source.shape):
            shape_mismatches.append({
                "key": key, "model_shape": list(target.shape),
                "checkpoint_shape": list(source.shape),
            })
        compatible_dtype = (
            source.is_floating_point() if target.is_floating_point()
            else source.dtype == target.dtype
        )
        if not compatible_dtype:
            dtype_mismatches.append({
                "key": key, "model_dtype": str(target.dtype),
                "checkpoint_dtype": str(source.dtype),
            })

    rejected_unexpected = unexpected if reject_unexpected else []
    if (absent_required or forbidden_missing or rejected_unexpected or malformed
            or shape_mismatches or dtype_mismatches):
        raise RuntimeError(
            "Spatial probe requires a complete, topology-matched S3 checkpoint; "
            f"absent_required_prefixes={absent_required[:20]}; "
            f"missing_active_keys={forbidden_missing[:20]}; "
            f"unexpected_checkpoint_keys={rejected_unexpected[:20]}; "
            f"invalid_tensor_keys={malformed[:20]}; "
            f"shape_mismatches={shape_mismatches[:10]}; "
            f"dtype_mismatches={dtype_mismatches[:10]}. "
            "Validation performed before load_state_dict; no missing active "
            "weights or refreshers were initialized from a fallback."
        )

    return {
        "validated": True,
        "validation_schema": "full_s3_spatial_probe_v1",
        "state_source": "torch.nn.Module.state_dict(model)",
        "model_key_count": len(current),
        "checkpoint_key_count": len(state_dict),
        "matched_key_count": len(matched),
        "matched_keys": matched,
        "required_matched_prefixes": list(required),
        "allowed_missing_prefixes": list(allowed),
        "allowed_missing_keys": permitted_missing,
        "missing_active_keys": [],
        "unexpected_checkpoint_keys": unexpected if not reject_unexpected else [],
        "load_performed": False,
    }


__all__ = ["validate_full_checkpoint"]
