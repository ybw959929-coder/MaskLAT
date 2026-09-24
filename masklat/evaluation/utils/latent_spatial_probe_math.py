"""Read-only FP32 mathematics for the latent spatial-correspondence probe.

This module imports only PyTorch and the Python standard library.  Nothing here
changes model parameters, the production forward, or a predicted mask.  Mask
arguments are probabilities, never logits or ground-truth annotations.

All tensors are detached.  Summary ``*_mean`` entries are arithmetic means of
the explicitly counted items, with accompanying ``*_sum``/``*_count`` entries
for aggregation across batches/ranks.  An unavailable mean is ``None`` rather
than NaN.  Numerical-empty thresholds below are diagnostics, not scientifically
validated cutoffs for deciding whether a latent represents an object.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Scalar = Union[int, float, bool, str, None]


def _finite_tensor(value: Tensor, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if any(int(n) <= 0 for n in value.shape):
        raise ValueError(f"{name} dimensions must all be nonzero")
    result = value.detach().float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _probabilities(value: Tensor, name: str, ndim: int) -> Tensor:
    result = _finite_tensor(value, name, ndim)
    if bool(((result < 0) | (result > 1)).any()):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return result


def _validity(valid: Optional[Tensor], reference: Tensor) -> Tensor:
    expected = (reference.shape[0], 1, *reference.shape[-2:])
    if valid is None:
        return reference.new_ones(expected)
    result = _probabilities(valid, "valid", 4)
    if tuple(result.shape) != expected:
        raise ValueError(f"valid must have shape {expected}")
    if result.device != reference.device:
        raise ValueError("valid and spatial maps must be on the same device")
    return result


def _summary(target: Dict[str, Scalar], name: str, values: Tensor) -> None:
    values = values.detach().double().reshape(-1)
    count = int(values.numel())
    target[f"{name}_count"] = count
    target[f"{name}_sum"] = float(values.sum().item())
    target[f"{name}_mean"] = float(values.mean().item()) if count else None
    target[f"{name}_min"] = float(values.min().item()) if count else None
    target[f"{name}_max"] = float(values.max().item()) if count else None


@torch.no_grad()
def reconstruct_mha_weights(
    module: nn.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    key_padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """Reconstruct head-averaged eval MHA probabilities as ``[B,L,Q]``.

    ``query`` and ``key`` must be the actual already-normalized/projector-input
    tensors passed to this MHA.  Uses its Q/K input projections, but does not
    call its forward or touch attention/dropout/module state.  Only standard
    batch-first, same-width MHA is supported.  A boolean padding mask uses
    PyTorch's convention: True excludes a key.  Float/additive/causal masks are
    deliberately unsupported; callers must not silently omit such masks.

    This FP32 reconstruction can differ slightly from a mixed-precision fused
    production attention kernel.  It is a diagnostic reconstruction, not the
    exact internal probability tensor of that kernel.
    """
    if not isinstance(module, nn.MultiheadAttention):
        raise TypeError("module must be nn.MultiheadAttention")
    if module.training:
        raise ValueError("MHA reconstruction requires eval mode (no dropout)")
    if not module.batch_first:
        raise ValueError("only batch_first=True MHA is supported")
    if not getattr(module, "_qkv_same_embed_dim", False):
        raise ValueError("only same-embedding-dimension MHA is supported")
    if module.bias_k is not None or module.bias_v is not None:
        raise ValueError("bias_k/bias_v MHA is unsupported")
    if module.add_zero_attn:
        raise ValueError("add_zero_attn MHA is unsupported")
    if module.in_proj_weight is None:
        raise ValueError("MHA must provide a packed in_proj_weight")
    q = _finite_tensor(query, "query", 3)
    k = _finite_tensor(key, "key", 3)
    width = int(module.embed_dim)
    if q.shape[0] != k.shape[0] or q.shape[-1] != width or k.shape[-1] != width:
        raise ValueError("query/key batch and embedding dimensions must match MHA")
    if q.device != k.device or q.device != module.in_proj_weight.device:
        raise ValueError("query, key and MHA must be on the same device")
    if key_padding_mask is not None:
        if not isinstance(key_padding_mask, Tensor) or key_padding_mask.dtype != torch.bool:
            raise TypeError("key_padding_mask must be boolean (True means excluded)")
        if tuple(key_padding_mask.shape) != (q.shape[0], k.shape[1]):
            raise ValueError("key_padding_mask must have shape [B,Q]")
        if key_padding_mask.device != q.device:
            raise ValueError("key_padding_mask must share the input device")
        if bool(key_padding_mask.all(dim=1).any()):
            raise ValueError("each sample must have at least one unmasked key")
    # Explicitly disable outer inference autocast: .float() alone is not enough
    # to stop a surrounding autocast context from downcasting F.linear/matmul.
    with torch.autocast(device_type=q.device.type, enabled=False):
        wq, wk, _ = module.in_proj_weight.detach().float().chunk(3, dim=0)
        bq = bk = None
        if module.in_proj_bias is not None:
            bq, bk, _ = module.in_proj_bias.detach().float().chunk(3, dim=0)
        q = F.linear(q, wq, bq)
        k = F.linear(k, wk, bk)
        batch, slots, _ = q.shape
        count = k.shape[1]
        heads = int(module.num_heads)
        head_dim = width // heads
        q = q.reshape(batch, slots, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, count, heads, head_dim).transpose(1, 2)
        logits = torch.matmul(q * (head_dim ** -0.5), k.transpose(-1, -2))
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
        result = logits.softmax(dim=-1).mean(dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("reconstructed attention probabilities are non-finite")
    return result


@torch.no_grad()
def spatial_support(
    weights: Tensor,
    masks: Tensor,
    valid_mask: Tensor,
    size: int = 32,
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate masks using attention weights, with padding-aware area resize.

    Args:
        weights: Row-normalized probabilities ``[B,L,Q]``.
        masks: Predicted mask probabilities ``[B,Q,H,W]``.
        valid_mask: Valid-image area weights ``[B,1,H,W]`` in [0,1].
        size: Square output resolution, positive integer.

    Returns ``(support, resized_masks, resized_valid)``.  ``resized_valid`` is
    fractional valid coverage, not a thresholded boolean mask.  Each resized
    mask is area(mask * valid) / area(valid), with zero in fully padded bins.
    Thus padding does not dilute a valid-region mean.  Downstream statistics
    must retain the returned valid coverage rather than count padded pixels.
    Entirely empty predicted masks and entirely padded samples are allowed.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    weights = _probabilities(weights, "weights", 3)
    masks = _probabilities(masks, "masks", 4)
    if weights.shape[0] != masks.shape[0] or weights.shape[2] != masks.shape[1]:
        raise ValueError("weights [B,L,Q] must match masks [B,Q,H,W]")
    if weights.device != masks.device:
        raise ValueError("weights and masks must share a device")
    if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=2e-5, rtol=2e-5):
        raise ValueError("weights must sum to one over Query dimension")
    valid = _validity(valid_mask, masks)
    with torch.autocast(device_type=masks.device.type, enabled=False):
        resized_valid = F.interpolate(valid, size=(size, size), mode="area")
        numerator = F.interpolate(masks * valid, size=(size, size), mode="area")
        resized_masks = torch.where(
            resized_valid > 0, numerator / resized_valid.clamp_min(1e-30), 0.0
        ).clamp(0.0, 1.0)
        support = torch.bmm(weights, resized_masks.flatten(2))
        support = support.reshape(weights.shape[0], weights.shape[1], size, size).clamp(0.0, 1.0)
    return support, resized_masks, resized_valid


@torch.no_grad()
def soft_iou_matrix(
    query_masks: Tensor,
    supports: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Product-intersection soft IoU, ``[B,Q,L]``; empty union maps to 0.

    Soft IoU uses intersection=sum(valid*x*y), not min(x,y).  Consequently two
    identical *soft* maps need not have IoU 1.  This is a numerical diagnostic,
    not an assertion that these maps identify the same physical object.
    """
    query_masks = _probabilities(query_masks, "query_masks", 4)
    supports = _probabilities(supports, "supports", 4)
    if query_masks.shape[0] != supports.shape[0] or query_masks.shape[-2:] != supports.shape[-2:]:
        raise ValueError("query_masks and supports must share batch/spatial shape")
    if query_masks.device != supports.device:
        raise ValueError("query_masks and supports must share a device")
    valid = _validity(valid, query_masks)
    with torch.autocast(device_type=query_masks.device.type, enabled=False):
        q = query_masks.flatten(2)
        s = supports.flatten(2)
        v = valid.flatten(2)
        intersection = torch.bmm(q * v, s.transpose(1, 2))
        union = (q * v).sum(dim=-1, keepdim=True) + (s * v).sum(dim=-1).unsqueeze(1) - intersection
        result = torch.where(union > 0, intersection / union.clamp_min(1e-30), 0.0)
    return result.clamp(0.0, 1.0)


@torch.no_grad()
def support_statistics(supports: Tensor, valid: Optional[Tensor] = None) -> Dict[str, Scalar]:
    """Summarize spatial support diversity, size and concentration.

    Spatial means/Euclidean distances use fractional valid-area weights.
    Entropy treats each bin with positive valid coverage as a spatial outcome,
    with probability proportional to support*coverage; normalized entropy and
    exp(entropy)/valid_bin_count lie in [0,1].  Empty supports have entropy and
    effective-area fraction 0.  Fully invalid samples are excluded.  Pairwise
    values count ordered off-diagonal pairs; cosine additionally excludes zero
    vectors.  ``near_empty`` means valid-weighted mean support <= 1e-6 only.
    """
    supports = _probabilities(supports, "supports", 4)
    valid = _validity(valid, supports)
    batch, slots = supports.shape[:2]
    stats: Dict[str, Scalar] = {
        "batch_size": int(batch), "latent_count": int(slots),
        "support_near_empty_mass_fraction_threshold": 1e-6,
        "support_pair_count_convention": "ordered_off_diagonal",
    }
    with torch.autocast(device_type=supports.device.type, enabled=False):
        s = supports.flatten(2)
        v = valid.flatten(2)
        area = v.sum(dim=-1)
        valid_samples = area[:, 0] > 0
        valid_slots = valid_samples[:, None].expand(batch, slots)
        stats["valid_sample_count"] = int(valid_samples.sum().item())
        stats["sample_slot_count"] = int(valid_slots.sum().item())
        mass = (s * v).sum(dim=-1)
        fraction = mass / area.clamp_min(1e-30)
        coverage = ((s > 0.5).float() * v).sum(dim=-1) / area.clamp_min(1e-30)
        _summary(stats, "support_probability_mass_fraction", fraction[valid_slots])
        _summary(stats, "support_coverage_gt_0p5_fraction", coverage[valid_slots])
        _summary(stats, "support_near_empty_fraction", (fraction <= 1e-6).float()[valid_slots])
        probability = s * v / mass.unsqueeze(-1).clamp_min(1e-30)
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1)
        bin_count = (v > 0).sum(dim=-1).float()
        entropy_normalized = torch.where(bin_count > 1, entropy / bin_count.clamp_min(2).log(), 0.0).clamp(0, 1)
        effective = torch.where(mass > 0, entropy.exp() / bin_count.clamp_min(1), 0.0).clamp(0, 1)
        _summary(stats, "support_entropy_normalized", entropy_normalized[valid_slots])
        _summary(stats, "support_effective_area_fraction", effective[valid_slots])
        weighted = s * v.sqrt()
        gram = torch.bmm(weighted, weighted.transpose(1, 2))
        squared_norm = (weighted.square()).sum(dim=-1)
        norms = squared_norm.sqrt()
        norm_products = norms.unsqueeze(2) * norms.unsqueeze(1)
        offdiag = ~torch.eye(slots, dtype=torch.bool, device=s.device)[None]
        pairs = offdiag & valid_samples[:, None, None]
        cosine_pairs = pairs & (norm_products > 0)
        cosine = (gram / norm_products.clamp_min(1e-30)).clamp(-1, 1)
        _summary(stats, "support_pairwise_cosine", cosine[cosine_pairs])
        # The direct-difference kernel gives exactly zero for identical maps
        # without clipping genuinely small differences to a chosen threshold.
        normalized = weighted / area.unsqueeze(-1).clamp_min(1e-30).sqrt()
        l2 = torch.cdist(normalized, normalized, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        _summary(stats, "support_pairwise_l2", l2[pairs])
        overlap = soft_iou_matrix(supports, supports, valid)
        _summary(stats, "support_pairwise_soft_iou", overlap[pairs])
        stats["pairwise_support_similarity"] = stats["support_pairwise_cosine_mean"]
    return stats


def _normalized_entropy(probability: Tensor) -> Tensor:
    count = probability.shape[-1]
    if count <= 1:
        return torch.zeros_like(probability[..., 0])
    return (-(probability * probability.clamp_min(1e-30).log()).sum(dim=-1) / math.log(count)).clamp(0, 1)


@torch.no_grad()
def routing_statistics(logits: Tensor, bias: Tensor) -> Dict[str, Scalar]:
    """Measure ``[B,Q,L]`` routing logits and a hypothetical soft-IoU bias.

    No biased scores are returned to the model.  Read rows normalize over L;
    write columns normalize over Q.  Std uses population (unbiased=False).
    Std ratio is ratio-of-means, not mean-of-ratios, and is None for baseline
    mean row std zero.  Constant threshold 1e-6 is numerical only.  Lambda-0
    checks cover shadow score/softmax algebra, NOT end-to-end model equality.
    Lambda-1 changes are explicitly offline hypothetical statistics.
    """
    logits = _finite_tensor(logits, "logits", 3)
    bias = _probabilities(bias, "bias", 3)
    if logits.shape != bias.shape or logits.device != bias.device:
        raise ValueError("logits and bias must have the same [B,Q,L] shape/device")
    batch, queries, slots = logits.shape
    stats: Dict[str, Scalar] = {
        "batch_size": int(batch), "query_count": int(queries), "latent_count": int(slots),
        "routing_read_row_count": int(batch * queries),
        "routing_write_row_count": int(batch * slots),
        "routing_near_constant_bias_row_std_threshold": 1e-6,
        "lambda0_check_scope": "shadow_logits_and_softmax_only_not_model_forward",
        "offline_lambda1_scope": "hypothetical_only_no_model_change",
    }
    with torch.autocast(device_type=logits.device.type, enabled=False):
        row_std = logits.std(dim=-1, unbiased=False)
        bias_row_std = bias.std(dim=-1, unbiased=False)
        _summary(stats, "routing_logit_row_std", row_std)
        _summary(stats, "routing_bias_row_std", bias_row_std)
        _summary(stats, "routing_logit_column_std", logits.std(dim=1, unbiased=False))
        _summary(stats, "routing_bias_column_std", bias.std(dim=1, unbiased=False))
        denominator = float(row_std.double().mean().item())
        stats["routing_bias_to_logit_row_std_ratio"] = float(bias_row_std.double().mean().item()) / denominator if denominator > 0 else None
        _summary(stats, "routing_near_constant_bias_row_fraction", (bias_row_std <= 1e-6).float())
        read = logits.softmax(dim=-1)
        write = logits.softmax(dim=1).transpose(1, 2)
        _summary(stats, "routing_read_entropy_normalized", _normalized_entropy(read))
        _summary(stats, "routing_write_entropy_normalized", _normalized_entropy(write))
        lambda0 = logits + 0.0 * bias
        stats["lambda0_logit_max_abs_delta"] = float((lambda0 - logits).abs().max().item())
        stats["lambda0_read_probability_max_abs_delta"] = float((lambda0.softmax(-1) - read).abs().max().item())
        stats["lambda0_write_probability_max_abs_delta"] = float((lambda0.softmax(1).transpose(1, 2) - write).abs().max().item())
        hypothetical = logits + bias
        read1 = hypothetical.softmax(dim=-1)
        write1 = hypothetical.softmax(dim=1).transpose(1, 2)
        _summary(stats, "offline_lambda1_read_total_variation", 0.5 * (read1 - read).abs().sum(dim=-1))
        _summary(stats, "offline_lambda1_write_total_variation", 0.5 * (write1 - write).abs().sum(dim=-1))
        _summary(stats, "offline_lambda1_read_top1_changed_fraction", (read1.argmax(-1) != read.argmax(-1)).float())
    return stats
