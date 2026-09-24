"""Strict spatial-validity helpers for padded SAM image canvases.

SAM valid regions use continuous input-pixel coordinates in
``(top, left, bottom, right)`` order and half-open interval semantics.  A
discrete grid location is valid exactly when its pixel center lies inside the
continuous region.  The same contract is used by topology construction,
region pooling, matching, and mask losses.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _spatial_size(size: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise TypeError(f"spatial size must be a two-element tuple/list, got {size!r}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial size must be positive, got {(height, width)}")
    return height, width


def validate_normalized_valid_boxes(
    boxes: Tensor,
    *,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Validate normalized ``[B,4]`` half-open boxes and return float32."""

    if not isinstance(boxes, Tensor):
        raise TypeError("normalized SAM valid boxes must be a tensor")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            "normalized SAM valid boxes must be [B,4] in "
            f"(top,left,bottom,right) order, got {tuple(boxes.shape)}"
        )
    if batch_size is not None and boxes.shape[0] != int(batch_size):
        raise ValueError(
            "normalized SAM valid-box batch does not match predictions: "
            f"{boxes.shape[0]} != {int(batch_size)}"
        )
    if device is not None and boxes.device != device:
        raise ValueError(
            "normalized SAM valid boxes must share the prediction device: "
            f"{boxes.device} != {device}"
        )
    boxes = boxes.to(dtype=torch.float32)
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError("normalized SAM valid boxes contain NaN or Inf")

    top, left, bottom, right = boxes.unbind(dim=1)
    valid = (
        top.ge(0.0)
        & left.ge(0.0)
        & bottom.le(1.0)
        & right.le(1.0)
        & bottom.gt(top)
        & right.gt(left)
    )
    if not bool(valid.all()):
        invalid = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise ValueError(
            "normalized SAM valid boxes must satisfy "
            "0 <= top < bottom <= 1 and 0 <= left < right <= 1; "
            f"invalid batch indices={invalid}, boxes={boxes[~valid].tolist()}"
        )
    return boxes


def normalize_sam_valid_regions(
    sam_input_size: Tensor,
    sam_valid_region: Tensor,
) -> Tensor:
    """Convert absolute SAM regions to normalized half-open ``[B,4]`` boxes."""

    if not isinstance(sam_input_size, Tensor) or not isinstance(
        sam_valid_region, Tensor
    ):
        raise TypeError("sam_input_size and sam_valid_region must be tensors")
    if sam_input_size.ndim != 2 or sam_input_size.shape[1] != 2:
        raise ValueError(
            f"sam_input_size must be [B,2], got {tuple(sam_input_size.shape)}"
        )
    if sam_valid_region.ndim != 2 or sam_valid_region.shape[1] != 4:
        raise ValueError(
            "sam_valid_region must be [B,4] in "
            f"(top,left,bottom,right) order, got {tuple(sam_valid_region.shape)}"
        )
    if sam_input_size.shape[0] != sam_valid_region.shape[0]:
        raise ValueError(
            "sam_input_size and sam_valid_region batch dimensions differ: "
            f"{sam_input_size.shape[0]} != {sam_valid_region.shape[0]}"
        )
    if sam_input_size.device != sam_valid_region.device:
        raise ValueError("SAM input sizes and valid regions must share one device")

    sizes = sam_input_size.to(dtype=torch.float32)
    regions = sam_valid_region.to(dtype=torch.float32)
    if not bool(torch.isfinite(sizes).all() and torch.isfinite(regions).all()):
        raise FloatingPointError("SAM spatial metadata contains NaN or Inf")
    if not bool(sizes.gt(0).all()):
        raise ValueError(f"SAM input sizes must be positive, got {sizes.tolist()}")

    heights, widths = sizes.unbind(dim=1)
    top, left, bottom, right = regions.unbind(dim=1)
    inside = (
        top.ge(0.0)
        & left.ge(0.0)
        & bottom.le(heights)
        & right.le(widths)
        & bottom.gt(top)
        & right.gt(left)
    )
    if not bool(inside.all()):
        invalid = torch.nonzero(~inside, as_tuple=False).flatten().tolist()
        raise ValueError(
            "sam_valid_region must be a non-empty half-open rectangle inside "
            "sam_input_size; "
            f"invalid batch indices={invalid}, "
            f"sizes={sizes[~inside].tolist()}, "
            f"regions={regions[~inside].tolist()}"
        )

    boxes = torch.stack(
        (
            top / heights,
            left / widths,
            bottom / heights,
            right / widths,
        ),
        dim=1,
    )
    return validate_normalized_valid_boxes(boxes)


def valid_mask_from_normalized_boxes(
    boxes: Tensor,
    spatial_size: Tuple[int, int],
    *,
    require_nonempty: bool = True,
) -> Tensor:
    """Return ``[B,1,H,W]`` validity using grid-pixel center membership."""

    boxes = validate_normalized_valid_boxes(boxes)
    height, width = _spatial_size(spatial_size)
    y = (
        torch.arange(height, device=boxes.device, dtype=torch.float32) + 0.5
    ) / float(height)
    x = (
        torch.arange(width, device=boxes.device, dtype=torch.float32) + 0.5
    ) / float(width)
    top = boxes[:, 0, None, None]
    left = boxes[:, 1, None, None]
    bottom = boxes[:, 2, None, None]
    right = boxes[:, 3, None, None]
    valid = (
        y[None, :, None].ge(top)
        & y[None, :, None].lt(bottom)
        & x[None, None, :].ge(left)
        & x[None, None, :].lt(right)
    ).unsqueeze(1)
    if require_nonempty:
        nonempty = valid.flatten(1).any(dim=1)
        if not bool(nonempty.all()):
            invalid = torch.nonzero(~nonempty, as_tuple=False).flatten().tolist()
            raise ValueError(
                "SAM valid region contains no pixel center at requested grid "
                f"{(height, width)} for batch indices {invalid}"
            )
    return valid


def scale_unit_coordinates_to_valid_boxes(
    coordinates: Tensor,
    boxes: Tensor,
) -> Tensor:
    """Map unit-square ``[...,2]`` coordinates into normalized valid boxes."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(
            f"coordinates must be [N,P,2], got {tuple(coordinates.shape)}"
        )
    if not bool(torch.isfinite(coordinates).all()):
        raise FloatingPointError("sampling coordinates contain NaN or Inf")
    if bool((coordinates < 0).any()) or bool((coordinates >= 1).any()):
        raise ValueError("unit sampling coordinates must lie in [0,1)")
    boxes = validate_normalized_valid_boxes(boxes, device=coordinates.device)
    if boxes.shape[0] not in (1, coordinates.shape[0]):
        raise ValueError(
            "valid-box batch must be one or match coordinate rows: "
            f"{boxes.shape[0]} not in (1,{coordinates.shape[0]})"
        )
    if boxes.shape[0] == 1 and coordinates.shape[0] != 1:
        boxes = boxes.expand(coordinates.shape[0], -1)
    top, left, bottom, right = boxes.unbind(dim=1)
    x = left[:, None] + coordinates[..., 0] * (right - left)[:, None]
    y = top[:, None] + coordinates[..., 1] * (bottom - top)[:, None]
    return torch.stack((x, y), dim=-1)


def _validate_masked_sampling_inputs(
    values: Tensor,
    source_valid_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    if values.ndim != 4:
        raise ValueError(f"values must be [B,C,H,W], got {tuple(values.shape)}")
    expected = (values.shape[0], 1, values.shape[2], values.shape[3])
    if source_valid_mask.shape != expected:
        raise ValueError(
            f"source_valid_mask must be {expected}, got {tuple(source_valid_mask.shape)}"
        )
    if source_valid_mask.device != values.device:
        raise ValueError("values and source_valid_mask must share one device")
    if source_valid_mask.dtype != torch.bool:
        raise TypeError("source_valid_mask must have dtype torch.bool")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("values for masked sampling contain NaN or Inf")
    return values.to(torch.float32), source_valid_mask.to(torch.float32)


def _apply_target_validity(
    sampled: Tensor,
    coverage: Tensor,
    target_valid_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if target_valid_mask is None:
        return sampled, coverage
    expected = (sampled.shape[0], 1, sampled.shape[2], sampled.shape[3])
    if target_valid_mask.shape != expected:
        raise ValueError(
            f"target_valid_mask must be {expected}, got {tuple(target_valid_mask.shape)}"
        )
    if target_valid_mask.device != sampled.device:
        raise ValueError("sampled values and target_valid_mask must share one device")
    if target_valid_mask.dtype != torch.bool:
        raise TypeError("target_valid_mask must have dtype torch.bool")
    target = target_valid_mask
    return sampled * target.to(sampled.dtype), coverage * target.to(coverage.dtype)


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    return eps


def masked_normalized_resize(
    values: Tensor,
    source_valid_mask: Tensor,
    spatial_size: Tuple[int, int],
    *,
    target_valid_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Bilinearly resize without padding-value leakage or edge attenuation."""

    eps = _validate_eps(eps)
    values_float, source = _validate_masked_sampling_inputs(
        values, source_valid_mask
    )
    spatial_size = _spatial_size(spatial_size)
    numerator = F.interpolate(
        values_float * source,
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    coverage = F.interpolate(
        source,
        size=spatial_size,
        mode="bilinear",
        align_corners=False,
    )
    sampled = torch.where(
        coverage.gt(eps),
        numerator / coverage.clamp_min(eps),
        torch.zeros_like(numerator),
    )
    return _apply_target_validity(sampled, coverage, target_valid_mask)


def masked_normalized_grid_sample(
    values: Tensor,
    source_valid_mask: Tensor,
    sampling_grid: Tensor,
    *,
    target_valid_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Grid-sample without reading invalid SAM padding values."""

    eps = _validate_eps(eps)
    values_float, source = _validate_masked_sampling_inputs(
        values, source_valid_mask
    )
    if sampling_grid.ndim != 4 or sampling_grid.shape[0] != values.shape[0]:
        raise ValueError(
            "sampling_grid must be [B,H,W,2] with matching batch, got "
            f"{tuple(sampling_grid.shape)}"
        )
    if sampling_grid.shape[-1] != 2:
        raise ValueError("sampling_grid final dimension must be 2")
    if sampling_grid.device != values.device:
        raise ValueError("sampling_grid and values must share one device")
    if not bool(torch.isfinite(sampling_grid).all()):
        raise FloatingPointError("sampling_grid contains NaN or Inf")
    numerator = F.grid_sample(
        values_float * source,
        sampling_grid.to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    coverage = F.grid_sample(
        source,
        sampling_grid.to(torch.float32),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = torch.where(
        coverage.gt(eps),
        numerator / coverage.clamp_min(eps),
        torch.zeros_like(numerator),
    )
    return _apply_target_validity(sampled, coverage, target_valid_mask)


__all__ = [
    "masked_normalized_grid_sample",
    "masked_normalized_resize",
    "normalize_sam_valid_regions",
    "scale_unit_coordinates_to_valid_boxes",
    "valid_mask_from_normalized_boxes",
    "validate_normalized_valid_boxes",
]
