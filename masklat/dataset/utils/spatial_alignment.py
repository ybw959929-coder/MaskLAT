"""Spatial-transform metadata shared by SAM and SigLIP preprocessing.

The current MaskLAT S3 pipeline applies deterministic aspect-preserving
resize/right-bottom padding for SAM and a deterministic resize (optionally
preceded by the existing centered square padding) for SigLIP.  This module
records both transforms without changing either processor's pixel output.
Random SAM crop/flip or an untracked SigLIP crop is marked unrecoverable so
topology mode can fail loudly instead of pretending the coordinate systems
are aligned.
"""

import importlib
import math
from numbers import Real
from typing import Any, Dict, Tuple

import torch
from PIL import Image


SAM_GEOMETRY_CONTRACT = "masklat_sam_direct_resize_right_bottom_pad_v1"
SIGLIP_GEOMETRY_CONTRACT = "siglip_direct_resize_v1"
SAM_RESOLVED_GEOMETRY_FIELD = "masklat_spatial_geometry_flags"
_SAM_RESOLVED_GEOMETRY_NAMES = ("do_resize", "do_crop", "do_flip", "do_pad")


def expand2square(image: Image.Image, background_color: Any) -> Image.Image:
    """Center a PIL image on a square canvas using an explicit floor offset."""

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"expand2square expects PIL.Image.Image, got {type(image).__name__}"
        )
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(
            f"expand2square requires a non-empty image, got {(width, height)}"
        )
    if width == height:
        return image
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    expanded = Image.new(image.mode, (side, side), background_color)
    expanded.paste(image, (left, top))
    return expanded


def _processor_objects(processor: Any):
    """Return every wrapper layer and the concrete image processor once."""

    if processor is None:
        return ()
    objects = []
    seen = set()
    current = processor
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ValueError("processor contains an image_processor wrapper cycle")
        seen.add(identity)
        objects.append(current)
        nested = getattr(current, "image_processor", None)
        if nested is None or nested is current:
            break
        current = nested
    return tuple(objects)


def unwrap_image_processor(
    processor: Any,
    *,
    name: str,
    require_image_mean: bool = False,
):
    """Resolve processor wrappers and validate the runtime preprocessing API."""

    if processor is None:
        return None
    seen = set()
    resolved = processor
    while hasattr(resolved, "image_processor"):
        identity = id(resolved)
        if identity in seen:
            raise ValueError(f"{name} contains an image_processor wrapper cycle")
        seen.add(identity)
        nested = getattr(resolved, "image_processor")
        if nested is None or nested is resolved:
            break
        resolved = nested
    if not callable(getattr(resolved, "preprocess", None)):
        raise TypeError(
            f"{name} must expose a callable preprocess method after unwrapping"
        )
    if require_image_mean and not hasattr(resolved, "image_mean"):
        raise TypeError(
            f"{name} must expose image_mean for deterministic square padding"
        )
    return resolved


def chw_spatial_size(image: Any, *, name: str) -> Tuple[int, int]:
    """Validate one RGB processor output as non-empty CHW."""

    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(
            f"{name} must be a single CHW image tensor/array, got shape {shape}"
        )
    channels, height, width = map(int, shape)
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{name} must have positive CHW dimensions, got {shape}")
    if channels != 3:
        raise ValueError(
            f"{name} must be RGB CHW with C=3, got shape {tuple(shape)}"
        )
    return height, width


def extract_single_pixel_values(processed: Any, *, name: str):
    """Require a processor result with exactly one RGB ``[1,3,H,W]`` image."""

    try:
        pixel_values = processed["pixel_values"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{name} output is missing pixel_values") from error
    shape = getattr(pixel_values, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(
            f"{name} pixel_values must be [1,3,H,W], got shape {shape}"
        )
    batch, channels, height, width = map(int, shape)
    if batch != 1 or channels != 3 or height <= 0 or width <= 0:
        raise ValueError(
            f"{name} pixel_values must be exactly [1,3,H,W] with positive "
            f"spatial dimensions, got {tuple(shape)}"
        )
    return pixel_values[0]


def _extract_single_size_field(
    processed: Any,
    field: str,
    *,
    name: str,
) -> Tuple[int, int]:
    """Require one finite, positive, integer-valued ``[1,2]`` size field."""

    try:
        sizes = processed[field]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{name} output is missing {field}") from error
    shape = getattr(sizes, "shape", None)
    if shape is None or tuple(map(int, shape)) != (1, 2):
        raise ValueError(
            f"{name} {field} must be exactly [1,2], got shape {shape}"
        )
    size = sizes[0]
    if isinstance(size, torch.Tensor):
        if not bool(torch.isfinite(size).all()):
            raise ValueError(f"{name} {field} contains NaN or Inf")
        size = size.detach().cpu().tolist()
    elif hasattr(size, "tolist"):
        size = size.tolist()
    return _size_tuple(size, f"{name} {field}[0]")


def extract_single_scaled_size(processed: Any, *, name: str) -> Tuple[int, int]:
    """Require exactly one positive, integer-valued SAM scaled size."""

    return _extract_single_size_field(processed, "scaled_sizes", name=name)


def extract_single_sam_geometry(
    processed: Any,
    *,
    expected_original_size: Any,
    pixel_size: Any,
    name: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Validate one SAM processor's original, scaled, and canvas sizes."""

    expected_original = _size_tuple(
        expected_original_size,
        f"{name} expected_original_size",
    )
    original = _extract_single_size_field(
        processed,
        "original_sizes",
        name=name,
    )
    if original != expected_original:
        raise ValueError(
            f"{name} original_sizes disagrees with the actual input image: "
            f"{original} != {expected_original}"
        )
    scaled = extract_single_scaled_size(processed, name=name)
    canvas = _size_tuple(pixel_size, f"{name} pixel_size")
    if scaled[0] > canvas[0] or scaled[1] > canvas[1]:
        raise ValueError(
            f"{name} scaled size cannot exceed the pixel canvas: "
            f"{scaled} > {canvas}"
        )
    return original, scaled


def extract_sam_resolved_geometry_options(
    processed: Any,
    *,
    name: str,
) -> Dict[str, bool]:
    """Read the per-call resolved SAM resize/crop/flip/pad option flags.

    The local ``SamImageProcessor`` returns one integer/bool row in the fixed
    order ``do_resize, do_crop, do_flip, do_pad``.  Requiring the exact shape
    prevents a caller from silently falling back to instance defaults when a
    per-call geometry override was used.  ``do_flip`` records the resolved
    option, not whether its random draw produced a non-None direction; enabled
    random flipping is deliberately rejected conservatively either way.
    """

    try:
        flags = processed[SAM_RESOLVED_GEOMETRY_FIELD]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{name} output is missing {SAM_RESOLVED_GEOMETRY_FIELD}"
        ) from error
    shape = getattr(flags, "shape", None)
    if shape is None or tuple(map(int, shape)) != (1, 4):
        raise ValueError(
            f"{name} {SAM_RESOLVED_GEOMETRY_FIELD} must be exactly [1,4], "
            f"got shape {shape}"
        )
    row = flags[0]
    if isinstance(row, torch.Tensor):
        row = row.detach().cpu().tolist()
    elif hasattr(row, "tolist"):
        row = row.tolist()
    if not isinstance(row, (tuple, list)) or len(row) != 4:
        raise ValueError(
            f"{name} {SAM_RESOLVED_GEOMETRY_FIELD}[0] must contain four flags"
        )
    resolved = {}
    for flag_name, value in zip(_SAM_RESOLVED_GEOMETRY_NAMES, row):
        if isinstance(value, bool):
            resolved[flag_name] = value
        elif isinstance(value, int) and value in (0, 1):
            resolved[flag_name] = bool(value)
        else:
            raise TypeError(
                f"{name} resolved geometry option {flag_name!r} must be bool "
                f"or integer 0/1, got {value!r}"
            )
    return resolved


def validate_mask_collection_canvas(
    masks: Any,
    expected_size: Any,
    *,
    name: str,
):
    """Require every returned mask to use the declared SAM ``(H,W)`` canvas."""

    if masks is None:
        return None
    expected = _size_tuple(expected_size, f"{name} expected_size")
    shape = getattr(masks, "shape", None)
    if shape is not None:
        if len(shape) not in (2, 3):
            raise ValueError(
                f"{name} must be [H,W] or [N,H,W], got shape {shape}"
            )
        actual = tuple(map(int, shape[-2:]))
        if actual != expected:
            raise ValueError(
                f"{name} canvas must match SAM input: {actual} != {expected}"
            )
        return masks
    if not isinstance(masks, (list, tuple)) or not masks:
        raise ValueError(
            f"{name} must expose spatial dimensions; got {type(masks).__name__}"
        )
    mismatches = []
    for index, mask in enumerate(masks):
        mask_shape = getattr(mask, "shape", None)
        if mask_shape is None or len(mask_shape) != 2:
            mismatches.append((index, mask_shape))
            continue
        actual = tuple(map(int, mask_shape[-2:]))
        if actual != expected:
            mismatches.append((index, actual))
    if mismatches:
        raise ValueError(
            f"every {name} entry must use SAM canvas {expected}; "
            f"mismatches={mismatches}"
        )
    return masks


def _processor_flag(processor: Any, name: str):
    """Return a visible boolean processor setting, including wrappers."""

    values = [
        getattr(candidate, name)
        for candidate in _processor_objects(processor)
        if hasattr(candidate, name)
    ]
    if not values:
        return None
    normalized_values = []
    for value in values:
        if isinstance(value, bool):
            normalized_values.append(value)
        elif isinstance(value, int) and value in (0, 1):
            normalized_values.append(bool(value))
        else:
            raise TypeError(
                f"processor geometry flag {name!r} must be bool or integer "
                f"0/1, got {values!r}"
            )
    normalized = set(normalized_values)
    if len(normalized) != 1:
        raise ValueError(
            f"processor wrapper layers disagree on geometry flag {name!r}: "
            f"{values!r}"
        )
    return normalized.pop()


def _has_supported_geometry_contract(processor: Any, *, role: str) -> bool:
    objects = _processor_objects(processor)
    concrete = objects[-1]
    declared = concrete.__class__.__dict__.get(
        "masklat_spatial_geometry_contract"
    )

    def declares(expected: str) -> bool:
        if declared == expected:
            return True
        return isinstance(declared, (tuple, list, set, frozenset)) and (
            expected in declared
        )

    if role == "sam":
        # Inherited/instance markers are deliberately insufficient: a
        # subclass that changes pad placement must explicitly redeclare the
        # contract on its own concrete class.
        return declares(SAM_GEOMETRY_CONTRACT)
    if role == "siglip":
        if declares(SIGLIP_GEOMETRY_CONTRACT):
            return True
        # Trust the exact imported Hugging Face type, not merely a matching
        # ``__module__``/``__name__`` pair that a custom class can spoof.
        try:
            siglip_module = importlib.import_module(
                "transformers.models.siglip.image_processing_siglip"
            )
        except (ImportError, ModuleNotFoundError):
            return False
        official_type = getattr(siglip_module, "SiglipImageProcessor", None)
        return official_type is not None and type(concrete) is official_type
    raise ValueError(f"unsupported processor geometry role {role!r}")


def _size_tuple(size: Any, name: str) -> Tuple[int, int]:
    if isinstance(size, torch.Tensor):
        size = size.detach().tolist()
    if isinstance(size, dict):
        if "height" not in size or "width" not in size:
            raise ValueError(f"{name} dict must contain height and width")
        height, width = size["height"], size["width"]
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        height, width = size
    else:
        raise TypeError(f"{name} must be a two-element size, got {size!r}")
    resolved = []
    for axis, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{name} {axis} must be a real integer-valued number, "
                f"got {value!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"{name} {axis} must be finite and integer-valued, "
                f"got {value!r}"
            )
        resolved.append(int(numeric))
    height, width = resolved
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def _declared_direct_resize_size(
    original_size: Tuple[int, int],
    declared_size: Any,
) -> Tuple[int, int] | None:
    """Resolve common deterministic SAM resize declarations when available."""

    if not isinstance(declared_size, dict):
        return None
    original_height, original_width = original_size
    if "longest_edge" in declared_size:
        edge = _size_tuple(
            (declared_size["longest_edge"], declared_size["longest_edge"]),
            "SAM size.longest_edge",
        )[0]
        scale = edge / max(original_height, original_width)
        return (
            int(original_height * scale + 0.5),
            int(original_width * scale + 0.5),
        )
    if "shortest_edge" in declared_size:
        edge = _size_tuple(
            (declared_size["shortest_edge"], declared_size["shortest_edge"]),
            "SAM size.shortest_edge",
        )[0]
        scale = edge / min(original_height, original_width)
        return (
            int(original_height * scale + 0.5),
            int(original_width * scale + 0.5),
        )
    if "height" in declared_size and "width" in declared_size:
        return _size_tuple(declared_size, "SAM size")
    return None


def _affine_scale_translate(
    scale_x: float,
    scale_y: float,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> torch.Tensor:
    """Return a homogeneous matrix that translates, then scales."""

    return torch.tensor(
        [
            [scale_x, 0.0, scale_x * translate_x],
            [0.0, scale_y, scale_y * translate_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def build_spatial_transform_metadata(
    *,
    original_size: Any,
    siglip_input_size: Any,
    sam_input_size: Any,
    sam_scaled_size: Any,
    pad_image_to_square: bool,
    siglip_processor: Any = None,
    sam_processor: Any = None,
    sam_preprocess_output: Any = None,
) -> Dict[str, Any]:
    """Describe the exact deterministic transforms used by ``BaseDataset``.

    Sizes use ``(height, width)`` and valid regions use
    ``(top, left, bottom, right)`` in continuous input-pixel coordinates.
    Returned matrices map original ``(x,y,1)`` coordinates into each model
    input.  Tensor fields are CPU float32 and are moved with ``DataSample``
    when a batch enters the model.
    """

    if not isinstance(pad_image_to_square, bool):
        raise TypeError(
            "pad_image_to_square must be a realized boolean decision, got "
            f"{pad_image_to_square!r}"
        )
    original_height, original_width = _size_tuple(original_size, "original_size")
    siglip_height, siglip_width = _size_tuple(siglip_input_size, "siglip_input_size")
    sam_height, sam_width = _size_tuple(sam_input_size, "sam_input_size")
    scaled_height, scaled_width = _size_tuple(sam_scaled_size, "sam_scaled_size")

    if scaled_height > sam_height or scaled_width > sam_width:
        raise ValueError(
            "SAM scaled valid region cannot exceed padded input: "
            f"scaled={(scaled_height, scaled_width)}, input={(sam_height, sam_width)}"
        )

    sam_instance_do_resize = _processor_flag(sam_processor, "do_resize")
    sam_instance_do_pad = _processor_flag(sam_processor, "do_pad")
    sam_instance_do_crop = _processor_flag(sam_processor, "do_crop")
    sam_instance_do_flip = _processor_flag(sam_processor, "do_flip")
    sam_do_resize = sam_instance_do_resize
    sam_do_pad = sam_instance_do_pad
    siglip_do_resize = _processor_flag(siglip_processor, "do_resize")
    sam_crop = bool(
        sam_instance_do_crop
        or _processor_flag(sam_processor, "do_center_crop")
    )
    sam_flip = bool(
        sam_instance_do_flip
        or _processor_flag(sam_processor, "do_random_flip")
    )
    siglip_crop = bool(
        _processor_flag(siglip_processor, "do_center_crop")
        or _processor_flag(siglip_processor, "do_crop")
        or _processor_flag(siglip_processor, "do_random_crop")
    )
    siglip_flip = bool(
        _processor_flag(siglip_processor, "do_flip")
        or _processor_flag(siglip_processor, "do_random_flip")
    )
    siglip_internal_pad = bool(_processor_flag(siglip_processor, "do_pad"))
    siglip_pan_and_scan = bool(
        _processor_flag(siglip_processor, "do_pan_and_scan")
    )
    reasons = []
    if sam_preprocess_output is not None:
        resolved_options = extract_sam_resolved_geometry_options(
            sam_preprocess_output,
            name="SAM processor",
        )
        for flag_name, instance_value in (
            ("do_resize", sam_instance_do_resize),
            ("do_crop", sam_instance_do_crop),
            ("do_flip", sam_instance_do_flip),
            ("do_pad", sam_instance_do_pad),
        ):
            if (
                instance_value is not None
                and resolved_options[flag_name] != instance_value
            ):
                reasons.append(
                    f"SAM per-call {flag_name}={resolved_options[flag_name]} "
                    f"overrides processor setting {instance_value}"
                )
        sam_do_resize = resolved_options["do_resize"]
        sam_do_pad = resolved_options["do_pad"]
        sam_crop = bool(sam_crop or resolved_options["do_crop"])
        sam_flip = bool(sam_flip or resolved_options["do_flip"])
    if sam_processor is None:
        reasons.append("SAM processor geometry contract is unavailable")
    elif not _has_supported_geometry_contract(sam_processor, role="sam"):
        reasons.append(
            "unsupported SAM processor geometry contract; expected "
            f"{SAM_GEOMETRY_CONTRACT}"
        )
    if siglip_processor is None:
        reasons.append("SigLIP processor geometry contract is unavailable")
    elif not _has_supported_geometry_contract(siglip_processor, role="siglip"):
        reasons.append(
            "unsupported SigLIP processor geometry contract; expected "
            "SiglipImageProcessor direct resize"
        )
    if sam_do_resize is None:
        reasons.append("SAM processor does not expose do_resize")
    if sam_do_pad is None:
        reasons.append("SAM processor does not expose do_pad")
    if siglip_do_resize is None:
        reasons.append(
            "SigLIP image processor does not expose deterministic do_resize"
        )
    if sam_crop:
        reasons.append("SAM random/dynamic crop parameters are not returned")
    if sam_flip:
        reasons.append("SAM random flip decision is not returned")
    if siglip_crop:
        reasons.append("SigLIP crop parameters are not returned")
    if siglip_flip:
        reasons.append("SigLIP flip parameters are not returned")
    if siglip_internal_pad:
        reasons.append(
            "SigLIP processor-internal padding is not part of the recorded "
            "center-square transform"
        )
    if siglip_pan_and_scan:
        reasons.append("SigLIP pan-and-scan geometry is not recorded")
    if sam_do_resize is False and (
        scaled_height != original_height or scaled_width != original_width
    ):
        reasons.append(
            "SAM do_resize=False but scaled_size differs from original_size"
        )
    if sam_do_resize is True and sam_processor is not None:
        sam_concrete = _processor_objects(sam_processor)[-1]
        declared_resize_size = _declared_direct_resize_size(
            (original_height, original_width),
            getattr(sam_concrete, "size", None),
        )
        if (
            declared_resize_size is not None
            and (scaled_height, scaled_width) != declared_resize_size
        ):
            reasons.append(
                "SAM scaled_size disagrees with the deterministic declared "
                f"resize size: {(scaled_height, scaled_width)} != "
                f"{declared_resize_size}"
            )
    if sam_do_pad is False and (
        sam_height != scaled_height or sam_width != scaled_width
    ):
        reasons.append(
            "SAM do_pad=False but input_size differs from scaled_size"
        )
    if sam_do_pad is True:
        sam_concrete = _processor_objects(sam_processor)[-1]
        declared_pad_size = getattr(sam_concrete, "pad_size", None)
        if declared_pad_size is None:
            reasons.append("SAM do_pad=True but pad_size is not exposed")
        elif _size_tuple(declared_pad_size, "SAM pad_size") != (
            sam_height,
            sam_width,
        ):
            reasons.append(
                "SAM declared pad_size differs from actual processor output"
            )

    siglip_preprocess_height = (
        max(original_height, original_width)
        if pad_image_to_square
        else original_height
    )
    siglip_preprocess_width = (
        max(original_height, original_width)
        if pad_image_to_square
        else original_width
    )
    if siglip_do_resize is False and (
        siglip_height != siglip_preprocess_height
        or siglip_width != siglip_preprocess_width
    ):
        reasons.append(
            "SigLIP do_resize=False but output size differs from its explicit "
            "preprocess input size"
        )
    recoverable = len(reasons) == 0

    original_to_sam = _affine_scale_translate(
        scaled_width / original_width,
        scaled_height / original_height,
    )
    sam_valid_region = torch.tensor(
        [0.0, 0.0, float(scaled_height), float(scaled_width)],
        dtype=torch.float32,
    )

    if pad_image_to_square:
        square_size = max(original_height, original_width)
        offset_x = float((square_size - original_width) // 2)
        offset_y = float((square_size - original_height) // 2)
        siglip_scale_x = siglip_width / square_size
        siglip_scale_y = siglip_height / square_size
        original_to_siglip = _affine_scale_translate(
            siglip_scale_x,
            siglip_scale_y,
            offset_x,
            offset_y,
        )
        siglip_valid_region = torch.tensor(
            [
                offset_y * siglip_scale_y,
                offset_x * siglip_scale_x,
                (offset_y + original_height) * siglip_scale_y,
                (offset_x + original_width) * siglip_scale_x,
            ],
            dtype=torch.float32,
        )
    else:
        original_to_siglip = _affine_scale_translate(
            siglip_width / original_width,
            siglip_height / original_height,
        )
        siglip_valid_region = torch.tensor(
            [0.0, 0.0, float(siglip_height), float(siglip_width)],
            dtype=torch.float32,
        )

    return {
        "original_size": torch.tensor(
            [float(original_height), float(original_width)],
            dtype=torch.float32,
        ),
        "original_to_sam": original_to_sam,
        "sam_input_size": torch.tensor(
            [float(sam_height), float(sam_width)],
            dtype=torch.float32,
        ),
        "sam_valid_region": sam_valid_region,
        "original_to_siglip": original_to_siglip,
        "siglip_input_size": torch.tensor(
            [float(siglip_height), float(siglip_width)],
            dtype=torch.float32,
        ),
        "siglip_valid_region": siglip_valid_region,
        "recoverable": recoverable,
        "unrecoverable_reason": "; ".join(reasons),
        "pad_image_to_square": bool(pad_image_to_square),
        "geometry_contract": (
            f"SigLIP={SIGLIP_GEOMETRY_CONTRACT}: original -> optional "
            "center-square pad -> direct resize; "
            f"SAM={SAM_GEOMETRY_CONTRACT}: original -> direct resize -> "
            "right/bottom pad"
        ),
    }


__all__ = [
    "build_spatial_transform_metadata",
    "chw_spatial_size",
    "expand2square",
    "extract_single_pixel_values",
    "extract_sam_resolved_geometry_options",
    "extract_single_sam_geometry",
    "extract_single_scaled_size",
    "SAM_GEOMETRY_CONTRACT",
    "SAM_RESOLVED_GEOMETRY_FIELD",
    "SIGLIP_GEOMETRY_CONTRACT",
    "unwrap_image_processor",
    "validate_mask_collection_canvas",
]
