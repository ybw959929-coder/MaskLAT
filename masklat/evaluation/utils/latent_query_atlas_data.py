"""Metadata-only, source-image-uniform sampling for the latent Query atlas.

No model, dataset package, CUDA or image decoder is imported while selecting
records. RefSegDataset.data contains one row per referring expression, so
sampling rows and deduplicating would over-sample images with more expressions.
Instead sample unique image IDs uniformly, then one expression per image.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import random
import shutil


SUPPORTED_DATASETS = frozenset((
    "refcoco_val_refseg", "refcoco+_val_refseg", "refcocog_val_refseg",
))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _size(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be (height, width), got {value!r}")
    size = [_integer(x, name) for x in value]
    if min(size) <= 0:
        raise ValueError(f"{name} must have positive dimensions")
    return size


def _raw_metadata(dataset):
    """Read annotation rows only; never call dataset.__getitem__()."""
    raw = getattr(dataset, "data", None)
    if raw is None:
        raise TypeError("expected a native RefSegDataset with raw .data metadata")
    if len(dataset) != len(raw):
        raise ValueError("atlas sampling requires an un-repeated, full evaluation dataset")
    if isinstance(raw, (list, tuple)):
        return raw
    # Optional Hugging Face-style annotation tables: project to metadata
    # columns first, excluding image columns/decoders. Native RefSeg uses lists.
    names = getattr(raw, "column_names", None)
    select = getattr(raw, "select_columns", None)
    required = ("image_id", "image_file", "image_size", "sampled_sents", "image_info")
    if isinstance(names, (list, tuple)) and callable(select) and set(required) <= set(names):
        return select(list(required))
    raise TypeError("unsupported raw annotation container; cannot safely decode dataset rows")


def _row_identity(row, index):
    if not isinstance(row, Mapping):
        raise TypeError(f"raw annotation {index} is not a mapping")
    image_id = _integer(row.get("image_id"), f"row {index} image_id")
    filename = row.get("image_file")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"row {index} has no source image_file")
    size = _size(row.get("image_size"), f"row {index} image_size")
    info = row.get("image_info")
    if not isinstance(info, Mapping):
        raise ValueError(f"row {index} has no native image_info")
    if _integer(info.get("image_id"), "image_info.image_id") != image_id:
        raise ValueError(f"row {index} has inconsistent source image IDs")
    if info.get("file_name") != filename:
        raise ValueError(f"row {index} has inconsistent source file names")
    if [_integer(info.get("height"), "image_info.height"),
            _integer(info.get("width"), "image_info.width")] != size:
        raise ValueError(f"row {index} has inconsistent source dimensions")
    phrases = row.get("sampled_sents")
    if not isinstance(phrases, (list, tuple)) or len(phrases) != 1 or not isinstance(phrases[0], str):
        raise ValueError(f"row {index} is not a single-expression evaluation record")
    if info.get("phrases") != list(phrases):
        raise ValueError(f"row {index} has inconsistent referring expressions")
    return image_id, filename, size


def select_unique_images(dataset, count=10, seed=1024, dataset_name=None):
    """Return deterministic JSON-safe selection records, before rank sharding.

    Every unique source image has equal probability regardless of its number
    of expression rows. Each dataset uses its own stable seeded RNG. Rank,
    world size, DataLoader workers and global random state are not involved.
    The same COCO image may legitimately appear in different datasets.
    """
    count, seed = _integer(count, "count"), _integer(seed, "seed")
    if count < 1 or seed < 0:
        raise ValueError("count must be positive and seed nonnegative")
    name = dataset_name if dataset_name is not None else getattr(dataset, "data_name", None)
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported atlas validation dataset: {name!r}")
    native_name = getattr(dataset, "data_name", name)
    if native_name != name:
        raise ValueError(f"requested dataset {name!r} differs from {native_name!r}")
    if getattr(dataset, "data_mode", "eval") != "eval" or getattr(dataset, "data_split", "val") != "val":
        raise ValueError("atlas accepts only native RefCOCO/+/g validation datasets")
    image_folder = getattr(dataset, "image_folder", None)
    if not image_folder:
        raise ValueError("dataset.image_folder is required to retain source images")
    folder = Path(image_folder)
    raw = _raw_metadata(dataset)
    groups, identities = {}, {}
    for index in range(len(raw)):
        image_id, filename, size = _row_identity(raw[index], index)
        identity = (filename, tuple(size))
        if image_id in identities and identities[image_id] != identity:
            raise ValueError(f"image {image_id} resolves to inconsistent files/dimensions")
        identities[image_id] = identity
        groups.setdefault(image_id, []).append(index)
    if len(groups) < count:
        raise ValueError(f"{name} has only {len(groups)} unique source images; need exactly {count}")
    material = f"latent-query-atlas-v1\0{name}\0{seed}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    selected_ids = rng.sample(sorted(groups), count)
    selected = []
    for image_id in selected_ids:
        index = rng.choice(groups[image_id])
        row = raw[index]
        _, filename, size = _row_identity(row, index)
        source = (folder / filename).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"selected source image is missing: {source}")
        info = row["image_info"]
        result = {
            "dataset": name, "dataset_index": index, "image_id": image_id,
            "image_file": filename, "image_path": str(source),
            "image_size": size, "expression": row["sampled_sents"][0],
            "available_expression_records": len(groups[image_id]),
        }
        for key in ("sample_id", "annotation_id", "sentence_id"):
            if key in info:
                result[key] = _integer(info[key], f"image_info.{key}")
        selected.append(result)
    return selected


def verify_batch_identity(batch, record):
    """Fail if one native collated sample is not the selected image/expression."""
    sample = batch.get("data_samples") if isinstance(batch, Mapping) else None
    metainfo = sample.get("metainfo") if isinstance(sample, Mapping) else getattr(sample, "metainfo", None)
    if not isinstance(metainfo, Mapping):
        raise ValueError("native collated batch is missing data_samples.metainfo")
    for key in ("image_infos", "image_files", "image_sizes"):
        if not isinstance(metainfo.get(key), (list, tuple)) or len(metainfo[key]) != 1:
            raise ValueError(f"atlas requires one sample in batch metadata {key}")
    info = metainfo["image_infos"][0]
    if not isinstance(info, Mapping) or info.get("image_id") != record["image_id"]:
        raise ValueError("runtime source image ID differs from selected image")
    if metainfo["image_files"][0] != record["image_file"] or info.get("file_name") != record["image_file"]:
        raise ValueError("runtime source file differs from selected image")
    expected = _size(record["image_size"], "selected image_size")
    actual = _size(metainfo["image_sizes"][0], "runtime image_size")
    if actual != expected or [info.get("height"), info.get("width")] != expected:
        raise ValueError("runtime image dimensions differ from selected image")
    if info.get("phrases") != [record["expression"]]:
        raise ValueError("runtime referring expression differs from selected record")
    for key in ("sample_id", "annotation_id", "sentence_id"):
        if key in record and info.get(key) != record[key]:
            raise ValueError(f"runtime {key} differs from selected expression record")
    return True


def load_source_rgb(record):
    """Decode RGB exactly as BaseDataset: no EXIF transpose, resize or crop."""
    from PIL import Image
    with Image.open(record["image_path"]) as source:
        image = source.convert("RGB")
    if [image.height, image.width] != _size(record["image_size"], "selected image_size"):
        raise ValueError("source pixels have dimensions different from annotation metadata")
    return image


def copy_source_image(record, destination):
    """Archive the original source bytes exclusively; never overwrite a file."""
    source, destination = Path(record["image_path"]), Path(destination)
    if destination.suffix.lower() != source.suffix.lower():
        raise ValueError("source archive must retain the original file extension")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    return str(destination)


def _plain_numeric(value):
    """Small tensor/array geometry fields to Python, without a torch import."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _vector(value, length, name):
    value = _plain_numeric(value)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must have exactly {length} entries")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite real coordinates")
    return [float(x) for x in value]


def _geometry_size(value, name):
    values = _vector(value, 2, name)
    if any(x <= 0 or abs(x - round(x)) > 1e-6 for x in values):
        raise ValueError(f"{name} must contain positive integer pixel dimensions")
    return [int(round(x)) for x in values]


def verify_source_geometry(batch, record, normalized_valid_box=None):
    """Validate full-source, orientation-preserving SAM resize/pad geometry.

    Source-image overlays are valid only if the complete original image maps
    to the recorded valid SAM rectangle, with positive axis-aligned scaling
    and optional padding translation. Crop, flip, rotation, shear and unknown
    transforms are rejected. Pixel tolerance permits only float32 rounding,
    not a one-pixel crop. An optional captured normalized box is cross-checked
    in SAM pixel coordinates before any overlay is rendered.
    """
    verify_batch_identity(batch, record)
    sample = batch["data_samples"]
    metainfo = sample.get("metainfo") if isinstance(sample, Mapping) else sample.metainfo
    transforms = metainfo.get("spatial_transforms")
    if not isinstance(transforms, (list, tuple)) or len(transforms) != 1:
        raise ValueError("atlas requires one recorded spatial_transform")
    transform = transforms[0]
    if not isinstance(transform, Mapping):
        raise ValueError("source geometry contract is missing")
    if _plain_numeric(transform.get("recoverable")) is not True:
        reason = transform.get("unrecoverable_reason", "unspecified")
        raise ValueError(f"source geometry is not recoverable: {reason}")
    original = _geometry_size(transform.get("original_size"), "original_size")
    if original != _size(record["image_size"], "selected image_size"):
        raise ValueError("geometry original_size differs from selected source image")
    sam_size = _geometry_size(transform.get("sam_input_size"), "sam_input_size")
    region = _vector(transform.get("sam_valid_region"), 4, "sam_valid_region")
    top, left, bottom, right = region
    sam_h, sam_w = sam_size
    pixel_tolerance = max(1e-3, 1e-6 * max(sam_size))
    if (top < -pixel_tolerance or left < -pixel_tolerance
            or bottom > sam_h + pixel_tolerance or right > sam_w + pixel_tolerance
            or bottom <= top or right <= left):
        raise ValueError("sam_valid_region must be nonempty and inside the SAM input canvas")
    matrix = _plain_numeric(transform.get("original_to_sam"))
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise ValueError("original_to_sam must be a 3 x 3 affine matrix")
    matrix = [_vector(row, 3, "original_to_sam row") for row in matrix]
    if (abs(matrix[0][1]) > 1e-7 or abs(matrix[1][0]) > 1e-7
            or abs(matrix[2][0]) > 1e-7 or abs(matrix[2][1]) > 1e-7
            or abs(matrix[2][2] - 1) > 1e-7
            or matrix[0][0] <= 0 or matrix[1][1] <= 0):
        raise ValueError("atlas requires positive axis-aligned resize: no flip, shear, rotation or perspective")
    scale_x, scale_y = matrix[0][0], matrix[1][1]
    offset_x, offset_y = matrix[0][2], matrix[1][2]
    mapped = [offset_y, offset_x,
              offset_y + original[0] * scale_y, offset_x + original[1] * scale_x]
    if any(abs(a - b) > pixel_tolerance for a, b in zip(mapped, region)):
        raise ValueError("full original image does not map to sam_valid_region; crop or inconsistent affine geometry")
    data_dict = batch.get("data_dict", {})
    pixels = data_dict.get("extra_pixel_values") if isinstance(data_dict, Mapping) else None
    if pixels is not None:
        if isinstance(pixels, (list, tuple)):
            if len(pixels) != 1:
                raise ValueError("atlas requires one SAM pixel tensor")
            pixels = pixels[0]
        shape = getattr(pixels, "shape", ())
        if len(shape) not in (3, 4) or list(shape[-2:]) != sam_size:
            raise ValueError("sam_input_size differs from actual SAM pixel tensor")
        if len(shape) == 4 and shape[0] != 1:
            raise ValueError("atlas requires a single-image SAM pixel tensor")
    expected_normalized = [top / sam_h, left / sam_w, bottom / sam_h, right / sam_w]
    if normalized_valid_box is not None:
        normalized = _plain_numeric(normalized_valid_box)
        if isinstance(normalized, (list, tuple)) and len(normalized) == 1:
            normalized = _plain_numeric(normalized[0])
        normalized = _vector(normalized, 4, "captured normalized valid box")
        actual_pixels = [x * divisor for x, divisor in zip(normalized, [sam_h, sam_w, sam_h, sam_w])]
        if any(abs(a - b) > pixel_tolerance for a, b in zip(actual_pixels, region)):
            raise ValueError("captured normalized valid box differs from source SAM geometry")
    return {
        "source_geometry_verified": True,
        "geometry_kind": "full-source positive axis-aligned resize plus padding only",
        "original_size": original, "sam_input_size": sam_size,
        "original_to_sam": matrix, "sam_valid_region": region,
        "sam_valid_boxes_normalized_tlbr": [expected_normalized],
        "pixel_rounding_tolerance": pixel_tolerance,
        "captured_valid_box_verified": normalized_valid_box is not None,
    }
