"""Small, model-free contracts shared only by the new mask-only evaluators."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile


PROTOCOL = "masklat_mask_only_v1"


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink output: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_image_coverage(expected_ids, observed_ids):
    """Require each GT image exactly once, including valid empty predictions."""
    expected_ids, observed_ids = list(expected_ids), list(observed_ids)
    for value in expected_ids + observed_ids:
        if type(value) not in (int, str) or (isinstance(value, str) and not value):
            raise ValueError(f"Invalid image ID: {value!r}")
    expected = Counter((type(value).__name__, value) for value in expected_ids)
    observed = Counter((type(value).__name__, value) for value in observed_ids)
    if not expected or any(count != 1 for count in expected.values()):
        raise ValueError("GT image IDs must be nonempty and unique")
    missing = set(expected) - set(observed)
    unexpected = set(observed) - set(expected)
    duplicate_count = sum(count - 1 for count in observed.values() if count > 1)
    if missing or unexpected or duplicate_count:
        raise ValueError(
            "Incomplete image coverage: "
            f"missing={len(missing)}, duplicate={duplicate_count}, unexpected={len(unexpected)}; "
            f"missing_examples={list(missing)[:5]}, unexpected_examples={list(unexpected)[:5]}"
        )
    return dict(num_ground_truth_images=len(expected_ids), num_prediction_images=len(observed_ids),
                missing_images=0, duplicate_images=0, unexpected_images=0)


def validate_summary(summary, *, expected_data_name=None, expected_world_size=None):
    if not isinstance(summary, dict):
        raise ValueError("Mask summary must be an object")
    required = dict(schema_version=1, evaluation_protocol=PROTOCOL, mask_only=True,
                    caption_scoring=False, complete_dataset=True, metric_unit="percent")
    for name, expected in required.items():
        value = summary.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Invalid mask summary field {name}: {value!r}")
    name = summary.get("data_name")
    if not isinstance(name, str) or not name or (expected_data_name is not None and name != expected_data_name):
        raise ValueError("Mask summary data_name mismatch")
    size = summary.get("world_size")
    if type(size) is not int or size < 1 or (expected_world_size is not None and size != expected_world_size):
        raise ValueError("Mask summary world_size mismatch")
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Missing image coverage")
    for key in ("num_ground_truth_images", "num_prediction_images", "missing_images",
                "duplicate_images", "unexpected_images", "empty_mask_images"):
        value = coverage.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid coverage count {key}")
    count = coverage["num_ground_truth_images"]
    if (count < 1 or coverage["num_prediction_images"] != count or
            any(coverage[key] != 0 for key in ("missing_images", "duplicate_images", "unexpected_images")) or
            coverage["empty_mask_images"] > count):
        raise ValueError("Mask summary is not full-coverage")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Missing mask metrics")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Invalid mask metric name")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError(f"Invalid percent metric {key}: {value!r}; use null only for undefined metrics")
    if not isinstance(summary.get("protocol_details"), dict):
        raise ValueError("Missing metric protocol details")
    return summary


def write_summary(output_dir, *, data_name, metrics, coverage, world_size, protocol_details):
    summary = dict(schema_version=1, evaluation_protocol=PROTOCOL, data_name=data_name,
                   mask_only=True, caption_scoring=False, complete_dataset=True,
                   world_size=world_size, metric_unit="percent", metrics=metrics,
                   coverage=coverage, protocol_details=protocol_details)
    validate_summary(summary)
    if output_dir is not None:
        atomic_json(Path(output_dir) / "summary.json", summary)
    return summary
