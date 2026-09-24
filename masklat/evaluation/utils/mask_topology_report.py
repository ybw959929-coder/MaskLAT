"""Pure-text reporting and serialization for mask-topology diagnostics.

The analysis code is responsible for aggregating every rank before calling this
module.  Values in ``report`` use their natural units:

* cIoU, gIoU, deltas, improve/worsen rates, and named ratios are in ``[0, 1]``;
* pairwise IoU values and count-like metrics are not scaled;
* runtime is supplied as ``runtime_minutes`` (``runtime_seconds`` is also
  accepted).

Only :func:`emit_report` writes ``table.txt`` or stdout.  It deliberately uses
one ``write`` and one ``flush`` so a redirected log receives the same complete
table as the file.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO


_MISSING = object()
_SEPARATOR = "=" * 60


class MaskTopologyReportError(RuntimeError):
    """Base class for mask-topology reporting errors."""


class ReportSchemaError(MaskTopologyReportError, ValueError):
    """Raised when a report is missing a value required by the fixed table."""


class MaskTopologySanityError(MaskTopologyReportError):
    """Raised after the complete table has been emitted with failed checks."""

    def __init__(self, failures: Sequence[str], table_path: os.PathLike[str] | str):
        self.failures = tuple(failures)
        self.table_path = Path(table_path)
        details = "; ".join(self.failures)
        super().__init__(
            f"Mask-topology sanity checks failed after writing "
            f"{self.table_path}: {details}"
        )


def _pick(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
    default: Any = _MISSING,
) -> Any:
    if not isinstance(mapping, Mapping):
        raise ReportSchemaError(f"{label} must be a mapping")
    for name in names:
        if name in mapping:
            return mapping[name]
    if default is not _MISSING:
        return default
    joined = " / ".join(names)
    raise ReportSchemaError(f"missing {label} ({joined})")


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ReportSchemaError(f"{label} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ReportSchemaError(f"{label} must be finite, got {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ReportSchemaError(f"{label} must be an integer, got {value!r}")
    return int(value)


def _boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return bool(value)
    raise ReportSchemaError(f"{label} must be boolean, got {value!r}")


def _all_booleans(value: Any, *, label: str) -> bool:
    if isinstance(value, Mapping):
        if not value:
            raise ReportSchemaError(f"{label} must not be empty")
        return all(
            _boolean(item, label=f"{label}[{key!r}]")
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ReportSchemaError(f"{label} must not be empty")
        return all(
            _boolean(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    return _boolean(value, label=label)


def _mean_metric(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> float:
    value = _pick(mapping, names, label=label)
    if isinstance(value, Mapping):
        value = _pick(value, ("mean", "value"), label=f"{label}.mean")
    return _finite_number(value, label=label)


def _metric_stat(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    stat_names: Sequence[str],
    *,
    label: str,
) -> float:
    value = _pick(mapping, names, label=label)
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"{label} must be a mapping with statistics")
    result = _pick(value, stat_names, label=f"{label}.{stat_names[0]}")
    return _finite_number(result, label=f"{label}.{stat_names[0]}")


def _ciou(metric: Mapping[str, Any], *, label: str) -> float:
    value = _pick(metric, ("cIoU", "ciou", "c_iou"), label=f"{label}.cIoU")
    return _finite_number(value, label=f"{label}.cIoU")


def _giou(metric: Mapping[str, Any], *, label: str) -> float:
    value = _pick(metric, ("gIoU", "giou", "g_iou"), label=f"{label}.gIoU")
    return _finite_number(value, label=f"{label}.gIoU")


def _delta(
    metric: Mapping[str, Any],
    names: Sequence[str],
    *,
    fallback: float,
    label: str,
) -> float:
    value = _pick(metric, names, label=label, default=fallback)
    return _finite_number(value, label=label)


def _rate(
    metric: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
    default: Any = _MISSING,
) -> float:
    value = _pick(metric, names, label=label, default=default)
    return _finite_number(value, label=label)


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}"


def _signed_points(value: float) -> str:
    return f"{value * 100.0:+.2f}"


def _plain(value: float) -> str:
    return f"{value:.2f}"


def _absolute_checkpoint(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ReportSchemaError("checkpoint must be a path string")
    text = os.fspath(value)
    if not text:
        raise ReportSchemaError("checkpoint must not be empty")
    return os.path.abspath(os.path.expanduser(text))


def _mask_size_text(value: Any) -> str:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        side = int(value)
        if side <= 0:
            raise ReportSchemaError("mask_size must be positive")
        return f"{side}x{side}"
    if isinstance(value, str):
        text = value.lower().replace(" ", "")
        parts = text.split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ReportSchemaError(f"invalid mask_size string: {value!r}")
        height, width = (int(part) for part in parts)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ReportSchemaError("mask_size sequence must contain H and W")
        height = _integer(value[0], label="mask_size[0]")
        width = _integer(value[1], label="mask_size[1]")
    else:
        raise ReportSchemaError(f"invalid mask_size: {value!r}")
    if height <= 0 or width <= 0:
        raise ReportSchemaError("mask_size dimensions must be positive")
    return f"{height}x{width}"


def _runtime_minutes(report: Mapping[str, Any]) -> float:
    if "runtime_minutes" in report:
        return _finite_number(report["runtime_minutes"], label="runtime_minutes")
    if "runtime_seconds" in report:
        seconds = _finite_number(report["runtime_seconds"], label="runtime_seconds")
        return seconds / 60.0
    value = _pick(report, ("runtime",), label="runtime_minutes")
    return _finite_number(value, label="runtime_minutes")


def _threshold_number(value: Any, *, label: str) -> float:
    if isinstance(value, str):
        text = value
        if text.startswith("threshold_"):
            text = text[len("threshold_") :]
        try:
            result = float(text)
        except ValueError as error:
            raise ReportSchemaError(f"{label} is not numeric: {value!r}") from error
    else:
        result = _finite_number(value, label=label)
    if not math.isfinite(result):
        raise ReportSchemaError(f"{label} must be finite")
    return result


def _threshold_rows(report: Mapping[str, Any]) -> list[tuple[float, Mapping[str, Any]]]:
    raw = _pick(
        report,
        ("thresholds", "threshold_metrics", "topology_thresholds"),
        label="thresholds",
    )
    rows: list[tuple[float, Mapping[str, Any]]] = []
    if isinstance(raw, Mapping):
        for key, metrics in raw.items():
            if not isinstance(metrics, Mapping):
                raise ReportSchemaError(f"thresholds[{key!r}] must be a mapping")
            threshold_value = metrics.get("threshold", key)
            threshold = _threshold_number(
                threshold_value, label=f"thresholds[{key!r}].threshold"
            )
            rows.append((threshold, metrics))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for index, metrics in enumerate(raw):
            if not isinstance(metrics, Mapping):
                raise ReportSchemaError(f"thresholds[{index}] must be a mapping")
            threshold_value = _pick(
                metrics,
                ("threshold", "group_threshold"),
                label=f"thresholds[{index}].threshold",
            )
            threshold = _threshold_number(
                threshold_value, label=f"thresholds[{index}].threshold"
            )
            rows.append((threshold, metrics))
    else:
        raise ReportSchemaError("thresholds must be a mapping or sequence")
    if not rows:
        raise ReportSchemaError("thresholds must not be empty")
    rows.sort(key=lambda item: item[0])
    for index in range(1, len(rows)):
        if math.isclose(rows[index - 1][0], rows[index][0], abs_tol=1e-12):
            raise ReportSchemaError(f"duplicate threshold {rows[index][0]!r}")
    return rows


@dataclasses.dataclass(frozen=True)
class _SanitySnapshot:
    top_score_matches: bool
    query_id_mismatch_count: int
    mask_mismatch_count: int
    ciou_absolute_difference: float
    giou_absolute_difference: float
    all_queries_assigned_exactly_once: bool
    distributed_total_samples_correct: bool
    nan_or_inf_count: int
    tolerance: float


def _sanity_snapshot(report: Mapping[str, Any]) -> _SanitySnapshot:
    sanity = _pick(report, ("sanity", "sanity_checks"), label="sanity")
    if not isinstance(sanity, Mapping):
        raise ReportSchemaError("sanity must be a mapping")

    query_mismatch = _integer(
        _pick(
            sanity,
            (
                "selected_query_id_mismatch_count",
                "query_id_mismatch_count",
            ),
            label="sanity.selected_query_id_mismatch_count",
        ),
        label="sanity.selected_query_id_mismatch_count",
    )
    mask_mismatch = _integer(
        _pick(
            sanity,
            (
                "selected_mask_mismatch_count",
                "mask_mismatch_count",
            ),
            label="sanity.selected_mask_mismatch_count",
        ),
        label="sanity.selected_mask_mismatch_count",
    )
    ciou_difference = _finite_number(
        _pick(
            sanity,
            (
                "cIoU_absolute_difference",
                "ciou_absolute_difference",
            ),
            label="sanity.cIoU_absolute_difference",
        ),
        label="sanity.cIoU_absolute_difference",
    )
    giou_difference = _finite_number(
        _pick(
            sanity,
            (
                "gIoU_absolute_difference",
                "giou_absolute_difference",
            ),
            label="sanity.gIoU_absolute_difference",
        ),
        label="sanity.gIoU_absolute_difference",
    )
    tolerance = _finite_number(
        _pick(
            sanity,
            ("absolute_difference_tolerance", "tolerance"),
            label="sanity.absolute_difference_tolerance",
            default=report.get("sanity_tolerance", 1e-6),
        ),
        label="sanity.absolute_difference_tolerance",
    )
    if tolerance < 0:
        raise ReportSchemaError("sanity tolerance must be non-negative")

    explicit_match = _pick(
        sanity,
        (
            "selected_group_top_score_equals_selected_query",
            "selected_group_top_score_query_matches_selected_query",
            "selected_group_top_score_query_mask_equals_selected_query_mask",
            "selected_group_top_score == selected_query",
        ),
        label="sanity.selected_group_top_score_equals_selected_query",
        default=True,
    )
    top_score_matches = (
        _all_booleans(
            explicit_match,
            label="sanity.selected_group_top_score_equals_selected_query",
        )
        and query_mismatch == 0
        and mask_mismatch == 0
        and ciou_difference <= tolerance
        and giou_difference <= tolerance
    )

    assigned = _all_booleans(
        _pick(
            sanity,
            ("all_queries_assigned_exactly_once",),
            label="sanity.all_queries_assigned_exactly_once",
        ),
        label="sanity.all_queries_assigned_exactly_once",
    )
    distributed_total = _all_booleans(
        _pick(
            sanity,
            ("distributed_total_samples_correct",),
            label="sanity.distributed_total_samples_correct",
        ),
        label="sanity.distributed_total_samples_correct",
    )
    nan_count = _integer(
        _pick(
            sanity,
            ("nan_or_inf_count",),
            label="sanity.nan_or_inf_count",
        ),
        label="sanity.nan_or_inf_count",
    )

    return _SanitySnapshot(
        top_score_matches=top_score_matches,
        query_id_mismatch_count=query_mismatch,
        mask_mismatch_count=mask_mismatch,
        ciou_absolute_difference=ciou_difference,
        giou_absolute_difference=giou_difference,
        all_queries_assigned_exactly_once=assigned,
        distributed_total_samples_correct=distributed_total,
        nan_or_inf_count=nan_count,
        tolerance=tolerance,
    )


def sanity_failures(report: Mapping[str, Any]) -> list[str]:
    """Return all critical sanity-check failures without emitting output."""

    sanity = _sanity_snapshot(report)
    failures: list[str] = []
    if not sanity.top_score_matches:
        failures.append("selected_group_top_score != selected_query")
    if sanity.query_id_mismatch_count != 0:
        failures.append(
            "selected_query_id_mismatch_count="
            f"{sanity.query_id_mismatch_count}"
        )
    if sanity.mask_mismatch_count != 0:
        failures.append(
            f"selected_mask_mismatch_count={sanity.mask_mismatch_count}"
        )
    if sanity.ciou_absolute_difference > sanity.tolerance:
        failures.append(
            "cIoU_absolute_difference="
            f"{sanity.ciou_absolute_difference:.9g}"
        )
    if sanity.giou_absolute_difference > sanity.tolerance:
        failures.append(
            "gIoU_absolute_difference="
            f"{sanity.giou_absolute_difference:.9g}"
        )
    if not sanity.all_queries_assigned_exactly_once:
        failures.append("all_queries_assigned_exactly_once=False")
    if not sanity.distributed_total_samples_correct:
        failures.append("distributed_total_samples_correct=False")
    if sanity.nan_or_inf_count != 0:
        failures.append(f"nan_or_inf_count={sanity.nan_or_inf_count}")
    two_stage_sanity = _two_stage_sanity_passed(report)
    if two_stage_sanity is False:
        failures.append("two_stage_oracle_decomposition.sanity.all_passed=False")
    return failures


def _baseline_parts(
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    baseline = _pick(
        report,
        ("baseline", "query_baseline"),
        label="baseline",
    )
    if not isinstance(baseline, Mapping):
        raise ReportSchemaError("baseline must be a mapping")
    selected = _pick(
        baseline,
        ("selected_query_mask",),
        label="baseline.selected_query_mask",
    )
    oracle = _pick(
        baseline,
        ("global_200q_oracle_query_mask",),
        label="baseline.global_200q_oracle_query_mask",
    )
    if not isinstance(selected, Mapping) or not isinstance(oracle, Mapping):
        raise ReportSchemaError("baseline variants must be mappings")
    return selected, oracle


def _raw_parts(report: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = _pick(
        report,
        ("raw_query", "raw_query_diagnostics"),
        label="raw_query",
    )
    if not isinstance(raw, Mapping):
        raise ReportSchemaError("raw_query must be a mapping")
    query_dup7 = _mean_metric(
        raw, ("query_dup7",), label="raw_query.query_dup7"
    )
    query_pair7_value = _pick(
        raw, ("query_pair7",), label="raw_query.query_pair7"
    )
    if isinstance(query_pair7_value, Mapping):
        query_pair7 = _finite_number(
            _pick(
                query_pair7_value,
                ("mean", "value"),
                label="raw_query.query_pair7.mean",
            ),
            label="raw_query.query_pair7.mean",
        )
        pair_defined = _finite_number(
            _pick(
                query_pair7_value,
                ("defined_sample_ratio", "defined_ratio"),
                label="raw_query.query_pair7.defined_sample_ratio",
            ),
            label="raw_query.query_pair7.defined_sample_ratio",
        )
    else:
        query_pair7 = _finite_number(
            query_pair7_value, label="raw_query.query_pair7"
        )
        pair_defined = _rate(
            raw,
            ("query_pair7_defined_sample_ratio", "query_pair7_defined_ratio"),
            label="raw_query.query_pair7_defined_sample_ratio",
        )
    good_sample = _rate(
        raw,
        ("good_query_sample_ratio",),
        label="raw_query.good_query_sample_ratio",
    )
    return query_dup7, query_pair7, pair_defined, good_sample


def _threshold_structure(
    metrics: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    groups = _metric_stat(
        metrics,
        ("topology_group_count",),
        ("mean",),
        label="topology_group_count",
    )
    group_size = _mean_metric(
        metrics,
        ("topology_mean_group_size",),
        label="topology_mean_group_size",
    )
    singleton = _mean_metric(
        metrics,
        ("topology_singleton_ratio",),
        label="topology_singleton_ratio",
    )
    largest = _mean_metric(
        metrics,
        ("topology_largest_group_size",),
        label="topology_largest_group_size",
    )
    good_groups = _metric_stat(
        metrics,
        ("good_query_group_count7",),
        ("mean",),
        label="good_query_group_count7",
    )
    eq1 = _metric_stat(
        metrics,
        ("good_query_group_count7",),
        ("ratio_eq_1", "eq1_ratio"),
        label="good_query_group_count7",
    )
    le2 = _metric_stat(
        metrics,
        ("good_query_group_count7",),
        ("ratio_le_2", "le2_ratio"),
        label="good_query_group_count7",
    )
    concentration = _mean_metric(
        metrics,
        ("good_query_concentration7",),
        label="good_query_concentration7",
    )
    purity = _mean_metric(
        metrics,
        ("good_query_group_purity7",),
        label="good_query_group_purity7",
    )
    return (
        groups,
        group_size,
        singleton,
        largest,
        good_groups,
        eq1,
        le2,
        concentration,
        purity,
    )


def _threshold_duplicate_bridge(
    metrics: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    medoid_dup = _mean_metric(
        metrics, ("group_medoid_dup7",), label="group_medoid_dup7"
    )
    medoid_pair_value = _pick(
        metrics, ("group_medoid_pair7",), label="group_medoid_pair7"
    )
    if not isinstance(medoid_pair_value, Mapping):
        raise ReportSchemaError(
            "group_medoid_pair7 must contain mean and defined_sample_ratio"
        )
    medoid_pair = _finite_number(
        _pick(
            medoid_pair_value,
            ("mean", "value"),
            label="group_medoid_pair7.mean",
        ),
        label="group_medoid_pair7.mean",
    )
    pair_defined = _finite_number(
        _pick(
            medoid_pair_value,
            ("defined_sample_ratio", "defined_ratio"),
            label="group_medoid_pair7.defined_sample_ratio",
        ),
        label="group_medoid_pair7.defined_sample_ratio",
    )
    bridge = _mean_metric(
        metrics,
        ("topology_bridge_group_ratio",),
        label="topology_bridge_group_ratio",
    )
    medoid_min_mean = _metric_stat(
        metrics,
        ("topology_medoid_min_iou",),
        ("mean",),
        label="topology_medoid_min_iou",
    )
    medoid_min_p10 = _metric_stat(
        metrics,
        ("topology_medoid_min_iou",),
        ("p10", "P10"),
        label="topology_medoid_min_iou",
    )
    pair_min_mean = _metric_stat(
        metrics,
        ("topology_group_min_pair_iou",),
        ("mean",),
        label="topology_group_min_pair_iou",
    )
    pair_min_p10 = _metric_stat(
        metrics,
        ("topology_group_min_pair_iou",),
        ("p10", "P10"),
        label="topology_group_min_pair_iou",
    )
    bad01 = _mean_metric(
        metrics,
        ("member_gt_iou_lt_01_ratio",),
        label="member_gt_iou_lt_01_ratio",
    )
    bad03 = _mean_metric(
        metrics,
        ("member_gt_iou_lt_03_ratio",),
        label="member_gt_iou_lt_03_ratio",
    )
    return (
        medoid_dup,
        medoid_pair,
        pair_defined,
        bridge,
        medoid_min_mean,
        medoid_min_p10,
        pair_min_mean,
        pair_min_p10,
        bad01,
        bad03,
    )


def _threshold_quality(
    metrics: Mapping[str, Any],
    selected_query: Mapping[str, Any],
    oracle_query: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float, float]:
    selected_medoid = _pick(
        metrics,
        ("selected_group_medoid_mask",),
        label="selected_group_medoid_mask",
    )
    oracle_medoid = _pick(
        metrics,
        ("global_group_medoid_oracle_mask",),
        label="global_group_medoid_oracle_mask",
    )
    if not isinstance(selected_medoid, Mapping) or not isinstance(
        oracle_medoid, Mapping
    ):
        raise ReportSchemaError("group medoid quality variants must be mappings")

    selected_c = _ciou(selected_medoid, label="selected_group_medoid_mask")
    selected_g = _giou(selected_medoid, label="selected_group_medoid_mask")
    oracle_c = _ciou(oracle_medoid, label="global_group_medoid_oracle_mask")
    oracle_g = _giou(oracle_medoid, label="global_group_medoid_oracle_mask")
    selected_dc = _delta(
        selected_medoid,
        ("dC", "delta_cIoU", "delta_ciou", "delta_c"),
        fallback=selected_c - _ciou(selected_query, label="selected_query_mask"),
        label="selected_group_medoid_mask.dC",
    )
    selected_dg = _delta(
        selected_medoid,
        ("dG", "delta_gIoU", "delta_giou", "delta_g"),
        fallback=selected_g - _giou(selected_query, label="selected_query_mask"),
        label="selected_group_medoid_mask.dG",
    )
    drop_c = _delta(
        oracle_medoid,
        (
            "dropQ-c",
            "dropQ_c",
            "drop_query_cIoU",
            "drop_query_ciou",
            "delta_to_global_query_cIoU",
        ),
        fallback=oracle_c - _ciou(oracle_query, label="global_200q_oracle_query_mask"),
        label="global_group_medoid_oracle_mask.dropQ-c",
    )
    drop_g = _delta(
        oracle_medoid,
        (
            "dropQ-g",
            "dropQ_g",
            "drop_query_gIoU",
            "drop_query_giou",
            "delta_to_global_query_gIoU",
        ),
        fallback=oracle_g - _giou(oracle_query, label="global_200q_oracle_query_mask"),
        label="global_group_medoid_oracle_mask.dropQ-g",
    )
    return (
        selected_c,
        selected_g,
        selected_dc,
        selected_dg,
        oracle_c,
        oracle_g,
        drop_c,
        drop_g,
    )


def _required_mapping(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    value = _pick(mapping, names, label=label)
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"{label} must be a mapping")
    return value


def _two_stage_sanity_passed(report: Mapping[str, Any]) -> bool | None:
    if "two_stage_oracle_decomposition" not in report:
        return None
    decomposition = _required_mapping(
        report,
        ("two_stage_oracle_decomposition",),
        label="two_stage_oracle_decomposition",
    )
    sanity = _required_mapping(
        decomposition,
        ("sanity",),
        label="two_stage_oracle_decomposition.sanity",
    )
    return _boolean(
        _pick(
            sanity,
            ("all_passed",),
            label="two_stage_oracle_decomposition.sanity.all_passed",
        ),
        label="two_stage_oracle_decomposition.sanity.all_passed",
    )


def _two_stage_decomposition_lines(report: Mapping[str, Any]) -> list[str]:
    """Format the optional two-stage oracle error decomposition.

    This section is deliberately optional so reports produced before the
    diagnostic was added retain their original seven-section layout.
    """

    if "two_stage_oracle_decomposition" not in report:
        return []
    decomposition = _required_mapping(
        report,
        ("two_stage_oracle_decomposition",),
        label="two_stage_oracle_decomposition",
    )
    topology_threshold = _finite_number(
        _pick(
            decomposition,
            ("topology_threshold",),
            label="two_stage_oracle_decomposition.topology_threshold",
        ),
        label="two_stage_oracle_decomposition.topology_threshold",
    )
    gt_threshold = _finite_number(
        _pick(
            decomposition,
            ("gt_iou_threshold",),
            label="two_stage_oracle_decomposition.gt_iou_threshold",
        ),
        label="two_stage_oracle_decomposition.gt_iou_threshold",
    )

    variants = _required_mapping(
        decomposition,
        ("variants",),
        label="two_stage_oracle_decomposition.variants",
    )
    variant_names = (
        "selected_query_mask",
        "selected_group_oracle_query_mask",
        "global_200q_oracle_query_mask",
    )
    variant_rows: list[tuple[str, float, float, float, float, bool]] = []
    for name in variant_names:
        label = f"two_stage_oracle_decomposition.variants.{name}"
        metric = _required_mapping(variants, (name,), label=label)
        ciou = _ciou(metric, label=label)
        giou = _giou(metric, label=label)
        dc = _delta(
            metric,
            ("dC", "delta_cIoU", "delta_ciou", "delta_c"),
            fallback=0.0,
            label=f"{label}.dC",
        )
        dg = _delta(
            metric,
            ("dG", "delta_gIoU", "delta_giou", "delta_g"),
            fallback=0.0,
            label=f"{label}.dG",
        )
        gt_select = _boolean(
            _pick(
                metric,
                ("gt_select", "GT-select"),
                label=f"{label}.gt_select",
            ),
            label=f"{label}.gt_select",
        )
        variant_rows.append((name, ciou, giou, dc, dg, gt_select))

    gaps = _required_mapping(
        decomposition,
        ("gaps",),
        label="two_stage_oracle_decomposition.gaps",
    )
    gap_names = (
        "within_group_member_gap",
        "cross_group_routing_gap",
        "total_oracle_gap",
    )
    gap_rows: list[tuple[str, float, float]] = []
    for name in gap_names:
        label = f"two_stage_oracle_decomposition.gaps.{name}"
        metric = _required_mapping(gaps, (name,), label=label)
        gap_rows.append(
            (name, _ciou(metric, label=label), _giou(metric, label=label))
        )
    member_share = _finite_number(
        _pick(
            gaps,
            ("member_gap_share",),
            label="two_stage_oracle_decomposition.gaps.member_gap_share",
        ),
        label="two_stage_oracle_decomposition.gaps.member_gap_share",
    )
    routing_share = _finite_number(
        _pick(
            gaps,
            ("routing_gap_share",),
            label="two_stage_oracle_decomposition.gaps.routing_gap_share",
        ),
        label="two_stage_oracle_decomposition.gaps.routing_gap_share",
    )

    categories = _required_mapping(
        decomposition,
        ("categories",),
        label="two_stage_oracle_decomposition.categories",
    )
    category_names = (
        "selected_good7",
        "wrong_member7",
        "wrong_group7",
        "no_good_query7",
    )
    category_rows: list[tuple[str, int, float, float, float, float]] = []
    for name in category_names:
        label = f"two_stage_oracle_decomposition.categories.{name}"
        category = _required_mapping(categories, (name,), label=label)
        samples = _integer(
            _pick(category, ("samples",), label=f"{label}.samples"),
            label=f"{label}.samples",
        )
        ratio = _finite_number(
            _pick(category, ("ratio",), label=f"{label}.ratio"),
            label=f"{label}.ratio",
        )
        mean_a = _finite_number(
            _pick(category, ("mean_a",), label=f"{label}.mean_a"),
            label=f"{label}.mean_a",
        )
        mean_b = _finite_number(
            _pick(category, ("mean_b",), label=f"{label}.mean_b"),
            label=f"{label}.mean_b",
        )
        mean_c = _finite_number(
            _pick(category, ("mean_c",), label=f"{label}.mean_c"),
            label=f"{label}.mean_c",
        )
        category_rows.append(
            (name, samples, ratio, mean_a, mean_b, mean_c)
        )
    total_category = _required_mapping(
        categories,
        ("total",),
        label="two_stage_oracle_decomposition.categories.total",
    )
    category_total_samples = _integer(
        _pick(
            total_category,
            ("samples",),
            label="two_stage_oracle_decomposition.categories.total.samples",
        ),
        label="two_stage_oracle_decomposition.categories.total.samples",
    )
    category_total_ratio = _finite_number(
        _pick(
            total_category,
            ("ratio",),
            label="two_stage_oracle_decomposition.categories.total.ratio",
        ),
        label="two_stage_oracle_decomposition.categories.total.ratio",
    )

    routing = _required_mapping(
        decomposition,
        ("routing",),
        label="two_stage_oracle_decomposition.routing",
    )
    same_group = _finite_number(
        _pick(
            routing,
            ("selected_group_same_as_global_oracle_group",),
            label=(
                "two_stage_oracle_decomposition.routing."
                "selected_group_same_as_global_oracle_group"
            ),
        ),
        label=(
            "two_stage_oracle_decomposition.routing."
            "selected_group_same_as_global_oracle_group"
        ),
    )
    contains_good = _finite_number(
        _pick(
            routing,
            ("selected_group_contains_good7",),
            label=(
                "two_stage_oracle_decomposition.routing."
                "selected_group_contains_good7"
            ),
        ),
        label=(
            "two_stage_oracle_decomposition.routing."
            "selected_group_contains_good7"
        ),
    )
    first_good = _required_mapping(
        routing,
        ("first_good_group_rank7",),
        label=(
            "two_stage_oracle_decomposition.routing."
            "first_good_group_rank7"
        ),
    )
    rank_mean = _finite_number(
        _pick(first_good, ("mean",), label="first_good_group_rank7.mean"),
        label="first_good_group_rank7.mean",
    )
    rank_median = _finite_number(
        _pick(
            first_good,
            ("median",),
            label="first_good_group_rank7.median",
        ),
        label="first_good_group_rank7.median",
    )
    rank_top1 = _finite_number(
        _pick(
            first_good,
            ("top1_ratio",),
            label="first_good_group_rank7.top1_ratio",
        ),
        label="first_good_group_rank7.top1_ratio",
    )
    rank_top3 = _finite_number(
        _pick(
            first_good,
            ("top3_ratio",),
            label="first_good_group_rank7.top3_ratio",
        ),
        label="first_good_group_rank7.top3_ratio",
    )
    rank_top5 = _finite_number(
        _pick(
            first_good,
            ("top5_ratio",),
            label="first_good_group_rank7.top5_ratio",
        ),
        label="first_good_group_rank7.top5_ratio",
    )
    two_stage_sanity = _two_stage_sanity_passed(report)
    assert two_stage_sanity is not None

    lines = [
        "",
        (
            "[8] Two-stage Oracle Decomposition "
            f"(Topology threshold={topology_threshold:.2f})"
        ),
        "",
        (
            f"{'variant':<36}{'cIoU':>7}{'gIoU':>7}{'dC':>7}"
            f"{'dG':>7}{'GT-select':>11}"
        ),
    ]
    for index, (name, ciou, giou, dc, dg, gt_select) in enumerate(
        variant_rows
    ):
        dc_text = f"{dc * 100.0:.2f}" if index == 0 else _signed_points(dc)
        dg_text = f"{dg * 100.0:.2f}" if index == 0 else _signed_points(dg)
        lines.append(
            f"{name:<36}{_percent(ciou):>7}{_percent(giou):>7}"
            f"{dc_text:>7}{dg_text:>7}{str(gt_select):>11}"
        )

    lines.extend(
        [
            "",
            "[9] Gap Decomposition",
            "",
            f"{'metric':<40}{'value':>12}",
        ]
    )
    for name, ciou, giou in gap_rows:
        lines.append(
            f"{name + '_cIoU':<40}{_percent(ciou):>12}"
        )
        lines.append(
            f"{name + '_gIoU':<40}{_percent(giou):>12}"
        )
    lines.append(
        f"{'member_gap_share':<40}{(_percent(member_share) + '%'):>12}"
    )
    lines.append(
        f"{'routing_gap_share':<40}{(_percent(routing_share) + '%'):>12}"
    )

    lines.extend(
        [
            "",
            (
                "[10] Sample Category Diagnostics "
                f"(GT IoU threshold={gt_threshold:.2f})"
            ),
            "",
            (
                f"{'category':<22}{'samples':>9}{'percent':>10}"
                f"{'mean-A':>10}{'mean-B':>10}{'mean-C':>10}"
            ),
        ]
    )
    for name, samples, ratio, mean_a, mean_b, mean_c in category_rows:
        lines.append(
            f"{name:<22}{samples:>9d}{_percent(ratio):>10}"
            f"{mean_a:>10.2f}{mean_b:>10.2f}{mean_c:>10.2f}"
        )
    lines.append(
        f"{'total':<22}{category_total_samples:>9d}"
        f"{_percent(category_total_ratio):>10}"
    )

    lines.extend(
        [
            "",
            "[11] Group Routing Diagnostics",
            "",
            f"{'metric':<48}{'value':>12}",
            (
                f"{'selected_group_same_as_global_oracle_group':<48}"
                f"{(_percent(same_group) + '%'):>12}"
            ),
            (
                f"{'selected_group_contains_good7':<48}"
                f"{(_percent(contains_good) + '%'):>12}"
            ),
            (
                f"{'first_good_group_rank7_mean':<48}"
                f"{_plain(rank_mean):>12}"
            ),
            (
                f"{'first_good_group_rank7_median':<48}"
                f"{_plain(rank_median):>12}"
            ),
            (
                f"{'first_good_group_top1':<48}"
                f"{(_percent(rank_top1) + '%'):>12}"
            ),
            (
                f"{'first_good_group_top3':<48}"
                f"{(_percent(rank_top3) + '%'):>12}"
            ),
            (
                f"{'first_good_group_top5':<48}"
                f"{(_percent(rank_top5) + '%'):>12}"
            ),
            (
                f"{'two_stage_sanity_checks':<48}"
                f"{'PASS' if two_stage_sanity else 'FAIL':>12}"
            ),
        ]
    )
    return lines


def build_table_text(report: Mapping[str, Any]) -> str:
    """Build the complete fixed-width terminal table.

    The function has no side effects and does not raise for failed sanity
    results.  Schema errors still raise immediately because a complete,
    truthful table cannot be constructed from missing metrics.  Reports with
    ``two_stage_oracle_decomposition`` receive four additional sections;
    legacy reports retain the original seven sections.
    """

    if not isinstance(report, Mapping):
        raise ReportSchemaError("report must be a mapping")
    dataset = _pick(report, ("dataset",), label="dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ReportSchemaError("dataset must be a non-empty string")
    checkpoint = _absolute_checkpoint(
        _pick(report, ("checkpoint",), label="checkpoint")
    )
    total_samples = _integer(
        _pick(report, ("total_samples",), label="total_samples"),
        label="total_samples",
    )
    world_size = _integer(
        _pick(report, ("world_size",), label="world_size"),
        label="world_size",
    )
    num_queries = _integer(
        _pick(report, ("num_queries", "query_count"), label="num_queries"),
        label="num_queries",
    )
    mask_size = _mask_size_text(
        _pick(report, ("mask_size",), label="mask_size")
    )
    mask_threshold = _finite_number(
        _pick(report, ("mask_threshold",), label="mask_threshold"),
        label="mask_threshold",
    )
    runtime_minutes = _runtime_minutes(report)
    selected, oracle = _baseline_parts(report)
    selected_c = _ciou(selected, label="selected_query_mask")
    selected_g = _giou(selected, label="selected_query_mask")
    oracle_c = _ciou(oracle, label="global_200q_oracle_query_mask")
    oracle_g = _giou(oracle, label="global_200q_oracle_query_mask")
    selected_dc = _delta(
        selected,
        ("dC", "delta_cIoU", "delta_ciou", "delta_c"),
        fallback=0.0,
        label="selected_query_mask.dC",
    )
    selected_dg = _delta(
        selected,
        ("dG", "delta_gIoU", "delta_giou", "delta_g"),
        fallback=0.0,
        label="selected_query_mask.dG",
    )
    selected_improve = _rate(
        selected,
        ("improve_ratio", "improve"),
        label="selected_query_mask.improve_ratio",
        default=0.0,
    )
    selected_worsen = _rate(
        selected,
        ("worsen_ratio", "worsen"),
        label="selected_query_mask.worsen_ratio",
        default=0.0,
    )
    oracle_dc = _delta(
        oracle,
        ("dC", "delta_cIoU", "delta_ciou", "delta_c"),
        fallback=oracle_c - selected_c,
        label="global_200q_oracle_query_mask.dC",
    )
    oracle_dg = _delta(
        oracle,
        ("dG", "delta_gIoU", "delta_giou", "delta_g"),
        fallback=oracle_g - selected_g,
        label="global_200q_oracle_query_mask.dG",
    )
    oracle_improve = _rate(
        oracle,
        ("improve_ratio", "improve"),
        label="global_200q_oracle_query_mask.improve_ratio",
    )
    oracle_worsen = _rate(
        oracle,
        ("worsen_ratio", "worsen"),
        label="global_200q_oracle_query_mask.worsen_ratio",
    )
    selected_gt = _boolean(
        _pick(
            selected,
            ("gt_select", "GT-select"),
            label="selected_query_mask.gt_select",
            default=False,
        ),
        label="selected_query_mask.gt_select",
    )
    oracle_gt = _boolean(
        _pick(
            oracle,
            ("gt_select", "GT-select"),
            label="global_200q_oracle_query_mask.gt_select",
            default=True,
        ),
        label="global_200q_oracle_query_mask.gt_select",
    )
    raw_dup, raw_pair, raw_pair_defined, raw_good_sample = _raw_parts(report)
    threshold_rows = _threshold_rows(report)
    sanity = _sanity_snapshot(report)

    lines: list[str] = [
        _SEPARATOR,
        "MaskLAT Mask-Topology Query Grouping",
        _SEPARATOR,
        f"{'dataset':<14}: {dataset}",
        f"{'checkpoint':<14}: {checkpoint}",
        f"{'total_samples':<14}: {total_samples}",
        f"{'world_size':<14}: {world_size}",
        f"{'num_queries':<14}: {num_queries}",
        f"{'mask_size':<14}: {mask_size}",
        f"{'mask_threshold':<14}: {mask_threshold:.2f}",
        f"{'runtime':<14}: {runtime_minutes:.2f} min",
        _SEPARATOR,
        "",
        "[1] MaskLAT Query Baseline",
        "",
        (
            f"{'variant':<36}{'cIoU':>7}{'gIoU':>7}{'dC':>7}{'dG':>7}"
            f"{'improve%':>10}{'worsen%':>9}{'GT-select':>10}"
        ),
        (
            f"{'selected_query_mask':<36}{_percent(selected_c):>7}"
            f"{_percent(selected_g):>7}{_signed_points(selected_dc):>7}"
            f"{_signed_points(selected_dg):>7}{_percent(selected_improve):>10}"
            f"{_percent(selected_worsen):>9}{str(selected_gt):>10}"
        ),
        (
            f"{'global_200q_oracle_query_mask':<36}{_percent(oracle_c):>7}"
            f"{_percent(oracle_g):>7}{_signed_points(oracle_dc):>7}"
            f"{_signed_points(oracle_dg):>7}{_percent(oracle_improve):>10}"
            f"{_percent(oracle_worsen):>9}{str(oracle_gt):>10}"
        ),
        "",
        "[2] Raw Query Duplicate Diagnostics",
        "",
        f"{'metric':<30}{'value':>10}",
        f"{'query_dup7':<30}{_plain(raw_dup):>10}",
        f"{'query_pair7':<30}{_plain(raw_pair):>10}",
        f"{'query_pair7_defined%':<30}{_percent(raw_pair_defined):>10}",
        f"{'good_query_sample%':<30}{_percent(raw_good_sample):>10}",
        "",
        "[3] Topology Group Structure",
        "",
        (
            f"{'thr':<6}{'groups':>8}{'grpSize':>8}{'single%':>9}"
            f"{'largest':>8}{'goodGrp7':>9}{'eq1%':>7}{'le2%':>7}"
            f"{'conc7%':>8}{'purity7%':>9}"
        ),
    ]

    structure_by_threshold: dict[float, tuple[float, ...]] = {}
    duplicate_by_threshold: dict[float, tuple[float, ...]] = {}
    quality_by_threshold: dict[float, tuple[float, ...]] = {}
    for threshold, metrics in threshold_rows:
        values = _threshold_structure(metrics)
        structure_by_threshold[threshold] = values
        (
            groups,
            group_size,
            singleton,
            largest,
            good_groups,
            eq1,
            le2,
            concentration,
            purity,
        ) = values
        lines.append(
            f"{threshold:<6.2f}{_plain(groups):>8}{_plain(group_size):>8}"
            f"{_percent(singleton):>9}{_plain(largest):>8}"
            f"{_plain(good_groups):>9}{_percent(eq1):>7}"
            f"{_percent(le2):>7}{_percent(concentration):>8}"
            f"{_percent(purity):>9}"
        )

    lines.extend(
        [
            "",
            "[4] Duplicate / Bridge Diagnostics",
            "",
            (
                f"{'thr':<6}{'medDup7':>8}{'medPair7':>9}{'pairDef%':>9}"
                f"{'bridge%':>8}{'medMin':>8}{'medP10':>8}"
                f"{'pairMin':>8}{'pairP10':>8}{'bad01%':>8}{'bad03%':>8}"
            ),
        ]
    )
    for threshold, metrics in threshold_rows:
        values = _threshold_duplicate_bridge(metrics)
        duplicate_by_threshold[threshold] = values
        (
            medoid_dup,
            medoid_pair,
            pair_defined,
            bridge,
            medoid_min_mean,
            medoid_min_p10,
            pair_min_mean,
            pair_min_p10,
            bad01,
            bad03,
        ) = values
        lines.append(
            f"{threshold:<6.2f}{_plain(medoid_dup):>8}"
            f"{_plain(medoid_pair):>9}{_percent(pair_defined):>9}"
            f"{_percent(bridge):>8}{_plain(medoid_min_mean):>8}"
            f"{_plain(medoid_min_p10):>8}{_plain(pair_min_mean):>8}"
            f"{_plain(pair_min_p10):>8}{_percent(bad01):>8}"
            f"{_percent(bad03):>8}"
        )

    lines.extend(
        [
            "",
            "[5] Group Medoid Mask Quality",
            "",
            (
                f"{'thr':<6}{'selMed-c':>9}{'selMed-g':>9}{'dC':>7}{'dG':>7}"
                f"{'oracleMed-c':>12}{'oracleMed-g':>12}"
                f"{'dropQ-c':>9}{'dropQ-g':>9}"
            ),
        ]
    )
    for threshold, metrics in threshold_rows:
        values = _threshold_quality(metrics, selected, oracle)
        quality_by_threshold[threshold] = values
        (
            selected_medoid_c,
            selected_medoid_g,
            selected_medoid_dc,
            selected_medoid_dg,
            oracle_medoid_c,
            oracle_medoid_g,
            drop_c,
            drop_g,
        ) = values
        lines.append(
            f"{threshold:<6.2f}{_percent(selected_medoid_c):>9}"
            f"{_percent(selected_medoid_g):>9}"
            f"{_signed_points(selected_medoid_dc):>7}"
            f"{_signed_points(selected_medoid_dg):>7}"
            f"{_percent(oracle_medoid_c):>12}"
            f"{_percent(oracle_medoid_g):>12}"
            f"{_signed_points(drop_c):>9}{_signed_points(drop_g):>9}"
        )

    lines.extend(
        [
            "",
            "[6] Sanity Checks",
            "",
            f"{'check':<50}{'value':>10}",
            (
                f"{'selected_group_top_score == selected_query':<50}"
                f"{'PASS' if sanity.top_score_matches else 'FAIL':>10}"
            ),
            (
                f"{'selected_query_id_mismatch_count':<50}"
                f"{sanity.query_id_mismatch_count:>10d}"
            ),
            (
                f"{'selected_mask_mismatch_count':<50}"
                f"{sanity.mask_mismatch_count:>10d}"
            ),
            (
                f"{'cIoU_absolute_difference':<50}"
                f"{sanity.ciou_absolute_difference:>10.6f}"
            ),
            (
                f"{'gIoU_absolute_difference':<50}"
                f"{sanity.giou_absolute_difference:>10.6f}"
            ),
            (
                f"{'all_queries_assigned_exactly_once':<50}"
                f"{'PASS' if sanity.all_queries_assigned_exactly_once else 'FAIL':>10}"
            ),
            (
                f"{'distributed_total_samples_correct':<50}"
                f"{'PASS' if sanity.distributed_total_samples_correct else 'FAIL':>10}"
            ),
            f"{'nan_or_inf_count':<50}{sanity.nan_or_inf_count:>10d}",
            "",
            "[7] Main Screenshot Summary",
            "",
            (
                f"{'method':<16}{'groups':>8}{'goodGrp7':>9}{'eq1%':>7}"
                f"{'conc7%':>8}{'purity7%':>9}{'medDup7':>9}"
                f"{'pair7':>8}{'oracle-c':>10}{'oracle-g':>10}"
            ),
            (
                f"{'Raw Query':<16}{'-':>8}{_plain(raw_dup):>9}{'-':>7}"
                f"{'-':>8}{'-':>9}{_plain(raw_dup):>9}"
                f"{_plain(raw_pair):>8}{_percent(oracle_c):>10}"
                f"{_percent(oracle_g):>10}"
            ),
        ]
    )
    for threshold, _ in threshold_rows:
        structure = structure_by_threshold[threshold]
        duplicate = duplicate_by_threshold[threshold]
        quality = quality_by_threshold[threshold]
        groups, _, _, _, good_groups, eq1, _, concentration, purity = structure
        medoid_dup, medoid_pair = duplicate[0], duplicate[1]
        oracle_medoid_c, oracle_medoid_g = quality[4], quality[5]
        method = f"Topology-{threshold:.2f}"
        lines.append(
            f"{method:<16}{_plain(groups):>8}{_plain(good_groups):>9}"
            f"{_percent(eq1):>7}{_percent(concentration):>8}"
            f"{_percent(purity):>9}{_plain(medoid_dup):>9}"
            f"{_plain(medoid_pair):>8}{_percent(oracle_medoid_c):>10}"
            f"{_percent(oracle_medoid_g):>10}"
        )

    lines.extend(_two_stage_decomposition_lines(report))
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def emit_report(
    report: Mapping[str, Any],
    output_dir: os.PathLike[str] | str,
    *,
    rank: int = 0,
    stream: TextIO | None = None,
) -> Path | None:
    """Write and print the complete table on rank 0, then enforce sanity.

    Non-zero ranks perform no formatting, filesystem writes, or stdout writes.
    On rank 0, failed critical checks are intentionally raised only after the
    same complete table has been written to ``table.txt`` and stdout.
    """

    if rank != 0:
        return None
    table_text = build_table_text(report)
    table_path = Path(output_dir) / "table.txt"
    _write_text(table_path, table_text)
    destination = sys.stdout if stream is None else stream
    destination.write(table_text)
    destination.flush()
    failures = sanity_failures(report)
    if failures:
        raise MaskTopologySanityError(failures, table_path)
    return table_path


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    module_name = type(value).__module__.split(".", maxsplit=1)[0]
    if module_name in {"numpy", "torch"}:
        if hasattr(value, "numel") and callable(value.numel) and value.numel() == 1:
            return _jsonable(value.item())
        if hasattr(value, "size") and not callable(value.size) and value.size == 1:
            return _jsonable(value.item())
        if hasattr(value, "tolist") and callable(value.tolist):
            return _jsonable(value.tolist())
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(
    path: os.PathLike[str] | str,
    value: Any,
    *,
    rank: int = 0,
) -> Path | None:
    """Write deterministic UTF-8 JSON on rank 0 and reject NaN/Infinity."""

    if rank != 0:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _jsonable(value),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return destination


def write_jsonl(
    path: os.PathLike[str] | str,
    rows: Iterable[Any],
    *,
    rank: int = 0,
) -> Path | None:
    """Write one compact JSON object per line on rank 0."""

    if rank != 0:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
    return destination


def _threshold_filename_token(threshold: float) -> str:
    return f"{threshold:.12g}"


def write_report_artifacts(
    report: Mapping[str, Any],
    output_dir: os.PathLike[str] | str,
    *,
    fragmentation_cases: Iterable[Any] = (),
    bridge_cases: Iterable[Any] = (),
    rank: int = 0,
    stream: TextIO | None = None,
) -> dict[str, Path] | None:
    """Serialize all requested artifacts and emit the terminal table last.

    ``summary.json`` receives the complete report.  Each threshold file receives
    its threshold plus that threshold's full metric mapping.  The table is the
    final artifact and the only stdout write.
    """

    if rank != 0:
        return None
    destination = Path(output_dir)
    paths: dict[str, Path] = {}
    summary_path = write_json(destination / "summary.json", report, rank=rank)
    assert summary_path is not None
    paths["summary"] = summary_path
    for threshold, metrics in _threshold_rows(report):
        payload = dict(metrics)
        payload["threshold"] = threshold
        key = _threshold_filename_token(threshold)
        threshold_path = write_json(
            destination / f"threshold_{key}.json", payload, rank=rank
        )
        assert threshold_path is not None
        paths[f"threshold_{key}"] = threshold_path
    fragmentation_path = write_jsonl(
        destination / "fragmentation_cases.jsonl",
        fragmentation_cases,
        rank=rank,
    )
    bridge_path = write_jsonl(
        destination / "bridge_cases.jsonl",
        bridge_cases,
        rank=rank,
    )
    assert fragmentation_path is not None and bridge_path is not None
    paths["fragmentation_cases"] = fragmentation_path
    paths["bridge_cases"] = bridge_path
    table_path = emit_report(report, destination, rank=rank, stream=stream)
    assert table_path is not None
    paths["table"] = table_path
    return paths


__all__ = [
    "MaskTopologyReportError",
    "MaskTopologySanityError",
    "ReportSchemaError",
    "build_table_text",
    "emit_report",
    "sanity_failures",
    "write_json",
    "write_jsonl",
    "write_report_artifacts",
]
