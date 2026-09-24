#!/usr/bin/env python

import argparse
import json
import math
import os
import os.path as osp
import re
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.utils import set_model_resource
from xtuner.utils.device import get_device

from masklat.dataset.collate_fns import masklat_collate_fn
from masklat.evaluation.utils import comm
from masklat.evaluation.utils.topology_group_diagnostics import (
    update_topology_group_diagnostics,
)
from masklat.evaluation.utils.variant_query_classification import (
    SUPPORTED_VARIANT_QUERY_DATASETS,
    VariantQueryClassificationDiagnostics,
    validate_variant_dataset_name,
)
from masklat.model.segmentors.mask2former.group_diagnostics import (
    GroupDiagnostics,
)
from masklat.model.segmentors.mask2former.group_matcher import (
    HierarchicalGroupMatcher,
)
from masklat.utils.checkpoint import load_checkpoint
from masklat.utils.config import setup_model_config
from masklat.utils.constants import DEFAULT_SEG_TOKEN
from masklat.utils.dist import setup_distributed
from masklat.utils.logging import print_log, set_default_logging_format
from masklat.utils.misc import data_dict_to_device
from masklat.utils.utils import register_function

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

MAX_SAMPLE_ERROR_DETAILS = 3


@dataclass(frozen=True)
class DatasetEvaluationResult:
    """Structured opt-in return value for aggregate evaluation reports.

    ``evaluate_dataset`` keeps returning the historical rank-local formal
    summary by default.  The standalone RefSeg/ReaSeg diagnostics CLI opts into
    this wrapper so it can collect globally reduced topology statistics without
    changing the periodic training hook's contract.
    """

    formal_summary: Optional[Dict]
    topology_diagnostics: Optional[Dict]


class DistributedEvalSampler(Sampler):
    """Shard a dataset across ranks without padding or duplicate samples."""

    def __init__(self, dataset, rank: int, num_replicas: int):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, but got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}], but got {rank}")

        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.num_replicas))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and models")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint for evaluation",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "evaluate only these exact evaluator data_name values; omitted "
            "means every segmentation dataset in the config"
        ),
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help=(
            "DataLoader workers per evaluation rank (default: 4); for "
            "multi-node 16-rank evaluation, 1 or 2 often avoids host I/O "
            "worker contention"
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher type",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--variant-query-classification",
        action="store_true",
        help=(
            "run the default-off ten-stage high-IoU Query classification "
            "diagnostic; requires exactly one supported single-expression "
            "RefCOCO dataset and original topology-off MaskLAT"
        ),
    )
    parser.add_argument(
        "--refseg-reaseg-diagnostics",
        action="store_true",
        help=(
            "write one rank-0 refseg_reaseg_summary.json and print compact "
            "formal/topology tables for selected RefSeg/ReaSeg datasets; this "
            "explicitly enables read-only topology diagnostics even when the "
            "training recipe disabled diagnostic logging"
        ),
    )
    parser.add_argument(
        "--refseg-reaseg-all-stage-gt",
        action="store_true",
        help=(
            "with --refseg-reaseg-diagnostics, run expensive restored 200-Query "
            "GT diagnostics at all ten stages; restricted to the single "
            "refcoco_val_refseg split (the default diagnoses GT only at stage 9)"
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help=(
            "optional non-negative training step recorded verbatim in the "
            "standalone RefSeg/ReaSeg summaries; omitted means null because "
            "pytorch_model.bin does not encode a reliable step"
        ),
    )
    return parser.parse_args()


def select_evaluation_configs(
    val_datasets,
    val_evaluators,
    selected_names=None,
):
    """Pair and optionally filter evaluation configs by exact data name."""

    if len(val_datasets) != len(val_evaluators):
        raise ValueError(
            "val_datasets and val_evaluators must have equal lengths: "
            f"{len(val_datasets)} != {len(val_evaluators)}"
        )
    pairs = tuple(zip(val_datasets, val_evaluators))
    if selected_names is None:
        return pairs
    selected_names = tuple(selected_names)
    if not selected_names:
        raise ValueError("--datasets requires at least one data_name")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError(f"--datasets contains duplicates: {selected_names}")

    named_pairs = []
    for dataset_cfg, evaluator_cfg in pairs:
        data_name = evaluator_cfg.get("data_name", None)
        if not data_name:
            raise ValueError("every evaluator config must define data_name")
        named_pairs.append((str(data_name), dataset_cfg, evaluator_cfg))
    available_names = {name for name, _, _ in named_pairs}
    missing_names = sorted(set(selected_names) - available_names)
    if missing_names:
        raise ValueError(
            "unknown --datasets value(s): "
            f"{missing_names}; available={sorted(available_names)}"
        )
    selected = set(selected_names)
    return tuple(
        (dataset_cfg, evaluator_cfg)
        for name, dataset_cfg, evaluator_cfg in named_pairs
        if name in selected
    )


def validate_variant_dataset_selection(selected_names) -> str:
    """Require exactly one supported dataset for the heavy diagnostic."""

    selected_names = tuple(selected_names or ())
    if len(selected_names) != 1:
        raise ValueError(
            "--variant-query-classification requires exactly one "
            "--datasets value from "
            f"{SUPPORTED_VARIANT_QUERY_DATASETS}"
        )
    return validate_variant_dataset_name(selected_names[0])


def validate_refseg_reaseg_dataset_selection(
    selected_names,
    *,
    all_stage_gt: bool = False,
) -> Tuple[str, ...]:
    """Require explicit, exclusively RefSeg/ReaSeg diagnostic datasets."""

    selected = tuple(selected_names or ())
    if not selected:
        raise ValueError(
            "--refseg-reaseg-diagnostics requires explicit --datasets; "
            "omitting it would also evaluate unrelated configured datasets"
        )
    invalid = [
        name for name in selected if not _is_refseg_reaseg_dataset(name)
    ]
    if invalid:
        raise ValueError(
            "--refseg-reaseg-diagnostics accepts only *_refseg/*_reaseg "
            f"datasets, got {invalid}"
        )
    if all_stage_gt and selected != ("refcoco_val_refseg",):
        raise ValueError(
            "--refseg-reaseg-all-stage-gt requires exactly "
            "--datasets refcoco_val_refseg"
        )
    return selected


def validate_variant_checkpoint_load(load_report: Dict) -> None:
    """Reject partial, stage-only, or topology weights for this diagnostic."""

    required_prefixes = (
        "llm.",
        "visual_encoder.",
        "segmentor.encoder.",
        "segmentor.pixel_decoder.",
        "segmentor.decoder.",
        "visual_projector.",
        "seg_projector.",
        "llm_projector.",
        "seg_connector.",
        "bg_embeds.",
        "vision_sampler.",
    )
    forbidden_fragments = (
        "topology_group_core.",
        "group_classifier.",
        "group_prediction_adapter.",
        "group_criterion.",
    )
    checkpoint_keys = tuple(load_report.get("checkpoint_keys", ()))
    model_keys = tuple(load_report.get("model_keys", ()))
    matched_keys = tuple(load_report.get("matched_keys", ()))
    unexpected_checkpoint_keys = tuple(
        load_report.get("unexpected_checkpoint_keys", ())
    )
    missing_model_keys = tuple(
        load_report.get("missing_model_keys", ())
    )
    if not checkpoint_keys or not matched_keys:
        raise RuntimeError(
            "variant Query diagnostics require a non-empty, matching "
            "original S3 checkpoint"
        )

    # Some configured modules are intentionally parameter-free.  In the
    # released S3 recipe ``vision_sampler`` uses NaiveSamplerModel, which only
    # contains AdaptiveMaxPool1d and therefore has no state-dict keys.  Require
    # a prefix only when the constructed model actually owns state under it.
    stateful_required_prefixes = tuple(
        prefix
        for prefix in required_prefixes
        if any(key.startswith(prefix) for key in model_keys)
    )
    absent_prefixes = [
        prefix
        for prefix in stateful_required_prefixes
        if not any(key.startswith(prefix) for key in matched_keys)
    ]
    missing_critical_keys = [
        key
        for key in missing_model_keys
        if key.startswith(required_prefixes)
    ]
    unexpected_critical_keys = [
        key
        for key in unexpected_checkpoint_keys
        if key.startswith(required_prefixes)
    ]
    forbidden_keys = [
        key
        for key in checkpoint_keys
        if any(fragment in key for fragment in forbidden_fragments)
    ]
    if (
        absent_prefixes
        or missing_critical_keys
        or unexpected_critical_keys
        or forbidden_keys
    ):
        raise RuntimeError(
            "variant Query diagnostics refuse a partial, stage-only, or "
            "topology checkpoint: "
            f"absent_module_prefixes={absent_prefixes}; "
            f"missing_critical_keys={missing_critical_keys[:20]}; "
            f"unexpected_critical_keys={unexpected_critical_keys[:20]}; "
            f"forbidden_topology_keys={forbidden_keys[:20]}; "
            f"checkpoint={load_report.get('checkpoint_path')!r}"
        )


def validate_checkpoint_load_contract(load_report: Dict, contract: Any) -> None:
    """Fail when a config's checkpoint does not match its model topology."""

    if contract is None:
        return
    if not hasattr(contract, "get"):
        raise TypeError("checkpoint_load_contract must be a mapping")

    required_matched_prefixes = tuple(
        contract.get("required_matched_prefixes", ())
    )
    allowed_missing_model_prefixes = tuple(
        contract.get("allowed_missing_model_prefixes", ())
    )
    reject_unexpected = bool(
        contract.get("reject_unexpected_checkpoint_keys", True)
    )
    matched_keys = tuple(load_report.get("matched_keys", ()))
    missing_model_keys = tuple(load_report.get("missing_model_keys", ()))
    unexpected_checkpoint_keys = tuple(
        load_report.get("unexpected_checkpoint_keys", ())
    )

    absent_prefixes = [
        prefix
        for prefix in required_matched_prefixes
        if not any(key.startswith(prefix) for key in matched_keys)
    ]
    disallowed_missing_keys = [
        key
        for key in missing_model_keys
        if not any(
            key.startswith(prefix)
            for prefix in allowed_missing_model_prefixes
        )
    ]
    rejected_unexpected_keys = (
        list(unexpected_checkpoint_keys) if reject_unexpected else []
    )
    if absent_prefixes or disallowed_missing_keys or rejected_unexpected_keys:
        raise RuntimeError(
            "checkpoint/model topology mismatch: "
            f"absent_required_prefixes={absent_prefixes}; "
            f"missing_model_keys={disallowed_missing_keys[:20]}; "
            f"unexpected_checkpoint_keys={rejected_unexpected_keys[:20]}; "
            f"checkpoint={load_report.get('checkpoint_path')!r}"
        )


def validate_refseg_reaseg_checkpoint_load(model: Any, load_report: Dict) -> None:
    """Reject topology-off, partial, or mismatched diagnostic checkpoints."""

    segmentor = getattr(model, "segmentor", None)
    config = getattr(segmentor, "dec_config", None)
    if not bool(
        config is not None
        and getattr(config, "use_topology_group_decoder", False)
    ):
        raise RuntimeError(
            "--refseg-reaseg-diagnostics requires "
            "use_topology_group_decoder=True"
        )

    base_prefixes = (
        "llm.",
        "visual_encoder.",
        "segmentor.encoder.",
        "segmentor.pixel_decoder.",
        "segmentor.decoder.",
        "visual_projector.",
        "seg_projector.",
        "llm_projector.",
        "seg_connector.",
        "bg_embeds.",
        "vision_sampler.",
    )
    topology_prefixes = (
        "segmentor.topology_group_core.",
        "segmentor.group_classifier.",
        "segmentor.group_prediction_adapter.",
        "segmentor.group_criterion.",
    )
    required_prefixes = base_prefixes + topology_prefixes
    checkpoint_keys = tuple(load_report.get("checkpoint_keys", ()))
    model_keys = tuple(load_report.get("model_keys", ()))
    matched_keys = tuple(load_report.get("matched_keys", ()))
    missing_model_keys = tuple(load_report.get("missing_model_keys", ()))
    unexpected_checkpoint_keys = tuple(
        load_report.get("unexpected_checkpoint_keys", ())
    )
    if not checkpoint_keys or not matched_keys:
        raise RuntimeError(
            "RefSeg/ReaSeg topology diagnostics require a non-empty, "
            "matching full checkpoint"
        )

    stateful_required_prefixes = tuple(
        prefix
        for prefix in required_prefixes
        if any(key.startswith(prefix) for key in model_keys)
    )
    stateful_topology_prefixes = tuple(
        prefix
        for prefix in topology_prefixes
        if any(key.startswith(prefix) for key in model_keys)
    )
    if "segmentor.topology_group_core." not in stateful_topology_prefixes:
        raise RuntimeError(
            "constructed topology model exposes no topology_group_core state"
        )
    absent_prefixes = [
        prefix
        for prefix in stateful_required_prefixes
        if not any(key.startswith(prefix) for key in matched_keys)
    ]
    missing_critical_keys = [
        key
        for key in missing_model_keys
        if key.startswith(required_prefixes)
    ]
    unexpected_critical_keys = [
        key
        for key in unexpected_checkpoint_keys
        if key.startswith(required_prefixes)
    ]
    if absent_prefixes or missing_critical_keys or unexpected_critical_keys:
        raise RuntimeError(
            "RefSeg/ReaSeg topology diagnostics refuse a partial or "
            "mismatched checkpoint: "
            f"absent_module_prefixes={absent_prefixes}; "
            f"missing_critical_keys={missing_critical_keys[:20]}; "
            f"unexpected_critical_keys={unexpected_critical_keys[:20]}; "
            f"checkpoint={load_report.get('checkpoint_path')!r}"
        )


def summarize_topology_diagnostics(diagnostics: Any) -> Dict:
    """Return the accumulator's public final-summary JSON contract.

    The accumulator owns the additive sufficient statistics and their global
    reduction.  The returned payload preserves every reported value and its
    documented validity denominator, but it is not a mergeable serialization
    of the private accumulator.  The evaluator must not reconstruct values
    through private methods because doing so can silently drop denominators or
    drift from the terminal diagnostics.
    """

    to_dict = getattr(diagnostics, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            "topology diagnostic accumulator must expose public to_dict()"
        )
    payload = to_dict()
    if not isinstance(payload, dict):
        raise TypeError("topology diagnostics to_dict() must return a dict")
    if payload.get("metric_unit") != "fraction":
        raise ValueError(
            "topology diagnostics must declare metric_unit='fraction'"
        )
    # Round-trip now so unsupported values and NaN/Inf fail before a report is
    # presented as valid.  ``allow_nan=False`` is also used by every writer.
    try:
        return json.loads(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"topology diagnostics are not finite JSON data: {error}"
        ) from error


def _is_refseg_reaseg_dataset(data_name: str) -> bool:
    name = str(data_name)
    return name.endswith("_refseg") or name.endswith("_reaseg")


def _foreground_metric_payload(summary: Dict) -> Dict:
    """Select the foreground row from a formal RefSeg/ReaSeg summary."""

    if not isinstance(summary, dict) or not summary:
        raise ValueError("formal evaluator returned an empty summary")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("formal summary is missing metrics")

    foreground_name = next(
        (name for name in ("refer", "reason") if name in metrics),
        None,
    )
    if foreground_name is None:
        candidates = [name for name in metrics if name != "ignore"]
        if len(candidates) != 1:
            raise ValueError(
                "cannot identify one RefSeg/ReaSeg foreground metric row: "
                f"{sorted(metrics)}"
            )
        foreground_name = candidates[0]

    foreground = metrics.get(foreground_name)
    if not isinstance(foreground, dict):
        raise ValueError(f"foreground metric row {foreground_name!r} is invalid")
    values = {}
    for metric_name in ("cIoU", "gIoU"):
        value = float(foreground[metric_name])
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(
                f"{foreground_name}.{metric_name} must be in [0,100], "
                f"got {value}"
            )
        values[metric_name] = value

    num_predictions = int(summary.get("num_predictions", -1))
    if num_predictions < 0:
        raise ValueError("formal summary has invalid num_predictions")
    return {
        "foreground_class": foreground_name,
        "num_predictions": num_predictions,
        **values,
    }


def build_refseg_reaseg_record(
    data_name: str,
    formal_summary: Dict,
    topology_diagnostics: Dict,
    *,
    world_size: int = 1,
    checkpoint_step: Optional[int] = None,
    all_stage_gt: bool = False,
) -> Dict:
    """Validate and build one complete aggregate dataset record."""

    if not _is_refseg_reaseg_dataset(data_name):
        raise ValueError(f"not a RefSeg/ReaSeg dataset: {data_name!r}")
    if checkpoint_step is not None and checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative or None")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    formal = _foreground_metric_payload(formal_summary)
    summary_name = formal_summary.get("data_name")
    if summary_name is not None and str(summary_name) != str(data_name):
        raise ValueError(
            "formal summary data_name mismatch: "
            f"{summary_name!r} != {data_name!r}"
        )
    try:
        json.dumps(
            formal_summary,
            ensure_ascii=False,
            allow_nan=False,
        )
        json.dumps(
            topology_diagnostics,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{data_name} summary contains non-finite or non-JSON data: "
            f"{error}"
        ) from error
    if not isinstance(topology_diagnostics, dict):
        raise TypeError(
            "standalone RefSeg/ReaSeg diagnostics require a topology summary"
        )
    if topology_diagnostics.get("metric_unit") != "fraction":
        raise ValueError("topology diagnostic metric_unit must be fraction")

    variants = topology_diagnostics.get("refseg_variants")
    formal_variant_name = topology_diagnostics.get("formal_variant_name")
    if not isinstance(variants, dict) or formal_variant_name not in variants:
        raise ValueError(
            "topology diagnostics do not contain their declared formal variant"
        )
    diagnostic_formal = variants[formal_variant_name]
    diagnostic_ciou = diagnostic_formal.get("cIoU")
    diagnostic_giou = diagnostic_formal.get("gIoU")
    if diagnostic_ciou is None or diagnostic_giou is None:
        raise ValueError("diagnostic formal cIoU/gIoU are unavailable")
    diagnostic_ciou_points = float(diagnostic_ciou) * 100.0
    diagnostic_giou_points = float(diagnostic_giou) * 100.0
    if not (
        math.isfinite(diagnostic_ciou_points)
        and math.isfinite(diagnostic_giou_points)
        and 0.0 <= diagnostic_ciou_points <= 100.0
        and 0.0 <= diagnostic_giou_points <= 100.0
    ):
        raise ValueError(
            "diagnostic formal cIoU/gIoU must be finite fractions in [0,1]"
        )
    ciou_delta = abs(float(formal["cIoU"]) - diagnostic_ciou_points)
    giou_delta = abs(float(formal["gIoU"]) - diagnostic_giou_points)

    topology_sanity = topology_diagnostics.get("sanity")
    if not isinstance(topology_sanity, dict):
        raise ValueError("topology diagnostics are missing sanity metadata")
    expected_stages = 10
    expected_queries = 200
    actual_stages = topology_sanity.get("stages")
    actual_queries = topology_sanity.get("queries")
    actual_samples = topology_sanity.get("global_diagnostic_samples")
    partition_sum = topology_sanity.get("routing_partition_sum7")
    partition_count = topology_sanity.get(
        "routing_partition_valid_count"
    )
    parity_tolerance = 1e-4
    partition_tolerance = 1e-6
    identity_tolerance = 1e-8

    routing = topology_diagnostics.get("routing_member")
    if not isinstance(routing, dict):
        raise ValueError("topology diagnostics are missing routing_member")

    def routing_value_and_count(metric_name: str):
        metric = routing.get(metric_name)
        if not isinstance(metric, dict):
            return None, None
        value = metric.get("value")
        count = metric.get("valid_count")
        return (
            None if value is None else float(value),
            count,
        )

    g_at1, g_at1_count = routing_value_and_count(
        "selected_group_contains_good7"
    )
    success7, success7_count = routing_value_and_count("success7")
    wrong_member7, wrong_member7_count = routing_value_and_count(
        "wrong_member7"
    )
    g_at1_rhs = (
        success7 + wrong_member7
        if success7 is not None and wrong_member7 is not None
        else None
    )

    route_gap_giou, route_gap_count = routing_value_and_count(
        "cross_group_routing_gap_giou"
    )
    member_gap_giou, member_gap_count = routing_value_and_count(
        "within_group_member_gap_giou"
    )
    query_oracle_variant = variants.get(
        "global_200q_oracle_query_mask",
        {},
    )
    query_oracle_giou = (
        query_oracle_variant.get("gIoU")
        if isinstance(query_oracle_variant, dict)
        else None
    )
    query_oracle_sample_count = (
        query_oracle_variant.get(
            "sufficient_statistics",
            {},
        ).get("sample_count")
        if isinstance(query_oracle_variant, dict)
        else None
    )
    diagnostic_formal_sample_count = (
        diagnostic_formal.get(
            "sufficient_statistics",
            {},
        ).get("sample_count")
        if isinstance(diagnostic_formal, dict)
        else None
    )
    giou_decomposition_rhs = (
        float(diagnostic_giou) + route_gap_giou + member_gap_giou
        if (
            route_gap_giou is not None
            and member_gap_giou is not None
        )
        else None
    )

    selected_cond_gt, selected_cond_gt_count = routing_value_and_count(
        "selected_cond_is_gt"
    )
    selected_cond_bg, selected_cond_bg_count = routing_value_and_count(
        "selected_cond_is_bg"
    )
    selected_cond_other, selected_cond_other_count = (
        routing_value_and_count("selected_cond_is_other")
    )
    selected_cond_partition_sum = (
        selected_cond_gt + selected_cond_bg + selected_cond_other
        if (
            selected_cond_gt is not None
            and selected_cond_bg is not None
            and selected_cond_other is not None
        )
        else None
    )

    expected_metric_count = formal["num_predictions"]
    checks = {
        "no_nan_or_inf": {
            "status": True,
        },
        "stages": {
            "status": actual_stages == expected_stages,
            "expected": expected_stages,
            "actual": actual_stages,
        },
        "queries": {
            "status": actual_queries == expected_queries,
            "expected": expected_queries,
            "actual": actual_queries,
        },
        "global_samples": {
            "status": actual_samples == formal["num_predictions"],
            "expected": formal["num_predictions"],
            "actual": actual_samples,
        },
        "routing_partition": {
            "status": (
                partition_sum is not None
                and math.isfinite(float(partition_sum))
                and abs(float(partition_sum) - 1.0)
                <= partition_tolerance
                and partition_count == formal["num_predictions"]
            ),
            "expected_sum": 1.0,
            "actual_sum": partition_sum,
            "sum_tolerance": partition_tolerance,
            "expected_valid_count": formal["num_predictions"],
            "actual_valid_count": partition_count,
        },
        "g_at1_identity": {
            "status": (
                g_at1 is not None
                and g_at1_rhs is not None
                and abs(g_at1 - g_at1_rhs) <= identity_tolerance
                and g_at1_count == expected_metric_count
                and success7_count == expected_metric_count
                and wrong_member7_count == expected_metric_count
            ),
            "identity": "G@1 = success7 + wrongM",
            "lhs": g_at1,
            "rhs": g_at1_rhs,
            "tolerance": identity_tolerance,
            "valid_counts": {
                "G@1": g_at1_count,
                "success7": success7_count,
                "wrongM": wrong_member7_count,
            },
            "expected_valid_count": expected_metric_count,
        },
        "giou_gap_decomposition": {
            "status": (
                query_oracle_giou is not None
                and giou_decomposition_rhs is not None
                and abs(
                    float(query_oracle_giou)
                    - giou_decomposition_rhs
                )
                <= identity_tolerance
                and route_gap_count == expected_metric_count
                and member_gap_count == expected_metric_count
                and query_oracle_sample_count == expected_metric_count
                and diagnostic_formal_sample_count
                == expected_metric_count
            ),
            "identity": (
                "QoracleG = formal_gIoU + routeDeltaG + memberDeltaG"
            ),
            "lhs": (
                None
                if query_oracle_giou is None
                else float(query_oracle_giou)
            ),
            "rhs": giou_decomposition_rhs,
            "tolerance": identity_tolerance,
            "valid_counts": {
                "routeDeltaG": route_gap_count,
                "memberDeltaG": member_gap_count,
                "QoracleG": query_oracle_sample_count,
                "formal_gIoU": diagnostic_formal_sample_count,
            },
            "expected_valid_count": expected_metric_count,
        },
        "selected_cond_partition": {
            "status": (
                selected_cond_partition_sum is not None
                and abs(selected_cond_partition_sum - 1.0)
                <= identity_tolerance
                and selected_cond_gt_count == expected_metric_count
                and selected_cond_bg_count == expected_metric_count
                and selected_cond_other_count
                == expected_metric_count
            ),
            "identity": "selGT + selBG + selOther = 1",
            "actual_sum": selected_cond_partition_sum,
            "expected_sum": 1.0,
            "tolerance": identity_tolerance,
            "valid_counts": {
                "selGT": selected_cond_gt_count,
                "selBG": selected_cond_bg_count,
                "selOther": selected_cond_other_count,
            },
            "expected_valid_count": expected_metric_count,
        },
        "global_reduction": {
            "status": (
                world_size == 1
                or topology_diagnostics.get("globally_reduced") is True
            ),
            "world_size": int(world_size),
            "globally_reduced": topology_diagnostics.get(
                "globally_reduced"
            ),
        },
        "formal_matches_selected_group_quality": {
            "status": (
                ciou_delta <= parity_tolerance
                and giou_delta <= parity_tolerance
            ),
            "formal_variant_name": formal_variant_name,
            "tolerance_points": parity_tolerance,
            "formal_cIoU": float(formal["cIoU"]),
            "diagnostic_cIoU": diagnostic_ciou_points,
            "cIoU_abs_delta_points": ciou_delta,
            "formal_gIoU": float(formal["gIoU"]),
            "diagnostic_gIoU": diagnostic_giou_points,
            "gIoU_abs_delta_points": giou_delta,
        },
    }
    if all_stage_gt:
        stagewise = topology_diagnostics.get(
            "stagewise_internal",
            {},
        ).get("stages")
        required_stage_metrics = (
            "query_oracle_giou",
            "no_good_query7",
            "selected_group_contains_good7",
            "routing_gap_giou",
        )
        stagewise_failures = []
        if not isinstance(stagewise, list) or len(stagewise) != expected_stages:
            stagewise_failures.append("stage_count")
        else:
            for expected_stage, stage_record in enumerate(stagewise):
                if stage_record.get("stage") != expected_stage:
                    stagewise_failures.append(
                        f"st{expected_stage}.stage_index"
                    )
                    continue
                for metric_name in required_stage_metrics:
                    metric_record = stage_record.get(metric_name)
                    if (
                        not isinstance(metric_record, dict)
                        or metric_record.get("value") is None
                        or metric_record.get("valid_count")
                        != formal["num_predictions"]
                    ):
                        stagewise_failures.append(
                            f"st{expected_stage}.{metric_name}"
                        )
                no_good_record = stage_record.get("no_good_query7", {})
                conditional_count = (
                    formal["num_predictions"]
                    * (1.0 - float(no_good_record["value"]))
                    if no_good_record.get("value") is not None
                    else None
                )
                for metric_name in (
                    "good_concentration7",
                    "best_group_purity7",
                ):
                    metric_record = stage_record.get(metric_name)
                    valid_count = (
                        metric_record.get("valid_count")
                        if isinstance(metric_record, dict)
                        else None
                    )
                    if (
                        conditional_count is None
                        or valid_count is None
                        or abs(float(valid_count) - conditional_count)
                        > partition_tolerance
                        or (
                            valid_count > 0
                            and metric_record.get("value") is None
                        )
                    ):
                        stagewise_failures.append(
                            f"st{expected_stage}.{metric_name}"
                        )
                for metric_name in ("wrong_member7", "member_gap_giou"):
                    metric_record = stage_record.get(metric_name)
                    expected_count = (
                        formal["num_predictions"]
                        if expected_stage == expected_stages - 1
                        else 0
                    )
                    if (
                        not isinstance(metric_record, dict)
                        or metric_record.get("valid_count") != expected_count
                        or (
                            expected_count > 0
                            and metric_record.get("value") is None
                        )
                        or (
                            expected_count == 0
                            and metric_record.get("value") is not None
                        )
                    ):
                        stagewise_failures.append(
                            f"st{expected_stage}.{metric_name}"
                        )
        checks["all_stage_gt_coverage"] = {
            "status": not stagewise_failures,
            "expected_stages": expected_stages,
            "expected_valid_count_per_stage": formal["num_predictions"],
            "failed_metrics": stagewise_failures,
        }
    failed_checks = [
        name for name, detail in checks.items() if detail["status"] is not True
    ]
    checks["all_passed"] = not failed_checks
    if failed_checks:
        raise RuntimeError(
            f"{data_name} diagnostic sanity failure(s): {failed_checks}; "
            f"cIoU_delta={ciou_delta:.8g} points; "
            f"gIoU_delta={giou_delta:.8g} points; "
            f"stages={actual_stages}; queries={actual_queries}; "
            f"samples={actual_samples}/{formal['num_predictions']}; "
            f"partition={partition_sum} count={partition_count}; "
            "refusing to write an ambiguous summary"
        )
    return {
        "data_name": str(data_name),
        "checkpoint_step": checkpoint_step,
        "diagnostic_scope": {
            "all_stage_gt": bool(all_stage_gt),
            "gt_restored_stages": (
                list(range(expected_stages))
                if all_stage_gt
                else [expected_stages - 1]
            ),
        },
        "formal": formal,
        "topology_diagnostics": topology_diagnostics,
        "sanity_checks": checks,
        "formal_summary": formal_summary,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _format_report_number(
    value: Optional[float],
    *,
    digits: int = 2,
) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _diagnostic_mean(section: Dict, name: str) -> Optional[float]:
    record = section.get(name)
    if not isinstance(record, dict):
        return None
    value = record.get("value")
    return None if value is None else float(value)


def _diagnostic_valid_count(
    section: Dict,
    name: str,
) -> Optional[float]:
    record = section.get(name)
    if not isinstance(record, dict):
        return None
    value = record.get("valid_count")
    return None if value is None else float(value)


def _percent(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value) * 100.0


def format_refseg_reaseg_tables(
    records: Sequence[Dict],
) -> Tuple[str, str, str]:
    """Return three narrow rank-zero tables for a complete evaluation."""

    if not records:
        raise ValueError("cannot format an empty RefSeg/ReaSeg report")
    formal_rows = []
    bottleneck_rows = []
    idea_rows = []
    for record in records:
        formal = record["formal"]
        formal_rows.append(
            (
                str(record["data_name"]),
                str(formal["num_predictions"]),
                _format_report_number(formal["cIoU"]),
                _format_report_number(formal["gIoU"]),
            )
        )

        topology = record["topology_diagnostics"]
        variants = topology["refseg_variants"]
        raw_query = variants.get("selected_query_mask", {})
        raw_query_ciou = _percent(raw_query.get("cIoU"))
        raw_query_giou = _percent(raw_query.get("gIoU"))
        raw_minus_formal_ciou = (
            raw_query_ciou - float(formal["cIoU"])
            if raw_query_ciou is not None
            else None
        )
        raw_minus_formal_giou = (
            raw_query_giou - float(formal["gIoU"])
            if raw_query_giou is not None
            else None
        )
        formal_rows[-1] = (
            *formal_rows[-1],
            _format_report_number(raw_query_ciou),
            _format_report_number(raw_query_giou),
            _format_report_number(raw_minus_formal_ciou),
            _format_report_number(raw_minus_formal_giou),
        )

        oracle = variants.get("global_200q_oracle_query_mask", {})
        selected_group_oracle = variants.get(
            "selected_group_oracle_member_mask",
            {},
        )
        oracle_ciou = _percent(oracle.get("cIoU"))
        oracle_giou = _percent(oracle.get("gIoU"))
        selected_group_oracle_ciou = _percent(
            selected_group_oracle.get("cIoU")
        )
        route_delta_ciou = (
            oracle_ciou - selected_group_oracle_ciou
            if (
                oracle_ciou is not None
                and selected_group_oracle_ciou is not None
            )
            else None
        )
        member_delta_ciou = (
            selected_group_oracle_ciou - float(formal["cIoU"])
            if selected_group_oracle_ciou is not None
            else None
        )
        routing = topology["routing_member"]
        idea = topology["idea_diagnostics"]
        no_good = _diagnostic_mean(routing, "no_good_query7")
        conditional_top3 = routing.get(
            "first_good_group_rank",
            {},
        ).get("top3_rate")
        good_group_at3 = (
            float(conditional_top3) * max(0.0, 1.0 - float(no_good))
            if conditional_top3 is not None and no_good is not None
            else None
        )
        bottleneck_rows.append(
            (
                str(record["data_name"]),
                _format_report_number(
                    oracle_ciou
                ),
                _format_report_number(
                    oracle_giou
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(routing, "success7"))
                ),
                _format_report_number(_percent(no_good)),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(
                            routing,
                            "selected_group_contains_good7",
                        )
                    )
                ),
                _format_report_number(_percent(good_group_at3)),
                _format_report_number(
                    _percent(_diagnostic_mean(routing, "wrong_group7"))
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(routing, "wrong_member7"))
                ),
                _format_report_number(
                    route_delta_ciou
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(
                            routing,
                            "cross_group_routing_gap_giou",
                        )
                    )
                ),
                _format_report_number(
                    member_delta_ciou
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(
                            routing,
                            "within_group_member_gap_giou",
                        )
                    )
                ),
            )
        )
        idea_rows.append(
            (
                str(record["data_name"]),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(routing, "selected_cond_is_gt")
                    )
                ),
                _format_report_number(
                    _diagnostic_mean(routing, "selected_cond_probability"),
                    digits=4,
                ),
                _format_report_number(
                    _diagnostic_mean(routing, "selected_cond_margin"),
                    digits=4,
                ),
                _format_report_number(
                    _diagnostic_mean(idea, "cond_iou_spearman"),
                    digits=4,
                ),
                _format_report_number(
                    _diagnostic_valid_count(
                        idea,
                        "cond_iou_spearman",
                    ),
                    digits=0,
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(idea, "variant_available7"))
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(idea, "variant_selected7"))
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(idea, "rescue_opportunity7")
                    )
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(idea, "rescue7"))
                ),
                _format_report_number(
                    _diagnostic_valid_count(idea, "rescue7"),
                    digits=0,
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(idea, "matched_good_lost7"))
                ),
                _format_report_number(
                    _diagnostic_valid_count(
                        idea,
                        "matched_good_lost7",
                    ),
                    digits=0,
                ),
                _format_report_number(
                    _diagnostic_mean(idea, "good_concentration7"),
                    digits=4,
                ),
                _format_report_number(
                    _diagnostic_mean(idea, "best_group_purity7"),
                    digits=4,
                ),
            )
        )

    formal_table = _markdown_table(
        (
            "dataset",
            "N",
            "cIoU",
            "gIoU",
            "rawQcIoU",
            "rawQgIoU",
            "raw-formalC",
            "raw-formalG",
        ),
        formal_rows,
    )
    bottleneck_table = _markdown_table(
        (
            "dataset",
            "QoracleC",
            "QoracleG",
            "success7",
            "noGood7",
            "G@1",
            "G@3",
            "wrongG",
            "wrongM",
            "routeΔC",
            "routeΔG",
            "memberΔC",
            "memberΔG",
        ),
        bottleneck_rows,
    )
    idea_table = _markdown_table(
        (
            "dataset",
            "selGT%",
            "selP",
            "margin",
            "condρ",
            "rhoN",
            "varAvail7",
            "varSelected7",
            "rescueOpp7",
            "rescue7",
            "rescueN",
            "matLost7",
            "matN",
            "goodConc7",
            "bestPur7",
        ),
        idea_rows,
    )
    return formal_table, bottleneck_table, idea_table


def format_refseg_reaseg_stage_table(records: Sequence[Dict]) -> str:
    """Format the optional ten-stage, descriptive internal trajectory."""

    if len(records) != 1:
        raise ValueError(
            "the all-stage internal trajectory requires exactly one dataset"
        )
    stagewise = records[0]["topology_diagnostics"].get(
        "stagewise_internal",
        {},
    )
    stages = stagewise.get("stages")
    if not isinstance(stages, list) or len(stages) != 10:
        raise ValueError(
            "all-stage internal diagnostics must contain exactly 10 stages"
        )
    rows = []
    for expected_stage, stage in enumerate(stages):
        if stage.get("stage") != expected_stage:
            raise ValueError(
                "all-stage internal diagnostics must be ordered st0..st9"
            )
        rho = stage.get("cond_iou_spearman", {})
        rows.append(
            (
                f"st{expected_stage}",
                _format_report_number(
                    _percent(
                        _diagnostic_mean(stage, "query_oracle_giou")
                    )
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(stage, "no_good_query7")
                    )
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(
                            stage,
                            "selected_group_contains_good7",
                        )
                    )
                ),
                _format_report_number(
                    _percent(
                        _diagnostic_mean(stage, "routing_gap_giou")
                    )
                ),
                _format_report_number(
                    _diagnostic_mean(stage, "cond_iou_spearman"),
                    digits=4,
                ),
                _format_report_number(
                    rho.get("valid_count")
                    if isinstance(rho, dict)
                    else None,
                    digits=0,
                ),
                _format_report_number(
                    _diagnostic_mean(stage, "good_concentration7"),
                    digits=4,
                ),
                _format_report_number(
                    _diagnostic_mean(stage, "best_group_purity7"),
                    digits=4,
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(stage, "wrong_member7"))
                ),
                _format_report_number(
                    _percent(_diagnostic_mean(stage, "member_gap_giou"))
                ),
            )
        )
    return _markdown_table(
        (
            "stage",
            "QoracleG",
            "noGood7",
            "G@1",
            "routeΔG",
            "condρ",
            "rhoN",
            "goodConc7",
            "bestPur7",
            "wrongM",
            "memberΔG",
        ),
        rows,
    )


def write_dataset_refseg_reaseg_summary(
    output_dir: str,
    record: Dict,
    *,
    checkpoint: str,
    config_path: str,
    world_size: int,
) -> str:
    """Atomically enrich one formal summary in the explicit diagnostic mode."""

    if not output_dir:
        raise ValueError("dataset evaluator output_dir is required")
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, "summary.json")
    temporary_path = output_path + ".tmp"
    payload = dict(record["formal_summary"])
    payload.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": record.get("checkpoint_step"),
            "config": str(config_path),
            "world_size": int(world_size),
            "diagnostic_scope": record["diagnostic_scope"],
            "topology_diagnostics": record["topology_diagnostics"],
            "sanity_checks": record["sanity_checks"],
        }
    )
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")
    os.replace(temporary_path, output_path)
    return output_path


def write_refseg_reaseg_report(
    work_dir: str,
    records: Sequence[Dict],
    *,
    checkpoint: str,
    checkpoint_step: Optional[int],
    config_path: str,
    world_size: int,
) -> str:
    """Atomically write the complete rank-zero aggregate JSON."""

    if not work_dir:
        raise ValueError("--refseg-reaseg-diagnostics requires --work-dir")
    if not records:
        raise ValueError("no RefSeg/ReaSeg dataset results were collected")
    output_dir = osp.abspath(work_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, "refseg_reaseg_summary.json")
    temporary_path = output_path + ".tmp"
    payload = {
        "schema_version": 2,
        "report_type": "refseg_reaseg_diagnostics",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "config": str(config_path),
        "world_size": int(world_size),
        "num_datasets": len(records),
        "diagnostic_scope": {
            "all_stage_gt": all(
                record["diagnostic_scope"]["all_stage_gt"]
                for record in records
            ),
            "per_dataset": {
                record["data_name"]: record["diagnostic_scope"]
                for record in records
            },
        },
        "sanity_checks": {
            "all_datasets_passed": all(
                record["sanity_checks"]["all_passed"]
                for record in records
            ),
            "checked_datasets": len(records),
        },
        "interpretation_notes": {
            "selected_query_mask": (
                "internal counterfactual selected over all 200 Query masks "
                "using the topology checkpoint's raw Query class logits; "
                "these Queries already received topology feedback, so this "
                "is neither a Group-root mask nor the standard query path"
            ),
            "raw_minus_formal": (
                "raw-query counterfactual cIoU/gIoU minus the formal topology "
                "result from the same checkpoint; positive favors raw Query "
                "selection and negative favors the formal Group path"
            ),
            "G@3": (
                "unconditional percentage over all expressions, derived as "
                "P(good Query exists) times conditional top-3 Group recall"
            ),
            "cIoU_counterfactual_gaps": (
                "QoracleC/routeDeltaC/memberDeltaC compare aggregate cIoU "
                "ratios of counterfactual prediction sets; routeDeltaC and "
                "memberDeltaC may be negative and are not per-sample losses"
            ),
            "gIoU_gap_decomposition": (
                "QoracleG equals formal gIoU plus non-negative routeDeltaG "
                "and memberDeltaG because these are means of per-expression "
                "IoU differences"
            ),
            "conditional_valid_counts": (
                "rhoN, rescueN, and matN are the exact globally reduced "
                "valid-expression denominators for condRho, rescue7, and "
                "matLost7"
            ),
            "metric_units": (
                "formal/table cIoU, gIoU, rates, and IoU gaps are percentage "
                "points; embedded topology rate/IoU records remain fractions, "
                "while their definitions identify logit, rank, correlation, "
                "count, norm, and loss fields"
            ),
        },
        "datasets": list(records),
    }
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")
    os.replace(temporary_path, output_path)
    return output_path


def get_gcg_phrases(input_ids, tokenizer, pstart_token_idx, pend_token_idx):
    pstart_idx = [i for i, x in enumerate(input_ids) if x == pstart_token_idx]
    pend_idx = [i + 1 for i, x in enumerate(input_ids) if x == pend_token_idx]
    phrases = []
    for ps, pe in zip(pstart_idx, pend_idx):
        phrase_ids = input_ids[ps + 1 : pe - 1]
        if (phrase_ids < 0).any():
            phrase = ""
        else:
            phrase = tokenizer.decode(phrase_ids).strip()
        phrases.append(phrase)
    return phrases


def get_gcg_caption(llm_generation_output):
    if DEFAULT_SEG_TOKEN not in llm_generation_output:
        return ""

    parts = llm_generation_output.split(".")
    sents = [part.strip() for part in parts if DEFAULT_SEG_TOKEN not in part]
    caption = ". ".join(sents)
    caption = re.sub(r"<.*?>", "", caption)
    caption = " ".join(caption.split()).strip("'").strip()
    return caption


def process_batch(
    model,
    data: Dict,
    data_name: str,
    metadata: Dict,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    mode: str = "tensor",
    collect_topology_raw: bool = False,
    collect_variant_raw: bool = False,
) -> Tuple[bool, Optional[torch.Tensor], Optional[object]]:
    """Process a single batch of data.

    Args:
        model: The model to evaluate
        data: Input data dictionary
        data_name: Name of the dataset
        generation_config: Generation configuration for LLM
        stop_criteria: Stopping criteria for LLM
        mode: Mode of the model

    Returns:
        Tuple of (success status, formal segmentation outputs, raw
        ``MaskLATSegmentorOutput``).  Raw output is populated only for an explicitly
        enabled diagnostic and comes from the same formal model forward.
    """
    data_samples = data["data_samples"]
    image_files = data_samples.image_files

    data_dict = {
        "input_ids": data["data_dict"].get("input_ids", None),
        "pixel_values": data["data_dict"].get("pixel_values", None),
        "extra_pixel_values": data["data_dict"].get("extra_pixel_values", None),
        "cond_ids": data["data_dict"].get("cond_ids", None),
        "seg_ids": data["data_dict"].get("seg_ids", None),
        "vprompt_masks": data["data_dict"].get("vprompt_masks", None),
    }

    llm_question_input = ""
    if data_dict["input_ids"] is not None:
        _input_ids = data_dict["input_ids"]
        llm_question_input = model.tokenizer.decode(_input_ids[_input_ids > 0])

    data_dict = data_dict_to_device(data_dict, device=model.device, dtype=model.dtype)

    collect_raw_seg_outputs = bool(
        collect_topology_raw or collect_variant_raw
    )
    with torch.no_grad():
        model_outputs = model(
            data_dict,
            data_samples,
            mode=mode,
            generation_config=generation_config,
            stopping_criteria=stop_criteria,
            metadata=metadata,
            do_postprocess=True,
            do_loss=False,
            return_raw_seg_outputs=collect_raw_seg_outputs,
            return_variant_diagnostic_stages=collect_variant_raw,
        )
    if collect_raw_seg_outputs:
        if not isinstance(model_outputs, tuple) or len(model_outputs) != 3:
            raise RuntimeError(
                "diagnostic evaluation requested a same-forward raw output, "
                "but the model did not return (llm, formal, raw)"
            )
        llm_outputs, seg_outputs, raw_seg_outputs = model_outputs
    else:
        if not isinstance(model_outputs, tuple) or len(model_outputs) != 2:
            raise RuntimeError(
                "baseline evaluation model must return (llm, segmentation)"
            )
        llm_outputs, seg_outputs = model_outputs
        raw_seg_outputs = None

    if seg_outputs is None:
        llm_generation_output = ""
        if llm_outputs is not None and hasattr(llm_outputs, "sequences"):
            llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)

        print_log(
            rf"Failed to get segmentation outputs: {image_files}, "
            rf"llm question_input: {repr(llm_question_input)}, "
            rf"llm generation_output: {repr(llm_generation_output)}",
            logger="current",
        )
        return False, None, raw_seg_outputs

    image_infos = data_samples.metainfo["image_infos"]
    if len(seg_outputs) != len(image_infos):
        raise ValueError(
            "postprocessed prediction and evaluator metadata batches differ; "
            f"predictions={len(seg_outputs)}, image_infos={len(image_infos)}"
        )
    if collect_raw_seg_outputs and raw_seg_outputs is None:
        raise RuntimeError(
            "diagnostic forward returned formal predictions without raw "
            "outputs"
        )

    if "gcg" in data_name and llm_outputs is not None and hasattr(llm_outputs, "sequences"):
        llm_generation_output = model.tokenizer.batch_decode(llm_outputs.sequences)
        gcg_phrases = [
            get_gcg_phrases(output_ids, model.tokenizer, model.pstart_token_idx, model.pend_token_idx)
            for output_ids in llm_outputs.sequences
        ]
        gcg_captions = [get_gcg_caption(output) for output in llm_generation_output]
        for i, segmentation_output in enumerate(seg_outputs):
            segmentation_output.update({"gcg_phrases": gcg_phrases[i], "gcg_caption": gcg_captions[i]})

    return True, seg_outputs, raw_seg_outputs


def build_topology_diagnostic_context(model, *, force: bool = False):
    """Build one dataset-local, deterministic topology diagnostic context."""

    segmentor = getattr(model, "segmentor", None)
    config = getattr(segmentor, "dec_config", None)
    enabled = bool(
        config is not None
        and getattr(config, "use_topology_group_decoder", False)
        and (
            force
            or getattr(config, "group_log_diagnostics", False)
        )
    )
    if not enabled:
        return None
    core = getattr(segmentor, "topology_group_core", None)
    criterion = getattr(segmentor, "group_criterion", None)
    if core is None or criterion is None:
        raise RuntimeError(
            "topology diagnostics require the configured decoder core and "
            "group criterion"
        )
    training_matcher = getattr(criterion, "matcher", None)
    normalizer = getattr(criterion, "target_normalizer", None)
    if training_matcher is None or normalizer is None:
        raise RuntimeError(
            "topology diagnostics require the criterion matcher and "
            "target normalizer"
        )
    # Use every configured matching cost, while replacing stochastic point
    # sampling with deterministic full-resolution evaluation matching.
    matcher = HierarchicalGroupMatcher(
        cost_class=training_matcher.cost_class,
        cost_mask=training_matcher.cost_mask,
        cost_dice=training_matcher.cost_dice,
        num_points=training_matcher.num_points,
        use_sample_point=False,
        group_shape_temperature=training_matcher.group_shape_temperature,
        cost_cls_type=training_matcher.cost_cls_type,
        alpha=training_matcher.alpha,
        gamma=training_matcher.gamma,
    )
    return {
        "diagnostics": GroupDiagnostics(num_stages=core.num_stages),
        "core": core,
        "normalizer": normalizer,
        "matcher": matcher,
        "mask_threshold": float(
            config.topology_diagnostic_query_mask_threshold
        ),
        "group_iou_threshold": float(config.group_iou_threshold),
    }


def _raise_on_rank_errors(local_error, *, phase: str, data_name: str) -> None:
    """Raise the same aggregated failure on every evaluation rank."""

    error_payloads = comm.all_gather(local_error)
    failures = [
        f"rank {failed_rank}: {message}"
        for failed_rank, message in enumerate(error_payloads)
        if message
    ]
    if failures:
        raise RuntimeError(
            f"{data_name} {phase} failed on {len(failures)} rank(s): "
            + "; ".join(failures)
        )


def evaluate_dataset(
    model,
    dataset,
    evaluator,
    rank: int,
    world_size: int,
    generation_config: Optional[GenerationConfig] = None,
    stop_criteria: Optional[StoppingCriteriaList] = None,
    data_name: Optional[str] = None,
    dataloader_num_workers: int = 4,
    collect_topology_diagnostics: bool = False,
    variant_query_classification: bool = False,
    checkpoint_path: Optional[str] = None,
    force_topology_diagnostics: bool = False,
    log_topology_diagnostics: bool = True,
    return_detailed_result: bool = False,
    collect_all_stage_gt: bool = False,
) -> Union[Optional[Dict], DatasetEvaluationResult]:
    """Evaluate model on a single dataset."""
    resolved_data_name = data_name or "<unknown>"

    # Every rank must finish dataset-local setup before any rank enters the
    # prediction/evaluator collectives.  Otherwise a rank-local worker or
    # diagnostics construction failure can strand its peers.
    setup_error = None
    try:
        evaluator_data_name = evaluator.data_name
        if data_name is not None and evaluator_data_name != data_name:
            raise ValueError(
                "dataset/evaluator data_name mismatch: "
                f"{data_name!r} != {evaluator_data_name!r}"
            )
        resolved_data_name = evaluator_data_name
        metadata = dataset.metadata
        output_ids_with_output = dataset.output_ids_with_output
        mode = "tensor" if output_ids_with_output else "predict"
        sampler = DistributedEvalSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=world_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=dataloader_num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=masklat_collate_fn,
        )
        topology_diagnostic_context = (
            build_topology_diagnostic_context(
                model,
                force=force_topology_diagnostics,
            )
            if collect_topology_diagnostics
            else None
        )
        variant_diagnostic_context = (
            VariantQueryClassificationDiagnostics(
                segmentor=model.segmentor,
                metadata=metadata,
                output_dir=evaluator.output_dir,
                expected_dataset_sample_count=len(dataset),
                checkpoint=checkpoint_path,
                dataset_name=resolved_data_name,
            )
            if variant_query_classification
            else None
        )
        if (
            topology_diagnostic_context is not None
            and variant_diagnostic_context is not None
        ):
            raise ValueError(
                "topology and base-model variant diagnostics cannot run "
                "in the same evaluation"
            )
        evaluator.reset()
    except Exception as error:
        setup_error = f"{type(error).__name__}: {error}"
    _raise_on_rank_errors(
        setup_error,
        phase="dataset-local setup",
        data_name=resolved_data_name,
    )
    data_name = resolved_data_name

    # Evaluation loop
    failed_cnt = 0
    local_errors = []
    print_log(f"Evaluating {data_name}...", logger="current")

    try:
        for data in tqdm(dataloader, desc=f"Evaluating {data_name}", disable=rank != 0):
            try:
                success, seg_outputs, raw_seg_outputs = process_batch(
                    model,
                    data,
                    data_name,
                    metadata,
                    generation_config,
                    stop_criteria,
                    mode,
                    collect_topology_raw=(
                        topology_diagnostic_context is not None
                    ),
                    collect_variant_raw=(
                        variant_diagnostic_context is not None
                    ),
                )
                if not success:
                    failed_cnt += 1
                    if len(local_errors) < MAX_SAMPLE_ERROR_DETAILS:
                        local_errors.append(
                            "process_batch returned no segmentation output"
                        )
                    elif len(local_errors) == MAX_SAMPLE_ERROR_DETAILS:
                        local_errors.append(
                            "additional per-sample errors suppressed"
                        )
                    continue

                if topology_diagnostic_context is not None:
                    update_topology_group_diagnostics(
                        topology_diagnostic_context["diagnostics"],
                        raw_seg_outputs,
                        data["data_samples"],
                        normalizer=topology_diagnostic_context["normalizer"],
                        matcher=topology_diagnostic_context["matcher"],
                        topology_group_core=topology_diagnostic_context["core"],
                        mask_threshold=topology_diagnostic_context[
                            "mask_threshold"
                        ],
                        group_iou_threshold=topology_diagnostic_context[
                            "group_iou_threshold"
                        ],
                        collect_all_stage_gt=collect_all_stage_gt,
                    )
                if variant_diagnostic_context is not None:
                    variant_diagnostic_context.update(
                        raw_seg_outputs,
                        data["data_samples"],
                        seg_outputs,
                    )

                image_infos = data["data_samples"].metainfo["image_infos"]
                evaluator.process(image_infos, seg_outputs)
            except Exception as error:
                # Keep this rank moving to the shared failure collective.  If
                # only one rank raised immediately, peers could otherwise
                # block forever while entering evaluator collectives.
                failed_cnt += 1
                error_detail = f"{type(error).__name__}: {error}"
                if len(local_errors) < MAX_SAMPLE_ERROR_DETAILS:
                    local_errors.append(error_detail)
                    print_log(
                        f"Error processing {data_name} sample on rank "
                        f"{rank}: {error}\n{traceback.format_exc()}",
                        logger="current",
                    )
                elif len(local_errors) == MAX_SAMPLE_ERROR_DETAILS:
                    local_errors.append(
                        "additional per-sample errors suppressed; use the "
                        "first failures above to fix the shared root cause"
                    )
                    print_log(
                        f"Suppressing further per-sample tracebacks for "
                        f"{data_name} on rank {rank}; the global failure "
                        "count will still include every failed sample.",
                        logger="current",
                    )
                continue
    except Exception as error:
        # DataLoader/worker failures occur outside the loop body.
        failed_cnt += 1
        local_errors.append(f"dataloader {type(error).__name__}: {error}")
        print_log(
            f"Error iterating {data_name} on rank {rank}: "
            f"{error}\n{traceback.format_exc()}",
            logger="current",
        )

    # Every rank must learn the global failure count before any evaluator
    # collective starts.  A partial prediction set must never be reported as
    # a successful metric run.
    failure_payloads = comm.all_gather((failed_cnt, tuple(local_errors)))
    global_failed_count = sum(payload[0] for payload in failure_payloads)
    if rank == 0:
        print_log(f"Failed number of {data_name}: {global_failed_count}", logger="current")
    if global_failed_count:
        error_details = "; ".join(
            f"rank {failed_rank}: {message}"
            for failed_rank, (_, messages) in enumerate(failure_payloads)
            for message in messages[:3]
        )
        if not error_details:
            error_details = "no per-sample detail was recorded"
        raise RuntimeError(
            f"{data_name} failed for {global_failed_count} sample/loader "
            f"operation(s); refusing to evaluate partial predictions. "
            f"{error_details}"
        )
    topology_diagnostic_summary = None
    if topology_diagnostic_context is not None:
        diagnostics = topology_diagnostic_context["diagnostics"]
        diagnostics.all_reduce_()
        if return_detailed_result:
            topology_diagnostic_summary = summarize_topology_diagnostics(
                diagnostics
            )
        if rank == 0 and log_topology_diagnostics:
            print_log(
                f"Topology group diagnostics for {data_name} "
                f"(restored raw-query probability > "
                f"{topology_diagnostic_context['mask_threshold']}):\n"
                f"{diagnostics.format_terminal_tables()}",
                logger="current",
            )
    print_log(f"Evaluating {data_name} done!", logger="current")
    summary = None
    formal_evaluator_error = None
    try:
        summary = evaluator.evaluate()
    except Exception as error:
        formal_evaluator_error = f"{type(error).__name__}: {error}"
    # RefSegEvaluator returns early on nonzero ranks after its prediction
    # gather, while rank zero still performs identity checks, metric
    # computation, and file writes.  Synchronize any rank-zero-only failure
    # before another diagnostic collective can begin.
    _raise_on_rank_errors(
        formal_evaluator_error,
        phase="formal evaluator",
        data_name=data_name,
    )
    if variant_diagnostic_context is not None:
        variant_diagnostic_context.finalize_and_write()
    if return_detailed_result:
        return DatasetEvaluationResult(
            formal_summary=summary,
            topology_diagnostics=topology_diagnostic_summary,
        )
    return summary


def main():
    """Main evaluation function."""
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(args)
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader-num-workers must be non-negative")
    if args.checkpoint_step is not None and args.checkpoint_step < 0:
        raise ValueError("--checkpoint-step must be non-negative")
    if args.refseg_reaseg_diagnostics and not args.work_dir:
        raise ValueError(
            "--refseg-reaseg-diagnostics requires an explicit --work-dir"
        )
    if args.refseg_reaseg_all_stage_gt and not args.refseg_reaseg_diagnostics:
        raise ValueError(
            "--refseg-reaseg-all-stage-gt requires "
            "--refseg-reaseg-diagnostics"
        )
    if args.refseg_reaseg_diagnostics:
        validate_refseg_reaseg_dataset_selection(
            args.datasets,
            all_stage_gt=args.refseg_reaseg_all_stage_gt,
        )
    if args.variant_query_classification:
        validate_variant_dataset_selection(args.datasets)

    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    if args.seed is not None:
        # Use args.seed
        set_random_seed(args.seed)
        print_log(
            f"Set the random seed to {args.seed}.",
            logger="current",
        )

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        else:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    # Build and place the model before any rank enters DDP construction.
    # A rank-local build/device failure must be propagated collectively;
    # otherwise healthy peers can block inside the next DDP collective.
    model = None
    model_setup_error = None
    try:
        model = BUILDER.build(cfg.model)
        if "llm" in cfg.model:
            model.llm.to(cfg.model.llm.torch_dtype)
        model.eval()
        model = model.to(get_device())
    except Exception as error:
        model_setup_error = f"{type(error).__name__}: {error}"
    _raise_on_rank_errors(
        model_setup_error,
        phase="model build/to-device",
        data_name="evaluation",
    )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank]).module

    # Loading happens independently on every rank.  Synchronize any local I/O,
    # deserialization, or checkpoint-contract failure before peers continue to
    # model setup or dataset collectives.
    checkpoint_load_report = None
    checkpoint_load_error = None
    try:
        checkpoint_load_report = load_checkpoint(model, args.pth_model)
        validate_checkpoint_load_contract(
            checkpoint_load_report,
            cfg.get("checkpoint_load_contract"),
        )
        if args.variant_query_classification:
            validate_variant_checkpoint_load(checkpoint_load_report)
        if args.refseg_reaseg_diagnostics:
            validate_refseg_reaseg_checkpoint_load(
                model,
                checkpoint_load_report,
            )
    except Exception as error:
        checkpoint_load_error = f"{type(error).__name__}: {error}"
    _raise_on_rank_errors(
        checkpoint_load_error,
        phase="checkpoint load/validation",
        data_name="evaluation",
    )
    stop_criteria, generation_config = setup_model_config(model, cfg)

    # Evaluate on all datasets
    evaluation_configs = select_evaluation_configs(
        cfg.val_datasets,
        cfg.val_evaluators,
        args.datasets,
    )
    print_log(f"Evaluating {len(evaluation_configs)} datasets...", logger="current")
    dataset_errors = []
    refseg_reaseg_records = []
    for dataset_cfg, evaluator_cfg in evaluation_configs:
        data_name = evaluator_cfg.get(
            "data_name",
            getattr(dataset_cfg, "data_name", "<unknown>"),
        )
        collect_refseg_reaseg = bool(
            args.refseg_reaseg_diagnostics
            and _is_refseg_reaseg_dataset(data_name)
        )
        setup_error = None
        try:
            dataset = BUILDER.build(dataset_cfg)
            model.postprocess_fn = dataset.postprocess_fn

            evaluator = BUILDER.build(evaluator_cfg)
            evaluator.metadata = dataset.metadata
            evaluator.output_dir = osp.join(args.work_dir, "pred_data", evaluator.data_name)
            if (
                collect_refseg_reaseg
                and hasattr(evaluator, "_terminal_summary_only")
            ):
                evaluator._terminal_summary_only = True
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
            print_log(
                f"Error setting up {data_name} on rank {rank}: "
                f"{error}\n{traceback.format_exc()}",
                logger="current",
            )
        try:
            _raise_on_rank_errors(
                setup_error,
                phase="dataset/evaluator construction",
                data_name=data_name,
            )
        except RuntimeError as error:
            dataset_errors.append((data_name, str(error)))
            print_log(str(error), logger="current")
            continue

        evaluation_error = None
        aggregate_record = None
        try:
            evaluation_result = evaluate_dataset(
                model,
                dataset,
                evaluator,
                rank,
                world_size,
                generation_config,
                stop_criteria,
                data_name=data_name,
                dataloader_num_workers=args.dataloader_num_workers,
                variant_query_classification=(
                    args.variant_query_classification
                ),
                checkpoint_path=args.pth_model,
                collect_topology_diagnostics=collect_refseg_reaseg,
                force_topology_diagnostics=collect_refseg_reaseg,
                log_topology_diagnostics=not collect_refseg_reaseg,
                return_detailed_result=collect_refseg_reaseg,
                collect_all_stage_gt=(
                    args.refseg_reaseg_all_stage_gt
                    if collect_refseg_reaseg
                    else False
                ),
            )
            if collect_refseg_reaseg and rank == 0:
                if not isinstance(
                    evaluation_result,
                    DatasetEvaluationResult,
                ):
                    raise TypeError(
                        "RefSeg/ReaSeg aggregate evaluation did not return "
                        "DatasetEvaluationResult"
                    )
                aggregate_record = build_refseg_reaseg_record(
                    data_name,
                    evaluation_result.formal_summary,
                    evaluation_result.topology_diagnostics,
                    world_size=world_size,
                    checkpoint_step=args.checkpoint_step,
                    all_stage_gt=args.refseg_reaseg_all_stage_gt,
                )
                dataset_summary_path = write_dataset_refseg_reaseg_summary(
                    evaluator.output_dir,
                    aggregate_record,
                    checkpoint=args.pth_model,
                    config_path=args.config,
                    world_size=world_size,
                )
                print_log(
                    f"Enriched {data_name} summary: "
                    f"{dataset_summary_path}",
                    logger="current",
                )
        except Exception as error:
            evaluation_error = f"{type(error).__name__}: {error}"
            print_log(
                f"Error evaluating {data_name} on rank {rank}: "
                f"{error}\n{traceback.format_exc()}",
                logger="current",
            )
        try:
            _raise_on_rank_errors(
                evaluation_error,
                phase="evaluation",
                data_name=data_name,
            )
        except RuntimeError as error:
            dataset_errors.append((data_name, str(error)))
            print_log(str(error), logger="current")
            continue
        if aggregate_record is not None:
            refseg_reaseg_records.append(aggregate_record)
    if dataset_errors:
        details = "\n".join(
            f"- {data_name}: {message}"
            for data_name, message in dataset_errors
        )
        raise RuntimeError(
            f"Evaluation failed for {len(dataset_errors)} dataset(s):\n{details}"
        )
    if args.refseg_reaseg_diagnostics and rank == 0:
        (
            formal_table,
            bottleneck_table,
            idea_table,
        ) = format_refseg_reaseg_tables(
            refseg_reaseg_records
        )
        stage_table = (
            format_refseg_reaseg_stage_table(refseg_reaseg_records)
            if args.refseg_reaseg_all_stage_gt
            else None
        )
        report_path = write_refseg_reaseg_report(
            args.work_dir,
            refseg_reaseg_records,
            checkpoint=args.pth_model,
            checkpoint_step=args.checkpoint_step,
            config_path=args.config,
            world_size=world_size,
        )
        report_message = (
            "RefSeg / ReaSeg Evaluation Summary\n"
            "rawQ is an internal counterfactual from the same topology "
            "checkpoint, not the standard query path; raw-formal is "
            "rawQ minus formal in percentage points\n"
            f"{formal_table}\n\n"
            "Topology Group Routing Diagnostics\n"
            "good Query IoU threshold = 0.70; G@3 is unconditional; "
            "route/member G gaps are a strict non-negative per-expression "
            "gIoU decomposition; C gaps are aggregate cIoU counterfactual "
            "differences and may be negative\n"
            f"{bottleneck_table}\n\n"
            "Idea Mechanism Diagnostics\n"
            "rhoN/rescueN/matN are the corresponding valid-expression "
            "denominators\n"
            f"{idea_table}\n\n"
        )
        if stage_table is not None:
            report_message += (
                "Internal Mechanism Stage Trajectory "
                "(descriptive statistics, not a causal conclusion)\n"
                f"{stage_table}\n\n"
            )
        report_message += f"Aggregate summary: {report_path}"
        print_log(
            report_message,
            logger="current",
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
