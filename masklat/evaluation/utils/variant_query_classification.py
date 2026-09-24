"""Read-only RefCOCO diagnostics for high-IoU unmatched MaskLAT Queries.

The diagnostic consumes tensors already produced by the original topology-off
MaskLAT forward.  It never changes model parameters, loss computation, formal
Query selection, or the RefSeg evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from masklat.dataset.process_fns.postprocess_fns.refseg_process_fn import (
    binarize_refseg_query_masks,
    restore_refseg_query_masks,
    select_refseg_query,
)
from masklat.dataset.utils.mask import decode_mask
from masklat.evaluation.utils import comm
from masklat.utils.logging import print_log


_NUM_STAGES = 10
_EXPECTED_QUERIES = 200
_GOOD_IOU_THRESHOLD = 0.70
_MASK_THRESHOLD = 0.50
_RESTORE_QUERY_CHUNK_SIZE = 16
SUPPORTED_VARIANT_QUERY_DATASETS = (
    "refcoco_val_refseg",
    "refcoco+_testB_refseg",
)
_REDUCE_FIELDS = (
    "stage_sample_count",
    "good_query_count",
    "matched_good_query_count",
    "variant_query_count",
    "matched_gt_count",
    "variant_gt_count",
    "variant_bg_count",
    "variant_wrong_count",
    "matched_gt_probability_sum",
    "variant_gt_probability_sum",
    "selected_good_count",
    "nan_or_inf_count",
    "selected_mask_mismatch_count",
)
_TABLE_FIELDS = (
    "layer",
    "goodQ",
    "varQ",
    "matGT%",
    "varGT%",
    "varBG%",
    "varWrong%",
    "matP",
    "varP",
    "drop",
    "sel7%",
)


def validate_variant_dataset_name(dataset_name: str) -> str:
    """Return a supported single-expression RefCOCO evaluator name."""

    dataset_name = str(dataset_name)
    if dataset_name not in SUPPORTED_VARIANT_QUERY_DATASETS:
        raise ValueError(
            "variant Query classification diagnostics support only one of "
            f"{SUPPORTED_VARIANT_QUERY_DATASETS}, got {dataset_name!r}"
        )
    return dataset_name


def stable_diagnostic_seed(
    dataset_name: str,
    sample_identity: Tuple[int, int],
    stage_index: int,
) -> int:
    """Create a process-independent matcher seed without Python ``hash``."""

    payload = (
        f"{dataset_name}\0{sample_identity[0]}\0"
        f"{sample_identity[1]}\0{stage_index}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


@contextmanager
def _fork_matcher_rng(device: torch.device, seed: int):
    """Make stochastic point matching reproducible without leaking RNG state."""

    devices = []
    if device.type == "cuda":
        devices = [
            torch.cuda.current_device()
            if device.index is None
            else int(device.index)
        ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def pairwise_binary_iou(
    query_masks: torch.Tensor,
    gt_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute exact pairwise IoU for binary ``[Q,H,W]`` and ``[N,H,W]``."""

    if query_masks.ndim != 3 or gt_masks.ndim != 3:
        raise ValueError(
            "pairwise IoU requires [Q,H,W] and [N,H,W], got "
            f"{tuple(query_masks.shape)} and {tuple(gt_masks.shape)}"
        )
    if query_masks.shape[-2:] != gt_masks.shape[-2:]:
        raise ValueError(
            "prediction/GT mask sizes differ after formal restoration: "
            f"{tuple(query_masks.shape[-2:])} != "
            f"{tuple(gt_masks.shape[-2:])}"
        )
    query_flat = query_masks.flatten(1).to(dtype=torch.float32)
    gt_flat = gt_masks.flatten(1).to(
        device=query_masks.device,
        dtype=torch.float32,
    )
    intersection = query_flat @ gt_flat.transpose(0, 1)
    union = (
        query_flat.sum(dim=1, keepdim=True)
        + gt_flat.sum(dim=1).unsqueeze(0)
        - intersection
    )
    return torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(intersection),
    )


def classify_refcoco_queries(
    class_logits: torch.Tensor,
    gt_condition_ids: torch.Tensor,
    loss_cls_type: str,
) -> Dict[str, torch.Tensor]:
    """Apply the original CE/focal classification semantics to each Query."""

    if class_logits.ndim != 2:
        raise ValueError(
            "class logits must be [queries, classes], got "
            f"{tuple(class_logits.shape)}"
        )
    gt_condition_ids = gt_condition_ids.to(
        device=class_logits.device,
        dtype=torch.long,
    )
    if gt_condition_ids.shape != (class_logits.shape[0],):
        raise ValueError(
            "one GT condition id is required per Query: "
            f"{tuple(gt_condition_ids.shape)} != "
            f"{(class_logits.shape[0],)}"
        )
    if bool((gt_condition_ids < 0).any()) or bool(
        (gt_condition_ids >= class_logits.shape[-1]).any()
    ):
        raise ValueError("GT condition id escapes the class-logit table")

    query_indices = torch.arange(
        class_logits.shape[0],
        device=class_logits.device,
    )
    if loss_cls_type == "ce_loss":
        probabilities = class_logits.softmax(dim=-1)
        predicted = probabilities.argmax(dim=-1)
        background_index = class_logits.shape[-1] - 1
        is_gt = predicted.eq(gt_condition_ids)
        is_background = predicted.eq(background_index)
        is_wrong = ~(is_gt | is_background)
        score_type = "softmax"
        has_explicit_background = True
    elif loss_cls_type == "focal_loss":
        probabilities = class_logits.sigmoid()
        gt_probability = probabilities[
            query_indices,
            gt_condition_ids,
        ]
        is_gt = gt_probability.ge(0.5)
        is_background = ~is_gt
        is_wrong = torch.zeros_like(is_gt)
        score_type = "sigmoid"
        has_explicit_background = False
    else:
        raise ValueError(
            "unsupported original classification loss for RefCOCO "
            f"diagnostics: {loss_cls_type!r}"
        )

    gt_probability = probabilities[query_indices, gt_condition_ids]
    return {
        "gt_probability": gt_probability,
        "is_gt": is_gt,
        "is_background": is_background,
        "is_wrong": is_wrong,
        "score_type": score_type,
        "has_explicit_background": has_explicit_background,
    }


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    scale: float = 1.0,
) -> Optional[float]:
    if denominator <= 0:
        return None
    value = scale * numerator / denominator
    if not math.isfinite(value):
        raise ValueError("diagnostic aggregation produced NaN/Inf")
    return float(value)


def summarize_variant_stage(
    raw: Mapping[str, float],
    *,
    single_condition: bool,
) -> Dict[str, Optional[float]]:
    """Convert globally summed stage counters into requested table metrics."""

    sample_count = raw["stage_sample_count"]
    mat_probability = _safe_ratio(
        raw["matched_gt_probability_sum"],
        raw["matched_good_query_count"],
    )
    var_probability = _safe_ratio(
        raw["variant_gt_probability_sum"],
        raw["variant_query_count"],
    )
    drop = (
        None
        if mat_probability is None or var_probability is None
        else mat_probability - var_probability
    )
    return {
        "goodQ": _safe_ratio(raw["good_query_count"], sample_count),
        "varQ": _safe_ratio(raw["variant_query_count"], sample_count),
        "matGT%": _safe_ratio(
            raw["matched_gt_count"],
            raw["matched_good_query_count"],
            scale=100.0,
        ),
        "varGT%": _safe_ratio(
            raw["variant_gt_count"],
            raw["variant_query_count"],
            scale=100.0,
        ),
        "varBG%": _safe_ratio(
            raw["variant_bg_count"],
            raw["variant_query_count"],
            scale=100.0,
        ),
        "varWrong%": (
            None
            if single_condition
            else _safe_ratio(
                raw["variant_wrong_count"],
                raw["variant_query_count"],
                scale=100.0,
            )
        ),
        "matP": mat_probability,
        "varP": var_probability,
        "drop": drop,
        "sel7%": _safe_ratio(
            raw["selected_good_count"],
            sample_count,
            scale=100.0,
        ),
    }


class VariantQueryClassificationDiagnostics:
    """Accumulate the minimal ten-stage diagnostic on a RefCOCO split."""

    def __init__(
        self,
        *,
        segmentor,
        metadata,
        output_dir: str,
        expected_dataset_sample_count: int,
        checkpoint: Optional[str],
        dataset_name: str,
    ):
        dataset_name = validate_variant_dataset_name(dataset_name)
        if bool(
            getattr(
                getattr(segmentor, "dec_config", None),
                "use_topology_group_decoder",
                False,
            )
        ):
            raise ValueError(
                "variant Query classification diagnostics require the "
                "original topology-off MaskLAT model"
            )
        criterion = getattr(segmentor, "criterion", None)
        matcher = getattr(criterion, "matcher", None)
        if criterion is None or matcher is None:
            raise RuntimeError(
                "the standard query criterion and matcher are unavailable"
            )
        loss_cls_type = str(getattr(criterion, "loss_cls_type", ""))
        if loss_cls_type not in {"ce_loss", "focal_loss"}:
            raise ValueError(
                "unsupported original classification loss: "
                f"{loss_cls_type!r}"
            )
        configured_queries = int(
            getattr(segmentor.dec_config, "num_queries", -1)
        )
        if configured_queries != _EXPECTED_QUERIES:
            raise ValueError(
                "this RefCOCO diagnostic requires the original 200-Query "
                f"checkpoint, config reports {configured_queries}"
            )
        configured_stages = int(
            getattr(segmentor.dec_config, "decoder_layers", -1)
        )
        if configured_stages != _NUM_STAGES:
            raise ValueError(
                "this RefCOCO diagnostic requires 10 configured prediction "
                f"stages, config reports {configured_stages}"
            )
        expected_matcher_cost = (
            loss_cls_type.split("_", 1)[0] + "_cost"
        )
        matcher_cost_cls_type = str(
            getattr(matcher, "cost_cls_type", "")
        )
        if matcher_cost_cls_type != expected_matcher_cost:
            raise ValueError(
                "criterion/matcher classification semantics differ: "
                f"{loss_cls_type!r} versus {matcher_cost_cls_type!r}"
            )
        if not output_dir:
            raise ValueError("diagnostic output directory is required")

        self.segmentor = segmentor
        self.matcher = matcher
        self.loss_cls_type = loss_cls_type
        self.score_type = (
            "softmax" if loss_cls_type == "ce_loss" else "sigmoid"
        )
        self.has_explicit_background = loss_cls_type == "ce_loss"
        self.matcher_cost_cls_type = matcher_cost_cls_type
        self.matcher_use_sample_point = bool(
            getattr(matcher, "use_sample_point", False)
        )
        self.output_dir = output_dir
        self.expected_dataset_sample_count = int(
            expected_dataset_sample_count
        )
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self._stats = [
            defaultdict(float) for _ in range(_NUM_STAGES)
        ]
        self._local_sample_ids: List[Tuple[int, int]] = []
        self._local_sample_id_set = set()
        self._device = next(segmentor.parameters()).device
        self._gt_index = self._load_gt_index(metadata)
        if len(self._gt_index) != self.expected_dataset_sample_count:
            raise ValueError(
                "RefCOCO diagnostic GT coverage differs from the dataset: "
                f"{len(self._gt_index)} != "
                f"{self.expected_dataset_sample_count}"
            )

    @staticmethod
    def _load_gt_index(metadata) -> Dict[Tuple[int, int], Dict]:
        gt_json = os.path.realpath(metadata.gt_json)
        with open(gt_json, "r", encoding="utf-8") as file:
            records = json.load(file)
        index = {}
        for record in records:
            image_info = record["image_info"]
            key = (
                int(record["image_id"]),
                int(image_info["sample_id"]),
            )
            if key in index:
                raise ValueError(
                    f"duplicate RefCOCO GT sample identity: {key!r}"
                )
            annotations = record["annotations"]
            if len(annotations) != 1:
                raise ValueError(
                    "RefCOCO diagnostic requires exactly one GT mask per "
                    f"record, got {len(annotations)} for {key!r}"
                )
            index[key] = record
        return index

    def _decode_original_gt(
        self,
        sample_identity: Tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            record = self._gt_index[sample_identity]
        except KeyError as error:
            raise ValueError(
                "RefCOCO diagnostic sample is absent from evaluator GT: "
                f"{sample_identity!r}"
            ) from error
        height, width = record["image_size"]
        mask = decode_mask(
            record["annotations"][0]["segmentation"],
            int(height),
            int(width),
        )
        return torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

    def _match_stage(
        self,
        stage_mask_logits: torch.Tensor,
        stage_class_logits: torch.Tensor,
        target_masks: torch.Tensor,
        target_labels: torch.Tensor,
        sample_identity: Tuple[int, int],
        stage_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = stable_diagnostic_seed(
            self.dataset_name,
            sample_identity,
            stage_index,
        )
        with _fork_matcher_rng(stage_mask_logits.device, seed):
            matched = self.matcher(
                stage_mask_logits.unsqueeze(0),
                stage_class_logits.unsqueeze(0),
                [target_masks],
                [target_labels],
            )
        if len(matched) != 1:
            raise RuntimeError(
                "single-record RefCOCO matcher returned "
                f"{len(matched)} assignments"
            )
        query_indices, target_indices = matched[0]
        query_indices = query_indices.to(stage_mask_logits.device)
        target_indices = target_indices.to(stage_mask_logits.device)
        if query_indices.shape != (1,) or target_indices.shape != (1,):
            raise RuntimeError(
                "single-GT RefCOCO matching must return exactly one pair, "
                f"got queries={tuple(query_indices.shape)} "
                f"targets={tuple(target_indices.shape)}"
            )
        if int(target_indices[0].item()) != 0:
            raise RuntimeError(
                "single-GT RefCOCO matcher returned a nonzero target index"
            )
        if not 0 <= int(query_indices[0].item()) < _EXPECTED_QUERIES:
            raise RuntimeError(
                "RefCOCO matcher returned a Query index outside [0, 199]"
            )
        return (
            query_indices,
            target_indices,
        )

    def _restored_pair_iou(
        self,
        mask_logits: torch.Tensor,
        original_gt: torch.Tensor,
        image_size,
        scaled_size,
        selected_query: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restore Queries in chunks, retaining only IoU and selected mask."""

        iou_chunks = []
        selected_mask = None
        for begin in range(0, mask_logits.shape[0], _RESTORE_QUERY_CHUNK_SIZE):
            end = min(
                begin + _RESTORE_QUERY_CHUNK_SIZE,
                mask_logits.shape[0],
            )
            sam_logits = self.segmentor.postprocess_masks_preds(
                (mask_logits[begin:end].unsqueeze(0),)
            )[0][0]
            restored_logits = restore_refseg_query_masks(
                sam_logits,
                image_size,
                scaled_size,
            )
            binary_masks = binarize_refseg_query_masks(
                restored_logits,
                _MASK_THRESHOLD,
            )
            iou_chunks.append(
                pairwise_binary_iou(binary_masks, original_gt)
            )
            if begin <= selected_query < end:
                selected_mask = binary_masks[
                    selected_query - begin
                ].clone()
            del sam_logits, restored_logits, binary_masks
        if selected_mask is None:
            raise AssertionError(
                f"selected Query {selected_query} was not restored"
            )
        return torch.cat(iou_chunks, dim=0), selected_mask

    @torch.no_grad()
    def update(
        self,
        raw_seg_outputs,
        data_samples,
        formal_seg_outputs: Sequence[Mapping],
    ) -> None:
        stage_classes = getattr(
            raw_seg_outputs,
            "diagnostic_stage_class_logits",
            None,
        )
        stage_masks = getattr(
            raw_seg_outputs,
            "diagnostic_stage_mask_logits",
            None,
        )
        if stage_classes is None or stage_masks is None:
            raise RuntimeError(
                "diagnostic stage tensors were not returned by the same "
                "standard model forward"
            )
        if len(stage_classes) != _NUM_STAGES or len(stage_masks) != _NUM_STAGES:
            raise ValueError(
                "the query diagnostic requires exactly 10 prediction "
                f"stages, got classes={len(stage_classes)} "
                f"masks={len(stage_masks)}"
            )

        image_infos = data_samples.metainfo["image_infos"]
        image_sizes = data_samples.metainfo["image_sizes"]
        scaled_sizes = data_samples.metainfo["scaled_sizes"]
        target_masks_batch = data_samples.mask_labels
        target_labels_batch = data_samples.class_labels
        batch_size = int(stage_classes[0].shape[0])
        lengths = {
            len(image_infos),
            len(image_sizes),
            len(scaled_sizes),
            len(target_masks_batch),
            len(target_labels_batch),
            len(formal_seg_outputs),
            batch_size,
        }
        if len(lengths) != 1:
            raise ValueError(
                "RefCOCO diagnostic batch fields disagree: "
                f"lengths={sorted(lengths)}"
            )

        for batch_index in range(batch_size):
            image_info = image_infos[batch_index]
            sample_identity = (
                int(image_info["image_id"]),
                int(image_info["sample_id"]),
            )
            if sample_identity in self._local_sample_id_set:
                raise ValueError(
                    "duplicate RefCOCO sample on one rank: "
                    f"{sample_identity!r}"
                )
            self._local_sample_ids.append(sample_identity)
            self._local_sample_id_set.add(sample_identity)
            original_gt = self._decode_original_gt(
                sample_identity,
                device=stage_masks[0].device,
            )
            target_masks = target_masks_batch[batch_index]
            target_labels = target_labels_batch[batch_index].to(
                device=stage_masks[0].device,
                dtype=torch.long,
            )
            if target_masks.shape[0] != 1 or target_labels.shape != (1,):
                raise ValueError(
                    f"{self.dataset_name} must contain one GT and one Cond id "
                    f"per record, got masks={tuple(target_masks.shape)} "
                    f"labels={tuple(target_labels.shape)}"
                )
            if int(target_labels[0].item()) != 0:
                raise ValueError(
                    "RefCOCO evaluation target Cond must be source-local id 0"
                )

            for stage_index, (class_batch, mask_batch) in enumerate(
                zip(stage_classes, stage_masks)
            ):
                if class_batch.shape[:2] != (
                    batch_size,
                    _EXPECTED_QUERIES,
                ):
                    raise ValueError(
                        f"stage{stage_index} class shape must start with "
                        f"[{batch_size},200], got {tuple(class_batch.shape)}"
                    )
                if class_batch.ndim != 3 or class_batch.shape[-1] != 2:
                    raise ValueError(
                        f"stage{stage_index} original RefCOCO class logits "
                        "must be [batch,200,2] for Cond/background, got "
                        f"{tuple(class_batch.shape)}"
                    )
                if mask_batch.shape[:2] != (
                    batch_size,
                    _EXPECTED_QUERIES,
                ):
                    raise ValueError(
                        f"stage{stage_index} mask shape must start with "
                        f"[{batch_size},200], got {tuple(mask_batch.shape)}"
                    )
                class_logits = class_batch[batch_index].detach()
                mask_logits = mask_batch[batch_index].detach()
                nonfinite = (
                    (~torch.isfinite(class_logits)).sum()
                    + (~torch.isfinite(mask_logits)).sum()
                )
                if int(nonfinite.item()) != 0:
                    raise ValueError(
                        f"stage{stage_index} contains "
                        f"{int(nonfinite.item())} NaN/Inf values"
                    )

                matched_queries, _ = self._match_stage(
                    mask_logits,
                    class_logits,
                    target_masks,
                    target_labels,
                    sample_identity,
                    stage_index,
                )
                _, selected_query_tensor = select_refseg_query(
                    class_logits
                )
                selected_query = int(selected_query_tensor.item())
                pair_iou, selected_mask = self._restored_pair_iou(
                    mask_logits,
                    original_gt,
                    image_sizes[batch_index],
                    scaled_sizes[batch_index],
                    selected_query,
                )
                if not bool(torch.isfinite(pair_iou).all()):
                    raise ValueError(
                        f"stage{stage_index} pair IoU contains NaN/Inf"
                    )
                if pair_iou.shape != (_EXPECTED_QUERIES, 1):
                    raise ValueError(
                        f"stage{stage_index} pair IoU must be [200,1], got "
                        f"{tuple(pair_iou.shape)}"
                    )
                best_iou, best_gt = pair_iou.max(dim=1)
                if bool(best_gt.ne(0).any()):
                    raise AssertionError(
                        "single-GT RefCOCO best-GT indices must all be zero"
                    )
                gt_condition_ids = target_labels[best_gt]
                good = best_iou.ge(_GOOD_IOU_THRESHOLD)
                matched_mask = torch.zeros(
                    _EXPECTED_QUERIES,
                    dtype=torch.bool,
                    device=good.device,
                )
                matched_mask[matched_queries] = True
                matched_good = good & matched_mask
                variant = good & ~matched_mask

                classification = classify_refcoco_queries(
                    class_logits,
                    gt_condition_ids,
                    self.loss_cls_type,
                )
                if classification["score_type"] != self.score_type:
                    raise AssertionError("classification score type changed")
                if (
                    classification["has_explicit_background"]
                    != self.has_explicit_background
                ):
                    raise AssertionError(
                        "classification background semantics changed"
                    )
                gt_probability = classification["gt_probability"]
                if not bool(torch.isfinite(gt_probability).all()):
                    raise ValueError(
                        f"stage{stage_index} probabilities contain NaN/Inf"
                    )

                selected_good = bool(good[selected_query].item())
                stats = self._stats[stage_index]
                stats["stage_sample_count"] += 1
                stats["good_query_count"] += int(good.sum().item())
                stats["matched_good_query_count"] += int(
                    matched_good.sum().item()
                )
                stats["variant_query_count"] += int(variant.sum().item())
                stats["matched_gt_count"] += int(
                    (
                        matched_good
                        & classification["is_gt"]
                    ).sum().item()
                )
                stats["variant_gt_count"] += int(
                    (variant & classification["is_gt"]).sum().item()
                )
                stats["variant_bg_count"] += int(
                    (
                        variant
                        & classification["is_background"]
                    ).sum().item()
                )
                stats["variant_wrong_count"] += int(
                    (
                        variant
                        & classification["is_wrong"]
                    ).sum().item()
                )
                stats["matched_gt_probability_sum"] += float(
                    gt_probability[matched_good].sum().item()
                )
                stats["variant_gt_probability_sum"] += float(
                    gt_probability[variant].sum().item()
                )
                stats["selected_good_count"] += int(selected_good)

                if stage_index == _NUM_STAGES - 1:
                    formal_mask = formal_seg_outputs[batch_index][
                        "segmentation"
                    ].eq(1)
                    if not torch.equal(selected_mask, formal_mask):
                        stats["selected_mask_mismatch_count"] += 1
                        raise ValueError(
                            "diagnostic stage9 selected mask differs from "
                            "the formal RefCOCO selected mask"
                        )

    def _global_reduce(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        gathered_ids = comm.all_gather(tuple(self._local_sample_ids))
        global_ids = [
            sample_id
            for rank_ids in gathered_ids
            for sample_id in rank_ids
        ]
        if len(global_ids) != len(set(global_ids)):
            raise ValueError(
                "distributed RefCOCO diagnostic encountered duplicate "
                "sample identities"
            )
        if len(global_ids) != self.expected_dataset_sample_count:
            raise ValueError(
                "global RefCOCO diagnostic sample count differs from the "
                f"dataset: {len(global_ids)} != "
                f"{self.expected_dataset_sample_count}"
            )

        reduced = torch.tensor(
            [
                [
                    self._stats[stage_index][field]
                    for field in _REDUCE_FIELDS
                ]
                for stage_index in range(_NUM_STAGES)
            ],
            dtype=torch.float64,
            device=self._device,
        )
        if comm.get_world_size() > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced.cpu(), global_ids

    @staticmethod
    def _format_value(
        field: str,
        value: Optional[float],
    ) -> str:
        if value is None:
            return "NA"
        if field in {"goodQ", "varQ"}:
            return f"{value:.2f}"
        if field.endswith("%"):
            return f"{value:.2f}"
        return f"{value:.4f}"

    @classmethod
    def _format_table(cls, rows: Sequence[Mapping]) -> str:
        widths = {
            "layer": 5,
            "goodQ": 7,
            "varQ": 7,
            "matGT%": 8,
            "varGT%": 8,
            "varBG%": 8,
            "varWrong%": 11,
            "matP": 7,
            "varP": 7,
            "drop": 7,
            "sel7%": 8,
        }
        header = " ".join(
            f"{field:>{widths[field]}}" for field in _TABLE_FIELDS
        )
        lines = [
            "Variant Query Classification Diagnostics",
            "good IoU threshold = 0.70",
            "",
            header,
        ]
        for row in rows:
            rendered = []
            for field in _TABLE_FIELDS:
                value = (
                    str(row[field])
                    if field == "layer"
                    else cls._format_value(field, row[field])
                )
                rendered.append(f"{value:>{widths[field]}}")
            lines.append(" ".join(rendered))
        return "\n".join(lines)

    def finalize_and_write(self) -> Optional[Dict]:
        """Reduce all ranks, validate coverage, and write only on rank zero."""

        reduced, global_ids = self._global_reduce()
        world_size = comm.get_world_size()
        rows = []
        raw_stage_records = []
        total_nonfinite = 0
        total_mask_mismatches = 0
        for stage_index in range(_NUM_STAGES):
            raw = {
                field: float(reduced[stage_index, field_index].item())
                for field_index, field in enumerate(_REDUCE_FIELDS)
            }
            if int(raw["stage_sample_count"]) != len(global_ids):
                raise ValueError(
                    f"stage{stage_index} sample count differs from global "
                    f"coverage: {int(raw['stage_sample_count'])} != "
                    f"{len(global_ids)}"
                )
            if raw["good_query_count"] != (
                raw["matched_good_query_count"]
                + raw["variant_query_count"]
            ):
                raise ValueError(
                    f"stage{stage_index} good Query partition is not closed"
                )
            if raw["variant_query_count"] != (
                raw["variant_gt_count"]
                + raw["variant_bg_count"]
                + raw["variant_wrong_count"]
            ):
                raise ValueError(
                    f"stage{stage_index} variant classification partition "
                    "is not closed"
                )
            if raw["variant_wrong_count"] != 0:
                raise ValueError(
                    f"stage{stage_index} single-Cond RefCOCO produced an "
                    "other-Cond classification"
                )
            if raw["matched_good_query_count"] > raw["stage_sample_count"]:
                raise ValueError(
                    f"stage{stage_index} has more matched-good Queries than "
                    "single-GT samples"
                )
            if raw["selected_good_count"] > raw["stage_sample_count"]:
                raise ValueError(
                    f"stage{stage_index} has more selected-good Queries than "
                    "samples"
                )
            for probability_sum_field, count_field in (
                (
                    "matched_gt_probability_sum",
                    "matched_good_query_count",
                ),
                (
                    "variant_gt_probability_sum",
                    "variant_query_count",
                ),
            ):
                probability_sum = raw[probability_sum_field]
                probability_count = raw[count_field]
                if (
                    probability_sum < -1e-8
                    or probability_sum > probability_count + 1e-8
                ):
                    raise ValueError(
                        f"stage{stage_index} {probability_sum_field} escapes "
                        f"[0, {probability_count}]"
                    )
            total_nonfinite += int(raw["nan_or_inf_count"])
            total_mask_mismatches += int(
                raw["selected_mask_mismatch_count"]
            )
            metrics = summarize_variant_stage(
                raw,
                single_condition=True,
            )
            rows.append({"layer": stage_index, **metrics})
            raw_stage_records.append(
                {
                    "layer": stage_index,
                    "metrics": metrics,
                    "counts": {
                        field: int(raw[field])
                        if not field.endswith("_sum")
                        else raw[field]
                        for field in _REDUCE_FIELDS
                    },
                }
            )
        if total_nonfinite:
            raise ValueError(
                f"diagnostic contains {total_nonfinite} NaN/Inf values"
            )
        if total_mask_mismatches:
            raise ValueError(
                "diagnostic/formal selected masks differ for "
                f"{total_mask_mismatches} stage9 samples"
            )

        sanity = {
            "stages": _NUM_STAGES,
            "queries": _EXPECTED_QUERIES,
            "global_samples": len(global_ids),
            "expected_dataset_samples": (
                self.expected_dataset_sample_count
            ),
            "distributed_aggregation": "PASS",
            "selected_stage9_mask_matches_formal": "PASS",
            "nan_or_inf_count": total_nonfinite,
        }
        report = {
            "schema_version": 1,
            "dataset_name": self.dataset_name,
            "checkpoint": self.checkpoint,
            "world_size": world_size,
            "global_sample_count": len(global_ids),
            "expected_dataset_sample_count": (
                self.expected_dataset_sample_count
            ),
            "good_iou_threshold": _GOOD_IOU_THRESHOLD,
            "mask_probability_threshold": _MASK_THRESHOLD,
            "score_type": self.score_type,
            "loss_cls_type": self.loss_cls_type,
            "has_explicit_background": self.has_explicit_background,
            "matcher_cost_cls_type": self.matcher_cost_cls_type,
            "matcher_use_sample_point": self.matcher_use_sample_point,
            "stages": raw_stage_records,
            "sanity_checks": sanity,
        }

        if comm.get_rank() != 0:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(
            self.output_dir,
            "variant_query_classification.csv",
        )
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_TABLE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: (
                            row[field]
                            if field == "layer"
                            else self._format_value(field, row[field])
                        )
                        for field in _TABLE_FIELDS
                    }
                )
        json_path = os.path.join(
            self.output_dir,
            "variant_query_classification.json",
        )
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")

        sanity_lines = [
            "Sanity Checks",
            f"- stages == {_NUM_STAGES}",
            f"- queries == {_EXPECTED_QUERIES}",
            (
                "- global_samples == dataset_samples: "
                f"{len(global_ids)} == "
                f"{self.expected_dataset_sample_count}"
            ),
            "- distributed_aggregation == PASS",
            "- selected_stage9_mask_matches_formal == PASS",
            "- nan_or_inf_count == 0",
            f"- CSV: {csv_path}",
            f"- JSON: {json_path}",
        ]
        print_log(
            self._format_table(rows)
            + "\n\n"
            + "\n".join(sanity_lines),
            logger="current",
        )
        return report
