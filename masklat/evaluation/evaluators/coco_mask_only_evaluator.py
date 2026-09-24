"""Structured COCO mask metrics with complete, all-rank image coverage.

Original instance/panoptic processing and dataset filtering are retained.
Semantic labels are mapped once into a shared contiguous space, with GT
derived from the original panoptic PNG/JSON. Instance coverage is measured
against metadata.gt_json, which can be the existing GenSegDataset's filtered
per-image annotation list.
"""

from copy import deepcopy
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..utils import comm
from ..utils.map import convert_to_coco_json
from ..utils.pq import pq_compute, pq_compute_single_core
from .genseg_evaluator import GenSegEvaluator
from .mask_only_common import atomic_json, validate_image_coverage, write_summary


_DATASETS = {
    "coco_panoptic_genseg": "panoptic",
    "coco_instance_genseg": "instance",
    "coco_panoptic_semantic_genseg": "semantic",
}


def _official_mask_ap(coco_gt, results):
    """Use official mask-area COCOeval, including an empty detection table."""
    results = deepcopy(results)
    for result in results:
        result.pop("bbox", None)
    with contextlib.redirect_stdout(io.StringIO()):
        if results:
            coco_dt = coco_gt.loadRes(results)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": deepcopy(coco_gt.dataset["images"]),
                "categories": deepcopy(coco_gt.dataset["categories"]),
                "annotations": [],
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "segm")
        evaluator.params.imgIds = sorted(coco_gt.getImgIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    if not np.all(np.isfinite(evaluator.stats)):
        raise ValueError("COCOeval returned non-finite metric values")
    return {
        name: float(value * 100.0) if value >= 0 else None
        for name, value in zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), evaluator.stats)
    }


def _semantic_metrics(confusion, num_classes):
    # These operations intentionally match GenSegEvaluator.semantic_evaluate,
    # including its GT-supported class set and FP32 arithmetic before reporting.
    acc = np.full(num_classes, np.nan, dtype=np.float32)
    iou = np.full(num_classes, np.nan, dtype=np.float32)
    tp = confusion.diagonal()[:-1].astype(np.float32)
    pos_gt = np.sum(confusion[:-1, :-1], axis=0).astype(np.float32)
    pos_pred = np.sum(confusion[:-1, :-1], axis=1).astype(np.float32)
    if not np.sum(pos_gt) > 0:
        return dict.fromkeys(("mIoU", "fwIoU", "mACC", "pACC"))
    class_weights = pos_gt / np.sum(pos_gt)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    union = pos_gt + pos_pred - tp
    iou_valid = np.logical_and(acc_valid, union > 0)
    iou[iou_valid] = tp[iou_valid] / union[iou_valid]
    values = (
        np.sum(iou[iou_valid]) / np.sum(iou_valid),
        np.sum(iou[iou_valid] * class_weights[iou_valid]),
        np.sum(acc[acc_valid]) / np.sum(acc_valid),
        np.sum(tp) / np.sum(pos_gt),
    )
    metrics = {}
    for name, value in zip(("mIoU", "fwIoU", "mACC", "pACC"), values):
        if not np.isfinite(value) or not -1e-6 <= value <= 1 + 1e-6:
            raise ValueError(f"Invalid semantic metric {name}: {value}")
        # FP32 class weights can sum to 1 + a tiny rounding tail for a perfect
        # prediction. Preserve normal values exactly; bound only that tail.
        metrics[name] = float(100 * min(max(value, 0.0), 1.0))
    return metrics


class COCOMaskOnlyEvaluator(GenSegEvaluator):
    def __init__(self, data_name, **kwargs):
        if data_name not in _DATASETS:
            raise ValueError(f"Unsupported COCO mask-only dataset: {data_name}")
        super().__init__(data_name=data_name, **kwargs)
        self._mask_metric_kind = _DATASETS[data_name]

    def reset(self):
        super().reset()
        self._observed_image_ids = []
        self._empty_mask_images = 0
        self._semantic_gt_annotations = None

    def semantic_process(self, inputs, outputs):
        """Map sampled COCO IDs once; derive semantic GT from panoptic labels.

        The legacy semantic path replaces local indices in-place with sparse
        COCO IDs. Here both predictions and GT use the metadata's contiguous
        class space, without changing the confusion-matrix metric formulas.
        """
        from panopticapi.utils import rgb2id

        mapping = self._metadata.dataset_id_to_contiguous_id
        if sorted(mapping.values()) != list(range(self._num_classes)):
            raise ValueError("Semantic COCO labels must be contiguous from zero")
        if self._semantic_gt_annotations is None:
            with Path(self._metadata.gt_json).open(encoding="utf-8") as source:
                ground_truth = json.load(source)
            validate_image_coverage(self._ground_truth_ids(ground_truth),
                                    [item["image_id"] for item in ground_truth["annotations"]])
            self._semantic_gt_annotations = {item["image_id"]: item for item in ground_truth["annotations"]}

        for input_, output in zip(inputs, outputs):
            local_prediction = output["segmentation"].to(self._cpu_device).numpy().astype(np.int64)
            if local_prediction.ndim != 2 or not local_prediction.size:
                raise ValueError("Semantic prediction must be a nonempty H x W label map")
            sampled_labels = output["sampled_labels"]
            if sampled_labels is None:
                prediction = local_prediction.copy()
                if np.any((prediction < 0) | (prediction >= self._num_classes)):
                    raise ValueError("Semantic prediction contains an invalid contiguous class")
            else:
                labels = [int(label) for label in sampled_labels]
                if np.any((local_prediction < 0) | (local_prediction >= len(labels))):
                    raise ValueError("Semantic prediction contains an invalid sampled-label index")
                try:
                    lookup = np.asarray([mapping[label] for label in labels], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError(f"Unknown sampled COCO semantic category: {exc.args[0]}") from exc
                prediction = lookup[local_prediction]

            annotation = self._semantic_gt_annotations[input_["image_id"]]
            panoptic = rgb2id(np.asarray(Image.open(
                Path(self._metadata.panseg_map_folder) / annotation["file_name"]), dtype=np.uint32))
            ground_truth = np.full(panoptic.shape, self._num_classes, dtype=np.int64)
            segments = {segment["id"]: segment for segment in annotation["segments_info"]}
            if len(segments) != len(annotation["segments_info"]):
                raise ValueError("Duplicate semantic GT panoptic segment IDs")
            for segment_id in np.unique(panoptic):
                if segment_id == 0:
                    continue
                if segment_id not in segments:
                    raise ValueError(f"GT PNG has an unknown panoptic segment: {segment_id}")
                category = segments[segment_id]["category_id"]
                if category not in mapping:
                    raise ValueError(f"Unknown GT COCO semantic category: {category}")
                ground_truth[panoptic == segment_id] = mapping[category]
            if prediction.shape != ground_truth.shape:
                raise ValueError("Semantic prediction and GT sizes do not match")
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * prediction.reshape(-1) + ground_truth.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            self._predictions.extend(self._encode_json_sem_seg(prediction, input_["file_name"]))

    def process(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise ValueError("COCO inputs and outputs must have equal image counts")
        image_ids = [item["image_id"] for item in inputs]
        if self._mask_metric_kind == "instance" and any("instances" not in item for item in outputs):
            raise ValueError("COCO instance output requires instances, including an empty instance container")
        before = len(self._predictions)
        super().process(inputs, outputs)
        self._observed_image_ids.extend(image_ids)
        if self._mask_metric_kind == "instance":
            self._empty_mask_images += sum(len(item["instances"]) == 0 for item in outputs)
        elif self._mask_metric_kind == "panoptic":
            self._empty_mask_images += sum(not item["segments_info"] for item in self._predictions[before:])
        # Semantic postprocessing emits a valid class for every pixel. A class-0
        # map is a real semantic prediction, never an empty/background flag.

    @staticmethod
    def _ground_truth_ids(ground_truth):
        if isinstance(ground_truth, list):
            return [item["image_id"] for item in ground_truth]
        if isinstance(ground_truth, dict):
            return [item["id"] for item in ground_truth["images"]]
        raise ValueError("COCO metadata GT must be an image list or COCO JSON object")

    def _instance_metrics(self, predictions, ground_truth, output_dir):
        validate_image_coverage(self._ground_truth_ids(ground_truth),
                                [item["image_id"] for item in predictions])
        atomic_json(output_dir / "predictions.json", predictions)
        converted_path = output_dir / f"{self.data_name}_coco_format.json"
        if isinstance(ground_truth, list):
            convert_to_coco_json(self.data_name, str(converted_path), ground_truth, allow_cached=False)
        else:
            converted = deepcopy(ground_truth)
            converted.setdefault("annotations", [])
            converted.setdefault("info", {})
            atomic_json(converted_path, converted)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(converted_path))
        mapping = self._metadata.thing_dataset_id_to_contiguous_id
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("COCO thing labels must be contiguous from zero")
        reverse_mapping = {value: key for key, value in mapping.items()}
        results = []
        for prediction in predictions:
            for original in prediction["instances"]:
                if original["category_id"] in reverse_mapping:
                    result = deepcopy(original)
                    result["category_id"] = reverse_mapping[result["category_id"]]
                    results.append(result)
        return _official_mask_ap(coco_gt, results)

    def _panoptic_metrics(self, predictions, ground_truth, output_dir):
        expected_ids = self._ground_truth_ids(ground_truth)
        validate_image_coverage(expected_ids, [item["image_id"] for item in ground_truth["annotations"]])
        validate_image_coverage(expected_ids, [item["image_id"] for item in predictions])
        with tempfile.TemporaryDirectory(prefix="coco_mask_only_panoptic_") as temporary:
            prediction_dir = Path(temporary)
            annotations = []
            file_names = set()
            for original in predictions:
                item = deepcopy(original)
                file_name = item["file_name"]
                if Path(file_name).name != file_name or file_name in file_names:
                    raise ValueError("Panoptic PNG names must be unique basenames")
                file_names.add(file_name)
                (prediction_dir / file_name).write_bytes(item.pop("png_string"))
                annotations.append(item)
            predicted_json = deepcopy(ground_truth)
            predicted_json["annotations"] = annotations
            prediction_path = output_dir / "predictions.json"
            atomic_json(prediction_path, predicted_json)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    result = pq_compute(
                        str(self._metadata.gt_json), str(prediction_path),
                        gt_folder=str(self._metadata.panseg_map_folder), pred_folder=str(prediction_dir),
                    )
                except ZeroDivisionError:
                    # The original reducer divides by zero for unsupported
                    # groups. Reuse its exact matching/counting and reduce only
                    # supported groups; any other data/metric error still fails.
                    by_id = {item["image_id"]: item for item in annotations}
                    pairs = [(item, by_id[item["image_id"]]) for item in ground_truth["annotations"]]
                    categories = {item["id"]: item for item in ground_truth["categories"]}
                    counts = pq_compute_single_core(0, pairs, str(self._metadata.panseg_map_folder),
                                                    str(prediction_dir), categories)
                    result = {}
                    unsupported = False
                    for name, isthing in (("All", None), ("Things", True), ("Stuff", False)):
                        supported = any(
                            (isthing is None or bool(category["isthing"]) == isthing)
                            and counts[label].tp + counts[label].fp + counts[label].fn > 0
                            for label, category in categories.items()
                        )
                        if supported:
                            result[name] = counts.pq_average(categories, isthing)[0]
                        else:
                            unsupported = True
                            result[name] = {"n": 0}
                    if not unsupported:
                        raise
        metrics = {}
        for group, suffix in (("All", ""), ("Things", "_th"), ("Stuff", "_st")):
            for metric in ("PQ", "SQ", "RQ"):
                metrics[metric + suffix] = (
                    float(result[group][metric.lower()] * 100.0) if result[group]["n"] else None
                )
        return metrics

    def evaluate(self):
        local = (self._conf_matrix, self._predictions, self._observed_image_ids, self._empty_mask_images)
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(local, dst=0)
            if not comm.is_main_process():
                return {}
            world_size = comm.get_world_size()
            if len(gathered) != world_size:
                raise ValueError("COCO metric gather did not include every distributed rank")
        else:
            gathered = [local]
            world_size = 1
        confusion = sum((entry[0] for entry in gathered), np.zeros_like(self._conf_matrix))
        predictions = list(itertools.chain.from_iterable(entry[1] for entry in gathered))
        image_ids = list(itertools.chain.from_iterable(entry[2] for entry in gathered))
        with Path(self._metadata.gt_json).open(encoding="utf-8") as source:
            ground_truth = json.load(source)
        coverage = validate_image_coverage(self._ground_truth_ids(ground_truth), image_ids)
        coverage["empty_mask_images"] = sum(entry[3] for entry in gathered)
        with tempfile.TemporaryDirectory(prefix="coco_mask_only_") as temporary:
            output_dir = Path(self._output_dir or temporary)
            output_dir.mkdir(parents=True, exist_ok=True)
            if self._mask_metric_kind == "instance":
                metrics = self._instance_metrics(predictions, ground_truth, output_dir)
            elif self._mask_metric_kind == "panoptic":
                metrics = self._panoptic_metrics(predictions, ground_truth, output_dir)
            else:
                predicted_json = deepcopy(ground_truth)
                predicted_json["annotations"] = predictions
                atomic_json(output_dir / "predictions.json", predicted_json)
                metrics = _semantic_metrics(confusion, self._num_classes)
            return write_summary(
                output_dir, data_name=self.data_name, metrics=metrics, coverage=coverage,
                world_size=world_size, protocol_details={
                    "task": self._mask_metric_kind,
                    "ground_truth": "metadata.gt_json; instance GT converted from original filtered dataset",
                    "coverage_basis": "metadata.gt_json images; preserve original GenSegDataset filtering",
                    "class_mapping": "original instance/panoptic postprocessing; semantic uses metadata contiguous IDs",
                    "instance_max_dets": [1, 10, 100],
                    "instance_area": "mask area; predicted bbox removed before COCOeval",
                    "semantic_classes": "original GT-supported classes; class 0 remains a valid class",
                    "semantic_ground_truth": "original panoptic PNG segments mapped through COCO category IDs; VOID ignored",
                    "semantic_label_mapping": "sampled labels mapped once to metadata contiguous IDs (no in-place substitution)",
                },
            )
