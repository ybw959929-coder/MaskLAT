import itertools
import json
import logging
import os
import os.path as osp
from typing import List, Optional

import numpy as np

from masklat.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.mask import calculate_iou, decode_mask, encode_mask
from ..utils import comm
from ..utils.iou import IouStat
from .base_evaluator import BaseEvaluator


class RefSegEvaluator(BaseEvaluator):

    def __init__(
        self,
        data_name: str = "refseg",
        cat_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        distributed: bool = True,
        terminal_summary_only: bool = False,
    ):
        """
        Args:
            metadata: metadata of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._distributed = distributed
        self._data_name = data_name
        self._metadata = MetadataCatalog.get(data_name)
        self._output_dir = output_dir
        self._terminal_summary_only = terminal_summary_only
        self.iou_stat = IouStat(
            cat_names=("ignore", "refer") if cat_names is None else cat_names
        )

        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)

    @property
    def data_name(self):
        return self._data_name

    def reset(self):
        self._predictions = []
        self.iou_stat.reset()

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise ValueError(
                "RefSeg evaluator inputs/outputs differ in length: "
                f"{len(inputs)} != {len(outputs)}"
            )
        for input, output in zip(inputs, outputs):
            pred_mask, segments_info = (
                output["segmentation"],
                output["segments_info"],
            )
            pred_mask = pred_mask.cpu().numpy()
            pred_mask[pred_mask == self._metadata.ignore_label] = 0
            pred_mask = pred_mask.astype(np.uint8)
            file_name = os.path.basename(input["file_name"])
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "sample_id": input["sample_id"],
                    "annotation_id": input.get("annotation_id"),
                    "sentence_id": input.get("sentence_id"),
                    "phrases": input.get("phrases"),
                    "file_name": file_name,
                    "pred_mask": encode_mask(pred_mask),
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")
        if len(predictions) == 0:
            logging.warning(f"{self.__class__.__name__} did not receive valid predictions.")
            return {}

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "predictions.json")
            print_log(f"Writing {self.data_name} predictions to {self._output_dir}...", logger="current")
            with open(file_path, "w") as f:
                json.dump(predictions, f)

        gt_json = osp.realpath(self._metadata.gt_json)

        summary = self._eval_predictions(predictions, gt_json)
        summary.update(
            {
                "schema_version": 2,
                "data_name": self.data_name,
                "num_predictions": len(predictions),
            }
        )
        if self._output_dir:
            summary_path = osp.join(self._output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
                file.write("\n")
            print_log(
                f"Writing {self.data_name} summary to {summary_path}",
                logger="current",
            )
        return summary

    def _eval_predictions(
        self,
        predictions,
        gt_json,
        *,
        require_complete: bool = True,
    ):
        with open(gt_json, "r") as f:
            gt_anns = json.load(f)

        def sample_key(image_id, sample_id):
            # A tuple avoids ambiguous concatenations such as
            # image=12/sample=34 versus image=123/sample=4.
            return image_id, sample_id

        gt_map = {}
        for data in gt_anns:
            image_info = data["image_info"]
            key = sample_key(data["image_id"], image_info["sample_id"])
            if key in gt_map:
                raise ValueError(f"duplicate RefSeg GT identity: {key!r}")
            annotations = data["annotations"]
            if len(annotations) != 1:
                raise ValueError(
                    "RefSeg evaluation requires exactly one annotation per "
                    f"expression, got {len(annotations)} for {key!r}"
                )
            gt_map[key] = data

        prediction_map = {}
        for pred in predictions:
            key = sample_key(pred["image_id"], pred["sample_id"])
            if key in prediction_map:
                raise ValueError(f"duplicate RefSeg prediction identity: {key!r}")
            prediction_map[key] = pred

        missing = set(gt_map) - set(prediction_map)
        unexpected = set(prediction_map) - set(gt_map)
        if unexpected or (require_complete and missing):
            raise ValueError(
                "RefSeg prediction/GT identities differ: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_examples={list(missing)[:3]} "
                f"unexpected_examples={list(unexpected)[:3]}"
            )

        for pred in predictions:
            image_id = pred["image_id"]
            sample_id = pred["sample_id"]
            key = sample_key(image_id, sample_id)
            gt_data = gt_map[key]
            gt_image_info = gt_data["image_info"]

            # Distributed ranks construct validation datasets independently.
            # Verify the semantic identity as well as the numeric sample id so
            # a rank-local order divergence can never silently score a mask
            # against the wrong expression/object.
            expected_annotation_id = gt_image_info.get("annotation_id")
            if pred.get("annotation_id") != expected_annotation_id:
                raise ValueError(
                    "RefSeg prediction/GT annotation identity mismatch for "
                    f"{key!r}: prediction={pred.get('annotation_id')!r}, "
                    f"gt={expected_annotation_id!r}"
                )
            expected_phrases = gt_image_info.get("phrases")
            expected_sentence_id = gt_image_info.get("sentence_id")
            if pred.get("sentence_id") != expected_sentence_id:
                raise ValueError(
                    "RefSeg prediction/GT sentence identity mismatch for "
                    f"{key!r}: prediction={pred.get('sentence_id')!r}, "
                    f"gt={expected_sentence_id!r}"
                )
            if pred.get("phrases") != expected_phrases:
                raise ValueError(
                    "RefSeg prediction/GT phrase identity mismatch for "
                    f"{key!r}: prediction={pred.get('phrases')!r}, "
                    f"gt={expected_phrases!r}"
                )

            pred_mask = pred["pred_mask"]
            height, width = pred_mask["size"]
            pred_mask = decode_mask(pred_mask, height, width)

            # segmentation is polygon
            gt_mask = gt_data["annotations"][0]["segmentation"]
            gt_mask = decode_mask(gt_mask, height, width)

            intersection, union, _ = calculate_iou(pred_mask, gt_mask, 2, self._metadata.ignore_label)
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        if not self._terminal_summary_only:
            print_log(
                f"{self.data_name} evaluation results:\n{self.iou_stat}",
                logger="current",
            )
        return {
            "evaluation_protocol": "one_prediction_per_referring_expression",
            "num_ground_truth": len(gt_anns),
            "num_predictions_evaluated": len(predictions),
            "complete_dataset": not missing and not unexpected,
            "metric_unit": "percent",
            "metric_range": [0.0, 100.0],
            "metric_definitions": {
                "cIoU": "100 * sum(intersection) / sum(union)",
                "gIoU": "100 * mean(per-expression intersection / union)",
            },
            "class_semantics": {
                self.iou_stat.cat_names[0]: (
                    "valid class-0 background pixels; despite the legacy "
                    "name, these pixels participate in IoU"
                ),
                self.iou_stat.cat_names[1]: "class-1 target foreground",
                "ignore_index": (
                    f"label {self._metadata.ignore_label}; excluded from IoU"
                ),
            },
            "metrics": {
                cat_name: {
                    "cIoU": float(self.iou_stat.ciou[index]),
                    "gIoU": float(self.iou_stat.giou[index]),
                }
                for index, cat_name in enumerate(self.iou_stat.cat_names)
            },
            "sufficient_statistics": {
                cat_name: {
                    "intersection": float(self.iou_stat.intersection[index]),
                    "union": float(self.iou_stat.union[index]),
                    "sample_iou_sum": float(self.iou_stat.acc_iou[index]),
                    "sample_count": int(self.iou_stat.count),
                }
                for index, cat_name in enumerate(self.iou_stat.cat_names)
            },
        }
