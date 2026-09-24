import json

import numpy as np

from masklat.utils.logging import print_log

from ...dataset.utils.mask import calculate_iou, decode_mask
from .refseg_evaluator import RefSegEvaluator


class ReaSegEvaluator(RefSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cat_names=["ignore", "reason"], **kwargs)

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
            # Keep image/sample components separate.  String concatenation can
            # alias distinct identities, for example (12, 34) and (123, 4).
            return image_id, sample_id

        gt_map = {}
        for data in gt_anns:
            image_info = data["image_info"]
            key = sample_key(data["image_id"], image_info["sample_id"])
            if key in gt_map:
                raise ValueError(f"duplicate ReaSeg GT identity: {key!r}")
            annotations = data["annotations"]
            if len(annotations) != 1:
                raise ValueError(
                    "ReaSeg evaluation requires exactly one annotation per "
                    f"reasoning question, got {len(annotations)} for {key!r}"
                )
            gt_map[key] = data

        prediction_map = {}
        for pred in predictions:
            key = sample_key(pred["image_id"], pred["sample_id"])
            if key in prediction_map:
                raise ValueError(
                    f"duplicate ReaSeg prediction identity: {key!r}"
                )
            prediction_map[key] = pred

        missing = set(gt_map) - set(prediction_map)
        unexpected = set(prediction_map) - set(gt_map)
        if unexpected or (require_complete and missing):
            raise ValueError(
                "ReaSeg prediction/GT identities differ: "
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
            gt_annotation = gt_data["annotations"][0]

            # Match RefSeg's semantic identity contract.  Archived LISA
            # records omit annotation_id/sentence_id, in which case both the
            # generated prediction metadata and the expected value are None.
            expected_annotation_id = gt_image_info.get("annotation_id")
            if pred.get("annotation_id") != expected_annotation_id:
                raise ValueError(
                    "ReaSeg prediction/GT annotation id mismatch for "
                    f"{key!r}: prediction={pred.get('annotation_id')!r}, "
                    f"gt={expected_annotation_id!r}"
                )
            expected_sentence_id = gt_image_info.get("sentence_id")
            if pred.get("sentence_id") != expected_sentence_id:
                raise ValueError(
                    "ReaSeg prediction/GT sentence id mismatch for "
                    f"{key!r}: prediction={pred.get('sentence_id')!r}, "
                    f"gt={expected_sentence_id!r}"
                )
            expected_phrases = gt_image_info.get("phrases")
            if pred.get("phrases") != expected_phrases:
                raise ValueError(
                    "ReaSeg prediction/GT phrase identity mismatch for "
                    f"{key!r}: prediction={pred.get('phrases')!r}, "
                    f"gt={expected_phrases!r}"
                )

            pred_mask = pred["pred_mask"]
            height, width = pred_mask["size"]
            pred_mask = decode_mask(pred_mask, height, width)

            if "ignore_mask" not in gt_annotation:
                raise ValueError(
                    f"ReaSeg GT is missing ignore_mask for {key!r}"
                )
            ignore_mask = gt_annotation["ignore_mask"]
            gt_mask = gt_annotation["segmentation"]
            ignore_mask = decode_mask(ignore_mask, height, width)
            gt_mask = decode_mask(gt_mask, height, width)
            if (
                pred_mask.shape != gt_mask.shape
                or ignore_mask.shape != gt_mask.shape
            ):
                raise ValueError(
                    "ReaSeg prediction/GT/ignore-mask shapes differ for "
                    f"{key!r}: prediction={pred_mask.shape}, "
                    f"gt={gt_mask.shape}, ignore={ignore_mask.shape}"
                )

            # Pixels explicitly marked by LISA's ignore polygons are excluded
            # from both foreground and background IoU.  The legacy class name
            # ``ignore`` below still denotes valid class-0 background pixels.
            pred_mask = np.where(
                ignore_mask == 1,
                self._metadata.ignore_label,
                pred_mask,
            )
            gt_mask = np.where(
                ignore_mask == 1,
                self._metadata.ignore_label,
                gt_mask,
            )
            intersection, union, _ = calculate_iou(
                pred_mask,
                gt_mask,
                2,
                self._metadata.ignore_label,
            )
            self.iou_stat.update(intersection, union, n=1)

        self.iou_stat.average()
        if not self._terminal_summary_only:
            print_log(
                f"{self.data_name} evaluation results:\n{self.iou_stat}",
                logger="current",
            )
        return {
            "evaluation_protocol": "one_prediction_per_reasoning_question",
            "num_ground_truth": len(gt_anns),
            "num_predictions_evaluated": len(predictions),
            "complete_dataset": not missing and not unexpected,
            "metric_unit": "percent",
            "metric_range": [0.0, 100.0],
            "metric_definitions": {
                "cIoU": "100 * sum(intersection) / sum(union)",
                "gIoU": "100 * mean(per-question intersection / union)",
            },
            "class_semantics": {
                self.iou_stat.cat_names[0]: (
                    "valid class-0 background pixels; despite the legacy "
                    "name, these pixels participate in IoU"
                ),
                self.iou_stat.cat_names[1]: (
                    "class-1 reasoning target foreground"
                ),
                "ignore_index": (
                    f"label {self._metadata.ignore_label}; LISA ignore-mask "
                    "pixels are mapped to this label and excluded from IoU"
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
                    "intersection": float(
                        self.iou_stat.intersection[index]
                    ),
                    "union": float(self.iou_stat.union[index]),
                    "sample_iou_sum": float(self.iou_stat.acc_iou[index]),
                    "sample_count": int(self.iou_stat.count),
                }
                for index, cat_name in enumerate(self.iou_stat.cat_names)
            },
        }
