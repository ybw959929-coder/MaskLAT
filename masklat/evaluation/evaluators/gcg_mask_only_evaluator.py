"""GCG mask metrics, without caption scoring or a text-evaluation model.

The model's normal autoregressive phrase/[SEG] generation is unchanged.  The
legacy GCG greedy positive-pair mIoU and constant-score, category-1 COCO AP are
retained, but *every* ground-truth image is evaluated, including valid outputs
whose masks are empty.  Missing model outputs are errors, not empty masks.
"""

import copy
import itertools
import json
import os

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from masklat.utils.logging import print_log

from ...dataset.utils.catalog import MetadataCatalog
from ...dataset.utils.coco import COCO
from ...dataset.utils.mask import decode_mask, encode_mask
from ..utils import comm
from ..utils.miou import compute_miou
from .base_evaluator import BaseEvaluator
from .mask_only_common import (
    atomic_json,
    validate_image_coverage,
    write_summary,
)


class GCGMaskOnlyEvaluator(BaseEvaluator):
    """Evaluate complete GCG image sets using segmentation outputs only."""

    def __init__(self, data_name="gcgseg", output_dir=None, distributed=True):
        self._data_name = data_name
        self._metadata = MetadataCatalog.get(data_name)
        self._distributed = distributed
        self.output_dir = output_dir
        self.reset()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def data_name(self):
        return self._data_name

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value
        if value is not None:
            os.makedirs(value, exist_ok=True)

    def reset(self):
        self._predictions = []

    def _get_rle_masks(self, segmentation):
        # Preserve the released GCG evaluator's area-descending mask order.
        # Equal AP confidence scores make that ordering part of the protocol.
        labels, areas = np.unique(segmentation, return_counts=True)
        labels = labels[np.argsort(-areas)]
        return [
            encode_mask((segmentation == label).astype(np.uint8))
            for label in labels
            if label < self.metadata.ignore_label
        ]

    def process(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise ValueError("GCG mask input/output batch lengths differ")
        for input_, output in zip(inputs, outputs):
            # Intentionally do not access gcg_caption, gcg_phrases, or caption GT.
            segmentation = output["segmentation"]
            if isinstance(segmentation, torch.Tensor):
                segmentation = segmentation.detach().cpu().numpy()
            segmentation = np.asarray(segmentation)
            if (
                segmentation.ndim != 2
                or not all(segmentation.shape)
                or not np.issubdtype(segmentation.dtype, np.number)
                or not np.all(np.isfinite(segmentation))
                or np.any(segmentation < 0)
                or np.any(segmentation != np.floor(segmentation))
            ):
                raise ValueError("GCG segmentation must be a finite 2D nonnegative label map")
            segmentation = segmentation.astype(np.int64)
            image_id = input_["image_id"]
            if isinstance(image_id, np.integer):
                image_id = int(image_id)
            self._predictions.append(
                {
                    "image_id": image_id,
                    "file_name": os.path.basename(input_["file_name"]),
                    "height": int(segmentation.shape[0]),
                    "width": int(segmentation.shape[1]),
                    "segmentation": self._get_rle_masks(segmentation),
                }
            )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(self._predictions, dst=0)
            if not comm.is_main_process():
                return {}
            world_size = comm.get_world_size()
            if len(gathered) != world_size:
                raise ValueError("GCG mask gather did not contain every rank")
            predictions = list(itertools.chain.from_iterable(gathered))
        else:
            world_size = 1
            predictions = list(self._predictions)

        gt_json = os.path.realpath(self.metadata.gt_json)
        with open(gt_json, "r", encoding="utf-8") as file:
            gt_data = json.load(file)
        image_ids = [image["id"] for image in gt_data["images"]]
        coverage = validate_image_coverage(
            image_ids, [prediction["image_id"] for prediction in predictions]
        )
        images = {image["id"]: image for image in gt_data["images"]}
        for prediction in predictions:
            image = images[prediction["image_id"]]
            shape = (image["height"], image["width"])
            if (prediction["height"], prediction["width"]) != shape:
                raise ValueError(
                    f"GCG mask shape mismatch for image {prediction['image_id']!r}: "
                    f"{(prediction['height'], prediction['width'])} != {shape}"
                )
            if any(tuple(mask["size"]) != shape for mask in prediction["segmentation"]):
                raise ValueError(f"GCG RLE shape mismatch for image {prediction['image_id']!r}")
        coverage["empty_mask_images"] = sum(
            not prediction["segmentation"] for prediction in predictions
        )

        mask_predictions = [
            {
                "image_id": prediction["image_id"],
                "category_id": 1,
                "segmentation": copy.deepcopy(mask),
                "score": 1.0,
            }
            for prediction in predictions
            for mask in prediction["segmentation"]
        ]
        gt_data.setdefault("info", {"description": "GCG mask-only evaluation"})
        coco_gt = COCO(dataset=gt_data)
        if mask_predictions:
            coco_dt = coco_gt.loadRes(mask_predictions)
        else:
            # COCO.loadRes([]) indexes anns[0].  Build a genuinely empty result
            # dataset instead of inventing a zero-area or low-confidence mask.
            coco_dt = COCO(
                dataset={
                    "info": copy.deepcopy(gt_data["info"]),
                    "images": copy.deepcopy(gt_data["images"]),
                    "categories": copy.deepcopy(gt_data["categories"]),
                    "annotations": [],
                }
            )

        mious = []
        for image_id in image_ids:
            image = images[image_id]
            height, width = image["height"], image["width"]
            gt_annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[image_id]))
            dt_annotations = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[image_id]))
            gt_masks = [decode_mask(ann["segmentation"], height, width) for ann in gt_annotations]
            dt_masks = [decode_mask(ann["segmentation"], height, width) for ann in dt_annotations]
            # The original formula returns zero if there are no positive pairs,
            # including an empty prediction on an otherwise valid image.
            mious.append(float(compute_miou(dt_masks, gt_masks)))

        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.imgIds = list(image_ids)
        coco_eval.params.catIds = [1]
        print_log(f"{self.data_name} mask-only COCO AP (all GT images):", logger="current")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics = {"mIoU": float(np.mean(mious) * 100.0)}
        for name, value in zip(("AP", "AP50", "AP75", "APs", "APm", "APl"), coco_eval.stats[:6]):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"GCG COCO metric {name} is not finite")
            metrics[name] = None if value == -1.0 else value * 100.0
        print_log(
            f"{self.data_name} mask-only mIoU={metrics['mIoU']:.2f}% "
            f"over {len(image_ids)} GT images "
            f"({coverage['empty_mask_images']} valid empty mask outputs).",
            logger="current",
        )

        if self.output_dir is not None:
            atomic_json(os.path.join(self.output_dir, "predictions.json"), predictions)
        return write_summary(
            self.output_dir,
            data_name=self.data_name,
            metrics=metrics,
            coverage=coverage,
            world_size=world_size,
            protocol_details={
                "task": "gcgseg",
                "image_scope": "complete_ground_truth_images_including_empty_mask_outputs",
                "mIoU": "legacy_greedy_one_to_one_positive_IoU_pairs_mean_per_image_then_mean_over_all_GT_images",
                "mIoU_is_refseg_gIoU": False,
                "empty_mask_image_mIoU": 0.0,
                "mask_order": "legacy_numpy_area_descending",
                "AP": "COCO_segmentation_AP_category_1_constant_score_1.0",
                "AP_category_id": 1,
                "AP_prediction_score": 1.0,
                "AP_undefined": None,
                "model_generation": "unchanged_autoregressive_phrase_and_SEG_generation",
            },
        )
