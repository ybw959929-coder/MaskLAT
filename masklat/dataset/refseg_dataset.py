# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import json
import os
import os.path as osp
import random
import tempfile

import numpy as np
import torch

from masklat.utils.logging import print_log

from .base_dataset import BaseDataset
from .utils.catalog import MetadataCatalog
from .utils.coco import COCO
from .utils.mask import decode_mask
from .utils.refer import REFER


def _as_single_condition_annotation(annotation):
    """Remap an evaluation annotation into its emitted one-condition sample."""

    annotation = copy.deepcopy(annotation)
    annotation["category_id"] = 0
    return annotation


class RefSegDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="refseg",
        dataset=None,
        data_root=None,
        data_split=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_path=None,
            dataset=dataset,
            task_name=task_name,
            data_root=data_root,
            data_split=data_split,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.dataset = kwargs.get("dataset", None)
        self.data_root = kwargs.get("data_root", None)
        self.data_split = kwargs.get("data_split", None)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=gt_json,
            data_name=self.data_name,
            ignore_label=self.ignore_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _convert_to_coco_format(self):
        refer_api = REFER(self.data_root, self.dataset)
        coco_data = {"images": [], "annotations": [], "categories": []}

        # images
        for img_id, img in refer_api.Imgs.items():
            ref = refer_api.imgToRefs[img_id]
            if ref[0]["split"] != self.data_split:
                continue
            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                }
            )

        # annotations
        for ann_id, ann in refer_api.Anns.items():
            assert (isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0) or isinstance(
                ann["segmentation"], dict
            )
            cur_ann = {
                "id": ann_id,
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": ann["segmentation"],
                "area": ann["area"],
                "bbox": ann["bbox"],
                "iscrowd": ann.get("iscrowd", 0),
            }
            ref = refer_api.annToRef.get(ann_id, None)
            # NOTE: one ref may have multiple sentences, but only one annotation.
            if ref:
                if ref["split"] != self.data_split:
                    continue
                cur_ann["refer_sents"] = [sent for sent in ref["sentences"]]

                # only add the annotation if it has refer expressions
                coco_data["annotations"].append(cur_ann)

        # categories as placeholder
        for cat_id, cat_name in refer_api.Cats.items():
            coco_data["categories"].append({"id": cat_id, "name": cat_name})

        return coco_data

    def _load_ann_data(self):
        coco_data = self._convert_to_coco_format()
        coco_api = COCO(dataset=coco_data)
        img_ids = sorted(coco_api.getImgIds())

        rets = []
        for img_id in img_ids:
            _img_info = coco_api.loadImgs(img_id)[0]
            ann_ids = coco_api.getAnnIds(imgIds=[img_id])
            anns = coco_api.loadAnns(ann_ids)
            _anns = [
                (dict(ann, segmentation=ann["segmentation"][0]) if isinstance(ann["segmentation"][0], dict) else ann)
                for ann in anns
            ]
            if len(_anns) == 0:
                self.woann_cnt += 1
                continue

            img_info = {
                "file_name": _img_info["file_name"],
                "image_id": _img_info["id"],
                "height": _img_info["height"],
                "width": _img_info["width"],
            }

            if self.data_split != "train":
                # RefCOCO's evaluation unit is one referring-expression
                # record, not one unique lower-cased string.  Preserve
                # duplicate text with distinct sentence ids and build a
                # deterministic order shared by every distributed rank.
                eval_conditions = []
                for ann in _anns:
                    refer_sents = ann.pop("refer_sents")
                    indexed_sents = list(enumerate(refer_sents))
                    indexed_sents.sort(
                        key=lambda item: (
                            item[1].get("sent_id") is None,
                            item[1].get("sent_id", item[0]),
                            item[0],
                        )
                    )
                    for original_index, sent_record in indexed_sents:
                        eval_conditions.append(
                            (
                                sent_record["sent"].lower(),
                                copy.deepcopy(ann),
                                sent_record.get("sent_id", original_index),
                            )
                        )

                for sample_id, (
                    sampled_sent,
                    sampled_ann,
                    sentence_id,
                ) in enumerate(eval_conditions):
                    sampled_ann = _as_single_condition_annotation(
                        sampled_ann
                    )
                    rets.append(
                        {
                            "image_file": _img_info["file_name"],
                            "image_id": _img_info["id"],
                            "image_size": (
                                _img_info["height"],
                                _img_info["width"],
                            ),
                            "sampled_sents": [sampled_sent],
                            "annotations": [sampled_ann],
                            "image_info": {
                                **img_info,
                                "sample_id": sample_id,
                                "annotation_id": sampled_ann["id"],
                                "sentence_id": sentence_id,
                                "phrases": [sampled_sent],
                                # Evaluation-only exact GT geometry for
                                # Query/Group diagnostics.  Keeping the
                                # original annotation RLE/polygon avoids
                                # measuring against a SAM-resized target that
                                # has already lost boundary pixels.
                                "diagnostic_gt_mask": copy.deepcopy(
                                    sampled_ann["segmentation"]
                                ),
                            },
                        }
                    )
                continue

            if getattr(self, "single_conversation", False):
                # Build the flat training unit directly from the official
                # annotation-expression relation.  Splitting the legacy
                # Cartesian expression combinations afterwards would repeat
                # every Cond/GT pair once per unrelated instance and silently
                # distort the RefCOCO sampling distribution.
                for annotation in _anns:
                    refer_sents = annotation.pop("refer_sents")
                    sampled_sents = sorted(
                        set(
                            sentence["sent"].lower()
                            for sentence in refer_sents
                        )
                    )
                    for sampled_sent in sampled_sents:
                        sampled_ann = copy.deepcopy(annotation)
                        sampled_ann["category_id"] = 0
                        rets.append(
                            {
                                "image_file": _img_info["file_name"],
                                "image_id": _img_info["id"],
                                "image_size": (
                                    _img_info["height"],
                                    _img_info["width"],
                                ),
                                "sampled_sents": [sampled_sent],
                                "annotations": [sampled_ann],
                                "image_info": {
                                    **img_info,
                                    "annotation_id": sampled_ann["id"],
                                    "phrases": [sampled_sent],
                                },
                            }
                        )
                continue

            # Keep the established stochastic multi-condition training
            # construction separate from the exact evaluation protocol.
            ann_sents = [
                sorted(
                    list(
                        set(
                            x["sent"].lower()
                            for x in ann.pop("refer_sents")
                        )
                    )
                )
                for ann in _anns
            ]
            num_combinations = sum(len(x) for x in ann_sents)
            sent_combinations = list(
                itertools.islice(
                    itertools.product(*ann_sents),
                    num_combinations,
                )
            )
            anns = [copy.deepcopy(ann) for ann in _anns]

            for sent_combination in sent_combinations:
                assert len(sent_combination) == len(anns)
                sampled_anns = copy.deepcopy(anns)
                sampled_sents = list(sent_combination)

                sampled_inds = random.sample(
                    range(len(sampled_sents)),
                    min(len(sampled_sents), self.num_class),
                )
                sampled_sents = [sampled_sents[i] for i in sampled_inds]
                sampled_anns = [sampled_anns[i] for i in sampled_inds]

                # Preserve the established training semantics: equal sampled
                # condition strings share the same condition id.
                for sampled_sent, sampled_ann in zip(
                    sampled_sents,
                    sampled_anns,
                ):
                    sampled_ann["category_id"] = sampled_sents.index(
                        sampled_sent
                    )

                rets.append(
                    {
                        "image_file": _img_info["file_name"],
                        "image_id": _img_info["id"],
                        "image_size": (
                            _img_info["height"],
                            _img_info["width"],
                        ),
                        "sampled_sents": sampled_sents,
                        "annotations": sampled_anns,
                        "image_info": {
                            **img_info,
                            "phrases": sampled_sents,
                        },
                    }
                )

        if self.data_split != "train":
            base_temp = tempfile.gettempdir()
            cache_dir = osp.join(base_temp, "masklat_cache")
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            print_log(f"Writing {self.data_name} gt_json to {temp_dir}...", logger="current")
            temp_file = osp.join(temp_dir, f"{self.data_name}.json")
            with open(temp_file, "w") as f:
                json.dump(rets, f)
            self._set_metadata(gt_json=temp_file)
        else:
            self._set_metadata()

        del coco_data
        return rets

    def _split_single_conversation_samples(self, data):
        """Turn every RefSeg expression/GT pair into one training sample."""

        if self.data_split != "train":
            return data
        expanded = []
        for sample_index, sample in enumerate(data):
            sentences = sample.get("sampled_sents")
            annotations = sample.get("annotations")
            if not isinstance(sentences, list) or not isinstance(annotations, list):
                raise TypeError(
                    "RefSeg single-conversation expansion requires list-valued "
                    f"sentences/annotations at sample {sample_index}"
                )
            if len(sentences) != len(annotations) or not sentences:
                raise ValueError(
                    "RefSeg Cond/SEG alignment is invalid at sample "
                    f"{sample_index}: sentences={len(sentences)}, "
                    f"annotations={len(annotations)}"
                )
            for sentence, annotation in zip(sentences, annotations):
                single = dict(sample)
                single_annotation = copy.deepcopy(annotation)
                # A one-dialogue sample has one local Cond, hence its only GT
                # must use local class id zero regardless of the old row id.
                single_annotation["category_id"] = 0
                single["sampled_sents"] = [sentence]
                single["annotations"] = [single_annotation]
                single["image_info"] = {
                    **sample["image_info"],
                    "phrases": [sentence],
                }
                expanded.append(single)
        return expanded

    def _decode_mask(self, data_dict):
        height, width = data_dict["image_size"]
        sampled_anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        for ann in sampled_anns:
            segmentation = ann["segmentation"]
            binary_mask = decode_mask(segmentation, height, width)
            mask_labels.append(binary_mask)
            class_labels.append(ann["category_id"])

        mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_labels])
        class_labels = torch.tensor(np.array(class_labels), dtype=torch.int64)

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
