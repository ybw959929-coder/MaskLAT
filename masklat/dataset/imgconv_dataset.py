import copy
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.utils import DEFAULT_IMAGE_TOKEN

from .base_dataset import BaseDataset
from .utils.load import load_jsonl
from .utils.spatial_alignment import (
    build_spatial_transform_metadata,
    chw_spatial_size,
    expand2square,
    extract_single_pixel_values,
    extract_single_sam_geometry,
)


class ImgConvDataset(BaseDataset):
    def __init__(
        self,
        *args,
        task_name="imgconv",
        offline_processed_text_folder=None,
        max_dataset_length=None,
        preprocess_text_data=False,
        is_multimodal=False,
        exclude_pure_text=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            task_name=task_name,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            preprocess_text_data=preprocess_text_data,
            is_multimodal=is_multimodal,
            exclude_pure_text=exclude_pure_text,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        self.offline_processed_text_folder = kwargs.get("offline_processed_text_folder", None)
        self.max_dataset_length = kwargs.get("max_dataset_length", None)
        self.preprocess_text_data = kwargs.get("preprocess_text_data", False)
        self.is_multimodal = kwargs.get("is_multimodal", False)
        self.exclude_pure_text = kwargs.get("exclude_pure_text", False)
        if not isinstance(self.exclude_pure_text, bool):
            raise TypeError("exclude_pure_text must be boolean")

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.data:
            cur_len = (
                sum(len(conv["value"].split()) for conv in data_dict["conversations"])
                if not self.preprocess_text_data
                else len(data_dict["input_ids"])
            )
            if data_dict.get("image_file", data_dict.get("image")) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def _load_ann_data(self):
        assert self.offline_processed_text_folder or (self.data_path and self.tokenizer)
        if self.offline_processed_text_folder and self.data_path:
            print_log(
                "Both `offline_processed_text_folder` and "
                "`data_path` are set, and we load dataset from"
                "`offline_processed_text_folder` "
                f"({self.offline_processed_text_folder})",
                logger="current",
                level=logging.WARNING,
            )

        if self.offline_processed_text_folder is not None:
            data = load_from_disk(self.offline_processed_text_folder)
        else:
            if self.data_path.endswith(".json"):
                with open(self.data_path, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
            elif self.data_path.endswith(".jsonl"):
                json_data = load_jsonl(self.data_path)
            else:
                raise NotImplementedError

            data = json_data
            if self.preprocess_text_data:
                for idx in range(len(json_data)):
                    if isinstance(json_data[idx]["id"], int):
                        json_data[idx]["id"] = str(json_data[idx]["id"])
                json_data = DatasetDict({"train": HFDataset.from_list(json_data)})
                text_data = process_hf_dataset(
                    dataset=json_data,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    dataset_map_fn=self.dataset_map_fn,
                    template_map_fn=self.template_map_fn,
                    split="train",
                    max_dataset_length=self.max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True,
                )
                data = text_data

        filtered_data = []
        excluded_pure_text = 0
        for d in data:
            if self.preprocess_fn is not None:
                processed = self.preprocess_fn(d)
                if processed is None:
                    raise ValueError("ImgConv preprocess_fn returned None")
                d = processed
            if "image" in d:
                d["image_file"] = d.pop("image")
            if d.get("image_file") is None and self.exclude_pure_text:
                excluded_pure_text += 1
                continue
            filtered_data.append(d)

        if excluded_pure_text:
            print_log(
                f"ImgConv {self.data_name}: excluded {excluded_pure_text} "
                "pure-text sample(s) from the multimodal latent path.",
                logger="current",
            )

        return filtered_data

    def _split_single_conversation_samples(self, data):
        """Split raw LLaVA multi-turn records into supervised turn pairs."""

        if self.preprocess_text_data or self.offline_processed_text_folder:
            raise ValueError(
                "single_conversation requires raw ImgConv conversations; "
                "disable offline/preprocessed text data"
            )
        expanded = []
        skipped_unpaired = 0
        for sample_index, sample in enumerate(data):
            messages = sample.get("conversations")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"ImgConv sample {sample_index} has no raw conversations"
                )
            pending_human = []
            pair_index = 0
            for message in messages:
                if not isinstance(message, dict):
                    raise TypeError(
                        f"ImgConv sample {sample_index} contains a non-dict turn"
                    )
                role = message.get("from")
                if role == "human":
                    pending_human.append(message)
                    continue
                if role != "gpt":
                    raise ValueError(
                        f"ImgConv sample {sample_index} has unsupported role {role!r}"
                    )
                # Match the established map function: leading/consecutive GPT
                # turns have no supervised input and are skipped.
                if not pending_human:
                    skipped_unpaired += 1
                    continue
                pair = pending_human + [message]
                pending_human = []
                if sample.get("image_file") is not None and not any(
                    DEFAULT_IMAGE_TOKEN in turn.get("value", "")
                    for turn in pair
                    if turn.get("from") == "human"
                ):
                    pair[0] = dict(pair[0])
                    pair[0]["value"] = (
                        DEFAULT_IMAGE_TOKEN + "\n" + pair[0].get("value", "")
                    ).strip()
                # __getitem__ deep-copies the selected record before mapping,
                # so a shallow record here is sufficient and avoids doubling
                # the 558K corpus in memory during one-turn expansion.
                single = dict(sample)
                single["conversations"] = pair
                single["single_conversation_index"] = pair_index
                expanded.append(single)
                pair_index += 1
            if pending_human:
                skipped_unpaired += 1
        if skipped_unpaired:
            print_log(
                f"ImgConv {self.data_name}: skipped {skipped_unpaired} "
                "unpaired dialogue fragment(s) with no supervised answer.",
                logger="current",
                level=logging.WARNING,
            )
        return expanded

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_file", None) is not None:
            image_file = data_dict["image_file"]
            pil_image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
            if self.image_processor is not None:
                image = pil_image
                if self.pad_image_to_square:
                    image = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                image = extract_single_pixel_values(
                    self.image_processor.preprocess(
                        image,
                        return_tensors="pt",
                    ),
                    name="ImgConv SigLIP processor",
                )
                data_dict["pixel_values"] = image
            if self.extra_image_processor is not None:
                seg_output = self.extra_image_processor.preprocess(pil_image, return_tensors="pt")
                data_dict["image_info"] = {"image_file": image_file}
                data_dict["extra_pixel_values"] = extract_single_pixel_values(
                    seg_output,
                    name="ImgConv SAM processor",
                )
                sam_input_size = chw_spatial_size(
                    data_dict["extra_pixel_values"],
                    name="ImgConv SAM extra_pixel_values",
                )
                (
                    data_dict["image_size"],
                    data_dict["scaled_size"],
                ) = extract_single_sam_geometry(
                    seg_output,
                    expected_original_size=(
                        pil_image.height,
                        pil_image.width,
                    ),
                    pixel_size=sam_input_size,
                    name="ImgConv SAM processor",
                )
                if self.image_processor is not None:
                    data_dict["spatial_transform"] = (
                        build_spatial_transform_metadata(
                            original_size=(pil_image.height, pil_image.width),
                            siglip_input_size=chw_spatial_size(
                                data_dict["pixel_values"],
                                name="ImgConv SigLIP pixel_values",
                            ),
                            sam_input_size=sam_input_size,
                            sam_scaled_size=data_dict["scaled_size"],
                            pad_image_to_square=self.pad_image_to_square,
                            siglip_processor=self.image_processor,
                            sam_processor=self.extra_image_processor,
                            sam_preprocess_output=seg_output,
                        )
                    )
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
        elif self.is_multimodal:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                elif hasattr(self.extra_image_processor, "pad_size"):
                    crop_size = self.extra_image_processor.pad_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["extra_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_file"] = None
                data_dict["image_size"] = {"height": crop_size["height"], "width": crop_size["width"]}
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["task_name"] = self.task_name
                if self.image_processor is not None:
                    data_dict["spatial_transform"] = (
                        build_spatial_transform_metadata(
                            original_size=(
                                crop_size["height"],
                                crop_size["width"],
                            ),
                            siglip_input_size=chw_spatial_size(
                                data_dict["pixel_values"],
                                name="ImgConv synthetic SigLIP pixel_values",
                            ),
                            sam_input_size=chw_spatial_size(
                                data_dict["extra_pixel_values"],
                                name="ImgConv synthetic SAM extra_pixel_values",
                            ),
                            sam_scaled_size=data_dict["scaled_size"],
                            pad_image_to_square=False,
                            siglip_processor=self.image_processor,
                            sam_processor=self.extra_image_processor,
                        )
                    )
            data_dict.update(self._get_input_ids(data_dict, with_image_token=False))
        else:
            data_dict.update(self._get_input_ids(data_dict, with_image_token=True))
        return data_dict
