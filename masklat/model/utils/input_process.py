from typing import List, Optional

import torch
from transformers import PreTrainedModel
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from ...utils.constants import REGION_TOKEN_INDEX


def build_group_bidirectional_causal_mask(
    attention_mask: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a causal additive mask with bidirectional Group-token blocks.

    ``attention_mask`` and ``group_ids`` describe the *same padded sequence*.
    A non-negative value in ``group_ids`` marks a Group token; ``-1`` marks
    every other token.  Each contiguous run of Group tokens is made
    bidirectional.  All other attention remains causal, so a Group block can
    read its preceding image tokens but cannot read later text tokens.

    The returned tensor has shape ``[batch, 1, length, length]`` and contains
    only ``0`` (visible) or ``torch.finfo(dtype).min`` (masked).  Keeping the
    mask finite avoids introducing NaN/Inf values before it reaches the LLM.
    This helper is intended for training and generation prefill; cached
    one-token decoding should keep using the ordinary 2-D attention mask.
    """

    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if not isinstance(group_ids, torch.Tensor):
        raise TypeError("group_ids must be a torch.Tensor")
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [batch, length], got "
            f"{tuple(attention_mask.shape)}"
        )
    if group_ids.ndim != 2:
        raise ValueError(
            "group_ids must have shape [batch, length], got "
            f"{tuple(group_ids.shape)}"
        )
    if group_ids.shape != attention_mask.shape:
        raise ValueError(
            "group_ids and attention_mask must describe the same sequence: "
            f"group_ids={tuple(group_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if group_ids.device != attention_mask.device:
        raise ValueError(
            "group_ids and attention_mask must be on the same device: "
            f"group_ids={group_ids.device}, "
            f"attention_mask={attention_mask.device}"
        )
    if group_ids.dtype == torch.bool or torch.is_floating_point(group_ids):
        raise TypeError(
            "group_ids must use an integer dtype with -1 for non-Group tokens"
        )
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating torch.dtype, got {dtype!r}")

    if attention_mask.dtype != torch.bool:
        is_binary = torch.logical_or(attention_mask == 0, attention_mask == 1)
        if not bool(is_binary.all().item()):
            raise ValueError("attention_mask must contain only boolean/0/1 values")
    valid_tokens = attention_mask.bool()

    valid_group_ids = torch.logical_or(group_ids == -1, group_ids >= 0)
    if not bool(valid_group_ids.all().item()):
        raise ValueError("group_ids may contain only -1 or non-negative ids")
    group_tokens = group_ids >= 0
    if bool(torch.logical_and(group_tokens, ~valid_tokens).any().item()):
        raise ValueError("a Group token cannot occupy a padded sequence position")

    batch_size, sequence_length = attention_mask.shape
    positions = torch.arange(sequence_length, device=attention_mask.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    allowed = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()
    allowed &= valid_tokens.unsqueeze(1)
    allowed &= valid_tokens.unsqueeze(2)

    # A sample may contain more than one image placeholder.  Treat every
    # contiguous Group run as its own bidirectional block rather than allowing
    # unrelated image blocks to attend to one another non-causally.
    for batch_index in range(batch_size):
        row = group_tokens[batch_index]
        if sequence_length == 0 or not bool(row.any().item()):
            continue
        padded_row = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=row.device),
                row,
                torch.zeros(1, dtype=torch.bool, device=row.device),
            )
        )
        boundaries = torch.nonzero(
            padded_row[1:] != padded_row[:-1], as_tuple=False
        ).flatten()
        for start, end in boundaries.reshape(-1, 2).tolist():
            allowed[batch_index, start:end, start:end] = True

    mask_value = torch.finfo(dtype).min
    additive_mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        mask_value,
        dtype=dtype,
        device=attention_mask.device,
    )
    additive_mask.masked_fill_(allowed.unsqueeze(1), 0)
    return additive_mask


def _validate_and_normalize_group_inputs(
    llm: PreTrainedModel,
    input_ids: torch.LongTensor,
    group_values: torch.Tensor,
    group_valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Validate padded Group embeddings and return a boolean validity mask."""

    if input_ids is None or input_ids.ndim != 2:
        raise ValueError(
            "Group insertion requires input_ids with shape [batch, length]"
        )
    if not isinstance(group_values, torch.Tensor):
        raise TypeError("group_values must be a torch.Tensor")
    if group_values.ndim != 3:
        raise ValueError(
            "group_values must have shape [batch, groups, hidden], got "
            f"{tuple(group_values.shape)}"
        )
    if group_values.shape[0] != input_ids.shape[0]:
        raise ValueError(
            "group_values batch size must match input_ids: "
            f"groups={group_values.shape[0]}, input_ids={input_ids.shape[0]}"
        )
    if not torch.is_floating_point(group_values):
        raise TypeError("group_values must use a floating dtype")
    if not bool(torch.isfinite(group_values).all().item()):
        raise ValueError("group_values contains NaN or Inf")

    input_embeddings = llm.get_input_embeddings()
    embedding_dim = getattr(input_embeddings, "embedding_dim", None)
    if embedding_dim is None and hasattr(input_embeddings, "weight"):
        embedding_dim = input_embeddings.weight.shape[-1]
    if embedding_dim is None:
        embedding_dim = getattr(llm.config, "hidden_size", None)
    if embedding_dim is None:
        raise ValueError("cannot determine the LLM input embedding dimension")
    if group_values.shape[-1] != int(embedding_dim):
        raise ValueError(
            "group_values hidden size must match the LLM input embeddings: "
            f"groups={group_values.shape[-1]}, llm={int(embedding_dim)}"
        )

    expected_mask_shape = group_values.shape[:2]
    if group_valid_mask is None:
        return torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=group_values.device,
        )
    if not isinstance(group_valid_mask, torch.Tensor):
        raise TypeError("group_valid_mask must be a torch.Tensor")
    if group_valid_mask.ndim != 2 or group_valid_mask.shape != expected_mask_shape:
        raise ValueError(
            "group_valid_mask must have shape [batch, groups] matching "
            f"group_values; got mask={tuple(group_valid_mask.shape)}, "
            f"groups={tuple(group_values.shape)}"
        )
    if group_valid_mask.dtype != torch.bool:
        is_binary = torch.logical_or(group_valid_mask == 0, group_valid_mask == 1)
        if not bool(is_binary.all().item()):
            raise ValueError("group_valid_mask must contain only boolean/0/1 values")
    return group_valid_mask.to(device=group_values.device, dtype=torch.bool)


def prepare_inputs_labels_for_multimodal(
    llm: PreTrainedModel,
    input_ids: torch.LongTensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    extra_pixel_values: Optional[torch.FloatTensor] = None,
    cond_ids: Optional[torch.LongTensor] = None,
    seg_ids: Optional[torch.LongTensor] = None,
    vprompt_feats: Optional[torch.FloatTensor] = None,
    group_values: Optional[torch.FloatTensor] = None,
    group_valid_mask: Optional[torch.Tensor] = None,
    expected_num_image_tokens: Optional[int] = None,
    **kwargs,
):
    if pixel_values is None:
        if group_values is not None or group_valid_mask is not None:
            raise ValueError(
                "Group tokens require pixel_values because they are inserted "
                "after an image-token block"
            )
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": None,
            "cond_ids": cond_ids,
            "vprompt_feats": vprompt_feats,
            "seg_ids": seg_ids,
            "labels": labels,
        }

    if group_values is None and group_valid_mask is not None:
        raise ValueError("group_valid_mask was provided without group_values")
    use_group_tokens = group_values is not None
    if use_group_tokens:
        group_valid_mask = _validate_and_normalize_group_inputs(
            llm,
            input_ids,
            group_values,
            group_valid_mask,
        )

    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 3:
        raise ValueError(
            "projected pixel_values must have shape [batch, tokens, hidden], "
            f"got {type(pixel_values).__name__} with shape "
            f"{getattr(pixel_values, 'shape', None)}"
        )
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError(
            "multimodal insertion requires input_ids with shape [batch, length]"
        )
    if pixel_values.shape[0] != input_ids.shape[0]:
        raise ValueError(
            "projected image-token batch must match input_ids: "
            f"images={pixel_values.shape[0]}, input_ids={input_ids.shape[0]}"
        )

    if extra_pixel_values is not None:
        if (
            not isinstance(extra_pixel_values, torch.Tensor)
            or extra_pixel_values.ndim != 3
        ):
            raise ValueError(
                "projected extra_pixel_values must have shape "
                "[batch, tokens, hidden]"
            )
        if (
            extra_pixel_values.shape[0] != pixel_values.shape[0]
            or extra_pixel_values.shape[2] != pixel_values.shape[2]
        ):
            raise ValueError(
                "SigLIP and direct SAM image-token tables must agree in batch "
                "and hidden dimensions: "
                f"siglip={tuple(pixel_values.shape)}, "
                f"sam={tuple(extra_pixel_values.shape)}"
            )
        pixel_values = torch.cat([pixel_values, extra_pixel_values], dim=1)

    if expected_num_image_tokens is not None:
        if (
            not isinstance(expected_num_image_tokens, int)
            or isinstance(expected_num_image_tokens, bool)
            or expected_num_image_tokens <= 0
        ):
            raise ValueError("expected_num_image_tokens must be a positive integer")
        actual_num_image_tokens = int(pixel_values.shape[1])
        if actual_num_image_tokens != expected_num_image_tokens:
            raise ValueError(
                "VLM image-token count violates the configured SAM-injection "
                "branch: "
                f"actual={actual_num_image_tokens}, "
                f"expected={expected_num_image_tokens}"
            )

    _input_ids = input_ids
    _cond_ids = cond_ids
    _seg_ids = seg_ids
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    else:
        attention_mask = attention_mask.bool().to(device=input_ids.device)
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if cond_ids is None:
        cond_ids = torch.full_like(input_ids, -1, dtype=torch.long, device=input_ids.device)
    if seg_ids is None:
        seg_ids = torch.full_like(input_ids, -1, dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX, device=input_ids.device)
    if vprompt_feats is None:
        vprompt_feats = [None] * len(input_ids)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    cond_ids = [cur_cond_ids[cur_attention_mask] for cur_cond_ids, cur_attention_mask in zip(cond_ids, attention_mask)]
    seg_ids = [cur_seg_ids[cur_attention_mask] for cur_seg_ids, cur_attention_mask in zip(seg_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_inputs_embeds = []
    new_input_ids = []
    new_cond_ids = []
    new_seg_ids = []
    new_labels = []
    new_group_ids = [] if use_group_tokens else None
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        num_regions = (cur_input_ids == REGION_TOKEN_INDEX).sum()
        if use_group_tokens:
            cur_valid_group_indices = torch.nonzero(
                group_valid_mask[batch_idx], as_tuple=False
            ).flatten()
            if int(num_images.item()) == 0 and cur_valid_group_indices.numel() > 0:
                raise ValueError(
                    "sample has valid Group tokens but no image placeholder: "
                    f"batch_index={batch_idx}, "
                    f"valid_groups={cur_valid_group_indices.numel()}"
                )
            if cur_valid_group_indices.numel() > 0 and int(num_images.item()) != 1:
                raise ValueError(
                    "st3 Group tokens require exactly one image placeholder "
                    "per source so a Group is inserted exactly once: "
                    f"batch_index={batch_idx}, image_placeholders="
                    f"{int(num_images.item())}"
                )

        if num_images == 0 and num_regions == 0:
            cur_pixel_values = pixel_values[cur_image_idx]
            # Every text row has one aligned visual-table row, including
            # ImgConv pure-text records backed by a synthetic zero image.
            cur_image_idx += 1
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            cur_inputs_embeds = torch.cat([cur_inputs_embeds, cur_pixel_values[0:0]], dim=0)
            if use_group_tokens:
                # Pure-text ImgConv rows intentionally insert no proposal
                # tokens.  Keep an empty view of their proposal table in the
                # autograd graph so every distributed rank still produces a
                # zero (rather than absent) gradient for proposal/prefix
                # parameters.  This preserves text-only semantics and keeps
                # ZeRO/DDP collective hooks identical to image-bearing ranks.
                empty_group_values = group_values[batch_idx, :0].to(
                    dtype=cur_inputs_embeds.dtype,
                )
                cur_inputs_embeds = torch.cat(
                    [cur_inputs_embeds, empty_group_values],
                    dim=0,
                )
            cur_cond_ids = cond_ids[batch_idx]
            cur_seg_ids = seg_ids[batch_idx]
            new_inputs_embeds.append(cur_inputs_embeds)
            new_input_ids.append(cur_input_ids)
            new_cond_ids.append(cur_cond_ids)
            new_seg_ids.append(cur_seg_ids)
            new_labels.append(labels[batch_idx])
            if use_group_tokens:
                new_group_ids.append(
                    torch.full_like(cur_input_ids, -1, dtype=torch.long)
                )
            continue

        image_token_indices = (
            [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )
        region_token_indices = (
            [-1] + torch.where(cur_input_ids == REGION_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )

        # Process both image and region tokens
        all_special_token_indices = sorted(
            [(idx, "image") for idx in image_token_indices[1:-1]]
            + [(idx, "region") for idx in region_token_indices[1:-1]]
        )
        all_special_token_indices = [(-1, "none")] + all_special_token_indices + [(cur_input_ids.shape[0], "none")]

        cur_input_ids_nospecial = []
        cur_cond_ids_nospecial = []
        cur_seg_ids_nospecial = []
        cur_cond_ids = cond_ids[batch_idx]
        cur_seg_ids = seg_ids[batch_idx]
        cur_labels = labels[batch_idx]
        cur_vprompt_feats = vprompt_feats[batch_idx]
        cur_labels_nospecial = []

        for i in range(len(all_special_token_indices) - 1):
            start_idx = all_special_token_indices[i][0] + 1
            end_idx = all_special_token_indices[i + 1][0]

            if start_idx < end_idx:
                cur_input_ids_nospecial.append(cur_input_ids[start_idx:end_idx])
                cur_labels_nospecial.append(cur_labels[start_idx:end_idx])
                cur_cond_ids_nospecial.append(cur_cond_ids[start_idx:end_idx])
                cur_seg_ids_nospecial.append(cur_seg_ids[start_idx:end_idx])

        split_sizes = [x.shape[0] for x in cur_labels_nospecial]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_nospecial)
            if cur_input_ids_nospecial
            else torch.tensor([], device=llm.device, dtype=llm.dtype)
        )

        if cur_inputs_embeds.numel() > 0:
            cur_inputs_embeds_no_special = torch.split(cur_inputs_embeds, split_sizes, dim=0)
            cur_cond_ids_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_cond_ids_nospecial)
                        if cur_cond_ids_nospecial
                        else torch.tensor([], device=cur_cond_ids.device, dtype=cur_cond_ids.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
            cur_seg_ids_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_seg_ids_nospecial)
                        if cur_seg_ids_nospecial
                        else torch.tensor([], device=cur_seg_ids.device, dtype=cur_seg_ids.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
            cur_labels_nospecial = (
                torch.split(
                    (
                        torch.cat(cur_labels_nospecial)
                        if cur_labels_nospecial
                        else torch.tensor([], device=cur_labels.device, dtype=cur_labels.dtype)
                    ),
                    split_sizes,
                    dim=0,
                )
                if split_sizes
                else []
            )
        else:
            cur_inputs_embeds_no_special = []
            cur_cond_ids_nospecial = []
            cur_seg_ids_nospecial = []
            cur_labels_nospecial = []

        cur_new_inputs_embeds = []
        cur_new_input_ids = []
        cur_new_cond_ids = []
        cur_new_seg_ids = []
        cur_new_labels = []
        cur_new_group_ids = [] if use_group_tokens else None

        segment_idx = 0
        cur_region_idx = 0
        for i in range(len(all_special_token_indices) - 1):
            if segment_idx < len(cur_inputs_embeds_no_special):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_special[segment_idx])
                cur_new_input_ids.append(cur_input_ids_nospecial[segment_idx])
                cur_new_cond_ids.append(cur_cond_ids_nospecial[segment_idx])
                cur_new_labels.append(cur_labels_nospecial[segment_idx])
                cur_new_seg_ids.append(cur_seg_ids_nospecial[segment_idx])
                if use_group_tokens:
                    cur_new_group_ids.append(
                        torch.full_like(
                            cur_input_ids_nospecial[segment_idx],
                            -1,
                            dtype=torch.long,
                        )
                    )
                segment_idx += 1

            # Insert special token (image or region) if present
            if i < len(all_special_token_indices) - 1 and all_special_token_indices[i + 1][1] != "none":
                token_type = all_special_token_indices[i + 1][1]

                if token_type == "image":
                    cur_pixel_values = pixel_values[cur_image_idx]
                    cur_image_idx += 1
                    image_embeds = cur_pixel_values.to(
                        dtype=(
                            cur_inputs_embeds.dtype
                            if cur_inputs_embeds.numel() > 0
                            else torch.float32
                        )
                    )
                    cur_new_inputs_embeds.append(image_embeds)
                    cur_new_input_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            IMAGE_TOKEN_INDEX,
                            device=cur_input_ids.device,
                            dtype=cur_input_ids.dtype,
                        )
                    )
                    cur_new_cond_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            -1,
                            device=cur_cond_ids.device,
                            dtype=cur_cond_ids.dtype,
                        )
                    )
                    cur_new_seg_ids.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            -1,
                            device=cur_seg_ids.device,
                            dtype=cur_seg_ids.dtype,
                        )
                    )
                    cur_new_labels.append(
                        torch.full(
                            (cur_pixel_values.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    if use_group_tokens:
                        cur_new_group_ids.append(
                            torch.full(
                                (cur_pixel_values.shape[0],),
                                -1,
                                device=cur_input_ids.device,
                                dtype=torch.long,
                            )
                        )
                        if cur_valid_group_indices.numel() > 0:
                            cur_group_embeds = group_values[
                                batch_idx, cur_valid_group_indices
                            ].to(
                                device=image_embeds.device,
                                dtype=image_embeds.dtype,
                            )
                            cur_new_inputs_embeds.append(cur_group_embeds)
                            cur_new_input_ids.append(
                                torch.full(
                                    (cur_group_embeds.shape[0],),
                                    IGNORE_INDEX,
                                    device=cur_input_ids.device,
                                    dtype=cur_input_ids.dtype,
                                )
                            )
                            cur_new_cond_ids.append(
                                torch.full(
                                    (cur_group_embeds.shape[0],),
                                    -1,
                                    device=cur_cond_ids.device,
                                    dtype=cur_cond_ids.dtype,
                                )
                            )
                            cur_new_seg_ids.append(
                                torch.full(
                                    (cur_group_embeds.shape[0],),
                                    -1,
                                    device=cur_seg_ids.device,
                                    dtype=cur_seg_ids.dtype,
                                )
                            )
                            cur_new_labels.append(
                                torch.full(
                                    (cur_group_embeds.shape[0],),
                                    IGNORE_INDEX,
                                    device=cur_labels.device,
                                    dtype=cur_labels.dtype,
                                )
                            )
                            cur_new_group_ids.append(
                                cur_valid_group_indices.to(
                                    device=cur_input_ids.device,
                                    dtype=torch.long,
                                )
                            )
                elif token_type == "region" and cur_vprompt_feats is not None:
                    cur_region_feats = cur_vprompt_feats[cur_region_idx][None, :]
                    cur_new_inputs_embeds.append(
                        cur_region_feats.to(
                            dtype=cur_inputs_embeds.dtype if cur_inputs_embeds.numel() > 0 else torch.float16
                        )
                    )
                    cur_new_input_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            REGION_TOKEN_INDEX,
                            device=cur_input_ids.device,
                            dtype=cur_input_ids.dtype,
                        )
                    )
                    cur_new_cond_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            cur_region_idx,
                            device=cur_cond_ids.device,
                            dtype=cur_cond_ids.dtype,
                        )
                    )
                    cur_new_seg_ids.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            -1,
                            device=cur_seg_ids.device,
                            dtype=cur_seg_ids.dtype,
                        )
                    )
                    cur_new_labels.append(
                        torch.full(
                            (cur_region_feats.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    if use_group_tokens:
                        cur_new_group_ids.append(
                            torch.full(
                                (cur_region_feats.shape[0],),
                                -1,
                                device=cur_input_ids.device,
                                dtype=torch.long,
                            )
                        )
                    cur_region_idx += 1

        if cur_new_inputs_embeds:
            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_input_ids = torch.cat(cur_new_input_ids)
            cur_new_cond_ids = torch.cat(cur_new_cond_ids)
            cur_new_seg_ids = torch.cat(cur_new_seg_ids)
            cur_new_labels = torch.cat(cur_new_labels)
            if use_group_tokens:
                cur_new_group_ids = torch.cat(cur_new_group_ids)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_input_ids.append(cur_new_input_ids)
            new_cond_ids.append(cur_new_cond_ids)
            new_seg_ids.append(cur_new_seg_ids)
            new_labels.append(cur_new_labels)
            if use_group_tokens:
                new_group_ids.append(cur_new_group_ids)
        else:
            # Handle empty case
            device = input_ids[0].device
            dtype = input_ids[0].dtype
            empty_embeds = torch.zeros((0, llm.config.hidden_size), device=device, dtype=dtype)
            new_inputs_embeds.append(empty_embeds)
            new_input_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_cond_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_seg_ids.append(torch.tensor([], device=device, dtype=dtype))
            new_labels.append(torch.tensor([], device=device, dtype=dtype))
            if use_group_tokens:
                new_group_ids.append(
                    torch.tensor([], device=device, dtype=torch.long)
                )

    # Combine them
    if not new_inputs_embeds:
        batch_size = _input_ids.shape[0]
        hidden_size = llm.config.hidden_size
        device = _input_ids.device
        dtype = torch.float32

        new_inputs_embeds = torch.zeros((batch_size, 0, hidden_size), device=device, dtype=dtype)
        result = {
            "input_ids": _input_ids,
            "position_ids": _position_ids,
            "attention_mask": _attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": new_inputs_embeds,
            "cond_ids": _cond_ids,
            "seg_ids": _seg_ids,
            "labels": _labels,
        }
        if use_group_tokens:
            result["group_ids"] = torch.full_like(_input_ids, -1)
        return result

    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    if use_group_tokens:
        max_position_embeddings = getattr(
            llm.config, "max_position_embeddings", None
        )
        if (
            not isinstance(max_position_embeddings, int)
            or isinstance(max_position_embeddings, bool)
            or max_position_embeddings <= 0
        ):
            raise ValueError(
                "Group insertion requires a positive integer "
                "llm.config.max_position_embeddings"
            )
        if max_len > max_position_embeddings:
            lengths = [int(value.shape[0]) for value in new_inputs_embeds]
            raise ValueError(
                "multimodal sequence exceeds the LLM context after Group "
                f"insertion: lengths={lengths}, "
                f"max_position_embeddings={max_position_embeddings}"
            )

    new_inputs_embeds_padded = []
    new_input_ids_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_input_ids[0].dtype,
        device=new_input_ids[0].device,
    )
    new_cond_ids_padded = torch.full(
        (batch_size, max_len),
        -1,
        dtype=new_cond_ids[0].dtype,
        device=new_cond_ids[0].device,
    )
    new_seg_ids_padded = torch.full(
        (batch_size, max_len),
        -1,
        dtype=new_seg_ids[0].dtype,
        device=new_seg_ids[0].device,
    )
    new_labels_padded = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=new_labels[0].dtype,
        device=new_labels[0].device,
    )
    if use_group_tokens:
        new_group_ids_padded = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.long,
            device=new_input_ids[0].device,
        )
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (
        cur_new_embed,
        cur_new_input_ids,
        cur_new_cond_ids,
        cur_new_seg_ids,
        cur_new_labels,
    ) in enumerate(
        zip(
            new_inputs_embeds,
            new_input_ids,
            new_cond_ids,
            new_seg_ids,
            new_labels,
        )
    ):
        cur_len = cur_new_embed.shape[0]
        new_inputs_embeds_padded.append(
            torch.cat(
                (
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                ),
                dim=0,
            )
        )
        if cur_len > 0:
            new_input_ids_padded[i, :cur_len] = cur_new_input_ids
            new_cond_ids_padded[i, :cur_len] = cur_new_cond_ids
            new_seg_ids_padded[i, :cur_len] = cur_new_seg_ids
            new_labels_padded[i, :cur_len] = cur_new_labels
            if use_group_tokens:
                new_group_ids_padded[i, :cur_len] = new_group_ids[i]
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    if _input_ids is None:
        new_input_ids = None
    else:
        new_input_ids = new_input_ids_padded

    if _cond_ids is None:
        new_cond_ids = None
    else:
        new_cond_ids = new_cond_ids_padded

    if _seg_ids is None:
        new_seg_ids = None
    else:
        new_seg_ids = new_seg_ids_padded

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None and not use_group_tokens:
        attention_mask = None
    elif _attention_mask is not None:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None and not use_group_tokens:
        position_ids = None

    result = {
        "input_ids": new_input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "inputs_embeds": new_inputs_embeds,
        "cond_ids": new_cond_ids,
        "seg_ids": new_seg_ids,
        "labels": new_labels,
    }
    if use_group_tokens:
        result["group_ids"] = new_group_ids_padded
    return result
