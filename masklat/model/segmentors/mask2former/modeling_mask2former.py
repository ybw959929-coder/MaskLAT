# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mask2Former model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import all_reduce, get_world_size, is_distributed
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    replace_return_docstrings,
    requires_backends,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_1
from transformers.utils.backbone_utils import load_backbone
from transformers.utils.import_utils import is_torchdynamo_compiling

from .configuration_mask2former import Mask2FormerConfig
from .topology_group_decoder import TopologyGroupDecoderCore, TopologyGroupStageOutput

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


_CONFIG_FOR_DOC = "Mask2FormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/mask2former-swin-small-coco-instance"
_IMAGE_PROCESSOR_FOR_DOC = "Mask2FormerImageProcessor"
def _finite_diagnostics_enabled() -> bool:
    return os.environ.get(
        "MASKLAT_FINITE_DIAGNOSTICS",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _nonfinite_tensor_report(name: str, tensor: Tensor) -> Optional[str]:
    """Describe a non-finite floating tensor without changing its values."""

    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        return None
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    if bool(finite_mask.all()):
        return None

    nan_count = int(torch.isnan(detached).sum().item())
    posinf_count = int(torch.isposinf(detached).sum().item())
    neginf_count = int(torch.isneginf(detached).sum().item())
    nonfinite_count = int((~finite_mask).sum().item())

    if detached.ndim == 0:
        affected_batch = [0]
    else:
        batch_finite = finite_mask.reshape(detached.shape[0], -1).all(dim=1)
        affected_batch = (
            (~batch_finite).nonzero(as_tuple=False).flatten().tolist()[:16]
        )

    finite_values = detached[finite_mask]
    if finite_values.numel() > 0:
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_max_abs = float(finite_values.abs().max().item())
    else:
        finite_min = float("nan")
        finite_max = float("nan")
        finite_max_abs = float("nan")

    return (
        f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"device={detached.device}, nonfinite={nonfinite_count}, "
        f"nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, "
        f"affected_batch={affected_batch}, finite_min={finite_min:.6g}, "
        f"finite_max={finite_max:.6g}, "
        f"finite_max_abs={finite_max_abs:.6g}"
    )


def _diagnostic_require_finite(name: str, tensor: Tensor) -> None:
    """Raise a value-preserving diagnostic at the first non-finite tensor."""

    report = _nonfinite_tensor_report(name, tensor)
    if report is not None:
        raise FloatingPointError(report)


@dataclass
class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.

    Args:
        multi_scale_features (`tuple(torch.FloatTensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attenntion based Pixel Decoder.
        mask_features (`torch.FloatTensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
    """

    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Transformer decoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions for mask predictions logits and a tuple of intermediate decoder activations,
    i.e. the output of each decoder layer, each of them gone through a layernorm.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. Returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. Returned when `output_attentions=True`.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`):
            Tuple of mask predictions from all layers of the transformer decoder.
        intermediate_hidden_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        group_stage_outputs (`tuple(TopologyGroupStageOutput)`, *optional*):
            Structural outputs for prediction stages st0--st9 when topology group decoding is enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    intermediate_hidden_states: Tuple[torch.FloatTensor] = None
    group_stage_outputs: Optional[Tuple[TopologyGroupStageOutput, ...]] = None
    stage_condition_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # Opt-in transport state export (group-supervised diagnostics or the
    # closed-loop final reader).  None preserves the legacy ModelOutput
    # key/tuple layout for every existing experiment.
    transport_final_latent_states: Optional[torch.FloatTensor] = None
    transport_relation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Mask2FormerDecoderContext:
    """Static tensors needed by a split Mask2Former decoder forward.

    Query-like tensors use the decoder's native query-first layout ``[Q,B,D]``;
    flattened visual tensors use ``[HW,B,D]``; mask features and optional
    spatial evidence are batch-first.  Keeping this boundary explicit avoids
    silently repeating image features before the image-only prefix has run.
    """

    initial_query_states: torch.FloatTensor
    query_position_embeddings: torch.FloatTensor
    encoder_hidden_states: Tuple[torch.FloatTensor, ...]
    positional_embeddings: Tuple[torch.FloatTensor, ...]
    pixel_embeddings: torch.FloatTensor
    feature_size_list: Tuple[Tuple[int, int], ...]
    siglip_spatial_features: Optional[torch.FloatTensor] = None
    siglip_valid_mask: Optional[torch.BoolTensor] = None
    spatial_metadata: Optional[Dict[str, torch.Tensor]] = None

    @property
    def batch_size(self) -> int:
        return int(self.pixel_embeddings.shape[0])


@dataclass
class Mask2FormerStage3State:
    """Continuation state after the formal st3 mask has been predicted.

    ``raw_query_states`` is the residual stream that must resume decoder layer
    3 (formal st4).  ``normalized_query_states`` is intentionally retained as
    a separate tensor because it produced ``mask_logits`` and is the correct
    representation for st3 proposal grouping.  Substituting the normalized
    tensor for the raw residual stream would change the pretrained decoder.
    """

    raw_query_states: torch.FloatTensor
    normalized_query_states: torch.FloatTensor
    mask_logits: torch.FloatTensor
    visual_attention_mask: torch.BoolTensor
    intermediate_hidden_states: Tuple[torch.FloatTensor, ...]
    masks_queries_logits: Tuple[torch.FloatTensor, ...]
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    next_layer_index: int = 3


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the `decoder` is a
    Multi-Scale Deformable Attention based decoder.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to multi-scale
    feature maps produced using **multi-scaling strategy** defined in the paper.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor`):
            Last hidden states (final feature map of shape `(batch_size, num_channels, height, width)`) of the last
            stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage. Returned if output_hidden_states is set to
            True.
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage.
    """

    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
            `output_hidden_states=True` is passed.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, , *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from each layer in the transformer decoder.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self attentions weights from transformer decoder.
    """

    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~Mask2FormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_logits (`List[Dict(str, torch.FloatTensor)]`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor,
    point_coordinates: torch.Tensor,
    add_dim=False,
    **kwargs,
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = torch.nn.functional.grid_sample(
        input_features.float(),
        (2.0 * point_coordinates - 1.0).float(),
        **kwargs,
    ).to(point_coordinates.dtype)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_masks: torch.Tensor,
    use_pos_weight: bool = False,
) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    num_positive = (labels == 1).sum()
    num_negative = (labels == 0).sum()
    pos_weight = (
        num_negative / num_positive if num_positive > 0 and use_pos_weight else torch.tensor(1.0).to(inputs.device)
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    loss = loss_pos + loss_neg
    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_masks: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    prob = inputs.sigmoid()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    p_t = prob * labels + (1 - prob) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the labels and the predictions of the network.

    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        num_points: int = 12544,
        use_sample_point: bool = False,
        cost_cls_type: str = "ce_cost",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
            num_points (`int`, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.use_sample_point = use_sample_point
        self.cost_cls_type = cost_cls_type
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: Optional[torch.Tensor] = None,
        mask_labels: torch.Tensor = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> List[Tuple[Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, height, width` with the predicted masks.
            class_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes` (where num_target_boxes is the number of ground-truth objects in the
                target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes, height, width` containing the target masks.

        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            if class_queries_logits is None:
                cost_class = None
            elif class_labels is None:
                cost_class = torch.zeros_like(class_queries_logits[i])
            elif self.cost_cls_type == "ce_cost":
                pred_probs = class_queries_logits[i].softmax(-1)
                # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be omitted.
                cost_class = -pred_probs[:, class_labels[i]]

            elif self.cost_cls_type == "focal_cost":
                pred_probs = class_queries_logits[i].sigmoid()
                neg_cost_class = (1 - self.alpha) * (pred_probs**self.gamma) * (-(1 - pred_probs + 1e-6).log())
                pos_cost_class = self.alpha * ((1 - pred_probs) ** self.gamma) * (-(pred_probs + 1e-6).log())
                cost_class = pos_cost_class[:, class_labels[i]] - neg_cost_class[:, class_labels[i]]

            pred_mask = masks_queries_logits[i]

            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device, dtype=pred_mask.dtype)

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            if self.use_sample_point:
                target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)

                pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)
            else:
                pred_mask = F.interpolate(
                    pred_mask,
                    size=target_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                pred_mask = pred_mask.flatten(1)
                target_mask = target_mask.flatten(1).to(pred_mask.dtype)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            if cost_class is None:
                cost_matrix = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            else:
                cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice

            # eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10, device=cost_matrix.device))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10, device=cost_matrix.device))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            cost_matrix = cost_matrix.cpu().to(torch.float32)
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix)
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        return matched_indices


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
class Mask2FormerLoss(nn.Module):
    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        requires_backends(self, ["scipy"])
        self.num_labels = config.num_labels
        self.loss_cls_type = config.loss_cls_type
        self.weight_dict = weight_dict

        # ce_loss configs
        self.eos_coef = config.no_object_weight
        # focal_loss configs
        self.alpha = config.alpha
        self.gamma = config.gamma

        # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio
        self.use_sample_point = config.use_sample_point
        self.use_nolabel_cls_loss = config.use_nolabel_cls_loss

        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
            use_sample_point=self.use_sample_point,
            cost_cls_type=self.loss_cls_type.split("_")[0] + "_cost",
            alpha=self.alpha,
            gamma=self.gamma,
        )

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # compute final size
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self,
        class_queries_logits: Optional[Tensor] = None,
        class_labels: Optional[List[Tensor]] = None,
        indices: Tuple[np.array] = None,
        num_masks: int = 1,
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cls** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        num_labels = pred_logits.shape[2]
        batch_size, num_queries, _ = pred_logits.shape
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)

        # for general_seg
        if class_labels is not None:
            target_classes_o = torch.cat(
                [target[j] for target, (_, j) in zip(class_labels, indices)]
            )  # shape of (batch_size, num_queries)
            target_classes = torch.full(
                (batch_size, num_queries),
                fill_value=(
                    num_labels - 1 if self.loss_cls_type == "ce_loss" else num_labels
                ),  # -1: add background class for ce_loss
                dtype=torch.int64,
                device=pred_logits.device,
            )
            target_classes[idx] = target_classes_o

            if self.loss_cls_type == "ce_loss":
                # Weight to apply to the null class
                empty_weight = torch.ones(
                    num_labels,
                    device=pred_logits.device,
                    dtype=pred_logits.dtype,
                )
                empty_weight[-1] = self.eos_coef
                criterion = nn.CrossEntropyLoss(weight=empty_weight)
                # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
                pred_logits_transposed = pred_logits.transpose(1, 2)
                loss_cls = criterion(pred_logits_transposed, target_classes)
            elif self.loss_cls_type == "focal_loss":
                target_classes_onehot = torch.zeros(
                    [
                        pred_logits.shape[0],
                        pred_logits.shape[1],
                        pred_logits.shape[2] + 1,
                    ],
                    dtype=pred_logits.dtype,
                    device=pred_logits.device,
                    layout=pred_logits.layout,
                )
                target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
                target_classes_onehot = target_classes_onehot[:, :, :-1]
                loss_cls = sigmoid_focal_loss(
                    pred_logits,
                    target_classes_onehot,
                    num_masks,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
        # for refer_seg/grounded_seg
        elif class_labels is None and self.use_nolabel_cls_loss:
            assert pred_logits.shape[2] == 1
            target_classes_onehot = torch.zeros(
                [pred_logits.shape[0], pred_logits.shape[1] + 1, pred_logits.shape[2]],
                dtype=pred_logits.dtype,
                device=pred_logits.device,
                layout=pred_logits.layout,
            )
            target_classes = torch.full(
                (batch_size, num_queries),
                fill_value=num_queries,
                dtype=torch.int64,
                device=pred_logits.device,
            )
            # the matched query indices will be the positive samples as no class_labels
            # indices: (src, tgt)
            target_classes_o = torch.cat([i for (i, _) in indices]).to(pred_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :-1, :]

            if self.loss_cls_type == "ce_loss":
                loss_cls = sigmoid_cross_entropy_loss(
                    pred_logits, target_classes_onehot, num_masks, use_pos_weight=True
                )
            elif self.loss_cls_type == "focal_loss":
                loss_cls = sigmoid_focal_loss(
                    pred_logits,
                    target_classes_onehot,
                    num_masks,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
        else:
            loss_cls = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype) * pred_logits.sum()

        losses = {"loss_cls": loss_cls}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        if self.use_sample_point:
            # Sample point coordinates
            # No need to upsample predictions as we are using normalized coordinates
            pred_masks = pred_masks[:, None]
            target_masks = target_masks[:, None]
            with torch.no_grad():
                point_coordinates = self.sample_points_using_uncertainty(
                    pred_masks,
                    lambda logits: self.calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                ).to(pred_masks.dtype)

                point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

            point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        else:
            # upsample predictions to the target size
            pred_masks = F.interpolate(
                pred_masks[:, None],
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            point_logits = pred_masks.flatten(1)
            point_labels = target_masks.flatten(1).to(point_logits.dtype)

        assert point_logits.shape == point_labels.shape

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions following indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels following indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Sam paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device, dtype=logits.dtype)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [
                    point_coordinates,
                    torch.rand(num_boxes, num_random_points, 2, device=logits.device),
                ],
                dim=1,
            )
        return point_coordinates

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`SamConfig`], then it contains the logits from
                the inner layers of the SamMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cls** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`SamConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(mask_labels, device=mask_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {**self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks)}
        if class_queries_logits is not None:
            losses.update(self.loss_labels(class_queries_logits, class_labels, indices, num_masks))
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(
                    masks_queries_logits,
                    class_queries_logits,
                    mask_labels,
                    class_labels,
                )
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, mask_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(masks) for masks in mask_labels])
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        world_size = 1
        # print_log(f"before reduce num_masks: {num_masks}, world_size: {world_size}", logger="current")
        if is_distributed():
            all_reduce(num_masks, "sum")
            world_size = get_world_size()

        # print_log(f"after reduce num_masks: {num_masks}, world_size: {world_size}", logger="current")
        num_masks = torch.clamp(num_masks / world_size, min=1)
        # print_log(f"after clamp num_masks: {num_masks}, world_size: {world_size}", logger="current")
        return num_masks


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_.float(), sampling_grid_l_.float(), mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_.to(value_l_.dtype))
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding with MaskFormer->Mask2Former
class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Modified from transformers.models.detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention
class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 128

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(
                [[shape[1], shape[0]] for shape in spatial_shapes_list],
                dtype=torch.long,
                device=reference_points.device,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention(value, spatial_shapes_list, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.embed_dim = config.feature_size
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=config.num_feature_levels,
            n_points=4,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of the backbone feature maps as a list of tuples.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights.transpose(1, 0),)

        return outputs


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder with DeformableDetrEncoder->Mask2FormerPixelDecoderEncoderOnly
class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`Mask2FormerPixelDecoderEncoderLayer`]. The encoder updates the flattened multi-scale feature maps through
    multiple deformable attention layers.

    Args:
        config: Mask2FormerConfig
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)]
        )

        self.gradient_checkpointing = False

    @staticmethod
    def get_reference_points(spatial_shapes_list, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of the backbone feature maps as a list of tuples.
            valid_ratios (`torch.FloatTensor`):
                Valid ratios of each feature map, has shape of `(batch_size, num_feature_levels, 2)`.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes_list):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of each feature map as a list of tuples.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(spatial_shapes_list, valid_ratios, device=inputs_embeds.device)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(1, 0),)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    reference_points,
                    spatial_shapes_list,
                    level_start_index,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states.transpose(1, 0),)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrModel with DeformableDetrModel->Mask2FormerPixelDecoder
class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels=None):
        super().__init__()

        self.config = config

        feature_dim = config.feature_size
        feature_channels = feature_channels if feature_channels is not None else config.feature_channels
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2

        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        self.num_feature_levels = config.num_feature_levels
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels :]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))

        # Create input projection layers
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        # Extra FPN levels
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Order convolutional layers from low to high resolution
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        input_embeds = []
        position_embeddings = []
        for level, x in enumerate(features[::-1][: self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x))
            position_embeddings.append(self.position_embedding(x))

        masks = [
            torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in input_embeds
        ]

        # Prepare encoder inputs (by flattening)
        spatial_shapes_list = [(embed.shape[2], embed.shape[3]) for embed in input_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=input_embeds_flat.device)
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)

        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = [x + self.level_embed[i].view(1, 1, -1) for i, x in enumerate(position_embeddings)]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype) for mask in masks], 1)

        # Send input_embeds_flat + masks_flat + level_pos_embed_flat (backbone + proj layer output) through encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds_flat,
                attention_mask=masks_flat,
                position_embeddings=level_pos_embed_flat,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        # We compute level_start_index_list separately from the tensor version level_start_index
        # to avoid iterating over a tensor which breaks torch.compile/export.
        level_start_index_list = [0]
        for height, width in spatial_shapes_list[:-1]:
            level_start_index_list.append(level_start_index_list[-1] + height * width)
        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = level_start_index_list[i + 1] - level_start_index_list[i]
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index_list[i]

        encoder_output = torch.split(last_hidden_state, split_sizes, dim=1)

        # Compute final features
        outputs = [
            x.transpose(1, 2).view(batch_size, -1, spatial_shapes_list[i][0], spatial_shapes_list[i][1])
            for i, x in enumerate(encoder_output)
        ]

        # Append extra FPN levels to outputs, ordered from low to high resolution
        for idx, feature in enumerate(features[: self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature)

            # Following FPN implementation, we use nearest upsampling here
            out = current_fpn + nn.functional.interpolate(
                outputs[-1], size=current_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            out = output_conv(out)
            outputs.append(out)

        num_cur_levels = 0
        multi_scale_features = []

        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1

        return Mask2FormerPixelDecoderOutput(
            mask_features=self.mask_projection(outputs[-1]),
            multi_scale_features=tuple(multi_scale_features),
            attentions=encoder_outputs.attentions,
        )


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        self.encoder = load_backbone(config) if config.use_backbone else None
        self.decoder = Mask2FormerPixelDecoder(
            config, feature_channels=self.encoder.channels if self.encoder else None
        )

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


# Modified from transformers.models.detr.modeling_detr.DetrAttention with Detr->Mask2Former
class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size * self.num_heads, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(target_len, batch_size * self.num_heads, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output).permute(1, 0, 2)

        return attn_output, attn_weights_reshaped


class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _prepare_query_self_attention_mask(
        self,
        attention_mask: Optional[Tensor],
        hidden_states: Tensor,
    ) -> Optional[Tensor]:
        """Normalize an optional group-local Query mask for self-attention.

        A boolean mask follows PyTorch's convention (``True`` means blocked).
        Callers may provide ``[Q,Q]``, ``[B,Q,Q]``, ``[B,H,Q,Q]``, or the
        already head-folded ``[B*H,Q,Q]`` additive form expected by
        :class:`Mask2FormerAttention`.  The legacy path supplies ``None`` and
        therefore remains byte-for-byte on its previous branch.
        """

        if attention_mask is None:
            return None
        if not isinstance(attention_mask, Tensor):
            raise TypeError("query self-attention mask must be a tensor")

        query_count, batch_size, _ = hidden_states.shape
        head_count = self.self_attn.num_heads
        expected_tail = (query_count, query_count)
        mask = attention_mask.to(device=hidden_states.device)

        if mask.ndim == 2:
            if tuple(mask.shape) != expected_tail:
                raise ValueError(
                    "2-D query self-attention mask must be [Q,Q], got "
                    f"{tuple(mask.shape)}"
                )
            mask = mask[None, None].expand(
                batch_size,
                head_count,
                query_count,
                query_count,
            )
        elif mask.ndim == 3:
            if tuple(mask.shape[-2:]) != expected_tail:
                raise ValueError(
                    "query self-attention mask has the wrong Query axes: "
                    f"{tuple(mask.shape)}"
                )
            if mask.shape[0] == batch_size:
                mask = mask[:, None].expand(
                    batch_size,
                    head_count,
                    query_count,
                    query_count,
                )
            elif mask.shape[0] == batch_size * head_count:
                mask = mask.reshape(
                    batch_size,
                    head_count,
                    query_count,
                    query_count,
                )
            else:
                raise ValueError(
                    "3-D query self-attention mask must be [B,Q,Q] or "
                    f"[B*H,Q,Q], got {tuple(mask.shape)}"
                )
        elif mask.ndim == 4:
            if tuple(mask.shape) != (
                batch_size,
                head_count,
                query_count,
                query_count,
            ):
                raise ValueError(
                    "4-D query self-attention mask must be [B,H,Q,Q], got "
                    f"{tuple(mask.shape)}"
                )
        else:
            raise ValueError(
                "query self-attention mask must have 2, 3, or 4 dimensions"
            )

        mask = mask.reshape(
            batch_size * head_count,
            query_count,
            query_count,
        )
        if mask.dtype == torch.bool:
            if bool(mask.all(dim=-1).any()):
                raise ValueError(
                    "query self-attention mask blocks every key for at least "
                    "one Query"
                )
            additive_mask = torch.zeros(
                mask.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            mask = additive_mask.masked_fill(
                mask,
                -torch.finfo(hidden_states.dtype).max,
            )
        else:
            mask = mask.to(dtype=hidden_states.dtype)
        return mask

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        query_self_attention_mask: Optional[torch.Tensor] = None,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states

        prepared_query_mask = self._prepare_query_self_attention_mask(
            query_self_attention_mask,
            hidden_states,
        )

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=prepared_query_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        query_self_attention_mask: Optional[torch.Tensor] = None,
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states

        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Self Attention Block
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        prepared_query_mask = self._prepare_query_self_attention_mask(
            query_self_attention_mask,
            hidden_states,
        )

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=prepared_query_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def forward_with_bipartite_transport(
        self,
        hidden_states: torch.Tensor,
        latent_states: torch.Tensor,
        transport_module: nn.Module,
        level_index: int = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        query_self_attention_mask: Optional[torch.Tensor] = None,
        proposal_mask_logits: Optional[torch.Tensor] = None,
        latent_proposal_refiner: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
        cached_vlm_latent_states: Optional[Tensor] = None,
        local_condition_states: Optional[Tensor] = None,
        local_condition_valid_mask: Optional[Tensor] = None,
        condition_refresh_module: Optional[nn.Module] = None,
    ):
        """Run one decoder layer with transport between visual/self attention.

        This method is used only by the explicitly enabled gate-free bipartite
        experiment.  The optional ``latent_proposal_refiner`` is used only by
        geometry st1--st3: after the original masked visual cross-attention it
        lets the preceding stage's mask pool fresh fused SAM+SigLIP evidence
        into Z before Q/Z transport.  The legacy ``forward_pre``/``forward_post``
        branches are intentionally untouched so old configurations and
        checkpoints retain their exact execution path.
        """

        if latent_states.ndim != 3:
            raise ValueError("transport latent states must be [B,L,D]")
        if latent_states.shape[0] != hidden_states.shape[1] or (
            latent_states.shape[-1] != hidden_states.shape[-1]
        ):
            raise ValueError(
                "transport latent batch/width must match decoder Queries"
            )
        if not callable(getattr(transport_module, "update_queries", None)) or (
            not callable(getattr(transport_module, "update_latents", None))
        ):
            raise TypeError(
                "transport_module must implement update_queries/update_latents"
            )
        if (proposal_mask_logits is None) != (latent_proposal_refiner is None):
            raise ValueError(
                "proposal_mask_logits and latent_proposal_refiner must be "
                "provided together"
            )
        refresh_values = (
            cached_vlm_latent_states,
            local_condition_states,
            local_condition_valid_mask,
            condition_refresh_module,
        )
        if any(value is not None for value in refresh_values) and not all(
            value is not None for value in refresh_values
        ):
            raise ValueError(
                "all late condition-refresh inputs must be provided together"
            )

        cross_attn_weights = None
        self_attn_weights = None

        # Masked visual cross-attention.  These operations mirror the original
        # pre/post-norm implementations exactly up to the new transport point.
        residual = hidden_states
        cross_input = (
            self.cross_attn_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        cross_output, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(
                cross_input,
                query_position_embeddings,
            ),
            key=self.with_pos_embed(
                encoder_hidden_states[level_index],
                position_embeddings[level_index],
            ),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )
        cross_output = nn.functional.dropout(
            cross_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + cross_output
        if not self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Geometry prefix only.  Q has now read the original Mask2Former
        # multi-scale visual feature.  M(previous) then selects the matching
        # regions from the separately fused SAM+SigLIP F64 map and refreshes Z
        # before Q and Z form their shared bidirectional relation.
        if latent_proposal_refiner is not None:
            refined_latents = latent_proposal_refiner(
                hidden_states.transpose(0, 1),
                latent_states,
                proposal_mask_logits,
            )
            if tuple(refined_latents.shape) != tuple(latent_states.shape):
                raise ValueError(
                    "latent_proposal_refiner must preserve latent shape: "
                    f"{tuple(refined_latents.shape)} != "
                    f"{tuple(latent_states.shape)}"
                )
            latent_states = refined_latents

        # Optional incremental st6/st9 experiment.  The current persistent Z
        # has already collected all preceding Query writebacks.  Recombine it
        # with the same cached post-VLM Z and then re-ground it in Cond+BG
        # before this stage forms its Query--latent relation.
        if condition_refresh_module is not None:
            latent_states = condition_refresh_module(
                latent_states,
                cached_vlm_latent_states,
                local_condition_states,
                local_condition_valid_mask,
            )

        # The shared Query--latent relation is formed here, before Query
        # self-attention, so semantic memory can affect the current stage's
        # inter-Query reasoning rather than only the following stage.
        transported_queries, relation_logits = (
            transport_module.update_queries(
                hidden_states.transpose(0, 1),
                latent_states,
            )
        )
        if tuple(transported_queries.shape) != (
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.shape[2],
        ):
            raise ValueError(
                "transport Query update must return [B,Q,D], got "
                f"{tuple(transported_queries.shape)}"
            )
        hidden_states = transported_queries.transpose(0, 1)

        # Original 200-Query self-attention.
        residual = hidden_states
        self_input = (
            self.self_attn_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        prepared_query_mask = self._prepare_query_self_attention_mask(
            query_self_attention_mask,
            self_input,
        )
        self_output, self_attn_weights = self.self_attn(
            hidden_states=self_input,
            position_embeddings=query_position_embeddings,
            attention_mask=prepared_query_mask,
            output_attentions=True,
        )
        self_output = nn.functional.dropout(
            self_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + self_output
        if not self.pre_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Column-normalizing the same relation writes the now self-attended
        # Queries back to their persistent latent slots.
        delayed_writeback = bool(getattr(
            transport_module, "writeback_after_prediction", False
        ))
        if not delayed_writeback:
            latent_states = transport_module.update_latents(
                hidden_states.transpose(0, 1),
                latent_states,
                relation_logits,
            )

        # Original decoder FFN.
        residual = hidden_states
        ffn_input = (
            self.final_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        ffn_output = self.activation_fn(self.fc1(ffn_input))
        ffn_output = nn.functional.dropout(
            ffn_output,
            p=self.activation_dropout,
            training=self.training,
        )
        ffn_output = self.fc2(ffn_output)
        ffn_output = nn.functional.dropout(
            ffn_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + ffn_output
        if not self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, latent_states)
        if delayed_writeback:
            # Explicit checkpoint output: never cache a live relation on a
            # module, since backward may recompute a different decoder stage.
            outputs += (relation_logits,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward_with_stagewise_trifusion(
        self,
        hidden_states: torch.Tensor,
        latent_states: torch.Tensor,
        condition_states: torch.Tensor,
        condition_anchor: torch.Tensor,
        condition_valid_mask: torch.Tensor,
        condition_update_mask: torch.Tensor,
        trifusion_module: nn.Module,
        level_index: int = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        query_self_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Run visual attention, one Q/Z/C update, then original self/FFN."""

        if latent_states.ndim != 3 or condition_states.ndim != 3:
            raise ValueError("trifusion latent/condition states must be [B,T,D]")
        if latent_states.shape[0] != hidden_states.shape[1] or (
            latent_states.shape[-1] != hidden_states.shape[-1]
        ):
            raise ValueError("trifusion latent batch/width must match Queries")
        if condition_states.shape[0] != hidden_states.shape[1] or (
            condition_states.shape[-1] != hidden_states.shape[-1]
        ):
            raise ValueError("trifusion condition batch/width must match Queries")
        if not callable(getattr(trifusion_module, "forward", None)):
            raise TypeError("trifusion_module must be callable")

        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states
        cross_input = (
            self.cross_attn_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        cross_output, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(
                cross_input,
                query_position_embeddings,
            ),
            key=self.with_pos_embed(
                encoder_hidden_states[level_index],
                position_embeddings[level_index],
            ),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )
        cross_output = nn.functional.dropout(
            cross_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + cross_output
        if not self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        query_states, latent_states, condition_states = trifusion_module(
            query_states=hidden_states.transpose(0, 1),
            latent_states=latent_states,
            condition_states=condition_states,
            condition_anchor=condition_anchor,
            condition_valid_mask=condition_valid_mask,
            condition_update_mask=condition_update_mask,
        )
        expected_query_shape = (
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.shape[2],
        )
        if tuple(query_states.shape) != expected_query_shape:
            raise ValueError(
                "trifusion Query update must return [B,Q,D], got "
                f"{tuple(query_states.shape)}"
            )
        hidden_states = query_states.transpose(0, 1)

        residual = hidden_states
        self_input = (
            self.self_attn_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        prepared_query_mask = self._prepare_query_self_attention_mask(
            query_self_attention_mask,
            self_input,
        )
        self_output, self_attn_weights = self.self_attn(
            hidden_states=self_input,
            position_embeddings=query_position_embeddings,
            attention_mask=prepared_query_mask,
            output_attentions=True,
        )
        self_output = nn.functional.dropout(
            self_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + self_output
        if not self.pre_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        ffn_input = (
            self.final_layer_norm(hidden_states)
            if self.pre_norm
            else hidden_states
        )
        ffn_output = self.activation_fn(self.fc1(ffn_input))
        ffn_output = nn.functional.dropout(
            ffn_output,
            p=self.activation_dropout,
            training=self.training,
        )
        ffn_output = self.fc2(ffn_output)
        ffn_output = nn.functional.dropout(
            ffn_output,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + ffn_output
        if not self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, latent_states, condition_states)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        query_self_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(1, seq_len, tgt_len, src_len)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the keys in the masked-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                Cross attention input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_attention_mask (`torch.FloatTensor`):
                Encoder attention mask of size`(1, seq_len, tgt_len, src_len)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        if self.pre_norm:
            outputs = self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                query_self_attention_mask=query_self_attention_mask,
            )
        else:
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                query_self_attention_mask=query_self_attention_mask,
            )

        return outputs


class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = config.num_feature_levels  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

    def _validate_split_context(
        self,
        context: Mask2FormerDecoderContext,
    ) -> None:
        if not isinstance(context, Mask2FormerDecoderContext):
            raise TypeError("context must be a Mask2FormerDecoderContext")
        batch_size = context.batch_size
        if context.initial_query_states.ndim != 3:
            raise ValueError("initial_query_states must be [Q,B,D]")
        query_count, query_batch, hidden_dim = context.initial_query_states.shape
        if query_batch != batch_size or hidden_dim != self.config.hidden_dim:
            raise ValueError(
                "initial Query/context shapes disagree: "
                f"queries={tuple(context.initial_query_states.shape)}, "
                f"context_batch={batch_size}, hidden_dim={self.config.hidden_dim}"
            )
        if tuple(context.query_position_embeddings.shape) != (
            query_count,
            batch_size,
            hidden_dim,
        ):
            raise ValueError(
                "query_position_embeddings must match initial_query_states"
            )
        if not (
            len(context.encoder_hidden_states)
            == len(context.positional_embeddings)
            == len(context.feature_size_list)
            == self.num_feature_levels
        ):
            raise ValueError(
                "split context must contain exactly one feature, position, "
                "and size entry per feature level"
            )
        for level, (features, positions, feature_size) in enumerate(
            zip(
                context.encoder_hidden_states,
                context.positional_embeddings,
                context.feature_size_list,
            )
        ):
            if features.ndim != 3 or positions.ndim != 3:
                raise ValueError(
                    f"flattened level {level} tensors must be [HW,B,D]"
                )
            if tuple(features.shape) != tuple(positions.shape):
                raise ValueError(
                    f"level {level} feature/position shapes differ: "
                    f"{tuple(features.shape)} != {tuple(positions.shape)}"
                )
            if features.shape[1:] != (batch_size, hidden_dim):
                raise ValueError(
                    f"level {level} has an invalid batch or hidden size"
                )
            if int(features.shape[0]) != int(feature_size[0]) * int(
                feature_size[1]
            ):
                raise ValueError(
                    f"level {level} flattened length does not match "
                    f"feature_size_list: {features.shape[0]} versus "
                    f"{tuple(feature_size)}"
                )
        if (
            context.pixel_embeddings.ndim != 4
            or context.pixel_embeddings.shape[0] != batch_size
        ):
            raise ValueError("pixel_embeddings must be [B,C,H,W]")

    def forward_to_stage3(
        self,
        context: Mask2FormerDecoderContext,
        *,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Mask2FormerStage3State:
        """Run the image-only prefix through formal prediction stage st3.

        The method executes st0 followed by decoder layers 0, 1, and 2.  No
        SEG embedding, topology feedback, or condition feature is introduced
        here.  The returned state retains the st3 visual attention prior so a
        query update between st3 and st4 does not create an extra mask stage.
        """

        self._validate_split_context(context)
        if len(self.layers) < 3:
            raise ValueError("forward_to_stage3 requires at least three decoder layers")
        if self.training and self.layerdrop > 0:
            raise RuntimeError(
                "split st3 execution requires layerdrop=0 so st0--st3 have "
                "stable semantics"
            )

        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else bool(output_attentions)
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else bool(output_hidden_states)
        )
        hidden_states = (
            context.initial_query_states
            if inputs_embeds is None
            else inputs_embeds
        )
        if tuple(hidden_states.shape) != tuple(
            context.initial_query_states.shape
        ):
            raise ValueError(
                "split-prefix inputs_embeds must match context initial Query "
                f"shape: {tuple(hidden_states.shape)} != "
                f"{tuple(context.initial_query_states.shape)}"
            )

        intermediate = ()
        intermediate_mask_predictions = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        normalized_query_states = self.layernorm(hidden_states)
        intermediate += (normalized_query_states,)
        predicted_mask, attention_mask = self.mask_predictor(
            normalized_query_states,
            context.pixel_embeddings,
            context.feature_size_list[0],
        )
        intermediate_mask_predictions += (predicted_mask,)

        for idx in range(3):
            decoder_layer = self.layers[idx]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            level_index = idx % self.num_feature_levels
            where = (
                attention_mask.sum(-1) != attention_mask.shape[-1]
            ).to(attention_mask.dtype)
            attention_mask = attention_mask * where.unsqueeze(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    level_index,
                    None,
                    context.positional_embeddings,
                    context.query_position_embeddings,
                    context.encoder_hidden_states,
                    attention_mask,
                    output_attentions,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=context.positional_embeddings,
                    query_position_embeddings=(
                        context.query_position_embeddings
                    ),
                    encoder_hidden_states=context.encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_self_attention_mask=None,
                )

            hidden_states = layer_outputs[0]
            normalized_query_states = self.layernorm(hidden_states)
            predicted_mask, attention_mask = self.mask_predictor(
                normalized_query_states,
                context.pixel_embeddings,
                context.feature_size_list[
                    (idx + 1) % self.num_feature_levels
                ],
            )
            intermediate += (normalized_query_states,)
            intermediate_mask_predictions += (predicted_mask,)
            if output_attentions:
                attentions += (layer_outputs[1],)

        return Mask2FormerStage3State(
            raw_query_states=hidden_states,
            normalized_query_states=normalized_query_states,
            mask_logits=predicted_mask,
            visual_attention_mask=attention_mask,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
            hidden_states=all_hidden_states,
            attentions=attentions,
            next_layer_index=3,
        )

    def forward_to_stage3_with_latent_transport(
        self,
        context: Mask2FormerDecoderContext,
        *,
        latent_initializer: Callable[[Tensor, Tensor], Tensor],
        transport_modules: Sequence[nn.Module],
        prefix_fused_refiner: Callable[
            [int, Tensor, Tensor, Tensor, Tensor], Tensor
        ],
        prefix_fused_features: Tensor,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[Mask2FormerStage3State, Tensor]:
        """Run st0--st3 with a recurrent proposal latent from st0 onward.

        This is an opt-in geometry-path entrypoint.  The ordinary
        :meth:`forward_to_stage3` remains byte-for-byte on the retained paths.
        ``latent_initializer`` consumes the formal st0 Query/mask prediction
        and returns ``Z0``.  Decoder layers 0, 1, and 2 retain their original
        masked visual cross-attention, then use M0/M1/M2 respectively to refresh
        Z from fused SAM+SigLIP features before Q/Z transport.  They produce
        st1, st2, and st3 respectively; no prefix fused refresh is run later.
        """

        self._validate_split_context(context)
        if len(self.layers) < 3:
            raise ValueError(
                "prefix latent transport requires at least three decoder layers"
            )
        if len(transport_modules) != 3:
            raise ValueError(
                "prefix transport_modules must cover exactly st1--st3"
            )
        if not callable(latent_initializer):
            raise TypeError("latent_initializer must be callable")
        if not callable(prefix_fused_refiner):
            raise TypeError("prefix_fused_refiner must be callable")
        if prefix_fused_features.ndim != 4 or (
            prefix_fused_features.shape[0] != context.batch_size
        ):
            raise ValueError("prefix_fused_features must be [B,C,H,W]")
        if self.training and self.layerdrop > 0:
            raise RuntimeError(
                "split st3 execution requires layerdrop=0 so st0--st3 have "
                "stable semantics"
            )

        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else bool(output_attentions)
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else bool(output_hidden_states)
        )
        hidden_states = (
            context.initial_query_states
            if inputs_embeds is None
            else inputs_embeds
        )
        if tuple(hidden_states.shape) != tuple(
            context.initial_query_states.shape
        ):
            raise ValueError(
                "split-prefix inputs_embeds must match context initial Query "
                f"shape: {tuple(hidden_states.shape)} != "
                f"{tuple(context.initial_query_states.shape)}"
            )

        intermediate = ()
        intermediate_mask_predictions = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        # Formal st0 exists before decoder layer 0.  It is the earliest point
        # where mask-pooled region evidence and geometry are well defined.
        normalized_query_states = self.layernorm(hidden_states)
        intermediate += (normalized_query_states,)
        predicted_mask, attention_mask = self.mask_predictor(
            normalized_query_states,
            context.pixel_embeddings,
            context.feature_size_list[0],
        )
        intermediate_mask_predictions += (predicted_mask,)
        latent_states = latent_initializer(
            normalized_query_states.transpose(0, 1),
            predicted_mask,
        )
        if latent_states.ndim != 3 or (
            latent_states.shape[0] != context.batch_size
            or latent_states.shape[-1] != hidden_states.shape[-1]
        ):
            raise ValueError(
                "latent_initializer must return [B,L,Ddecoder]"
            )

        for idx in range(3):
            decoder_layer = self.layers[idx]
            transport_module = transport_modules[idx]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            level_index = idx % self.num_feature_levels
            where = (
                attention_mask.sum(-1) != attention_mask.shape[-1]
            ).to(attention_mask.dtype)
            attention_mask = attention_mask * where.unsqueeze(-1)

            def refine_stage_latents(
                layer_query_states,
                layer_latent_states,
                layer_mask_logits,
                _stage_index=idx,
            ):
                return prefix_fused_refiner(
                    _stage_index,
                    layer_query_states,
                    layer_latent_states,
                    layer_mask_logits,
                    prefix_fused_features,
                )

            if self.gradient_checkpointing and self.training:
                def prefix_transport_forward(
                    layer_hidden_states,
                    layer_latent_states,
                    layer_mask_logits,
                    layer_fused_features,
                    _decoder_layer=decoder_layer,
                    _transport_module=transport_module,
                    _level_index=level_index,
                    _attention_mask=attention_mask,
                    _stage_index=idx,
                ):
                    def checkpointed_latent_proposal_refiner(
                        layer_query_states,
                        current_latent_states,
                        current_mask_logits,
                    ):
                        return prefix_fused_refiner(
                            _stage_index,
                            layer_query_states,
                            current_latent_states,
                            current_mask_logits,
                            layer_fused_features,
                        )

                    return _decoder_layer.forward_with_bipartite_transport(
                        layer_hidden_states,
                        layer_latent_states,
                        _transport_module,
                        level_index=_level_index,
                        position_embeddings=context.positional_embeddings,
                        query_position_embeddings=(
                            context.query_position_embeddings
                        ),
                        encoder_hidden_states=context.encoder_hidden_states,
                        encoder_attention_mask=_attention_mask,
                        output_attentions=output_attentions,
                        query_self_attention_mask=None,
                        proposal_mask_logits=layer_mask_logits,
                        latent_proposal_refiner=(
                            checkpointed_latent_proposal_refiner
                        ),
                    )

                layer_outputs = self._gradient_checkpointing_func(
                    prefix_transport_forward,
                    hidden_states,
                    latent_states,
                    predicted_mask,
                    prefix_fused_features,
                )
            else:
                layer_outputs = (
                    decoder_layer.forward_with_bipartite_transport(
                        hidden_states,
                        latent_states,
                        transport_module,
                        level_index=level_index,
                        position_embeddings=context.positional_embeddings,
                        query_position_embeddings=(
                            context.query_position_embeddings
                        ),
                        encoder_hidden_states=context.encoder_hidden_states,
                        encoder_attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        query_self_attention_mask=None,
                        proposal_mask_logits=predicted_mask,
                        latent_proposal_refiner=refine_stage_latents,
                    )
                )

            hidden_states = layer_outputs[0]
            latent_states = layer_outputs[1]
            normalized_query_states = self.layernorm(hidden_states)
            predicted_mask, attention_mask = self.mask_predictor(
                normalized_query_states,
                context.pixel_embeddings,
                context.feature_size_list[
                    (idx + 1) % self.num_feature_levels
                ],
            )
            intermediate += (normalized_query_states,)
            intermediate_mask_predictions += (predicted_mask,)
            if output_attentions:
                attentions += (layer_outputs[2],)

        state = Mask2FormerStage3State(
            raw_query_states=hidden_states,
            normalized_query_states=normalized_query_states,
            mask_logits=predicted_mask,
            visual_attention_mask=attention_mask,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
            hidden_states=all_hidden_states,
            attentions=attentions,
            next_layer_index=3,
        )
        return state, latent_states

    def expand_stage3_state(
        self,
        state: Mask2FormerStage3State,
        row_source_indices: Tensor,
    ) -> Mask2FormerStage3State:
        """Gather a B-level st3 state into source-major effective SEG rows."""

        if not isinstance(state, Mask2FormerStage3State):
            raise TypeError("state must be a Mask2FormerStage3State")
        if row_source_indices.ndim != 1:
            raise ValueError("row_source_indices must be one-dimensional")
        source_batch = int(state.mask_logits.shape[0])
        indices = row_source_indices.to(
            device=state.mask_logits.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            raise ValueError("row_source_indices must not be empty")
        if bool((indices < 0).any()) or bool((indices >= source_batch).any()):
            raise IndexError(
                "row_source_indices contains an index outside the st3 batch"
            )

        def gather_query_first(tensor: Tensor) -> Tensor:
            return tensor.index_select(
                1,
                indices.to(device=tensor.device),
            )

        def gather_batch_first(tensor: Tensor) -> Tensor:
            return tensor.index_select(
                0,
                indices.to(device=tensor.device),
            )

        query_count = int(state.raw_query_states.shape[0])
        head_count = int(self.config.num_attention_heads)
        visual_mask = state.visual_attention_mask
        if visual_mask.ndim != 3 or visual_mask.shape[0] != source_batch * head_count:
            raise ValueError(
                "st3 visual_attention_mask must be [B*H,Q,HW], got "
                f"{tuple(visual_mask.shape)}"
            )
        if visual_mask.shape[1] != query_count:
            raise ValueError("st3 visual attention Query count is inconsistent")
        visual_mask = visual_mask.reshape(
            source_batch,
            head_count,
            query_count,
            visual_mask.shape[-1],
        ).index_select(
            0,
            indices.to(device=visual_mask.device),
        ).flatten(0, 1)

        hidden_states = (
            None
            if state.hidden_states is None
            else tuple(gather_query_first(x) for x in state.hidden_states)
        )
        attentions = (
            None
            if state.attentions is None
            else tuple(gather_batch_first(x) for x in state.attentions)
        )
        return Mask2FormerStage3State(
            raw_query_states=gather_query_first(state.raw_query_states),
            normalized_query_states=gather_query_first(
                state.normalized_query_states
            ),
            mask_logits=gather_batch_first(state.mask_logits),
            visual_attention_mask=visual_mask,
            intermediate_hidden_states=tuple(
                gather_query_first(x)
                for x in state.intermediate_hidden_states
            ),
            masks_queries_logits=tuple(
                gather_batch_first(x) for x in state.masks_queries_logits
            ),
            hidden_states=hidden_states,
            attentions=attentions,
            next_layer_index=state.next_layer_index,
        )

    def forward_from_stage3(
        self,
        context: Mask2FormerDecoderContext,
        state: Mask2FormerStage3State,
        *,
        resumed_query_states: Optional[Tensor] = None,
        query_self_attention_mask: Optional[Tensor] = None,
        pre_prediction_refiner: Optional[
            Callable[
                [int, Tensor, Mask2FormerDecoderContext],
                Tensor,
            ]
        ] = None,
        stage_refiner: Optional[
            Callable[
                [
                    int,
                    Tensor,
                    Tensor,
                    Tensor,
                    Mask2FormerDecoderContext,
                ],
                Optional[Tensor],
            ]
        ] = None,
        transport_latent_states: Optional[Tensor] = None,
        transport_modules: Optional[Sequence[nn.Module]] = None,
        transport_cached_vlm_latent_states: Optional[Tensor] = None,
        transport_local_condition_states: Optional[Tensor] = None,
        transport_local_condition_valid_mask: Optional[Tensor] = None,
        transport_condition_refreshers: Optional[nn.ModuleDict] = None,
        export_transport_latent_states: bool = False,
        trifusion_latent_states: Optional[Tensor] = None,
        trifusion_condition_states: Optional[Tensor] = None,
        trifusion_condition_anchor: Optional[Tensor] = None,
        trifusion_condition_valid_mask: Optional[Tensor] = None,
        trifusion_condition_update_mask: Optional[Tensor] = None,
        trifusion_modules: Optional[Sequence[nn.Module]] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        """Resume at decoder layer 3 and produce formal stages st4--st9.

        ``resumed_query_states`` and callback Query tensors are batch-first
        ``[B,Q,D]``.  ``pre_prediction_refiner`` runs after the ordinary
        decoder layer but before layer normalization and the formal mask head;
        its result therefore defines both the current stage prediction and the
        next stage's visual attention prior.  The older ``stage_refiner`` runs
        after mask prediction and remains available for legacy Group paths.

        The optional ``transport_*`` arguments select the independent
        gate-free bipartite path.  In that path each st4--st9 layer executes
        visual cross-attention, latent-to-Query transport, original Query
        self-attention, Query-to-latent transport, and the original FFN.  The
        legacy callback paths are deliberately left unchanged.
        """

        self._validate_split_context(context)
        if not isinstance(state, Mask2FormerStage3State):
            raise TypeError("state must be a Mask2FormerStage3State")
        if state.next_layer_index != 3:
            raise ValueError(
                "forward_from_stage3 requires next_layer_index=3, got "
                f"{state.next_layer_index}"
            )
        if self.training and self.layerdrop > 0:
            raise RuntimeError(
                "split st3 execution requires layerdrop=0 so st4--st9 have "
                "stable semantics"
            )
        transport_enabled = (
            transport_latent_states is not None
            or transport_modules is not None
        )
        trifusion_values = (
            trifusion_latent_states,
            trifusion_condition_states,
            trifusion_condition_anchor,
            trifusion_condition_valid_mask,
            trifusion_condition_update_mask,
            trifusion_modules,
        )
        trifusion_enabled = any(value is not None for value in trifusion_values)
        if trifusion_enabled and not all(
            value is not None for value in trifusion_values
        ):
            raise ValueError("all stagewise trifusion inputs must be provided")
        if transport_enabled and trifusion_enabled:
            raise ValueError("bipartite transport and stagewise trifusion are exclusive")
        if transport_enabled and (
            transport_latent_states is None or transport_modules is None
        ):
            raise ValueError(
                "transport_latent_states and transport_modules must be "
                "provided together"
            )
        if not isinstance(export_transport_latent_states, bool):
            raise TypeError("export_transport_latent_states must be boolean")
        if export_transport_latent_states and not transport_enabled:
            raise ValueError(
                "export_transport_latent_states requires bipartite transport"
            )
        refresh_values = (
            transport_cached_vlm_latent_states,
            transport_local_condition_states,
            transport_local_condition_valid_mask,
            transport_condition_refreshers,
        )
        refresh_enabled = any(value is not None for value in refresh_values)
        if refresh_enabled and not all(
            value is not None for value in refresh_values
        ):
            raise ValueError(
                "all transport condition-refresh inputs must be provided together"
            )
        if refresh_enabled and not transport_enabled:
            raise ValueError(
                "late condition refresh requires bipartite transport"
            )
        if (transport_enabled or trifusion_enabled) and (
            pre_prediction_refiner is not None or stage_refiner is not None
        ):
            raise ValueError(
                "latent transport is mutually exclusive with legacy "
                "stage refinement callbacks"
            )
        if transport_enabled:
            if transport_latent_states.ndim != 3:
                raise ValueError(
                    "transport_latent_states must be [B,L,D]"
                )
            expected_transport_depth = (
                len(self.layers) - state.next_layer_index
            )
            if len(transport_modules) != expected_transport_depth:
                raise ValueError(
                    "transport_modules must cover exactly st4--st9: "
                    f"{len(transport_modules)} != "
                    f"{expected_transport_depth}"
                )
            if refresh_enabled:
                if (
                    transport_cached_vlm_latent_states.shape
                    != transport_latent_states.shape
                ):
                    raise ValueError(
                        "cached VLM and persistent transport latents must "
                        "have equal shapes"
                    )
                if transport_local_condition_states.ndim != 3 or tuple(
                    transport_local_condition_valid_mask.shape
                ) != tuple(transport_local_condition_states.shape[:2]):
                    raise ValueError(
                        "transport local conditions must be [B,C,D] and [B,C]"
                    )
                if transport_local_condition_valid_mask.dtype != torch.bool:
                    raise TypeError(
                        "transport_local_condition_valid_mask must be boolean"
                    )
                valid_refresh_keys = {
                    f"st{stage}"
                    for stage in range(
                        state.next_layer_index + 1,
                        len(self.layers) + 1,
                    )
                }
                unexpected_refresh_keys = set(
                    transport_condition_refreshers.keys()
                ) - valid_refresh_keys
                if unexpected_refresh_keys:
                    raise ValueError(
                        "condition refreshers target invalid formal stages: "
                        f"{sorted(unexpected_refresh_keys)}"
                    )
        if trifusion_enabled:
            if trifusion_latent_states.ndim != 3 or (
                trifusion_condition_states.ndim != 3
                or trifusion_condition_anchor.ndim != 3
            ):
                raise ValueError("trifusion Z/C states must be [B,T,D]")
            if trifusion_condition_states.shape != trifusion_condition_anchor.shape:
                raise ValueError("trifusion condition state/anchor shapes differ")
            if trifusion_condition_valid_mask.shape != (
                trifusion_condition_states.shape[:2]
            ) or trifusion_condition_update_mask.shape != (
                trifusion_condition_states.shape[:2]
            ):
                raise ValueError("trifusion condition masks must be [B,C]")
            expected_trifusion_depth = len(self.layers) - state.next_layer_index
            if len(trifusion_modules) != expected_trifusion_depth:
                raise ValueError(
                    "trifusion_modules must cover exactly st4--st9: "
                    f"{len(trifusion_modules)} != {expected_trifusion_depth}"
                )
        return_dict = (
            self.config.use_return_dict
            if return_dict is None
            else bool(return_dict)
        )

        expected_internal_shape = tuple(state.raw_query_states.shape)
        if expected_internal_shape[1] != context.batch_size:
            raise ValueError(
                "expanded st3 state and decoder context batches differ: "
                f"{expected_internal_shape[1]} != {context.batch_size}"
            )
        if resumed_query_states is None:
            hidden_states = state.raw_query_states
        else:
            expected_batch_first = (
                context.batch_size,
                expected_internal_shape[0],
                expected_internal_shape[2],
            )
            if tuple(resumed_query_states.shape) != expected_batch_first:
                raise ValueError(
                    "resumed_query_states must be [B,Q,D], got "
                    f"{tuple(resumed_query_states.shape)}; expected "
                    f"{expected_batch_first}"
                )
            hidden_states = resumed_query_states.transpose(0, 1)

        if transport_enabled:
            expected_latent_batch_width = (
                context.batch_size,
                expected_internal_shape[2],
            )
            actual_latent_batch_width = (
                transport_latent_states.shape[0],
                transport_latent_states.shape[2],
            )
            if actual_latent_batch_width != expected_latent_batch_width:
                raise ValueError(
                    "transport latent batch/width must match decoder state: "
                    f"{actual_latent_batch_width} != "
                    f"{expected_latent_batch_width}"
                )
        if trifusion_enabled:
            expected_batch_width = (
                context.batch_size,
                expected_internal_shape[2],
            )
            for name, value in (
                ("latent", trifusion_latent_states),
                ("condition", trifusion_condition_states),
                ("condition anchor", trifusion_condition_anchor),
            ):
                actual_batch_width = (value.shape[0], value.shape[2])
                if actual_batch_width != expected_batch_width:
                    raise ValueError(
                        f"trifusion {name} batch/width must match decoder: "
                        f"{actual_batch_width} != {expected_batch_width}"
                    )

        attention_mask = state.visual_attention_mask
        intermediate = state.intermediate_hidden_states
        intermediate_mask_predictions = state.masks_queries_logits
        all_hidden_states = state.hidden_states
        attentions = state.attentions
        output_hidden_states = all_hidden_states is not None
        output_attentions = attentions is not None
        stage_condition_states = () if trifusion_enabled else None
        delayed_transport = transport_enabled and all(
            bool(getattr(module, "writeback_after_prediction", False))
            for module in transport_modules
        )
        if transport_enabled and any(
            bool(getattr(module, "writeback_after_prediction", False))
            for module in transport_modules
        ) and not delayed_transport:
            raise ValueError("delayed writeback must be enabled at every st4--st9 stage")
        if delayed_transport and (pre_prediction_refiner is not None or stage_refiner is not None):
            raise ValueError("delayed writeback cannot be combined with legacy prediction refiners")
        transport_relations = (
            () if delayed_transport and self.training and torch.is_grad_enabled() else None
        )

        for idx in range(state.next_layer_index, len(self.layers)):
            decoder_layer = self.layers[idx]
            stage_index = idx + 1
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            level_index = idx % self.num_feature_levels
            where = (
                attention_mask.sum(-1) != attention_mask.shape[-1]
            ).to(attention_mask.dtype)
            attention_mask = attention_mask * where.unsqueeze(-1)

            if trifusion_enabled:
                trifusion_module = trifusion_modules[
                    idx - state.next_layer_index
                ]
                if self.gradient_checkpointing and self.training:
                    def trifusion_forward(
                        layer_hidden_states,
                        layer_latent_states,
                        layer_condition_states,
                        layer_condition_anchor,
                        _decoder_layer=decoder_layer,
                        _trifusion_module=trifusion_module,
                        _level_index=level_index,
                        _attention_mask=attention_mask,
                    ):
                        return _decoder_layer.forward_with_stagewise_trifusion(
                            layer_hidden_states,
                            layer_latent_states,
                            layer_condition_states,
                            layer_condition_anchor,
                            trifusion_condition_valid_mask,
                            trifusion_condition_update_mask,
                            _trifusion_module,
                            level_index=_level_index,
                            position_embeddings=context.positional_embeddings,
                            query_position_embeddings=(
                                context.query_position_embeddings
                            ),
                            encoder_hidden_states=context.encoder_hidden_states,
                            encoder_attention_mask=_attention_mask,
                            output_attentions=output_attentions,
                            query_self_attention_mask=(
                                query_self_attention_mask
                            ),
                        )

                    layer_outputs = self._gradient_checkpointing_func(
                        trifusion_forward,
                        hidden_states,
                        trifusion_latent_states,
                        trifusion_condition_states,
                        trifusion_condition_anchor,
                    )
                else:
                    layer_outputs = (
                        decoder_layer.forward_with_stagewise_trifusion(
                            hidden_states,
                            trifusion_latent_states,
                            trifusion_condition_states,
                            trifusion_condition_anchor,
                            trifusion_condition_valid_mask,
                            trifusion_condition_update_mask,
                            trifusion_module,
                            level_index=level_index,
                            position_embeddings=context.positional_embeddings,
                            query_position_embeddings=(
                                context.query_position_embeddings
                            ),
                            encoder_hidden_states=context.encoder_hidden_states,
                            encoder_attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            query_self_attention_mask=(
                                query_self_attention_mask
                            ),
                        )
                    )
                raw_layer_hidden_states = layer_outputs[0]
                trifusion_latent_states = layer_outputs[1]
                trifusion_condition_states = layer_outputs[2]
                stage_condition_states += (trifusion_condition_states,)
                attention_output_index = 3
            elif transport_enabled:
                transport_module = transport_modules[
                    idx - state.next_layer_index
                ]
                condition_refresh_module = None
                if refresh_enabled:
                    refresh_key = f"st{stage_index}"
                    if refresh_key in transport_condition_refreshers:
                        condition_refresh_module = (
                            transport_condition_refreshers[refresh_key]
                        )
                if self.gradient_checkpointing and self.training:
                    if condition_refresh_module is None:
                        # Preserve the original transport checkpoint call for
                        # every stage without a refresher.
                        def transport_forward(
                            layer_hidden_states,
                            layer_latent_states,
                            _decoder_layer=decoder_layer,
                            _transport_module=transport_module,
                            _level_index=level_index,
                            _attention_mask=attention_mask,
                        ):
                            return _decoder_layer.forward_with_bipartite_transport(
                                layer_hidden_states,
                                layer_latent_states,
                                _transport_module,
                                level_index=_level_index,
                                position_embeddings=(
                                    context.positional_embeddings
                                ),
                                query_position_embeddings=(
                                    context.query_position_embeddings
                                ),
                                encoder_hidden_states=(
                                    context.encoder_hidden_states
                                ),
                                encoder_attention_mask=_attention_mask,
                                output_attentions=output_attentions,
                                query_self_attention_mask=(
                                    query_self_attention_mask
                                ),
                            )

                        layer_outputs = self._gradient_checkpointing_func(
                            transport_forward,
                            hidden_states,
                            transport_latent_states,
                        )
                    else:
                        # Tensor refresh inputs are explicit checkpoint inputs
                        # so a future non-frozen variant also propagates their
                        # gradients correctly during recomputation.
                        def transport_refresh_forward(
                            layer_hidden_states,
                            layer_latent_states,
                            layer_cached_vlm_latent_states,
                            layer_local_condition_states,
                            layer_local_condition_valid_mask,
                            _decoder_layer=decoder_layer,
                            _transport_module=transport_module,
                            _condition_refresh_module=(
                                condition_refresh_module
                            ),
                            _level_index=level_index,
                            _attention_mask=attention_mask,
                        ):
                            return _decoder_layer.forward_with_bipartite_transport(
                                layer_hidden_states,
                                layer_latent_states,
                                _transport_module,
                                level_index=_level_index,
                                position_embeddings=(
                                    context.positional_embeddings
                                ),
                                query_position_embeddings=(
                                    context.query_position_embeddings
                                ),
                                encoder_hidden_states=(
                                    context.encoder_hidden_states
                                ),
                                encoder_attention_mask=_attention_mask,
                                output_attentions=output_attentions,
                                query_self_attention_mask=(
                                    query_self_attention_mask
                                ),
                                cached_vlm_latent_states=(
                                    layer_cached_vlm_latent_states
                                ),
                                local_condition_states=(
                                    layer_local_condition_states
                                ),
                                local_condition_valid_mask=(
                                    layer_local_condition_valid_mask
                                ),
                                condition_refresh_module=(
                                    _condition_refresh_module
                                ),
                            )

                        layer_outputs = self._gradient_checkpointing_func(
                            transport_refresh_forward,
                            hidden_states,
                            transport_latent_states,
                            transport_cached_vlm_latent_states,
                            transport_local_condition_states,
                            transport_local_condition_valid_mask,
                        )
                else:
                    layer_outputs = (
                        decoder_layer.forward_with_bipartite_transport(
                            hidden_states,
                            transport_latent_states,
                            transport_module,
                            level_index=level_index,
                            position_embeddings=(
                                context.positional_embeddings
                            ),
                            query_position_embeddings=(
                                context.query_position_embeddings
                            ),
                            encoder_hidden_states=(
                                context.encoder_hidden_states
                            ),
                            encoder_attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            query_self_attention_mask=(
                                query_self_attention_mask
                            ),
                            cached_vlm_latent_states=(
                                transport_cached_vlm_latent_states
                                if condition_refresh_module is not None
                                else None
                            ),
                            local_condition_states=(
                                transport_local_condition_states
                                if condition_refresh_module is not None
                                else None
                            ),
                            local_condition_valid_mask=(
                                transport_local_condition_valid_mask
                                if condition_refresh_module is not None
                                else None
                            ),
                            condition_refresh_module=condition_refresh_module,
                        )
                    )
                raw_layer_hidden_states = layer_outputs[0]
                transport_latent_states = layer_outputs[1]
                attention_output_index = 3 if delayed_transport else 2
            elif self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    level_index,
                    None,
                    context.positional_embeddings,
                    context.query_position_embeddings,
                    context.encoder_hidden_states,
                    attention_mask,
                    output_attentions,
                    query_self_attention_mask,
                )
                raw_layer_hidden_states = layer_outputs[0]
                attention_output_index = 1
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=context.positional_embeddings,
                    query_position_embeddings=(
                        context.query_position_embeddings
                    ),
                    encoder_hidden_states=context.encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_self_attention_mask=query_self_attention_mask,
                )
                raw_layer_hidden_states = layer_outputs[0]
                attention_output_index = 1
            if pre_prediction_refiner is not None:
                refined_batch_first = pre_prediction_refiner(
                    stage_index,
                    raw_layer_hidden_states.transpose(0, 1),
                    context,
                )
                expected_batch_first = (
                    context.batch_size,
                    raw_layer_hidden_states.shape[0],
                    raw_layer_hidden_states.shape[2],
                )
                if tuple(refined_batch_first.shape) != expected_batch_first:
                    raise ValueError(
                        "pre_prediction_refiner must return [B,Q,D]; "
                        f"st{stage_index} returned "
                        f"{tuple(refined_batch_first.shape)}, expected "
                        f"{expected_batch_first}"
                    )
                raw_layer_hidden_states = refined_batch_first.transpose(0, 1)

            normalized_query_states = self.layernorm(
                raw_layer_hidden_states
            )
            predicted_mask, attention_mask = self.mask_predictor(
                normalized_query_states,
                context.pixel_embeddings,
                context.feature_size_list[
                    (idx + 1) % self.num_feature_levels
                ],
            )
            intermediate += (normalized_query_states,)
            intermediate_mask_predictions += (predicted_mask,)

            if delayed_transport:
                # M_s has now been predicted from these exact normalized Q_s.
                # Reuse the SAME R from before Query self-attention; update Z
                # from Q_s only here, after FFN and formal mask prediction.
                stage_relation = layer_outputs[2]
                latent_update = transport_module.update_latents(
                    normalized_query_states.transpose(0, 1),
                    transport_latent_states,
                    stage_relation,
                )
                if bool(getattr(
                    transport_module, "returns_group_supervision_weights", False,
                )):
                    if not isinstance(latent_update, tuple) or len(latent_update) != 2:
                        raise RuntimeError(
                            "independent transport must return latent states and "
                            "latent<-Query attention weights"
                        )
                    transport_latent_states, stage_relation = latent_update
                else:
                    transport_latent_states = latent_update
                if transport_relations is not None:
                    transport_relations += (stage_relation,)

            refined_batch_first = None
            if stage_refiner is not None:
                refined_batch_first = stage_refiner(
                    stage_index,
                    raw_layer_hidden_states.transpose(0, 1),
                    normalized_query_states.transpose(0, 1),
                    predicted_mask,
                    context,
                )
                if refined_batch_first is not None:
                    expected_batch_first = (
                        context.batch_size,
                        raw_layer_hidden_states.shape[0],
                        raw_layer_hidden_states.shape[2],
                    )
                    if tuple(refined_batch_first.shape) != expected_batch_first:
                        raise ValueError(
                            "stage_refiner must return [B,Q,D] or None; "
                            f"st{stage_index} returned "
                            f"{tuple(refined_batch_first.shape)}, expected "
                            f"{expected_batch_first}"
                        )

            if idx < len(self.layers) - 1:
                hidden_states = (
                    raw_layer_hidden_states
                    if refined_batch_first is None
                    else refined_batch_first.transpose(0, 1)
                )
            else:
                # st9 has no following decoder layer.  Its formal prediction
                # and legacy last_hidden_state both use the unrefined stream.
                hidden_states = raw_layer_hidden_states

            if output_attentions:
                attentions += (layer_outputs[attention_output_index],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        final_hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [
                final_hidden_states,
                all_hidden_states,
                attentions,
                intermediate,
                intermediate_mask_predictions,
                stage_condition_states,
            ]
            if delayed_transport or export_transport_latent_states:
                outputs.extend((transport_latent_states, transport_relations))
            return tuple(v for v in outputs if v is not None)

        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=final_hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
            group_stage_outputs=None,
            stage_condition_states=stage_condition_states,
            transport_final_latent_states=(
                transport_latent_states
                if delayed_transport or export_transport_latent_states
                else None
            ),
            transport_relation_logits=transport_relations,
        )

    def _forward_with_topology_groups(
        self,
        inputs_embeds: torch.Tensor,
        multi_stage_positional_embeddings: List[torch.Tensor],
        pixel_embeddings: torch.Tensor,
        encoder_hidden_states: List[torch.Tensor],
        query_position_embeddings: torch.Tensor,
        feature_size_list: List,
        siglip_spatial_features: torch.Tensor,
        siglip_valid_mask: Optional[torch.Tensor],
        spatial_metadata: Dict[str, torch.Tensor],
        topology_group_core: TopologyGroupDecoderCore,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        """Run all ten prediction stages with topology feedback.

        Stage predictions stored in the output are always computed before that
        stage's feedback.  Legacy V1 recomputes the cross-attention mask after
        feedback.  V2 deliberately reuses the current formal stage mask as the
        next layer's prior, so its mask predictor runs exactly once for each of
        st0--st9.  Stage 9 is prediction-only in both versions.
        """

        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required for topology group decoding")
        expected_num_stages = len(self.layers) + 1
        if topology_group_core.num_stages != expected_num_stages:
            raise AssertionError(
                "topology_group_core stage count must match the Mask2Former "
                f"prediction stages: {topology_group_core.num_stages} != {expected_num_stages}"
            )
        topology_group_version = int(
            getattr(topology_group_core, "topology_group_version", 1)
        )
        if topology_group_version not in (1, 2):
            raise ValueError(
                "unsupported topology group version "
                f"{topology_group_version}"
            )
        configured_topology_version = int(
            getattr(self.config, "topology_group_version", 1)
        )
        if configured_topology_version != topology_group_version:
            raise ValueError(
                "decoder config and topology core versions disagree: "
                f"{configured_topology_version} != {topology_group_version}"
            )

        hidden_states = inputs_embeds
        intermediate = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()
        group_stage_outputs = ()

        # st0: initial learned queries, before the first transformer layer.
        intermediate_hidden_states = self.layernorm(hidden_states)
        intermediate += (intermediate_hidden_states,)
        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states,
            pixel_embeddings,
            feature_size_list[0],
        )
        intermediate_mask_predictions += (predicted_mask,)
        stage_output = topology_group_core.build_stage(
            query_states=intermediate_hidden_states.transpose(0, 1),
            mask_logits=predicted_mask,
            sam_mask_features=pixel_embeddings,
            siglip_spatial_features=siglip_spatial_features,
            spatial_metadata=spatial_metadata,
            stage_index=0,
            siglip_valid_mask=siglip_valid_mask,
        )
        group_stage_outputs += (stage_output,)

        feedback_query_states = topology_group_core.apply_feedback(
            raw_query_states=hidden_states.transpose(0, 1),
            stage_output=stage_output,
        )
        hidden_states = feedback_query_states.transpose(0, 1)
        if topology_group_version == 1:
            feedback_hidden_states = self.layernorm(hidden_states)
            _, attention_mask = self.mask_predictor(
                feedback_hidden_states,
                pixel_embeddings,
                feature_size_list[0],
            )

        # st1--st9: all decoder layers are executed in topology mode.  In
        # particular, stochastic layerdrop must not remove a formal stage.
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            level_index = idx % self.num_feature_levels
            where = (attention_mask.sum(-1) != attention_mask.shape[-1]).to(attention_mask.dtype)
            attention_mask = attention_mask * where.unsqueeze(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    level_index,
                    None,
                    multi_stage_positional_embeddings,
                    query_position_embeddings,
                    encoder_hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=multi_stage_positional_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            raw_layer_hidden_states = layer_outputs[0]
            intermediate_hidden_states = self.layernorm(raw_layer_hidden_states)
            attention_mask_target_size = feature_size_list[(idx + 1) % self.num_feature_levels]
            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                attention_mask_target_size,
            )
            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)

            stage_index = idx + 1
            stage_output = topology_group_core.build_stage(
                query_states=intermediate_hidden_states.transpose(0, 1),
                mask_logits=predicted_mask,
                sam_mask_features=pixel_embeddings,
                siglip_spatial_features=siglip_spatial_features,
                spatial_metadata=spatial_metadata,
                stage_index=stage_index,
                siglip_valid_mask=siglip_valid_mask,
                previous_stage_output=group_stage_outputs[-1],
            )
            group_stage_outputs += (stage_output,)

            if stage_index < expected_num_stages - 1:
                feedback_query_states = topology_group_core.apply_feedback(
                    raw_query_states=raw_layer_hidden_states.transpose(0, 1),
                    stage_output=stage_output,
                )
                hidden_states = feedback_query_states.transpose(0, 1)
                if topology_group_version == 1:
                    feedback_hidden_states = self.layernorm(hidden_states)
                    _, attention_mask = self.mask_predictor(
                        feedback_hidden_states,
                        pixel_embeddings,
                        attention_mask_target_size,
                    )
            else:
                # st9 is a formal prediction stage with no feedback.
                hidden_states = raw_layer_hidden_states

            if output_attentions:
                attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [
                hidden_states,
                all_hidden_states,
                attentions,
                intermediate,
                intermediate_mask_predictions,
                group_stage_outputs,
            ]
            return tuple(v for v in outputs if v is not None)

        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
            group_stage_outputs=group_stage_outputs,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        query_position_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        siglip_spatial_features: Optional[torch.Tensor] = None,
        siglip_valid_mask: Optional[torch.Tensor] = None,
        spatial_metadata: Optional[Dict[str, torch.Tensor]] = None,
        topology_group_core: Optional[TopologyGroupDecoderCore] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                The query embeddings that are passed into the decoder.
            multi_stage_positional_embeddings (`torch.FloatTensor` of shape `(height*width, batch_size, num_channels)`):
                Position embeddings that are added to the keys in each cross(masked)-attention layer.
            pixel_embeddings (`torch.FloatTensor`):
                Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel
                Decoder.
            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross(masked)-attention of the decoder.
            feature_size_list (`List[torch.Size]`):
                This is a list containing shapes (height & width) of multi-scale features from the Pixel Decoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if topology_group_core is not None:
            return self._forward_with_topology_groups(
                inputs_embeds=inputs_embeds,
                multi_stage_positional_embeddings=multi_stage_positional_embeddings,
                pixel_embeddings=pixel_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                query_position_embeddings=query_position_embeddings,
                feature_size_list=feature_size_list,
                siglip_spatial_features=siglip_spatial_features,
                siglip_valid_mask=siglip_valid_mask,
                spatial_metadata=spatial_metadata,
                topology_group_core=topology_group_core,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # intermediate hidden states with layernorm applied - required for predicting class logits
        intermediate = ()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        # intermediate mask predictions from transformer decoder layers
        intermediate_mask_predictions = ()

        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, pixel_embeddings, feature_size_list[0]
        )
        intermediate_mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = torch.rand([])

            if self.training and (dropout_probability < self.layerdrop):
                continue

            level_index = idx % self.num_feature_levels
            where = (attention_mask.sum(-1) != attention_mask.shape[-1]).to(attention_mask.dtype)
            # Multiply the attention mask instead of indexing to avoid issue in torch.export.
            attention_mask = attention_mask * where.unsqueeze(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    level_index,
                    None,
                    multi_stage_positional_embeddings,
                    query_position_embeddings,
                    encoder_hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    level_index=level_index,
                    position_embeddings=multi_stage_positional_embeddings,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            intermediate_hidden_states = self.layernorm(layer_outputs[0])

            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )

            intermediate_mask_predictions += (predicted_mask,)

            # add intermediate hidden states with layer norm applied which will be used for predicting class logits
            intermediate += (intermediate_hidden_states,)

            hidden_states = layer_outputs[0]

            if output_attentions:
                attentions += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [
                hidden_states,
                all_hidden_states,
                attentions,
                intermediate,
                intermediate_mask_predictions,
                None,
            ]
            return tuple(v for v in outputs if v is not None)

        return Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            intermediate_hidden_states=intermediate,
            masks_queries_logits=intermediate_mask_predictions,
            group_stage_outputs=None,
        )


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock with MaskFormer->Mask2Former
class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        """
        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also
        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained
        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next
        decoder layer as input.

        Args:
            hidden_size (`int`):
                The feature dimension of the Mask2FormerMaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the Mask2FormerMaskedAttentionDecoder
            mask_feature_size (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):
        if _finite_diagnostics_enabled():
            _diagnostic_require_finite(
                "mask_predictor.input_query_states",
                outputs,
            )
            _diagnostic_require_finite(
                "mask_predictor.input_pixel_embeddings",
                pixel_embeddings,
            )

        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        if _finite_diagnostics_enabled():
            _diagnostic_require_finite(
                "mask_predictor.mask_embeddings_after_mlp",
                mask_embeddings,
            )

        is_tracing = torch.jit.is_tracing() or isinstance(outputs, torch.fx.Proxy) or is_torchdynamo_compiling()
        # Sum up over the channels
        if is_tracing and not is_torch_greater_or_equal_than_2_1:
            # Equivalent to einsum('bqc, bchw -> bqhw') but jit friendly
            batch_size, num_queries, num_channels = mask_embeddings.shape
            _, _, height, width = pixel_embeddings.shape
            outputs_mask = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                outputs_mask += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]

        else:
            outputs_mask = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings)

        if _finite_diagnostics_enabled():
            _diagnostic_require_finite(
                "mask_predictor.outputs_mask_after_einsum",
                outputs_mask,
            )

        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        in_features = config.feature_size
        self.num_feature_levels = config.num_feature_levels
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def prepare_decoder_context(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        *,
        siglip_spatial_features: Optional[Tensor] = None,
        siglip_valid_mask: Optional[Tensor] = None,
        spatial_metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Mask2FormerDecoderContext:
        """Prepare a B-level, SEG-free context for split decoder execution.

        Unlike :meth:`forward`, this method deliberately performs no
        ``cond_lens`` repeat and adds no SEG embedding.  The resulting learned
        Query tensor can therefore run st0--st3 once per source image.
        """

        if len(multi_scale_features) != self.num_feature_levels:
            raise ValueError(
                "multi_scale_features must contain exactly "
                f"{self.num_feature_levels} levels"
            )
        if mask_features.ndim != 4:
            raise ValueError("mask_features must be [B,C,H,W]")
        batch_size = int(mask_features.shape[0])
        if not all(
            features.ndim == 4 and features.shape[0] == batch_size
            for features in multi_scale_features
        ):
            raise ValueError(
                "all multi-scale features must be [B,C,H,W] with the mask "
                "feature batch"
            )

        feature_size_list = []
        positional_embeddings = []
        encoder_hidden_states = []
        for level in range(self.num_feature_levels):
            level_features = multi_scale_features[level]
            feature_size_list.append(
                (int(level_features.shape[-2]), int(level_features.shape[-1]))
            )
            positions = self.position_embedder(
                level_features,
                None,
            ).flatten(2)
            projected = (
                self.input_projections[level](level_features).flatten(2)
                + self.level_embed.weight[level][None, :, None]
            )
            positional_embeddings.append(positions.permute(2, 0, 1))
            encoder_hidden_states.append(projected.permute(2, 0, 1))

        if siglip_spatial_features is not None and (
            siglip_spatial_features.ndim < 1
            or siglip_spatial_features.shape[0] != batch_size
        ):
            raise ValueError(
                "siglip_spatial_features must have the source image batch"
            )
        if siglip_valid_mask is not None and (
            siglip_valid_mask.ndim < 1
            or siglip_valid_mask.shape[0] != batch_size
        ):
            raise ValueError("siglip_valid_mask must have the source image batch")
        if spatial_metadata is not None:
            for key, value in spatial_metadata.items():
                if not isinstance(value, Tensor):
                    raise TypeError(
                        f"spatial_metadata[{key!r}] must be a tensor"
                    )
                if value.ndim < 1 or value.shape[0] != batch_size:
                    raise ValueError(
                        f"spatial_metadata[{key!r}] must have the source "
                        "image batch"
                    )

        query_position_embeddings = self.queries_embedder.weight.unsqueeze(
            1
        ).repeat(1, batch_size, 1)
        initial_query_states = self.queries_features.weight.unsqueeze(1).repeat(
            1,
            batch_size,
            1,
        )
        return Mask2FormerDecoderContext(
            initial_query_states=initial_query_states,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=tuple(encoder_hidden_states),
            positional_embeddings=tuple(positional_embeddings),
            pixel_embeddings=mask_features,
            feature_size_list=tuple(feature_size_list),
            siglip_spatial_features=siglip_spatial_features,
            siglip_valid_mask=siglip_valid_mask,
            spatial_metadata=spatial_metadata,
        )

    @staticmethod
    def expand_decoder_context(
        context: Mask2FormerDecoderContext,
        row_source_indices: Tensor,
    ) -> Mask2FormerDecoderContext:
        """Gather a source-image context into effective SEG-row order."""

        if not isinstance(context, Mask2FormerDecoderContext):
            raise TypeError("context must be a Mask2FormerDecoderContext")
        if row_source_indices.ndim != 1:
            raise ValueError("row_source_indices must be one-dimensional")
        indices = row_source_indices.to(
            device=context.pixel_embeddings.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            raise ValueError("row_source_indices must not be empty")
        if bool((indices < 0).any()) or bool(
            (indices >= context.batch_size).any()
        ):
            raise IndexError(
                "row_source_indices contains an index outside the context batch"
            )

        def gather_query_first(tensor: Tensor) -> Tensor:
            return tensor.index_select(
                1,
                indices.to(device=tensor.device),
            )

        def gather_batch_first(tensor: Tensor) -> Tensor:
            return tensor.index_select(
                0,
                indices.to(device=tensor.device),
            )

        return Mask2FormerDecoderContext(
            initial_query_states=gather_query_first(
                context.initial_query_states
            ),
            query_position_embeddings=gather_query_first(
                context.query_position_embeddings
            ),
            encoder_hidden_states=tuple(
                gather_query_first(x) for x in context.encoder_hidden_states
            ),
            positional_embeddings=tuple(
                gather_query_first(x) for x in context.positional_embeddings
            ),
            pixel_embeddings=gather_batch_first(context.pixel_embeddings),
            feature_size_list=context.feature_size_list,
            siglip_spatial_features=(
                None
                if context.siglip_spatial_features is None
                else gather_batch_first(context.siglip_spatial_features)
            ),
            siglip_valid_mask=(
                None
                if context.siglip_valid_mask is None
                else gather_batch_first(context.siglip_valid_mask)
            ),
            spatial_metadata=(
                None
                if context.spatial_metadata is None
                else {
                    key: gather_batch_first(value)
                    for key, value in context.spatial_metadata.items()
                }
            ),
        )

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        seg_embeddings: Optional[Tensor] = None,
        cond_lens: Optional[List] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        siglip_spatial_features: Optional[Tensor] = None,
        siglip_valid_mask: Optional[Tensor] = None,
        spatial_metadata: Optional[Dict[str, Tensor]] = None,
        topology_group_core: Optional[TopologyGroupDecoderCore] = None,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        if cond_lens is not None:
            assert mask_features.shape[0] == len(cond_lens), (
                "mask_features batch must match cond_lens before repeat: "
                f"{mask_features.shape[0]} != {len(cond_lens)}"
            )
            repeat_multi_scale_features = ()
            for i in range(self.num_feature_levels):
                repeat_single_scale_features = torch.cat(
                    [
                        single_scale_features.unsqueeze(0).repeat(cond_len, 1, 1, 1)
                        for single_scale_features, cond_len in zip(multi_scale_features[i], cond_lens)
                    ],
                    dim=0,
                )
                repeat_multi_scale_features += (repeat_single_scale_features,)

            mask_features = torch.cat(
                [
                    mask_feature.unsqueeze(0).repeat(cond_len, 1, 1, 1)
                    for mask_feature, cond_len in zip(mask_features, cond_lens)
                ],
                dim=0,
            )

            if siglip_spatial_features is not None:
                assert siglip_spatial_features.shape[0] == len(cond_lens), (
                    "SigLIP spatial feature batch must match cond_lens before repeat: "
                    f"{siglip_spatial_features.shape[0]} != {len(cond_lens)}"
                )
                siglip_spatial_features = torch.cat(
                    [
                        single_siglip_features.unsqueeze(0).repeat(cond_len, 1, 1, 1)
                        for single_siglip_features, cond_len in zip(siglip_spatial_features, cond_lens)
                    ],
                    dim=0,
                )

            if siglip_valid_mask is not None:
                assert siglip_valid_mask.shape[0] == len(cond_lens), (
                    "SigLIP valid-mask batch must match cond_lens before repeat: "
                    f"{siglip_valid_mask.shape[0]} != {len(cond_lens)}"
                )
                siglip_valid_mask = torch.cat(
                    [
                        single_valid_mask.unsqueeze(0).repeat(cond_len, 1, 1, 1)
                        for single_valid_mask, cond_len in zip(siglip_valid_mask, cond_lens)
                    ],
                    dim=0,
                )

            if spatial_metadata is not None:
                repeated_spatial_metadata = {}
                for key, value in spatial_metadata.items():
                    assert isinstance(value, Tensor), f"spatial_metadata[{key!r}] must be a tensor"
                    assert value.shape[0] == len(cond_lens), (
                        f"spatial_metadata[{key!r}] batch must match cond_lens "
                        f"before repeat: {value.shape[0]} != {len(cond_lens)}"
                    )
                    repeated_spatial_metadata[key] = torch.cat(
                        [
                            single_value.unsqueeze(0).repeat((cond_len,) + (1,) * single_value.ndim)
                            for single_value, cond_len in zip(value, cond_lens)
                        ],
                        dim=0,
                    )
                spatial_metadata = repeated_spatial_metadata

            multi_scale_features = repeat_multi_scale_features

        effective_batch_size = mask_features.shape[0]
        assert all(features.shape[0] == effective_batch_size for features in multi_scale_features), (
            "all multi-scale features must have the same batch as mask_features"
        )
        if siglip_spatial_features is not None:
            assert siglip_spatial_features.shape[0] == effective_batch_size, (
                "SigLIP spatial feature batch must match repeated mask_features: "
                f"{siglip_spatial_features.shape[0]} != {effective_batch_size}"
            )
        if siglip_valid_mask is not None:
            assert siglip_valid_mask.shape[0] == effective_batch_size, (
                "SigLIP valid-mask batch must match repeated mask_features: "
                f"{siglip_valid_mask.shape[0]} != {effective_batch_size}"
            )
        if spatial_metadata is not None:
            assert all(value.shape[0] == effective_batch_size for value in spatial_metadata.values()), (
                "all spatial metadata fields must match the repeated mask_features batch"
            )
        if topology_group_core is not None:
            assert siglip_spatial_features is not None, (
                "topology group decoding requires siglip_spatial_features"
            )
            assert spatial_metadata is not None, "topology group decoding requires spatial_metadata"

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        if _finite_diagnostics_enabled():
            _diagnostic_require_finite(
                "transformer.positional_query_parameter",
                self.queries_embedder.weight,
            )
            _diagnostic_require_finite(
                "transformer.learned_query_parameter_before_repeat",
                self.queries_features.weight,
            )

        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        if _finite_diagnostics_enabled():
            _diagnostic_require_finite(
                "transformer.learned_query_features",
                query_features,
            )
            _diagnostic_require_finite(
                "transformer.pixel_decoder_mask_features",
                mask_features,
            )

        if seg_embeddings is not None:
            assert seg_embeddings.shape[1] == 1
            if _finite_diagnostics_enabled():
                _diagnostic_require_finite(
                    "transformer.seg_embeddings_before_query_fusion",
                    seg_embeddings,
                )
            query_features = query_features + seg_embeddings.transpose(0, 1)
            if _finite_diagnostics_enabled():
                _diagnostic_require_finite(
                    "transformer.query_features_after_seg_fusion",
                    query_features,
                )

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            siglip_spatial_features=siglip_spatial_features,
            siglip_valid_mask=siglip_valid_mask,
            spatial_metadata=spatial_metadata,
            topology_group_core=topology_group_core,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_output


MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASK2FORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.preprocess`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.
"""


class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)


@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(config)

        self.post_init()

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerModelOutput:
        r"""
        Returns:
            `Mask2FormerModelOutput`

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoImageProcessor, Mask2FormerModel

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load image preprocessor and Mask2FormerModel trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # model outputs last hidden states of shape (batch_size, num_queries, hidden_size)
        >>> print(outputs.transformer_decoder_last_hidden_state.shape)
        torch.Size([1, 100, 256])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output


@add_start_docstrings(
    "The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)

        self.weight_dict: Dict[str, float] = {
            "loss_cls": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []  # type: ignore

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        r"""
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `Mask2FormerUniversalSegmentationOutput`

        Examples:

        Instance segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-coco-instance"
        ... )

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get instance segmentation map
        >>> pred_instance_map = image_processor.post_process_instance_segmentation(
        ...     outputs, target_sizes=[(image.height, image.width)]
        ... )[0]
        >>> print(pred_instance_map.shape)
        torch.Size([480, 640])
        ```

        Semantic segmentation example:
        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on ADE20k semantic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get semantic segmentation map
        >>> pred_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[(image.height, image.width)]
        ... )[0]
        >>> print(pred_semantic_map.shape)
        torch.Size([512, 683])
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on CityScapes panoptic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
        >>> model = Mask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-cityscapes-panoptic"
        ... )

        >>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get panoptic segmentation map
        >>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[(image.height, image.width)]
        ... )[0]["segmentation"]
        >>> print(pred_panoptic_map.shape)
        torch.Size([338, 676])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output


__all__ = ["Mask2FormerForUniversalSegmentation", "Mask2FormerModel", "Mask2FormerPreTrainedModel"]
