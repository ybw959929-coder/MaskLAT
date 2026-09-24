"""Hierarchical Hungarian matching for topology-group predictions.

Matching happens at two levels:

1. For every topology group/target pair, choose the member with the minimum
   weighted mask BCE + Dice cost.
2. Run one-to-one Hungarian assignment between valid groups and targets,
   adding the root row's group-classification cost to that minimum shape
   cost.

The selected member is always one of the original decoder queries.  No union,
average, or synthesized group mask is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # Package import.
    from .spatial_validity import (
        masked_normalized_grid_sample,
        masked_normalized_resize,
        scale_unit_coordinates_to_valid_boxes,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )
except (ImportError, ValueError):  # Standalone import used by pure unit tests.
    from spatial_validity import (  # type: ignore
        masked_normalized_grid_sample,
        masked_normalized_resize,
        scale_unit_coordinates_to_valid_boxes,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """Mask2Former pairwise Dice cost for ``[P,K]`` mask samples."""

    inputs = inputs.sigmoid().flatten(1)
    labels = labels.flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.transpose(0, 1))
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """Mask2Former pairwise sigmoid-BCE cost for ``[P,K]`` mask samples."""

    inputs = inputs.flatten(1)
    labels = labels.flatten(1)
    num_samples = inputs.shape[1]
    if num_samples <= 0:
        raise ValueError("pairwise mask costs require at least one sampled point")
    loss_pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    loss_neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    positive_cost = torch.matmul(loss_pos / num_samples, labels.transpose(0, 1))
    negative_cost = torch.matmul(
        loss_neg / num_samples, (1 - labels).transpose(0, 1)
    )
    return positive_cost + negative_cost


def _target_fields(target: Any) -> Tuple[Tensor, Tensor]:
    if hasattr(target, "masks") and hasattr(target, "condition_ids"):
        return target.masks, target.condition_ids
    if hasattr(target, "masks") and hasattr(target, "labels"):
        return target.masks, target.labels
    if isinstance(target, dict):
        masks = target.get("masks", target.get("mask_labels"))
        labels = target.get(
            "condition_ids",
            target.get(
                "target_condition_ids",
                target.get("labels", target.get("class_labels")),
            ),
        )
        if masks is not None and labels is not None:
            return masks, labels
    raise TypeError(
        "each normalized target must expose masks and condition_ids"
    )


@dataclass(frozen=True)
class SampleGroupMatch:
    """Hierarchical assignment diagnostics for one batch element."""

    batch_index: int
    group_indices: Tensor
    target_indices: Tensor
    member_indices: Tensor
    assignment_costs: Tensor
    class_costs: Tensor
    mask_costs: Tensor
    dice_costs: Tensor
    unmatched_group_indices: Tensor
    unmatched_target_indices: Tensor
    num_valid_groups: int
    num_targets: int

    def __len__(self) -> int:
        return int(self.group_indices.numel())

    @property
    def matched_indices(self) -> Tuple[Tensor, Tensor]:
        return self.group_indices, self.target_indices


@dataclass(frozen=True)
class HierarchicalMatchingDiagnostics(Sequence[SampleGroupMatch]):
    """Sequence-compatible matching result for one prediction stage."""

    samples: Tuple[SampleGroupMatch, ...]
    stage_index: Optional[int] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def __iter__(self) -> Iterator[SampleGroupMatch]:
        return iter(self.samples)

    @property
    def num_matches(self) -> int:
        return sum(len(sample) for sample in self.samples)

    @property
    def num_valid_groups(self) -> int:
        return sum(sample.num_valid_groups for sample in self.samples)

    @property
    def num_targets(self) -> int:
        return sum(sample.num_targets for sample in self.samples)


class HierarchicalGroupMatcher(nn.Module):
    """Assign topology groups to targets using minimum member-mask costs.

    Args mirror Mask2Former's matcher.  ``cost_cls_type`` accepts
    ``"ce_cost"`` and ``"focal_cost"``.  Full-resolution matching is useful
    for deterministic tests; point matching reproduces the original training
    path's common random points.
    """

    def __init__(
        self,
        *,
        cost_class: float = 2.0,
        cost_mask: float = 5.0,
        cost_dice: float = 5.0,
        num_points: int = 12544,
        use_sample_point: bool = True,
        group_shape_temperature: float = 0.1,
        cost_cls_type: str = "ce_cost",
        alpha: float = 0.25,
        gamma: float = 2.0,
        full_resolution_query_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if float(cost_class) == float(cost_mask) == float(cost_dice) == 0.0:
            raise ValueError("at least one matching cost must be non-zero")
        if any(float(value) < 0 for value in (cost_class, cost_mask, cost_dice)):
            raise ValueError("matching cost weights must be non-negative")
        if int(num_points) <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if float(group_shape_temperature) <= 0:
            raise ValueError(
                "group_shape_temperature must be positive, got "
                f"{group_shape_temperature}"
            )
        if cost_cls_type not in {"ce_cost", "focal_cost"}:
            raise ValueError(
                "cost_cls_type must be 'ce_cost' or 'focal_cost', got "
                f"{cost_cls_type!r}"
            )
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if int(full_resolution_query_chunk_size) <= 0:
            raise ValueError(
                "full_resolution_query_chunk_size must be positive, got "
                f"{full_resolution_query_chunk_size}"
            )
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)
        self.num_points = int(num_points)
        self.use_sample_point = bool(use_sample_point)
        # The matcher uses a hard minimum by contract.  Keeping the shared
        # temperature here makes configuration parity explicit; the criterion
        # alone applies it to differentiable soft-best supervision.
        self.group_shape_temperature = float(group_shape_temperature)
        self.cost_cls_type = cost_cls_type
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.full_resolution_query_chunk_size = int(
            full_resolution_query_chunk_size
        )

    @staticmethod
    def _validate_stage(
        stage_output: Any,
        targets: Sequence[Any],
        embed_masks: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        required = (
            "query_mask_logits",
            "query_to_group",
            "group_valid_mask",
            "sam_valid_boxes_normalized",
        )
        missing = [name for name in required if not hasattr(stage_output, name)]
        if missing:
            raise AttributeError(f"stage output is missing fields: {missing}")
        query_masks = stage_output.query_mask_logits
        class_logits = getattr(stage_output, "group_class_logits", None)
        query_to_group = stage_output.query_to_group
        group_valid_mask = stage_output.group_valid_mask

        if query_masks.ndim != 4:
            raise ValueError(
                f"query_mask_logits must be [B,Q,H,W], got {tuple(query_masks.shape)}"
            )
        batch_size, num_queries = query_masks.shape[:2]
        sam_valid_boxes = validate_normalized_valid_boxes(
            stage_output.sam_valid_boxes_normalized,
            batch_size=batch_size,
            device=query_masks.device,
        )
        if len(targets) != batch_size:
            raise ValueError(
                f"target batch size {len(targets)} != prediction batch {batch_size}"
            )
        if query_to_group.shape != (batch_size, num_queries):
            raise ValueError(
                "query_to_group must match [B,Q], got "
                f"{tuple(query_to_group.shape)}"
            )
        if group_valid_mask.shape != (batch_size, num_queries):
            raise ValueError(
                "group_valid_mask must match [B,Q], got "
                f"{tuple(group_valid_mask.shape)}"
            )
        if query_to_group.device != query_masks.device or (
            group_valid_mask.device != query_masks.device
        ):
            raise ValueError("all stage topology tensors must share one device")
        if query_to_group.dtype != torch.long:
            raise TypeError("query_to_group must have dtype torch.long")
        if min(query_masks.shape) <= 0:
            raise ValueError(
                f"query_mask_logits dimensions must be positive, got "
                f"{tuple(query_masks.shape)}"
            )
        if not bool(torch.isfinite(query_masks).all()):
            raise FloatingPointError("query_mask_logits contains NaN or Inf")
        if bool((query_to_group < 0).any()) or bool(
            (query_to_group >= num_queries).any()
        ):
            raise ValueError("query_to_group contains an out-of-range query id")

        valid = group_valid_mask.to(dtype=torch.bool)
        query_ids = torch.arange(num_queries, device=query_masks.device)
        expected_valid = query_to_group.eq(query_ids.unsqueeze(0))
        if not bool(torch.equal(valid, expected_valid)):
            raise ValueError(
                "group_valid_mask must be true exactly at physical root rows"
            )
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid group")
        if not bool(valid.gather(1, query_to_group).all()):
            raise ValueError("every query must map to a valid physical root")

        if class_logits is not None:
            if class_logits.ndim != 3 or class_logits.shape[:2] != (
                batch_size,
                num_queries,
            ):
                raise ValueError(
                    "group_class_logits must be [B,Q,C], got "
                    f"{tuple(class_logits.shape)}"
                )
            if class_logits.device != query_masks.device:
                raise ValueError("group_class_logits must share prediction device")
            if not bool(torch.isfinite(class_logits).all()):
                raise FloatingPointError("group_class_logits contains NaN or Inf")
        if embed_masks is not None:
            if class_logits is None:
                raise ValueError("embed_masks requires group_class_logits")
            if embed_masks.shape != (
                batch_size,
                class_logits.shape[-1],
            ):
                raise ValueError(
                    "embed_masks must match [B,C], got "
                    f"{tuple(embed_masks.shape)}"
                )
            if not bool(embed_masks.to(dtype=torch.bool)[:, -1].all()):
                raise ValueError("final background class must be valid")
        return query_masks, class_logits, query_to_group, valid, sam_valid_boxes

    def _classification_cost(
        self,
        root_logits: Optional[Tensor],
        labels: Tensor,
        *,
        num_groups: int,
        embed_mask: Optional[Tensor],
    ) -> Tensor:
        if self.cost_class == 0:
            return torch.zeros(
                num_groups,
                labels.numel(),
                dtype=torch.float32,
                device=labels.device,
            )
        if root_logits is None:
            raise ValueError("non-zero class cost requires group_class_logits")
        num_classes = root_logits.shape[-1]
        if labels.numel() > 0 and (
            bool((labels < 0).any()) or bool((labels >= num_classes).any())
        ):
            raise ValueError("target label is outside group_class_logits columns")
        logits = root_logits.float()
        if embed_mask is not None:
            valid_classes = embed_mask.to(device=logits.device, dtype=torch.bool)
            if labels.numel() > 0 and not bool(
                valid_classes.gather(0, labels).all()
            ):
                raise ValueError("target label references an invalid class column")
            logits = logits.masked_fill(~valid_classes.unsqueeze(0), -1e9)
        if self.cost_cls_type == "ce_cost":
            probabilities = logits.softmax(dim=-1)
            return -probabilities[:, labels]

        probabilities = logits.sigmoid()
        negative = (1 - self.alpha) * probabilities.pow(self.gamma)
        negative = negative * (-(1 - probabilities + 1e-6).log())
        positive = self.alpha * (1 - probabilities).pow(self.gamma)
        positive = positive * (-(probabilities + 1e-6).log())
        return positive[:, labels] - negative[:, labels]

    def _mask_costs(
        self,
        query_masks: Tensor,
        target_masks: Tensor,
        sam_valid_box: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_queries = query_masks.shape[0]
        num_targets = target_masks.shape[0]
        if num_targets == 0:
            empty = query_masks.new_empty((num_queries, 0), dtype=torch.float32)
            return empty, empty
        if target_masks.ndim != 3:
            raise ValueError(
                f"target masks must be [N,H,W], got {tuple(target_masks.shape)}"
            )
        if target_masks.shape[-2] <= 0 or target_masks.shape[-1] <= 0:
            raise ValueError("non-empty target masks need positive H and W")

        predictions = query_masks[:, None].float()
        labels = target_masks[:, None].to(
            device=query_masks.device, dtype=torch.float32
        )
        valid_box = validate_normalized_valid_boxes(
            sam_valid_box.reshape(1, 4),
            batch_size=1,
            device=query_masks.device,
        )
        prediction_valid_single = valid_mask_from_normalized_boxes(
            valid_box,
            query_masks.shape[-2:],
        )
        target_valid_single = valid_mask_from_normalized_boxes(
            valid_box,
            target_masks.shape[-2:],
        )
        target_valid = target_valid_single.expand(
            num_targets, -1, -1, -1
        )
        if self.use_sample_point:
            prediction_valid = prediction_valid_single.expand(
                num_queries,
                -1,
                -1,
                -1,
            )
            unit_coordinates = torch.rand(
                1,
                self.num_points,
                2,
                device=query_masks.device,
                dtype=torch.float32,
            )
            coordinates = scale_unit_coordinates_to_valid_boxes(
                unit_coordinates,
                valid_box,
            )
            prediction_grid = (
                2.0
                * coordinates.expand(num_queries, -1, -1).unsqueeze(2)
                - 1.0
            )
            target_grid = (
                2.0
                * coordinates.expand(num_targets, -1, -1).unsqueeze(2)
                - 1.0
            )
            prediction_points, _ = masked_normalized_grid_sample(
                predictions,
                prediction_valid,
                prediction_grid,
            )
            target_points, _ = masked_normalized_grid_sample(
                labels,
                target_valid,
                target_grid,
            )
            prediction_points = prediction_points.squeeze(1).squeeze(-1)
            target_points = target_points.squeeze(1).squeeze(-1)
            return (
                pair_wise_sigmoid_cross_entropy_loss(
                    prediction_points, target_points
                ),
                pair_wise_dice_loss(prediction_points, target_points),
            )

        # Deterministic evaluation matching uses full-resolution masks.  Work
        # in Query chunks so Q=200 and a 1024-square target cannot
        # materialize several multi-gigabyte temporary tensors at once.
        valid_pixels = target_valid_single[0, 0]
        target_points = labels[:, 0, valid_pixels]
        mask_cost_chunks = []
        dice_cost_chunks = []
        for start in range(
            0,
            num_queries,
            self.full_resolution_query_chunk_size,
        ):
            prediction_chunk = predictions[
                start : start + self.full_resolution_query_chunk_size
            ]
            chunk_size = prediction_chunk.shape[0]
            resized, _ = masked_normalized_resize(
                prediction_chunk,
                prediction_valid_single.expand(
                    chunk_size,
                    -1,
                    -1,
                    -1,
                ),
                labels.shape[-2:],
                target_valid_mask=target_valid_single.expand(
                    chunk_size,
                    -1,
                    -1,
                    -1,
                ),
            )
            prediction_points = resized[:, 0, valid_pixels]
            mask_cost_chunks.append(
                pair_wise_sigmoid_cross_entropy_loss(
                    prediction_points,
                    target_points,
                )
            )
            dice_cost_chunks.append(
                pair_wise_dice_loss(
                    prediction_points,
                    target_points,
                )
            )
        return torch.cat(mask_cost_chunks, dim=0), torch.cat(
            dice_cost_chunks,
            dim=0,
        )

    @torch.no_grad()
    def forward(
        self,
        stage_output: Any,
        targets: Sequence[Any],
        *,
        embed_masks: Optional[Tensor] = None,
    ) -> HierarchicalMatchingDiagnostics:
        (
            query_masks,
            class_logits,
            query_to_group,
            valid,
            sam_valid_boxes,
        ) = self._validate_stage(stage_output, targets, embed_masks)
        device = query_masks.device
        samples = []

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "HierarchicalGroupMatcher requires scipy.optimize"
            ) from error

        for batch_index in range(query_masks.shape[0]):
            target_masks, target_labels = _target_fields(targets[batch_index])
            target_masks = target_masks.to(device=device, dtype=torch.float32)
            target_labels = target_labels.to(device=device, dtype=torch.long)
            if target_masks.ndim != 3 or target_labels.ndim != 1:
                raise ValueError("normalized target masks/labels have invalid rank")
            if target_masks.shape[0] != target_labels.shape[0]:
                raise ValueError("normalized target mask/label counts differ")

            roots = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            num_groups = int(roots.numel())
            num_targets = int(target_labels.numel())
            sample_embed_mask = (
                None if embed_masks is None else embed_masks[batch_index]
            )
            root_logits = (
                None
                if class_logits is None
                else class_logits[batch_index, roots]
            )
            class_cost = self._classification_cost(
                root_logits,
                target_labels,
                num_groups=num_groups,
                embed_mask=sample_embed_mask,
            )

            if num_targets == 0:
                empty_long = torch.empty(0, dtype=torch.long, device=device)
                empty_float = torch.empty(
                    0, dtype=query_masks.dtype, device=device
                )
                samples.append(
                    SampleGroupMatch(
                        batch_index=batch_index,
                        group_indices=empty_long,
                        target_indices=empty_long.clone(),
                        member_indices=empty_long.clone(),
                        assignment_costs=empty_float,
                        class_costs=empty_float.clone(),
                        mask_costs=empty_float.clone(),
                        dice_costs=empty_float.clone(),
                        unmatched_group_indices=roots.detach().clone(),
                        unmatched_target_indices=empty_long.clone(),
                        num_valid_groups=num_groups,
                        num_targets=0,
                    )
                )
                continue

            mask_cost, dice_cost = self._mask_costs(
                query_masks[batch_index],
                target_masks,
                sam_valid_boxes[batch_index],
            )
            weighted_member_cost = (
                self.cost_mask * mask_cost + self.cost_dice * dice_cost
            )
            best_member_rows = []
            best_member_mask_costs = []
            best_member_dice_costs = []
            for root in roots:
                members = torch.nonzero(
                    query_to_group[batch_index].eq(root),
                    as_tuple=False,
                ).flatten()
                group_member_cost = weighted_member_cost[members]
                best_positions = group_member_cost.argmin(dim=0)
                best_member_rows.append(members[best_positions])
                target_ids = torch.arange(num_targets, device=device)
                best_member_mask_costs.append(
                    mask_cost[members[best_positions], target_ids]
                )
                best_member_dice_costs.append(
                    dice_cost[members[best_positions], target_ids]
                )
            best_members = torch.stack(best_member_rows, dim=0)
            group_mask_cost = torch.stack(best_member_mask_costs, dim=0)
            group_dice_cost = torch.stack(best_member_dice_costs, dim=0)
            total_cost = (
                self.cost_class * class_cost
                + self.cost_mask * group_mask_cost
                + self.cost_dice * group_dice_cost
            )
            if not bool(torch.isfinite(total_cost).all()):
                raise FloatingPointError(
                    f"stage {getattr(stage_output, 'stage_index', None)} "
                    f"sample {batch_index} Hungarian cost contains NaN or Inf"
                )
            finite_cost = total_cost
            row_array, column_array = linear_sum_assignment(
                finite_cost.detach().cpu().to(torch.float32).numpy()
            )
            row_indices = torch.as_tensor(
                row_array, dtype=torch.long, device=device
            )
            target_indices = torch.as_tensor(
                column_array, dtype=torch.long, device=device
            )
            group_indices = roots[row_indices]
            member_indices = best_members[row_indices, target_indices]
            matched_costs = finite_cost[row_indices, target_indices]
            matched_class_costs = class_cost[row_indices, target_indices]
            matched_mask_costs = group_mask_cost[row_indices, target_indices]
            matched_dice_costs = group_dice_cost[row_indices, target_indices]

            unmatched_group_mask = torch.ones(
                num_groups, dtype=torch.bool, device=device
            )
            unmatched_group_mask[row_indices] = False
            unmatched_target_mask = torch.ones(
                num_targets, dtype=torch.bool, device=device
            )
            unmatched_target_mask[target_indices] = False
            samples.append(
                SampleGroupMatch(
                    batch_index=batch_index,
                    group_indices=group_indices.detach(),
                    target_indices=target_indices.detach(),
                    member_indices=member_indices.detach(),
                    assignment_costs=matched_costs.detach(),
                    class_costs=matched_class_costs.detach(),
                    mask_costs=matched_mask_costs.detach(),
                    dice_costs=matched_dice_costs.detach(),
                    unmatched_group_indices=roots[
                        unmatched_group_mask
                    ].detach(),
                    unmatched_target_indices=torch.arange(
                        num_targets, device=device
                    )[unmatched_target_mask].detach(),
                    num_valid_groups=num_groups,
                    num_targets=num_targets,
                )
            )

        return HierarchicalMatchingDiagnostics(
            samples=tuple(samples),
            stage_index=getattr(stage_output, "stage_index", None),
        )


__all__ = [
    "HierarchicalGroupMatcher",
    "HierarchicalMatchingDiagnostics",
    "SampleGroupMatch",
    "pair_wise_dice_loss",
    "pair_wise_sigmoid_cross_entropy_loss",
]
