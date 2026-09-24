"""Losses for the mask-topology grouped hierarchical decoder.

Every prediction stage is matched independently at group level.  Version 1
keeps the archived soft-best mask, absolute quality, relative quality, and
coverage objectives.  Version 2 keeps the same joint group Hungarian matcher
but applies mask supervision only to its hard-best member and trains the final
intra-group selector with a single regret-weighted pairwise ranking objective.

This module is intentionally independent of MMEngine and Transformers so its
mathematics can be tested with plain ``unittest`` and PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:  # Package import.
    from .group_matcher import (
        HierarchicalGroupMatcher,
        HierarchicalMatchingDiagnostics,
    )
    from .target_normalizer import (
        NormalizedTargetBatch,
        UnifiedTargetNormalizer,
    )
    from .spatial_validity import (
        masked_normalized_grid_sample,
        masked_normalized_resize,
        scale_unit_coordinates_to_valid_boxes,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )
except (ImportError, ValueError):  # Standalone import used by pure unit tests.
    from group_matcher import (  # type: ignore
        HierarchicalGroupMatcher,
        HierarchicalMatchingDiagnostics,
    )
    from target_normalizer import (  # type: ignore
        NormalizedTargetBatch,
        UnifiedTargetNormalizer,
    )
    from spatial_validity import (  # type: ignore
        masked_normalized_grid_sample,
        masked_normalized_resize,
        scale_unit_coordinates_to_valid_boxes,
        valid_mask_from_normalized_boxes,
        validate_normalized_valid_boxes,
    )

if TYPE_CHECKING:
    from .topology_group_decoder import TopologyGroupStageOutput


def _aligned_bce_and_dice(point_logits: Tensor, point_labels: Tensor) -> Tuple[Tensor, Tensor]:
    """Return per-row Mask2Former BCE and Dice losses in float32."""

    point_logits = point_logits.float().flatten(1)
    point_labels = point_labels.float().flatten(1)
    if point_logits.shape != point_labels.shape:
        raise ValueError(
            "point logits/labels shape mismatch: "
            f"{tuple(point_logits.shape)} != {tuple(point_labels.shape)}"
        )
    if point_logits.shape[1] <= 0:
        raise ValueError("mask loss requires at least one spatial sample")
    bce = F.binary_cross_entropy_with_logits(
        point_logits, point_labels, reduction="none"
    ).mean(dim=1)
    probabilities = point_logits.sigmoid()
    numerator = 2 * (probabilities * point_labels).sum(dim=1)
    denominator = probabilities.sum(dim=1) + point_labels.sum(dim=1)
    dice = 1 - (numerator + 1) / (denominator + 1)
    return bce, dice


def _sigmoid_focal_elements(
    logits: Tensor,
    labels: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tensor:
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    p_t = probabilities * labels + (1 - probabilities) * (1 - labels)
    loss = cross_entropy * (1 - p_t).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss
    return loss


@dataclass(frozen=True)
class SoftBestGroupDiagnostics:
    """Detached member losses and weights for one matched group/target."""

    batch_index: int
    group_index: int
    target_index: int
    member_indices: Tensor
    member_bce: Tensor
    member_dice: Tensor
    weights: Tensor


@dataclass(frozen=True)
class StageCriterionDiagnostics:
    """One stage's independent assignment and soft-best statistics."""

    stage_index: int
    matching: HierarchicalMatchingDiagnostics
    soft_best_groups: Tuple[SoftBestGroupDiagnostics, ...]
    raw_class_loss: Tensor
    raw_mask_loss: Tensor
    raw_dice_loss: Tensor

    @property
    def num_matches(self) -> int:
        return self.matching.num_matches


@dataclass(frozen=True)
class HierarchicalCriterionDiagnostics(Sequence[StageCriterionDiagnostics]):
    """Detached diagnostics returned beside the weighted loss dictionary."""

    stages: Tuple[StageCriterionDiagnostics, ...]
    final_stage_index: Optional[int]
    local_num_targets: int
    normalized_num_targets: Tensor
    unmatched_final_targets: int
    skipped: bool = False

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index):
        return self.stages[index]

    def __iter__(self) -> Iterator[StageCriterionDiagnostics]:
        return iter(self.stages)


class HierarchicalGroupCriterion(nn.Module):
    """Compute the complete group-level segmentation objective.

    The returned loss values are already multiplied by all configured
    weights.  Final-stage keys are ``loss_group_cls``, ``loss_group_mask``,
    and ``loss_group_dice``.  Earlier stages append their actual stage index
    (for example ``loss_group_mask_0``).  Final-only keys are
    Version 1 final-only keys are ``loss_quality_reg``,
    ``loss_quality_rank``, and ``loss_unmatched_coverage``.  Version 2 instead
    returns only ``loss_member_selector`` in addition to the per-stage group
    classification/mask/Dice losses.
    """

    def __init__(
        self,
        *,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        no_object_weight: float = 0.1,
        loss_cls_type: str = "ce_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        use_sample_point: bool = True,
        group_shape_temperature: float = 0.1,
        group_quality_temperature: float = 0.1,
        quality_iou_query_chunk_size: int = 8,
        quality_reg_weight: float = 1.0,
        quality_rank_weight: float = 1.0,
        unmatched_coverage_weight: float = 1.0,
        topology_group_version: int = 1,
        member_selector_weight: float = 1.0,
        member_selector_tie_epsilon: float = 1.0e-4,
        group_loss_stage_weights: Optional[Sequence[float]] = None,
        matcher: Optional[HierarchicalGroupMatcher] = None,
        target_normalizer: Optional[UnifiedTargetNormalizer] = None,
    ) -> None:
        super().__init__()
        scalar_weights = {
            "class_weight": class_weight,
            "mask_weight": mask_weight,
            "dice_weight": dice_weight,
            "quality_reg_weight": quality_reg_weight,
            "quality_rank_weight": quality_rank_weight,
            "unmatched_coverage_weight": unmatched_coverage_weight,
            "member_selector_weight": member_selector_weight,
        }
        for name, value in scalar_weights.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if float(no_object_weight) < 0:
            raise ValueError(
                f"no_object_weight must be non-negative, got {no_object_weight}"
            )
        if loss_cls_type not in {"ce_loss", "focal_loss"}:
            raise ValueError(
                "loss_cls_type must be 'ce_loss' or 'focal_loss', got "
                f"{loss_cls_type!r}"
            )
        if int(train_num_points) <= 0:
            raise ValueError(
                f"train_num_points must be positive, got {train_num_points}"
            )
        if float(oversample_ratio) < 1:
            raise ValueError(
                f"oversample_ratio must be >= 1, got {oversample_ratio}"
            )
        if not 0 <= float(importance_sample_ratio) <= 1:
            raise ValueError(
                "importance_sample_ratio must be in [0,1], got "
                f"{importance_sample_ratio}"
            )
        if float(group_shape_temperature) <= 0:
            raise ValueError("group_shape_temperature must be positive")
        if float(group_quality_temperature) <= 0:
            raise ValueError("group_quality_temperature must be positive")
        if int(quality_iou_query_chunk_size) <= 0:
            raise ValueError(
                "quality_iou_query_chunk_size must be positive, got "
                f"{quality_iou_query_chunk_size}"
            )
        if int(topology_group_version) not in (1, 2):
            raise ValueError(
                "topology_group_version must be 1 or 2, got "
                f"{topology_group_version}"
            )
        if float(member_selector_tie_epsilon) < 0:
            raise ValueError(
                "member_selector_tie_epsilon must be non-negative, got "
                f"{member_selector_tie_epsilon}"
            )
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.no_object_weight = float(no_object_weight)
        self.loss_cls_type = loss_cls_type
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.train_num_points = int(train_num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.use_sample_point = bool(use_sample_point)
        self.group_shape_temperature = float(group_shape_temperature)
        self.group_quality_temperature = float(group_quality_temperature)
        self.quality_iou_query_chunk_size = int(quality_iou_query_chunk_size)
        self.quality_reg_weight = float(quality_reg_weight)
        self.quality_rank_weight = float(quality_rank_weight)
        self.unmatched_coverage_weight = float(unmatched_coverage_weight)
        self.topology_group_version = int(topology_group_version)
        self.member_selector_weight = float(member_selector_weight)
        self.member_selector_tie_epsilon = float(
            member_selector_tie_epsilon
        )
        self.group_loss_stage_weights = (
            None
            if group_loss_stage_weights is None
            else tuple(float(value) for value in group_loss_stage_weights)
        )
        if self.group_loss_stage_weights is not None and any(
            value < 0 for value in self.group_loss_stage_weights
        ):
            raise ValueError("group_loss_stage_weights must be non-negative")

        self.target_normalizer = (
            UnifiedTargetNormalizer()
            if target_normalizer is None
            else target_normalizer
        )
        self.matcher = (
            HierarchicalGroupMatcher(
                cost_class=self.class_weight,
                cost_mask=self.mask_weight,
                cost_dice=self.dice_weight,
                num_points=self.train_num_points,
                use_sample_point=self.use_sample_point,
                group_shape_temperature=self.group_shape_temperature,
                cost_cls_type=(
                    "ce_cost"
                    if self.loss_cls_type == "ce_loss"
                    else "focal_cost"
                ),
                alpha=self.alpha,
                gamma=self.gamma,
            )
            if matcher is None
            else matcher
        )

    @staticmethod
    def _distributed_normalizer(
        local_count: int,
        *,
        device: torch.device,
    ) -> Tensor:
        """Mask2Former DDP normalization: global sum / world size, clamped."""

        count = torch.tensor(float(local_count), dtype=torch.float32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count = count / dist.get_world_size()
        return count.clamp_min(1.0)

    def _ce_weight_normalizer(self, local_weight_sum: Tensor) -> Tensor:
        """Return the CE denominator used before DDP gradient averaging.

        Version 1 deliberately preserves the historical rank-local weighted
        mean.  Version 2 instead divides every rank-local numerator by
        ``global_weight_sum / world_size``.  Because DDP subsequently averages
        gradients across ranks, this is exactly the global weighted Group
        mean even when ranks contain different numbers of dynamic Groups.
        """

        normalizer = local_weight_sum.detach().clone()
        if (
            self.topology_group_version >= 2
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        return normalizer.clamp_min(1.0e-6)

    def _uncertain_coordinates(
        self,
        logits: Tensor,
        sam_valid_box: Tensor,
    ) -> Tensor:
        num_masks = logits.shape[0]
        valid_box = validate_normalized_valid_boxes(
            sam_valid_box.reshape(1, 4),
            batch_size=1,
            device=logits.device,
        )
        expanded_boxes = valid_box.expand(num_masks, -1)
        source_valid = valid_mask_from_normalized_boxes(
            valid_box,
            logits.shape[-2:],
        ).expand(num_masks, -1, -1, -1)
        num_sampled = int(self.train_num_points * self.oversample_ratio)
        coordinates = scale_unit_coordinates_to_valid_boxes(
            torch.rand(
                num_masks,
                num_sampled,
                2,
                dtype=torch.float32,
                device=logits.device,
            ),
            expanded_boxes,
        )
        with torch.no_grad():
            point_logits, _ = masked_normalized_grid_sample(
                logits.float(),
                source_valid,
                2.0 * coordinates.unsqueeze(2) - 1.0,
            )
            point_logits = point_logits.squeeze(1).squeeze(-1)
            uncertainty = -point_logits.abs()
            num_uncertain = int(
                self.importance_sample_ratio * self.train_num_points
            )
            num_random = self.train_num_points - num_uncertain
            if num_uncertain > 0:
                indices = torch.topk(
                    uncertainty, k=num_uncertain, dim=1
                ).indices
                offsets = num_sampled * torch.arange(
                    num_masks, device=logits.device
                )
                flat_indices = (indices + offsets[:, None]).reshape(-1)
                selected = coordinates.reshape(-1, 2)[flat_indices]
                coordinates = selected.reshape(num_masks, num_uncertain, 2)
            else:
                coordinates = coordinates[:, :0]
            if num_random > 0:
                random_coordinates = scale_unit_coordinates_to_valid_boxes(
                    torch.rand(
                        num_masks,
                        num_random,
                        2,
                        dtype=torch.float32,
                        device=logits.device,
                    ),
                    expanded_boxes,
                )
                coordinates = torch.cat(
                    (
                        coordinates,
                        random_coordinates,
                    ),
                    dim=1,
                )
        return coordinates

    def _member_mask_losses(
        self,
        member_logits: Tensor,
        target_mask: Tensor,
        sam_valid_box: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Per-member BCE/Dice against one target, preserving mask gradients."""

        if member_logits.ndim != 3 or target_mask.ndim != 2:
            raise ValueError("member logits must be [K,H,W] and target [Ht,Wt]")
        num_members = member_logits.shape[0]
        if num_members <= 0 or min(target_mask.shape) <= 0:
            raise ValueError("member and target mask dimensions must be non-empty")
        predictions = member_logits[:, None].float()
        labels = target_mask[None, None].to(
            device=member_logits.device, dtype=torch.float32
        )
        labels = labels.expand(num_members, -1, -1, -1)
        valid_box = validate_normalized_valid_boxes(
            sam_valid_box.reshape(1, 4),
            batch_size=1,
            device=member_logits.device,
        )
        prediction_valid = valid_mask_from_normalized_boxes(
            valid_box,
            member_logits.shape[-2:],
        ).expand(num_members, -1, -1, -1)
        target_valid_single = valid_mask_from_normalized_boxes(
            valid_box,
            target_mask.shape,
        )
        target_valid = target_valid_single.expand(
            num_members, -1, -1, -1
        )
        if self.use_sample_point:
            coordinates = self._uncertain_coordinates(
                predictions,
                valid_box[0],
            )
            sampling_grid = 2.0 * coordinates.unsqueeze(2) - 1.0
            point_logits, _ = masked_normalized_grid_sample(
                predictions,
                prediction_valid,
                sampling_grid,
            )
            with torch.no_grad():
                point_labels, _ = masked_normalized_grid_sample(
                    labels,
                    target_valid,
                    sampling_grid,
                )
            point_logits = point_logits.squeeze(1).squeeze(-1)
            point_labels = point_labels.squeeze(1).squeeze(-1)
        else:
            resized, _ = masked_normalized_resize(
                predictions,
                prediction_valid,
                target_mask.shape,
                target_valid_mask=target_valid,
            )
            valid_pixels = target_valid_single[0, 0]
            point_logits = resized[:, 0, valid_pixels]
            point_labels = labels[:, 0, valid_pixels]
        bce, dice = _aligned_bce_and_dice(point_logits, point_labels)
        if not bool(torch.isfinite(bce).all() and torch.isfinite(dice).all()):
            raise FloatingPointError("member BCE/Dice contains NaN or Inf")
        return bce, dice

    def _classification_loss(
        self,
        stage_output: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
        embed_masks: Optional[Tensor],
        num_targets: Tensor,
    ) -> Tensor:
        logits = getattr(stage_output, "group_class_logits", None)
        if logits is None:
            if self.class_weight != 0:
                raise ValueError("group classification loss requires class logits")
            return stage_output.query_mask_logits.sum() * 0.0
        if logits.ndim != 3:
            raise ValueError("group_class_logits must be [B,Q,C]")
        num_classes = logits.shape[-1]
        if num_classes <= 0:
            raise ValueError("group_class_logits needs at least one class")

        ce_logits = []
        ce_targets = []
        focal_loss = logits.float().sum() * 0.0
        for batch_index, sample_match in enumerate(matching):
            roots = torch.nonzero(
                stage_output.group_valid_mask[batch_index].to(torch.bool),
                as_tuple=False,
            ).flatten()
            sample_logits = logits[batch_index, roots].float()
            valid_conditions = (
                torch.ones(
                    num_classes, dtype=torch.bool, device=logits.device
                )
                if embed_masks is None
                else embed_masks[batch_index].to(
                    device=logits.device, dtype=torch.bool
                )
            )
            if not bool(valid_conditions[-1]):
                raise ValueError("final background condition must be valid")
            sample_logits = sample_logits.masked_fill(
                ~valid_conditions.unsqueeze(0), -1e9
            )
            target_ids = torch.full(
                (roots.numel(),),
                num_classes - 1,
                dtype=torch.long,
                device=logits.device,
            )
            if len(sample_match) > 0:
                root_to_row = torch.full(
                    (logits.shape[1],),
                    -1,
                    dtype=torch.long,
                    device=logits.device,
                )
                root_to_row[roots] = torch.arange(
                    roots.numel(), device=logits.device
                )
                matched_rows = root_to_row[sample_match.group_indices]
                if bool((matched_rows < 0).any()):
                    raise RuntimeError("matcher returned a non-root group row")
                matched_conditions = targets[
                    batch_index
                ].condition_ids[sample_match.target_indices]
                target_ids[matched_rows] = matched_conditions

            if self.loss_cls_type == "ce_loss":
                ce_logits.append(sample_logits)
                ce_targets.append(target_ids)
            else:
                one_hot = torch.zeros_like(sample_logits)
                one_hot.scatter_(1, target_ids[:, None], 1.0)
                elements = _sigmoid_focal_elements(
                    sample_logits,
                    one_hot,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                elements = elements * valid_conditions[None].to(elements.dtype)
                focal_loss = focal_loss + elements.mean(dim=0).sum() / num_targets

        if self.loss_cls_type == "focal_loss":
            return focal_loss
        all_logits = torch.cat(ce_logits, dim=0)
        all_targets = torch.cat(ce_targets, dim=0)
        class_weights = torch.ones(
            num_classes, dtype=all_logits.dtype, device=all_logits.device
        )
        class_weights[-1] = self.no_object_weight
        target_weights = class_weights[all_targets]
        weight_sum = target_weights.sum()
        if self.topology_group_version < 2 and not bool(weight_sum > 0):
            return all_logits.sum() * 0.0
        negative_log_likelihood = -F.log_softmax(
            all_logits, dim=-1
        ).gather(1, all_targets[:, None]).squeeze(1)
        weighted_loss_sum = (
            negative_log_likelihood * target_weights
        ).sum()
        return weighted_loss_sum / self._ce_weight_normalizer(weight_sum)

    def _soft_best_mask_losses(
        self,
        stage_output: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
        num_targets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tuple[SoftBestGroupDiagnostics, ...]]:
        query_masks = stage_output.query_mask_logits
        mask_sum = query_masks.float().sum() * 0.0
        dice_sum = query_masks.float().sum() * 0.0
        records = []
        for batch_index, sample_match in enumerate(matching):
            for match_index in range(len(sample_match)):
                root = sample_match.group_indices[match_index]
                target_index = sample_match.target_indices[match_index]
                members = torch.nonzero(
                    stage_output.query_to_group[batch_index].eq(root),
                    as_tuple=False,
                ).flatten()
                member_bce, member_dice = self._member_mask_losses(
                    query_masks[batch_index, members],
                    targets[batch_index].masks[target_index],
                    stage_output.sam_valid_boxes_normalized[batch_index],
                )
                combined = (
                    self.mask_weight * member_bce
                    + self.dice_weight * member_dice
                )
                weights = torch.softmax(
                    -combined.detach() / self.group_shape_temperature,
                    dim=0,
                )
                mask_sum = mask_sum + (weights * member_bce).sum()
                dice_sum = dice_sum + (weights * member_dice).sum()
                records.append(
                    SoftBestGroupDiagnostics(
                        batch_index=batch_index,
                        group_index=int(root.item()),
                        target_index=int(target_index.item()),
                        member_indices=members.detach().clone(),
                        member_bce=member_bce.detach().clone(),
                        member_dice=member_dice.detach().clone(),
                        weights=weights.detach().clone(),
                    )
                )
        return mask_sum / num_targets, dice_sum / num_targets, tuple(records)

    def _hard_best_mask_losses(
        self,
        stage_output: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
        num_targets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tuple[SoftBestGroupDiagnostics, ...]]:
        """Supervise exactly the member selected by the joint matcher.

        ``SampleGroupMatch.member_indices`` is the hard minimum of the
        configured mask BCE + Dice matching cost inside each matched group.
        Keeping the choice detached preserves the usual Hungarian contract:
        assignment is discrete while the chosen mask losses remain
        differentiable.  A one-element diagnostic record is returned through
        the legacy ``soft_best_groups`` field so V1 diagnostic consumers keep
        a stable schema.
        """

        query_masks = stage_output.query_mask_logits
        mask_sum = query_masks.float().sum() * 0.0
        dice_sum = query_masks.float().sum() * 0.0
        records = []
        for batch_index, sample_match in enumerate(matching):
            for match_index in range(len(sample_match)):
                root = sample_match.group_indices[match_index]
                target_index = sample_match.target_indices[match_index]
                member = sample_match.member_indices[match_index].reshape(1)
                if not bool(
                    stage_output.query_to_group[batch_index, member[0]].eq(root)
                ):
                    raise RuntimeError(
                        "matcher hard-best member does not belong to its "
                        "matched group"
                    )
                member_bce, member_dice = self._member_mask_losses(
                    query_masks[batch_index, member],
                    targets[batch_index].masks[target_index],
                    stage_output.sam_valid_boxes_normalized[batch_index],
                )
                mask_sum = mask_sum + member_bce[0]
                dice_sum = dice_sum + member_dice[0]
                records.append(
                    SoftBestGroupDiagnostics(
                        batch_index=batch_index,
                        group_index=int(root.item()),
                        target_index=int(target_index.item()),
                        member_indices=member.detach().clone(),
                        member_bce=member_bce.detach().clone(),
                        member_dice=member_dice.detach().clone(),
                        weights=torch.ones_like(member_bce).detach(),
                    )
                )
        return mask_sum / num_targets, dice_sum / num_targets, tuple(records)

    @torch.no_grad()
    def _binary_ious(
        self,
        member_logits: Tensor,
        target_mask: Tensor,
        sam_valid_box: Tensor,
    ) -> Tensor:
        valid_box = validate_normalized_valid_boxes(
            sam_valid_box.reshape(1, 4),
            batch_size=1,
            device=member_logits.device,
        )
        source_valid_single = valid_mask_from_normalized_boxes(
            valid_box,
            member_logits.shape[-2:],
        )
        target_valid_single = valid_mask_from_normalized_boxes(
            valid_box,
            target_mask.shape,
        )
        valid_pixels = target_valid_single[0, 0]
        target = target_mask.to(
            device=member_logits.device, dtype=torch.float32
        )[valid_pixels].ge(0.5)
        iou_chunks = []

        # A collapsed topology group can contain almost all Q=200 members.
        # Restoring every member to the 1024-square training target at once
        # materializes several ~800 MiB float32 temporaries.  IoU is independent
        # per member, so chunking this dimension is mathematically equivalent
        # while bounding the transient full-resolution allocation.
        for start in range(
            0,
            member_logits.shape[0],
            self.quality_iou_query_chunk_size,
        ):
            member_chunk = member_logits[
                start : start + self.quality_iou_query_chunk_size
            ]
            chunk_size = member_chunk.shape[0]
            resized, _ = masked_normalized_resize(
                member_chunk[:, None].float(),
                source_valid_single.expand(chunk_size, -1, -1, -1),
                target_mask.shape,
                target_valid_mask=target_valid_single.expand(
                    chunk_size, -1, -1, -1
                ),
            )
            predictions = resized[:, 0, valid_pixels].sigmoid().ge(0.5)
            intersections = (predictions & target.unsqueeze(0)).sum(1)
            unions = (predictions | target.unsqueeze(0)).sum(1)
            iou_chunks.append(
                torch.where(
                    unions > 0,
                    intersections.float() / unions.clamp_min(1).float(),
                    torch.ones_like(unions, dtype=torch.float32),
                )
            )
            del resized, predictions, intersections, unions
        return torch.cat(iou_chunks, dim=0).detach()

    def _quality_losses(
        self,
        final_stage: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
    ) -> Tuple[Tensor, Tensor]:
        quality_logits = getattr(final_stage, "member_quality_logits", None)
        if quality_logits is None:
            if self.quality_reg_weight != 0 or self.quality_rank_weight != 0:
                raise ValueError(
                    "non-zero quality loss weights require final "
                    "member_quality_logits"
                )
            zero = final_stage.query_mask_logits.sum() * 0.0
            return zero, zero
        if quality_logits.shape != final_stage.query_to_group.shape:
            raise ValueError("member_quality_logits must be [B,Q]")
        if not bool(torch.isfinite(quality_logits).all()):
            raise FloatingPointError("member_quality_logits contains NaN or Inf")

        reg_sum = quality_logits.float().sum() * 0.0
        rank_sum = quality_logits.float().sum() * 0.0
        local_members = 0
        local_groups = 0
        for batch_index, sample_match in enumerate(matching):
            for match_index in range(len(sample_match)):
                root = sample_match.group_indices[match_index]
                target_index = sample_match.target_indices[match_index]
                members = torch.nonzero(
                    final_stage.query_to_group[batch_index].eq(root),
                    as_tuple=False,
                ).flatten()
                member_logits = quality_logits[batch_index, members].float()
                iou_targets = self._binary_ious(
                    final_stage.query_mask_logits[batch_index, members],
                    targets[batch_index].masks[target_index],
                    final_stage.sam_valid_boxes_normalized[batch_index],
                )
                reg_sum = reg_sum + F.smooth_l1_loss(
                    member_logits.sigmoid(),
                    iou_targets,
                    reduction="sum",
                )
                if members.numel() > 1:
                    target_distribution = torch.softmax(
                        iou_targets / self.group_quality_temperature,
                        dim=0,
                    )
                    rank_sum = rank_sum - (
                        target_distribution
                        * F.log_softmax(member_logits, dim=0)
                    ).sum()
                local_members += int(members.numel())
                local_groups += 1

        member_normalizer = self._distributed_normalizer(
            local_members, device=quality_logits.device
        )
        group_normalizer = self._distributed_normalizer(
            local_groups, device=quality_logits.device
        )
        return reg_sum / member_normalizer, rank_sum / group_normalizer

    def _member_selector_loss(
        self,
        final_stage: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
    ) -> Tensor:
        """Regret-weighted pairwise ranking for the V2 final selector.

        The matcher is used only to identify which group corresponds to which
        GT.  Within that matched group the ordering target is the evaluator-
        aligned binary IoU after restoring every original member mask to the
        target canvas.  This deliberately does not replace the matcher's
        sampled BCE+Dice hard member: matching, mask supervision, and final
        representative selection keep their separate responsibilities.
        """

        selector_logits = getattr(
            final_stage, "member_quality_logits", None
        )
        if selector_logits is None:
            if self.member_selector_weight != 0:
                raise ValueError(
                    "non-zero member_selector_weight requires final "
                    "member_quality_logits"
                )
            return final_stage.query_mask_logits.sum() * 0.0
        if selector_logits.shape != final_stage.query_to_group.shape:
            raise ValueError("member_quality_logits must be [B,Q]")
        if not bool(torch.isfinite(selector_logits).all()):
            raise FloatingPointError(
                "member_quality_logits contains NaN or Inf"
            )

        loss_sum = selector_logits.float().sum() * 0.0
        weight_sum = selector_logits.new_zeros((), dtype=torch.float32)
        for batch_index, sample_match in enumerate(matching):
            for match_index in range(len(sample_match)):
                root = sample_match.group_indices[match_index]
                target_index = sample_match.target_indices[match_index]
                members = torch.nonzero(
                    final_stage.query_to_group[batch_index].eq(root),
                    as_tuple=False,
                ).flatten()
                if members.numel() <= 1:
                    continue
                member_scores = selector_logits[
                    batch_index, members
                ].float()
                iou_targets = self._binary_ious(
                    final_stage.query_mask_logits[batch_index, members],
                    targets[batch_index].masks[target_index],
                    final_stage.sam_valid_boxes_normalized[batch_index],
                )
                upper = torch.triu_indices(
                    members.numel(),
                    members.numel(),
                    offset=1,
                    device=members.device,
                )
                target_delta = (
                    iou_targets[upper[0]] - iou_targets[upper[1]]
                )
                pair_weight = target_delta.abs()
                valid_pair = pair_weight.gt(
                    self.member_selector_tie_epsilon
                )
                if not bool(valid_pair.any()):
                    continue
                target_sign = target_delta[valid_pair].sign()
                score_delta = (
                    member_scores[upper[0][valid_pair]]
                    - member_scores[upper[1][valid_pair]]
                )
                selected_weight = pair_weight[valid_pair].detach()
                loss_sum = loss_sum + (
                    selected_weight
                    * F.softplus(-target_sign * score_delta)
                ).sum()
                weight_sum = weight_sum + selected_weight.sum()

        # Match the repository's DDP normalization convention.  DDP averages
        # gradients across ranks, so every rank divides by global/world_size.
        normalizer = weight_sum.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        normalizer = normalizer.clamp_min(1.0e-6)
        return loss_sum / normalizer

    def _coverage_loss(
        self,
        final_stage: Any,
        targets: NormalizedTargetBatch,
        matching: HierarchicalMatchingDiagnostics,
        num_targets: Tensor,
    ) -> Tuple[Tensor, int]:
        query_masks = final_stage.query_mask_logits
        coverage_sum = query_masks.float().sum() * 0.0
        unmatched_count = 0
        for batch_index, sample_match in enumerate(matching):
            for target_index in sample_match.unmatched_target_indices:
                member_bce, member_dice = self._member_mask_losses(
                    query_masks[batch_index],
                    targets[batch_index].masks[target_index],
                    final_stage.sam_valid_boxes_normalized[batch_index],
                )
                member_cost = (
                    self.mask_weight * member_bce
                    + self.dice_weight * member_dice
                )
                coverage_sum = coverage_sum + member_cost.min()
                unmatched_count += 1
        return coverage_sum / num_targets, unmatched_count

    @staticmethod
    def _stage_index(stage_output: Any) -> int:
        stage_index = getattr(stage_output, "stage_index", None)
        if not isinstance(stage_index, int) or stage_index < 0:
            raise ValueError(
                f"every stage output needs a non-negative integer stage_index, "
                f"got {stage_index!r}"
            )
        return stage_index

    def forward(
        self,
        group_stage_outputs: Sequence["TopologyGroupStageOutput"],
        targets: Any,
        embed_masks: Optional[Tensor] = None,
        *,
        task_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Tensor], HierarchicalCriterionDiagnostics]:
        """Return ``(weighted_loss_dict, detached_diagnostics)``.

        ``group_stage_outputs`` is a sequence of ``TopologyGroupStageOutput``.
        Every stage is independently Hungarian-matched.  ``targets`` accepts
        every format supported by :class:`UnifiedTargetNormalizer`.
        """

        stages = tuple(group_stage_outputs)
        if not stages:
            if str(task_name).strip().lower() == "imgconv" and targets is None:
                diagnostics = HierarchicalCriterionDiagnostics(
                    stages=(),
                    final_stage_index=None,
                    local_num_targets=0,
                    normalized_num_targets=torch.tensor(1.0),
                    unmatched_final_targets=0,
                    skipped=True,
                )
                return {}, diagnostics
            raise ValueError("group_stage_outputs must contain at least one stage")

        stage_indices = tuple(self._stage_index(stage) for stage in stages)
        if len(set(stage_indices)) != len(stage_indices):
            raise ValueError(f"duplicate stage indices are not allowed: {stage_indices}")
        expected_stage_indices = tuple(
            range(stage_indices[0], stage_indices[0] + len(stages))
        )
        if stage_indices != expected_stage_indices:
            raise ValueError(
                "group stage indices must be contiguous and ordered: "
                f"{stage_indices} != {expected_stage_indices}"
            )
        if self.group_loss_stage_weights is None:
            stage_weights = (1.0,) * len(stages)
        else:
            stage_weights = self.group_loss_stage_weights
            if len(stage_weights) != len(stages):
                raise ValueError(
                    "group_loss_stage_weights length must equal prediction "
                    f"stages: {len(stage_weights)} != {len(stages)}"
                )

        final_stage = stages[-1]
        final_masks = final_stage.query_mask_logits
        if final_masks.ndim != 4:
            raise ValueError("final query_mask_logits must be [B,Q,H,W]")
        batch_size = final_masks.shape[0]
        final_class_logits = getattr(final_stage, "group_class_logits", None)
        num_classes = (
            final_class_logits.shape[-1]
            if final_class_logits is not None
            else (embed_masks.shape[-1] if embed_masks is not None else None)
        )
        normalized_targets = self.target_normalizer(
            targets,
            task_name=task_name,
            batch_size=batch_size,
            device=final_masks.device,
            mask_dtype=torch.float32,
            num_classes=num_classes,
            embed_masks=embed_masks,
        )
        if normalized_targets is None:
            diagnostics = HierarchicalCriterionDiagnostics(
                stages=(),
                final_stage_index=stage_indices[-1],
                local_num_targets=0,
                normalized_num_targets=torch.tensor(
                    1.0, device=final_masks.device
                ),
                unmatched_final_targets=0,
                skipped=True,
            )
            return {}, diagnostics

        local_num_targets = normalized_targets.num_masks
        num_targets = self._distributed_normalizer(
            local_num_targets, device=final_masks.device
        )
        losses: Dict[str, Tensor] = {}
        stage_diagnostics = []
        final_matching = None
        for position, (stage, stage_weight) in enumerate(
            zip(stages, stage_weights)
        ):
            if stage.query_mask_logits.device != final_masks.device:
                raise ValueError("all prediction stages must share one device")
            matching = self.matcher(
                stage,
                normalized_targets,
                embed_masks=embed_masks,
            )
            raw_class = self._classification_loss(
                stage,
                normalized_targets,
                matching,
                embed_masks,
                num_targets,
            )
            if self.topology_group_version >= 2:
                raw_mask, raw_dice, soft_best = (
                    self._hard_best_mask_losses(
                        stage,
                        normalized_targets,
                        matching,
                        num_targets,
                    )
                )
            else:
                raw_mask, raw_dice, soft_best = (
                    self._soft_best_mask_losses(
                        stage,
                        normalized_targets,
                        matching,
                        num_targets,
                    )
                )
            suffix = "" if position == len(stages) - 1 else f"_{stage_indices[position]}"
            losses[f"loss_group_cls{suffix}"] = (
                stage_weight * self.class_weight * raw_class
            )
            losses[f"loss_group_mask{suffix}"] = (
                stage_weight * self.mask_weight * raw_mask
            )
            losses[f"loss_group_dice{suffix}"] = (
                stage_weight * self.dice_weight * raw_dice
            )
            stage_diagnostics.append(
                StageCriterionDiagnostics(
                    stage_index=stage_indices[position],
                    matching=matching,
                    soft_best_groups=soft_best,
                    raw_class_loss=raw_class.detach(),
                    raw_mask_loss=raw_mask.detach(),
                    raw_dice_loss=raw_dice.detach(),
                )
            )
            if position == len(stages) - 1:
                final_matching = matching

        if final_matching is None:  # pragma: no cover - guarded by non-empty stages.
            raise RuntimeError("final stage matching was not produced")
        if self.topology_group_version >= 2:
            selector = self._member_selector_loss(
                final_stage,
                normalized_targets,
                final_matching,
            )
            losses["loss_member_selector"] = (
                self.member_selector_weight * selector
            )
            unmatched_count = sum(
                int(sample.unmatched_target_indices.numel())
                for sample in final_matching
            )
        else:
            quality_reg, quality_rank = self._quality_losses(
                final_stage,
                normalized_targets,
                final_matching,
            )
            coverage, unmatched_count = self._coverage_loss(
                final_stage,
                normalized_targets,
                final_matching,
                num_targets,
            )
            losses["loss_quality_reg"] = (
                self.quality_reg_weight * quality_reg
            )
            losses["loss_quality_rank"] = (
                self.quality_rank_weight * quality_rank
            )
            losses["loss_unmatched_coverage"] = (
                self.unmatched_coverage_weight * coverage
            )

        for name, value in losses.items():
            if value.ndim != 0:
                raise RuntimeError(f"{name} must be scalar, got {tuple(value.shape)}")
            if not bool(torch.isfinite(value)):
                raise FloatingPointError(f"{name} is NaN or Inf")
        diagnostics = HierarchicalCriterionDiagnostics(
            stages=tuple(stage_diagnostics),
            final_stage_index=stage_indices[-1],
            local_num_targets=local_num_targets,
            normalized_num_targets=num_targets.detach().clone(),
            unmatched_final_targets=unmatched_count,
            skipped=False,
        )
        return losses, diagnostics


__all__ = [
    "HierarchicalCriterionDiagnostics",
    "HierarchicalGroupCriterion",
    "SoftBestGroupDiagnostics",
    "StageCriterionDiagnostics",
]
