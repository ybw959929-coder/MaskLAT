"""Versioned, training-only supervision of latent reads using mask groups.

This module does not mutate the query table, mask predictions, or attention.
Each image is grouped independently from detached predicted masks.  Greedy
complete-link grouping prevents a chain of overlapping masks from joining two
incompatible endpoints.  Groups are sorted by member count (descending), then
minimum query index (ascending); at most one group per available latent is
supervised.  Unmatched latents receive no auxiliary background/diversity loss.

Only the actual latent-to-query attention carries gradients.  Its full rows are
normalized to distributions, including probability mass on queries excluded
from grouping: sending attention to an invalid/unselected query is NOT silently
forgiven.  Mask grouping and Hungarian assignments are detached.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskGroupAuxiliaryLoss(nn.Module):
    """Hungarian-matched KL loss for ``attention[B, latent, query]``.

    Args:
        mask_threshold: Foreground probability threshold for grouping only.
        group_iou_threshold: Every pair in a group must meet this binary IoU.
        spatial_size: Maximum grouping raster size per axis; smaller inputs are
            not upsampled.  This has no effect on segmentation predictions.
        min_mask_area: Minimum foreground pixels on that valid grouping raster.
        eps: Floor used only when taking logs of attention probabilities.

    ``spatial_valid_mask`` uses True/one for real image pixels and False/zero
    for padding; it may be [B,H,W] or [B,1,H,W], at any spatial resolution.
    ``query_valid_mask`` is [B,Q], True for queries eligible for grouping.
    Invalid query indices remain in the attention denominator.
    """

    def __init__(
        self,
        mask_threshold: float = 0.5,
        group_iou_threshold: float = 0.7,
        spatial_size: Union[int, tuple[int, int]] = 64,
        min_mask_area: int = 4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        if not 0.0 < group_iou_threshold <= 1.0:
            raise ValueError("group_iou_threshold must be in (0, 1]")
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if len(spatial_size) != 2 or any(int(x) != x or x <= 0 for x in spatial_size):
            raise ValueError("spatial_size must contain two positive integers")
        if int(min_mask_area) != min_mask_area or min_mask_area < 1:
            raise ValueError("min_mask_area must be a positive integer")
        if not math.isfinite(eps) or not 0.0 < eps < 1.0:
            raise ValueError("eps must be finite and in (0, 1)")
        self.mask_threshold = float(mask_threshold)
        self.group_iou_threshold = float(group_iou_threshold)
        self.spatial_size = tuple(int(x) for x in spatial_size)
        self.min_mask_area = int(min_mask_area)
        self.eps = float(eps)

    @staticmethod
    def _check_finite(name: str, value: Tensor) -> None:
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} contains non-finite values")

    @torch.no_grad()
    def build_mask_groups(
        self,
        mask_logits: Tensor,
        num_latents: int,
        spatial_valid_mask: Optional[Tensor] = None,
        query_valid_mask: Optional[Tensor] = None,
    ) -> dict:
        """Return selected query-index groups and detached per-image counts.

        No mask thresholding/clustering decision participates in autograd.
        ``num_groups`` counts all eligible groups BEFORE truncation, while
        ``num_selected_groups`` and ``selected_query_count`` describe the
        actual supervised targets.  Empty examples are valid and skipped.
        """
        if mask_logits.ndim != 4 or not mask_logits.is_floating_point():
            raise ValueError("mask_logits must be floating point [B,Q,H,W]")
        if int(num_latents) != num_latents or num_latents < 1:
            raise ValueError("num_latents must be a positive integer")
        batch_size, num_queries, height, width = mask_logits.shape
        if height == 0 or width == 0:
            raise ValueError("mask_logits must have nonempty spatial dimensions")
        self._check_finite("mask_logits", mask_logits)
        device = mask_logits.device
        if query_valid_mask is None:
            query_valid = torch.ones(batch_size, num_queries, dtype=torch.bool, device=device)
        else:
            if query_valid_mask.shape != (batch_size, num_queries):
                raise ValueError("query_valid_mask must have shape [B,Q]")
            if query_valid_mask.dtype != torch.bool:
                raise ValueError("query_valid_mask must have boolean dtype")
            query_valid = query_valid_mask.detach().to(device=device)

        if spatial_valid_mask is None:
            spatial_valid = torch.ones(batch_size, 1, height, width, device=device)
        else:
            spatial_valid = spatial_valid_mask.detach().to(device=device)
            if spatial_valid.ndim == 3:
                spatial_valid = spatial_valid.unsqueeze(1)
            if (
                spatial_valid.ndim != 4
                or spatial_valid.shape[:2] != (batch_size, 1)
                or min(spatial_valid.shape[-2:]) <= 0
            ):
                raise ValueError("spatial_valid_mask must have shape [B,1,H,W] or [B,H,W]")
            self._check_finite("spatial_valid_mask", spatial_valid)
            if ((spatial_valid != 0) & (spatial_valid != 1)).any().item():
                raise ValueError("spatial_valid_mask must contain only zero/one values")
            spatial_valid = spatial_valid.float()
            if spatial_valid.shape[-2:] != (height, width):
                spatial_valid = F.interpolate(spatial_valid, (height, width), mode="nearest")

        raster_size = (min(height, self.spatial_size[0]), min(width, self.spatial_size[1]))
        probabilities = mask_logits.detach().float().sigmoid()
        if raster_size != (height, width) and batch_size and num_queries:
            # Padding logits must never leak into valid pixels when resizing.
            weighted = F.interpolate(
                probabilities * spatial_valid, raster_size, mode="bilinear", align_corners=False
            )
            coverage = F.interpolate(
                spatial_valid, raster_size, mode="bilinear", align_corners=False
            )
            probabilities = weighted / coverage.clamp_min(self.eps)
            spatial_valid = F.interpolate(spatial_valid, raster_size, mode="nearest")
        binary_masks = (probabilities > self.mask_threshold) & spatial_valid.bool()
        flat = binary_masks.flatten(2).float()
        areas = flat.sum(-1)
        eligible = query_valid & (areas >= self.min_mask_area)

        selected_groups = []
        all_group_counts = []
        selected_counts = []
        selected_query_counts = []
        for sample_index in range(batch_size):
            indices = torch.nonzero(eligible[sample_index], as_tuple=False).flatten()
            query_indices = indices.cpu().tolist()
            groups: list[list[int]] = []
            if query_indices:
                masks = flat[sample_index].index_select(0, indices)
                with torch.autocast(device_type=device.type, enabled=False):
                    intersection = masks @ masks.transpose(0, 1)
                selected_areas = areas[sample_index].index_select(0, indices)
                union = selected_areas[:, None] + selected_areas[None, :] - intersection
                compatible = (intersection / union.clamp_min(1.0) >= self.group_iou_threshold).cpu().tolist()
                # Local indices are ordered by original query ID.  The first
                # eligible complete-link group wins; no transitive merge.
                local_groups: list[list[int]] = []
                for index in range(len(query_indices)):
                    for group in local_groups:
                        if all(compatible[index][member] for member in group):
                            group.append(index)
                            break
                    else:
                        local_groups.append([index])
                groups = [[query_indices[member] for member in group] for group in local_groups]
            groups.sort(key=lambda group: (-len(group), group[0]))
            all_group_counts.append(len(groups))
            chosen = groups[:num_latents]
            selected_groups.append(chosen)
            selected_counts.append(len(chosen))
            selected_query_counts.append(sum(len(group) for group in chosen))

        def counts(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        valid_counts = eligible.sum(-1)
        query_counts = counts(selected_query_counts)
        return {
            "selected_groups": selected_groups,
            "num_groups": counts(all_group_counts),
            "num_selected_groups": counts(selected_counts),
            "num_valid_queries": valid_counts,
            "selected_query_count": query_counts,
            "selected_query_fraction": query_counts.float() / valid_counts.clamp_min(1).float(),
        }

    def forward(
        self,
        attention: Tensor,
        mask_logits: Tensor,
        spatial_valid_mask: Optional[Tensor] = None,
        query_valid_mask: Optional[Tensor] = None,
    ) -> dict:
        """Return scalar/per-image KL losses plus detached grouping diagnostics.

        ``loss`` averages only samples with at least one selected group.
        ``per_sample_loss`` includes graph-connected zero entries for empty
        samples; callers can therefore normalize across stages or SEG rows.
        """
        if attention.ndim != 3 or not attention.is_floating_point():
            raise ValueError("attention must be floating point [B,L,Q]")
        if mask_logits.ndim != 4 or attention.shape[0] != mask_logits.shape[0] or attention.shape[2] != mask_logits.shape[1]:
            raise ValueError("attention batch/query dimensions must match mask_logits")
        if attention.device != mask_logits.device:
            raise ValueError("attention and mask_logits must be on the same device")
        if attention.shape[1] < 1:
            raise ValueError("attention must contain at least one latent")
        self._check_finite("attention", attention)
        if (attention < 0).any().item():
            raise ValueError("attention must contain nonnegative probabilities, not logits")
        probabilities = attention.float()
        row_mass = probabilities.sum(-1, keepdim=True)
        if attention.shape[-1] and (row_mass <= 0).any().item():
            raise ValueError("each attention row must have positive probability mass")
        # Do not mask out excluded queries before normalization: their mass
        # belongs outside every selected target group and must be penalized.
        probabilities = probabilities / row_mass.clamp_min(self.eps)
        log_probabilities = probabilities.clamp_min(self.eps).log()
        result = self.build_mask_groups(
            mask_logits,
            num_latents=attention.shape[1],
            spatial_valid_mask=spatial_valid_mask,
            query_valid_mask=query_valid_mask,
        )
        losses = []
        for sample_index, groups in enumerate(result["selected_groups"]):
            if not groups:
                losses.append(probabilities[sample_index].sum() * 0.0)
                continue
            # Existing training dependencies provide scipy; importing lazily
            # keeps grouping/empty batches usable without the matcher runtime.
            from scipy.optimize import linear_sum_assignment

            targets = torch.zeros(len(groups), attention.shape[-1], device=attention.device, dtype=torch.float32)
            target_entropy = torch.empty(len(groups), device=attention.device, dtype=torch.float32)
            for group_index, members in enumerate(groups):
                targets[group_index, members] = 1.0 / len(members)
                target_entropy[group_index] = math.log(len(members))
            with torch.autocast(device_type=attention.device.type, enabled=False):
                cross_entropy = -(targets @ log_probabilities[sample_index].transpose(0, 1))
            group_indices, latent_indices = linear_sum_assignment(cross_entropy.detach().cpu().numpy())
            group_indices = torch.as_tensor(group_indices, dtype=torch.long, device=attention.device)
            latent_indices = torch.as_tensor(latent_indices, dtype=torch.long, device=attention.device)
            group_kl = cross_entropy[group_indices, latent_indices] - target_entropy[group_indices]
            losses.append(group_kl.mean())
        per_sample_loss = torch.stack(losses) if losses else probabilities.sum((1, 2)) * 0.0
        valid_sample_mask = result["num_selected_groups"] > 0
        result.update(
            loss=per_sample_loss.sum() / valid_sample_mask.sum().clamp_min(1),
            per_sample_loss=per_sample_loss,
            valid_sample_mask=valid_sample_mask,
        )
        return result
