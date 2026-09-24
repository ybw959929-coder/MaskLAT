from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import sem_seg_postprocess
from .topology_group_filter import get_topology_group_valid_mask


def restore_refseg_query_masks(
    mask_logits: torch.Tensor,
    image_size,
    scaled_size,
) -> torch.Tensor:
    """Restore every Query mask with the formal RefSeg geometry path."""

    return sem_seg_postprocess(
        mask_logits,
        scaled_size,
        image_size[0],
        image_size[1],
    )


def refseg_foreground_scores(class_logits: torch.Tensor) -> torch.Tensor:
    """Return the exact foreground score used by formal MaskLAT RefSeg."""

    if class_logits.ndim != 2:
        raise ValueError(
            "RefSeg class logits must be [queries, classes], got "
            f"{tuple(class_logits.shape)}"
        )
    if class_logits.shape[-1] != 2:
        raise ValueError(
            "RefSeg requires one condition plus explicit background, got "
            f"{class_logits.shape[-1]} class columns"
        )
    return F.softmax(class_logits, dim=-1)[:, 0]


def select_refseg_query(
    class_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one Query exactly as the formal RefSeg postprocessor does."""

    return refseg_foreground_scores(class_logits).max(dim=0)


def binarize_refseg_query_masks(
    restored_mask_logits: torch.Tensor,
    mask_threshold: float,
) -> torch.Tensor:
    """Apply the formal RefSeg sigmoid and strict probability threshold."""

    return restored_mask_logits.sigmoid() > mask_threshold


def refseg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    **kwargs,
) -> List[Dict]:
    # [batch_size, num_queries, num_classes+1]
    class_queries_logits = outputs.class_queries_logits
    # [batch_size, num_queries, height, width]
    masks_queries_logits = outputs.masks_queries_logits
    group_valid_mask = get_topology_group_valid_mask(
        outputs,
        class_queries_logits,
        masks_queries_logits,
    )
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes

    batch_size = class_queries_logits.shape[0]
    num_labels = class_queries_logits.shape[-1] - 1
    assert num_labels == 1

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

        if group_valid_mask is not None:
            valid_indices = torch.nonzero(group_valid_mask[i], as_tuple=False).flatten()
            mask_cls = mask_cls[valid_indices]
            mask_pred = mask_pred[valid_indices]

        # Every RefCOCO expression denotes an existing target.  Rank candidate
        # masks by that condition's foreground probability after discarding
        # topology padding rows.  A background score from an unrelated
        # candidate must not veto the best foreground candidate.
        foreground_scores = F.softmax(mask_cls, dim=-1)[:, 0]
        top_score, top_index = foreground_scores.max(dim=0)
        mask_prob = mask_pred[top_index].sigmoid()
        if mask_prob.ndim != 2:
            raise ValueError(
                "RefSeg selected mask must be two-dimensional, got "
                f"{tuple(mask_prob.shape)}"
            )

        # 255 is the ignore index
        segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)
        segmentation[mask_prob > mask_threshold] = 1

        segments_info = {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results
