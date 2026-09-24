"""Validation helpers for topology-group post-processing.

The topology decoder keeps a physical ``Q``-row prediction table so existing
task post-processors can consume its outputs.  Only rows marked by
``group_valid_mask`` are real group predictions; the other rows are padding
used to preserve that interface.
"""

from typing import Optional

import torch


def get_topology_group_valid_mask(
    outputs,
    class_queries_logits: torch.Tensor,
    masks_queries_logits: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return a validated boolean root mask, or ``None`` for legacy outputs.

    Presence of ``outputs.group_valid_mask`` is the topology-mode contract.
    Legacy outputs intentionally take the unchanged post-processing path.
    """

    group_valid_mask = getattr(outputs, "group_valid_mask", None)
    if group_valid_mask is None:
        return None
    if not isinstance(group_valid_mask, torch.Tensor):
        raise TypeError("outputs.group_valid_mask must be a torch.Tensor")
    if group_valid_mask.shape != class_queries_logits.shape[:2]:
        raise ValueError(
            "outputs.group_valid_mask must match the batch/query dimensions "
            f"{tuple(class_queries_logits.shape[:2])}, got {tuple(group_valid_mask.shape)}"
        )
    if masks_queries_logits is not None and masks_queries_logits.shape[:2] != class_queries_logits.shape[:2]:
        raise ValueError(
            "outputs.masks_queries_logits must match class-query batch/query "
            f"dimensions {tuple(class_queries_logits.shape[:2])}, "
            f"got {tuple(masks_queries_logits.shape[:2])}"
        )

    group_valid_mask = group_valid_mask.to(device=class_queries_logits.device, dtype=torch.bool)
    if not bool(group_valid_mask.any(dim=1).all()):
        raise ValueError("every topology sample must contain at least one valid group")
    return group_valid_mask
