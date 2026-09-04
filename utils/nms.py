# Non-Maximum Suppression (NMS) post-processing for instance predictions.
# Used to filter duplicate boxes/masks around the same target at inference time.
import torch

from .box_ops import box_iou


def mask_iou(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of binary masks.
    Args:
        masks1: Tensor of shape (N, H, W), binary (0/1) masks.
        masks2: Tensor of shape (M, H, W), binary (0/1) masks.
    Returns:
        Tensor of shape (N, M) with pairwise mask IoU.
    """
    masks1 = masks1.flatten(1).float()
    masks2 = masks2.flatten(1).float()
    inter = masks1 @ masks2.t()  # (N, M)
    area1 = masks1.sum(1).unsqueeze(1)  # (N, 1)
    area2 = masks2.sum(1).unsqueeze(0)  # (1, M)
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def nms_from_iou(scores: torch.Tensor, iou_matrix: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Greedy NMS given a precomputed IoU matrix.
    Args:
        scores: Tensor of shape (N,) with confidence scores.
        iou_matrix: Tensor of shape (N, N) with pairwise IoU.
        iou_threshold: instances with IoU >= threshold (w.r.t. a kept instance
            with higher score) are suppressed.
    Returns:
        LongTensor of indices to keep.
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    for idx in order:
        idx = idx.item()
        if suppressed[idx]:
            continue
        keep.append(idx)
        # suppress all instances that overlap enough with the kept one
        suppressed |= iou_matrix[idx] >= iou_threshold
    return torch.tensor(keep, dtype=torch.long, device=scores.device)


def box_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """NMS on xyxy boxes."""
    iou_matrix, _ = box_iou(boxes, boxes)
    return nms_from_iou(scores, iou_matrix, iou_threshold)


def mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """NMS on binary masks (mask IoU)."""
    iou_matrix = mask_iou(masks, masks)
    return nms_from_iou(scores, iou_matrix, iou_threshold)
