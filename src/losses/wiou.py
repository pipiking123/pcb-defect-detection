"""Wise-IoU v3 (WIoU v3) bounding-box regression loss.

Reference:
    Tong, Z., Chen, Y., Xu, Z., & Yu, R. (2023). Wise-IoU: Bounding Box
    Regression Loss with Dynamic Focusing Mechanism. arXiv:2301.10051.

Implements Sec. 3 of the paper:
    - WIoU v1 (Sec. 3.1, Eq. 5-6): a distance-attention gain R_WIoU that
      multiplies the plain IoU loss L_IoU. R_WIoU's denominator (the
      squared diagonal of the enclosing box) is detached from the
      autograd graph so it acts purely as a non-negative gain and does
      not itself accelerate convergence for already well-aligned boxes.
    - WIoU v3 (Sec. 3.3, Eq. 12-14): an outlier degree beta, computed
      from an exponential-moving-average running mean of L_IoU, drives
      a non-monotonic focusing coefficient r(beta). r down-weights both
      low-quality (harmful/outlier) and high-quality (already easy)
      anchors, concentrating gradient on medium-quality anchors.

Citation context: DECISIONS.md D-004, Item 4.
"""

import torch

EPS = 1e-7


def wiou_v3_loss(pred_bboxes, target_bboxes, running_mean_ref, alpha=1.9, delta=3.0, momentum=0.02):
    """
    Compute per-box Wise-IoU v3 loss.

    Args:
        pred_bboxes (torch.Tensor): (N, 4) predicted boxes in xyxy, already filtered to fg_mask.
        target_bboxes (torch.Tensor): (N, 4) target boxes in xyxy, already filtered to fg_mask.
        running_mean_ref (torch.Tensor): mutable single-element buffer tracking the EMA of
            L_IoU across calls; updated in place.
        alpha (float): non-monotonic focusing coefficient shape parameter.
        delta (float): non-monotonic focusing coefficient shape parameter.
        momentum (float): EMA momentum for running_mean_ref.

    Returns:
        (torch.Tensor): per-box L_WIoUv3, shape (N,). Caller applies weight and reduction.
    """
    # 1. Standard IoU
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + EPS
    iou = inter / union
    L_IoU = 1.0 - iou  # (N, 1)

    # 2. Centers
    cx_p, cy_p = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx_t, cy_t = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    # 3. Enclosing box
    W_g = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    H_g = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    # 4. R_WIoU — only the denominator is detached, per paper
    R_WIoU = torch.exp(
        ((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2) / ((W_g**2 + H_g**2).detach() + EPS)
    )

    # 5. WIoU v1
    L_WIoUv1 = R_WIoU * L_IoU

    # 6. EMA update of the running mean of L_IoU (no grad), clamped to avoid division by zero
    with torch.no_grad():
        running_mean_ref.mul_(1 - momentum).add_(momentum * L_IoU.detach().mean())
        running_mean_ref.clamp_(min=EPS)

    # 7. Outlier degree
    beta = L_IoU.detach() / running_mean_ref

    # 8. Non-monotonic focusing coefficient
    r = beta / (delta * alpha ** (beta - delta))

    # 9. WIoU v3
    return (r * L_WIoUv1).squeeze(-1)
