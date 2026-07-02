"""Runtime monkey-patch of ultralytics.utils.loss.BboxLoss for WIoU v3.

Gated by the IOU_TYPE env var so the stock CIoU path is untouched unless
explicitly requested (IOU_TYPE=wiou). No site-packages files are edited —
this patches the class object in memory, at import time, in the running
process only.
"""

import os

import torch

from .wiou import wiou_v3_loss


def apply_wiou_patch():
    """Patch BboxLoss to use WIoU v3 instead of CIoU, if IOU_TYPE=wiou."""
    if os.environ.get("IOU_TYPE", "").lower() != "wiou":
        return

    import ultralytics.utils.loss as _loss_mod

    BboxLoss = _loss_mod.BboxLoss

    if getattr(BboxLoss, "_wiou_patched", False):
        print("[wiou] patch already applied.")
        return

    bbox2dist = _loss_mod.bbox2dist

    _orig_init = BboxLoss.__init__
    _orig_forward = BboxLoss.forward
    BboxLoss._orig_init = _orig_init
    BboxLoss._orig_forward = _orig_forward

    def _patched_init(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        _orig_init(self, reg_max)
        self.register_buffer("_wiou_running_mean", torch.tensor(1.0), persistent=False)

    def _patched_forward(
        self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    ):
        """IoU loss (WIoU v3)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        l_wiou = wiou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], self._wiou_running_mean)
        loss_iou = (l_wiou.unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    BboxLoss.__init__ = _patched_init
    BboxLoss.forward = _patched_forward
    BboxLoss._wiou_patched = True

    print("[wiou] BboxLoss patched — using WIoU v3 (alpha=1.9, delta=3, momentum=0.02).")
