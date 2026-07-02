"""Vanilla Coordinate Attention (CA) module.

Reference:
    Hou, Q., Zhou, D., & Feng, J. (2021). Coordinate Attention for
    Efficient Mobile Network Design. CVPR 2021. arXiv:2103.02907.

    Citation context and novelty-claim rationale for this project are
    recorded in DECISIONS.md (D-003, D-006) — no local PDF copy of the
    source paper is checked into this repo.

This is the vanilla module as described in the paper: no gating
variants, no channel shuffling, no initialization hacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def h_swish(x):
        return x * F.relu6(x + 3, inplace=True) / 6

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()

        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.h_swish(y)

        f_h, f_w = torch.split(y, [h, w], dim=2)  # (B, mip, H, 1), (B, mip, W, 1)
        f_w = f_w.permute(0, 1, 3, 2)  # (B, mip, 1, W)

        a_h = self.conv_h(f_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(f_w).sigmoid()  # (B, C, 1, W)

        return identity * a_h * a_w
