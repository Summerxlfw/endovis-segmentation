# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from: https://arxiv.org/pdf/2401.16456 (CVPR2024-SHSA)
"""Single-Head Self-Attention module for efficient attention computation."""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution with BatchNorm and optional activation."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SHSA_GroupNorm(nn.GroupNorm):
    """Group Normalization with 1 group.

    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SHSA(nn.Module):
    """Single-Head Self-Attention.

    This module splits the input channels and applies efficient single-head
    attention on one half while keeping the other half unchanged.

    Args:
        dim (int): Number of input channels.
        qk_dim (int): Dimension for query and key projections. Default: 16.
    """

    def __init__(self, dim, qk_dim=16):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        pdim = dim // 2
        self.pdim = pdim

        self.pre_norm = SHSA_GroupNorm(pdim)

        self.qkv = Conv(pdim, qk_dim * 2 + pdim, act=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            Conv(dim, dim, act=False)
        )

        # Initialize projection weights to zero for residual learning
        nn.init.constant_(self.proj[1].bn.weight, 0.0)
        nn.init.constant_(self.proj[1].bn.bias, 0)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x
