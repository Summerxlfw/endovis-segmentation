# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from HCFNet: https://mp.weixin.qq.com/s/26H0PgN5sikD1MoSkIBJzg
"""Pyramid Parallel Attention module for multi-scale feature aggregation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SpatialAttentionModule(nn.Module):
    """Spatial Attention Module using channel-wise pooling."""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class LocalGlobalAttention(nn.Module):
    """Local-Global Attention with patch-based processing.

    Args:
        output_dim (int): Output dimension.
        patch_size (int): Size of local patches.
    """

    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = nn.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = nn.Parameter(
            torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Ensure H and W are divisible by patch_size
        if H % P != 0 or W % P != 0:
            pad_h = (P - H % P) % P
            pad_w = (P - W % P) % P
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1)
            _, H, W, _ = x.shape

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, C, P, P)
        local_patches = local_patches.reshape(B, -1, C, P * P)  # (B, num_patches, C, P*P)
        local_patches = local_patches.mean(dim=2)  # (B, num_patches, P*P)

        local_patches = self.mlp1(local_patches)  # (B, num_patches, output_dim // 2)
        local_patches = self.norm(local_patches)
        local_patches = self.mlp2(local_patches)  # (B, num_patches, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)
        local_out = local_patches * local_attention

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(
            self.prompt[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(
            local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    """Efficient Channel Attention.

    Args:
        in_channel (int): Number of input channels.
        gamma (int): Hyperparameter for kernel size calculation. Default: 2.
        b (int): Hyperparameter for kernel size calculation. Default: 1.
    """

    def __init__(self, in_channel, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=1, kernel_size=kernel_size,
                padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class PPA(nn.Module):
    """Pyramid Parallel Attention.

    Combines local-global attention at multiple scales with spatial and
    channel attention for comprehensive feature aggregation.

    Args:
        in_features (int): Number of input channels.
        filters (int): Number of output channels.
    """

    def __init__(self, in_features, filters):
        super().__init__()

        self.skip = Conv(in_features, filters, act=False)
        self.c1 = Conv(in_features, filters, 3)
        self.c2 = Conv(filters, filters, 3)
        self.c3 = Conv(filters, filters, 3)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.silu = nn.SiLU()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, filters, H, W).
        """
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x
