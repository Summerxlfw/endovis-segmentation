# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from: https://mmcheng.net/wp-content/uploads/2025/06/25PAMI_LRFormer.pdf
"""Low-Resolution Self-Attention module for efficient global context modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LRSA(nn.Module):
    """Low-Resolution Self-Attention.

    This module performs self-attention at multiple low resolutions through
    pyramid pooling, enabling efficient global context modeling.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 4.
        qkv_bias (bool): If True, add learnable bias to qkv. Default: False.
        qk_scale (float, optional): Override default qk scale. Default: None.
        attn_drop (float): Dropout ratio of attention weight. Default: 0.
        proj_drop (float): Dropout ratio of output. Default: 0.
        pooled_sizes (list): List of pooled sizes for K, V. Default: [11, 8, 6, 4].
        q_pooled_size (int): Pooled size for Q. Default: 16.
        q_conv (bool): Whether to use conv for Q. Default: False.
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., pooled_sizes=[11, 8, 6, 4],
                 q_pooled_size=16, q_conv=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001

        self.norm = nn.LayerNorm(dim)

        self.q_pooled_size = q_pooled_size

        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

        self.d_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
            for _ in pooled_sizes
        ])

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.size()
        N = H * W
        x_flat = x.flatten(2).permute(0, 2, 1)  # B C H W -> B N C

        if self.q_pooled_size > 1:
            # Keep the W/H ratio of the features
            if W >= H:
                q_pooled_size = (
                    self.q_pooled_size,
                    round(W * float(self.q_pooled_size) / H + self.eps)
                )
            else:
                q_pooled_size = (
                    round(H * float(self.q_pooled_size) / W + self.eps),
                    self.q_pooled_size
                )

            # Conduct fixed pooled size pooling on q
            q = F.adaptive_avg_pool2d(x, q_pooled_size)
            _, _, H1, W1 = q.shape
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)
            q = self.q[0](q).reshape(
                B, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3).contiguous()
        else:
            H1, W1 = H, W
            if self.q_conv is not None:
                q = x + self.q_conv(x)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q[0](q).reshape(
                    B, -1, self.num_heads, C // self.num_heads
                ).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q[0](x_flat).reshape(
                    B, -1, self.num_heads, C // self.num_heads
                ).permute(0, 2, 1, 3).contiguous()

        # Conduct Pyramid Pooling on K, V
        pools = []
        for (pooled_size, l) in zip(self.pooled_sizes, self.d_convs):
            if W >= H:
                ps = (pooled_size, round(W * pooled_size / H + self.eps))
            else:
                ps = (round(H * pooled_size / W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x, ps)
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv[0](pools).reshape(
            B, -1, 2, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)  # B num_heads N C//num_heads
        out = out.transpose(1, 2).reshape(B, -1, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        # Bilinear upsampling for residual connection
        out = out.transpose(1, 2).reshape(B, C, H1, W1)
        if self.q_pooled_size > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out
