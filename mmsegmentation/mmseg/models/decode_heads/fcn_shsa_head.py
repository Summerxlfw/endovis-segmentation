# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .fcn_head import FCNHead
from ..plugins.shsa import SHSA


@MODELS.register_module()
class FCNSHSAHead(FCNHead):
    """FCN Head with Single-Head Self-Attention for boundary refinement.

    This head extends FCNHead by adding SHSA attention module after the
    convolution layers for better boundary detection. The attention is
    applied at a reduced resolution to save memory.

    Args:
        shsa_qk_dim (int): Dimension for query and key in SHSA. Default: 16.
        shsa_residual (bool): Whether to use residual connection. Default: True.
        shsa_downsample (int): Downsample factor before attention. Default: 8.
        **kwargs: Other arguments for FCNHead.
    """

    def __init__(self, shsa_qk_dim=16, shsa_residual=True, shsa_downsample=8, **kwargs):
        super().__init__(**kwargs)
        self.shsa_residual = shsa_residual
        self.shsa_downsample = shsa_downsample
        self.shsa = SHSA(dim=self.channels, qk_dim=shsa_qk_dim)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # Get features from parent FCNHead
        feats = super()._forward_feature(inputs)

        H, W = feats.shape[2:]

        # Downsample for memory efficiency
        if self.shsa_downsample > 1:
            feats_down = F.adaptive_avg_pool2d(
                feats, (H // self.shsa_downsample, W // self.shsa_downsample))
        else:
            feats_down = feats

        # Apply SHSA attention at lower resolution
        attn_out = self.shsa(feats_down)

        # Upsample back to original resolution
        if self.shsa_downsample > 1:
            attn_out = F.interpolate(
                attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Apply residual connection
        if self.shsa_residual:
            feats = feats + attn_out
        else:
            feats = attn_out

        return feats
