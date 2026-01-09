# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .fcn_head import FCNHead
from ..plugins.lrsa import LRSA


@MODELS.register_module()
class FCNLRSAHead(FCNHead):
    """FCN Head with Low-Resolution Self-Attention for global context.

    This head extends FCNHead by adding LRSA module after the convolution
    layers for better global semantic understanding. The attention is
    applied at a reduced resolution to save memory.

    Args:
        lrsa_num_heads (int): Number of attention heads in LRSA. Default: 4.
        lrsa_pooled_sizes (list): List of pooled sizes for K, V.
            Default: [8, 6, 4, 2].
        lrsa_q_pooled_size (int): Pooled size for Q. Default: 8.
        lrsa_residual (bool): Whether to use residual connection. Default: True.
        lrsa_downsample (int): Downsample factor before attention. Default: 8.
        **kwargs: Other arguments for FCNHead.
    """

    def __init__(self, lrsa_num_heads=4, lrsa_pooled_sizes=[8, 6, 4, 2],
                 lrsa_q_pooled_size=8, lrsa_residual=True, lrsa_downsample=8, **kwargs):
        super().__init__(**kwargs)
        self.lrsa_residual = lrsa_residual
        self.lrsa_downsample = lrsa_downsample
        self.lrsa = LRSA(
            dim=self.channels,
            num_heads=lrsa_num_heads,
            pooled_sizes=lrsa_pooled_sizes,
            q_pooled_size=lrsa_q_pooled_size
        )

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
        if self.lrsa_downsample > 1:
            feats_down = F.adaptive_avg_pool2d(
                feats, (H // self.lrsa_downsample, W // self.lrsa_downsample))
        else:
            feats_down = feats

        # Apply LRSA attention at lower resolution
        attn_out = self.lrsa(feats_down)

        # Upsample back to original resolution
        if self.lrsa_downsample > 1:
            attn_out = F.interpolate(
                attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Apply residual connection
        if self.lrsa_residual:
            feats = feats + attn_out
        else:
            feats = attn_out

        return feats
