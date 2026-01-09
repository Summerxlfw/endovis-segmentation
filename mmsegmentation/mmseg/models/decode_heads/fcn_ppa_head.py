# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .fcn_head import FCNHead
from ..plugins.ppa import PPA


@MODELS.register_module()
class FCNPPAHead(FCNHead):
    """FCN Head with Pyramid Parallel Attention for fine-grained segmentation.

    This head extends FCNHead by adding PPA module after the convolution
    layers for better local-global feature aggregation. The attention is
    applied at a reduced resolution to save memory.

    Args:
        ppa_filters (int, optional): Number of filters in PPA. If None,
            uses self.channels. Default: None.
        ppa_downsample (int): Downsample factor before attention. Default: 8.
        **kwargs: Other arguments for FCNHead.
    """

    def __init__(self, ppa_filters=None, ppa_downsample=8, **kwargs):
        super().__init__(**kwargs)
        self.ppa_downsample = ppa_downsample
        filters = ppa_filters if ppa_filters is not None else self.channels
        self.ppa = PPA(in_features=self.channels, filters=filters)

        # If filters != channels, add a projection layer
        if filters != self.channels:
            self.ppa_proj = ConvModule(
                filters,
                self.channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        else:
            self.ppa_proj = None

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
        if self.ppa_downsample > 1:
            feats_down = F.adaptive_avg_pool2d(
                feats, (H // self.ppa_downsample, W // self.ppa_downsample))
        else:
            feats_down = feats

        # Apply PPA attention at lower resolution
        attn_out = self.ppa(feats_down)

        # Project back to original channels if needed
        if self.ppa_proj is not None:
            attn_out = self.ppa_proj(attn_out)

        # Upsample back to original resolution
        if self.ppa_downsample > 1:
            attn_out = F.interpolate(
                attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Residual connection
        feats = feats + attn_out

        return feats
