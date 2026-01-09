# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class MultiTaskEncoderDecoder(BaseSegmentor):
    """Multi-Task Encoder Decoder segmentor for EndoVis 2017.

    This segmentor uses a shared backbone and three separate decode heads
    for binary, parts, and type segmentation tasks.

    Args:
        backbone (ConfigType): The config for the backbone.
        decode_head_binary (ConfigType): The config for binary decode head.
        decode_head_parts (ConfigType): The config for parts decode head.
        decode_head_type (ConfigType): The config for type decode head.
        neck (OptConfigType): The config for the neck. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (OptConfigType): The pre-process config.
        pretrained (str, optional): The path for pretrained model.
        init_cfg (dict, optional): The weight initialized config.
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head_binary: ConfigType,
                 decode_head_parts: ConfigType,
                 decode_head_type: ConfigType,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Initialize backbone
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)

        # Initialize neck
        if neck is not None:
            self.neck = MODELS.build(neck)

        # Initialize three decode heads
        self.decode_head_binary = MODELS.build(decode_head_binary)
        self.decode_head_parts = MODELS.build(decode_head_parts)
        self.decode_head_type = MODELS.build(decode_head_type)

        # Set decode_head to binary for compatibility with base class
        self.decode_head = self.decode_head_binary
        self.align_corners = self.decode_head_binary.align_corners
        self.num_classes = self.decode_head_binary.num_classes
        self.out_channels = self.decode_head_binary.out_channels

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_decode_head(self) -> bool:
        return hasattr(self, 'decode_head_binary') and self.decode_head_binary is not None

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head_binary.predict(x, batch_img_metas,
                                                     self.test_cfg)
        return seg_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(inputs)
        losses = dict()

        # Binary task loss
        loss_binary = self._task_forward_train(
            x, data_samples, self.decode_head_binary, 'binary')
        losses.update(add_prefix(loss_binary, 'binary'))

        # Parts task loss
        loss_parts = self._task_forward_train(
            x, data_samples, self.decode_head_parts, 'parts')
        losses.update(add_prefix(loss_parts, 'parts'))

        # Type task loss
        loss_type = self._task_forward_train(
            x, data_samples, self.decode_head_type, 'type')
        losses.update(add_prefix(loss_type, 'type'))

        return losses

    def _task_forward_train(self, inputs: List[Tensor],
                            data_samples: SampleList,
                            decode_head,
                            task_name: str) -> dict:
        """Run forward function and calculate loss for a specific task.

        Args:
            inputs (List[Tensor]): Feature maps from backbone.
            data_samples (SampleList): Data samples.
            decode_head: The decode head for this task.
            task_name (str): Name of the task ('binary', 'parts', 'type').

        Returns:
            dict: Loss dictionary for this task.
        """
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        # Create modified data samples with the appropriate gt_seg_map
        modified_samples = []
        for sample in data_samples:
            # Get the task-specific ground truth
            gt_key = f'gt_seg_map_{task_name}'
            if hasattr(sample, gt_key):
                gt_seg = getattr(sample, gt_key)
                # Create a copy of the sample with standard gt_sem_seg
                new_sample = SegDataSample()
                new_sample.set_metainfo(sample.metainfo)
                new_sample.gt_sem_seg = PixelData(data=gt_seg.data)
                modified_samples.append(new_sample)
            else:
                modified_samples.append(sample)

        losses = decode_head.loss(inputs, modified_samples, self.train_cfg)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples.

        For multi-task, we predict all three tasks and store them separately.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results with predictions
                for all three tasks.
        """
        if data_samples is None:
            data_samples = [None] * inputs.shape[0]

        batch_img_metas = []
        for data_sample in data_samples:
            if data_sample is not None:
                batch_img_metas.append(data_sample.metainfo)
            else:
                batch_img_metas.append(dict())

        x = self.extract_feat(inputs)

        # Predict all three tasks
        seg_logits_binary = self.decode_head_binary.predict(
            x, batch_img_metas, self.test_cfg)
        seg_logits_parts = self.decode_head_parts.predict(
            x, batch_img_metas, self.test_cfg)
        seg_logits_type = self.decode_head_type.predict(
            x, batch_img_metas, self.test_cfg)

        # Store predictions in data samples
        return self.postprocess_result_multitask(
            seg_logits_binary, seg_logits_parts, seg_logits_type,
            data_samples)

    def postprocess_result_multitask(self,
                                     seg_logits_binary: Tensor,
                                     seg_logits_parts: Tensor,
                                     seg_logits_type: Tensor,
                                     data_samples: OptSampleList = None) -> SampleList:
        """Convert multi-task results to SegDataSample format.

        Args:
            seg_logits_binary: Binary segmentation logits.
            seg_logits_parts: Parts segmentation logits.
            seg_logits_type: Type segmentation logits.
            data_samples: Original data samples.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results.
        """
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        batch_size = seg_logits_binary.shape[0]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            # Get predictions by argmax
            pred_binary = seg_logits_binary[i].argmax(dim=0)
            pred_parts = seg_logits_parts[i].argmax(dim=0)
            pred_type = seg_logits_type[i].argmax(dim=0)

            # Store in data sample
            if data_samples[i] is None:
                data_samples[i] = SegDataSample()

            # Main prediction (binary for compatibility)
            data_samples[i].pred_sem_seg = PixelData(data=pred_binary)

            # Store all task predictions
            data_samples[i].pred_sem_seg_binary = PixelData(data=pred_binary)
            data_samples[i].pred_sem_seg_parts = PixelData(data=pred_parts)
            data_samples[i].pred_sem_seg_type = PixelData(data=pred_type)

            # Store logits
            data_samples[i].seg_logits = PixelData(data=seg_logits_binary[i])

        return data_samples

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process for ONNX export.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg data samples.

        Returns:
            Tensor: Binary segmentation logits (for compatibility).
        """
        x = self.extract_feat(inputs)
        return self.decode_head_binary.forward(x)
