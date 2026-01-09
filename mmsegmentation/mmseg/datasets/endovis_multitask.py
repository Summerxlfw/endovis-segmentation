# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EndoVis2017MultiTaskDataset(BaseSegDataset):
    """EndoVis 2017 Multi-Task Segmentation dataset.

    Multi-task learning dataset for surgical instrument segmentation.
    Loads binary, parts, and type annotations for each image.

    Directory structure:
        endovis2017_multitask_fold{0-3}/
        ├── images/
        │   ├── train/
        │   └── val/
        └── annotations/
            ├── binary/
            │   ├── train/
            │   └── val/
            ├── parts/
            │   ├── train/
            │   └── val/
            └── type/
                ├── train/
                └── val/
    """
    METAINFO = dict(
        classes_binary=('background', 'instrument'),
        classes_parts=('background', 'shaft', 'wrist', 'claspers'),
        classes_type=('background', 'Bipolar_Forceps', 'Prograsp_Forceps',
                      'Large_Needle_Driver', 'Vessel_Sealer', 'Grasping_Retractor',
                      'Monopolar_Curved_Scissors', 'Other'),
        palette_binary=[[0, 0, 0], [255, 255, 255]],
        palette_parts=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        palette_type=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                      [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory.

        Returns:
            list[dict]: All data info of dataset, with multi-task annotations.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)

        # Get annotation directories for each task
        # The data_prefix should have img_path like 'images/train'
        # We derive annotation paths from it
        subset = osp.basename(img_dir)  # 'train' or 'val'
        data_root = self.data_root

        ann_dir_binary = osp.join(data_root, 'annotations', 'binary', subset)
        ann_dir_parts = osp.join(data_root, 'annotations', 'parts', subset)
        ann_dir_type = osp.join(data_root, 'annotations', 'type', subset)

        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):

            img_name = img[:-_suffix_len]
            seg_map = img_name + self.seg_map_suffix

            data_info = dict(
                img_path=osp.join(img_dir, img),
                seg_map_path_binary=osp.join(ann_dir_binary, seg_map),
                seg_map_path_parts=osp.join(ann_dir_parts, seg_map),
                seg_map_path_type=osp.join(ann_dir_type, seg_map),
            )
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
