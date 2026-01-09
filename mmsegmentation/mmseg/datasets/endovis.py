# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class EndoVis2017BinaryDataset(BaseSegDataset):
    """EndoVis 2017 Binary Segmentation dataset.

    Binary segmentation task for surgical instrument segmentation.
    0 = background, 1 = instrument.
    """
    METAINFO = dict(
        classes=('background', 'instrument'),
        palette=[[0, 0, 0], [255, 255, 255]])

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


@DATASETS.register_module()
class EndoVis2017PartsDataset(BaseSegDataset):
    """EndoVis 2017 Parts Segmentation dataset.

    Parts segmentation task for surgical instruments.
    0 = background, 1 = shaft, 2 = wrist, 3 = claspers, 4 = other.
    """
    METAINFO = dict(
        classes=('background', 'shaft', 'wrist', 'claspers', 'other'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])

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


@DATASETS.register_module()
class EndoVis2017TypeDataset(BaseSegDataset):
    """EndoVis 2017 Instrument Type Segmentation dataset.

    Instrument type segmentation task.
    0 = background, 1 = Bipolar Forceps, 2 = Prograsp Forceps,
    3 = Large Needle Driver, 4 = Vessel Sealer, 5 = Grasping Retractor,
    6 = Monopolar Curved Scissors, 7 = Other.
    """
    METAINFO = dict(
        classes=('background', 'Bipolar_Forceps', 'Prograsp_Forceps',
                 'Large_Needle_Driver', 'Vessel_Sealer', 'Grasping_Retractor',
                 'Monopolar_Curved_Scissors', 'Other'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128]])

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
