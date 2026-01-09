#!/usr/bin/env python3
"""Create 4-fold cross-validation config files for MMSegmentation."""

import os

BASE_DIR = '/home/summer/endovis/mmsegmentation/configs/endovis'

# Binary config template
BINARY_TEMPLATE = '''_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root = '/home/summer/endovis/endovis2017_binary_fold{fold}'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1088), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline,
        metainfo=dict(classes=('background', 'instrument'))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline,
        metainfo=dict(classes=('background', 'instrument'))
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# Model settings
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='whole')
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
]

# Training settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/endovis2017_binary_fold{fold}'
'''

# Parts config template
PARTS_TEMPLATE = '''_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root = '/home/summer/endovis/endovis2017_parts_fold{fold}'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1088), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline,
        metainfo=dict(classes=('background', 'shaft', 'wrist', 'claspers'))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline,
        metainfo=dict(classes=('background', 'shaft', 'wrist', 'claspers'))
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# Model settings
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    test_cfg=dict(mode='whole')
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
]

# Training settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/endovis2017_parts_fold{fold}'
'''

# Type config template
TYPE_TEMPLATE = '''_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root = '/home/summer/endovis/endovis2017_type_fold{fold}'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1088), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline,
        metainfo=dict(classes=(
            'background',
            'Bipolar_Forceps',
            'Prograsp_Forceps',
            'Large_Needle_Driver',
            'Vessel_Sealer',
            'Grasping_Retractor',
            'Monopolar_Curved_Scissors',
            'Other'
        ))
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline,
        metainfo=dict(classes=(
            'background',
            'Bipolar_Forceps',
            'Prograsp_Forceps',
            'Large_Needle_Driver',
            'Vessel_Sealer',
            'Grasping_Retractor',
            'Monopolar_Curved_Scissors',
            'Other'
        ))
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# Model settings
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
    test_cfg=dict(mode='whole')
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
]

# Training settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/endovis2017_type_fold{fold}'
'''

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    templates = {
        'binary': BINARY_TEMPLATE,
        'parts': PARTS_TEMPLATE,
        'type': TYPE_TEMPLATE,
    }

    for task, template in templates.items():
        for fold in range(4):
            config_path = os.path.join(BASE_DIR, f'endovis2017_{task}_fold{fold}.py')
            content = template.format(fold=fold)
            with open(config_path, 'w') as f:
                f.write(content)
            print(f"Created: {config_path}")

    print("\nAll 12 config files created successfully!")

if __name__ == '__main__':
    main()
