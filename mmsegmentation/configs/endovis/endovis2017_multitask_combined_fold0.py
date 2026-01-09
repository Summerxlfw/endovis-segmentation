_base_ = [
    '../_base_/default_runtime.py',
]

# Dataset settings
dataset_type = 'EndoVis2017MultiTaskDataset'
data_root = '/home/summer/endovis/data/multitask/endovis2017_multitask_fold0'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMultiTaskAnnotations'),
    dict(type='RandomResize', scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackMultiTaskSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1088), keep_ratio=False),
    dict(type='LoadMultiTaskAnnotations'),
    dict(type='PackMultiTaskSegInputs')
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
        data_prefix=dict(img_path='images/train'),
        pipeline=train_pipeline
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
        data_prefix=dict(img_path='images/val'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

# Model: 注意力模块 + 类别权重 + 调整任务权重
model = dict(
    type='MultiTaskEncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size
    ),
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False
    ),
    # Binary head: SHSA + 提高任务权重
    decode_head_binary=dict(
        type='FCNSHSAHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        shsa_qk_dim=16,
        shsa_residual=True,
        shsa_downsample=8,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.5  # 提高权重 1.0 -> 1.5
        )
    ),
    # Parts head: PPA + 类别权重 + 提高任务权重
    decode_head_parts=dict(
        type='FCNPPAHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        ppa_filters=64,
        ppa_downsample=8,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,  # 提高权重 1.5 -> 2.0
            class_weight=[1.0, 6.67, 9.03, 5.65]
        )
    ),
    # Type head: LRSA + 类别权重
    decode_head_type=dict(
        type='FCNLRSAHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        lrsa_num_heads=4,
        lrsa_pooled_sizes=[8, 6, 4, 2],
        lrsa_q_pooled_size=8,
        lrsa_residual=True,
        lrsa_downsample=8,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.5,  # 降低权重 2.0 -> 1.5 (已经表现好)
            class_weight=[1.0, 7.14, 6.50, 6.15, 11.56, 20.0, 10.93, 8.66]
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01)
)

# Learning rate scheduler - 100k iterations
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=2000, end=100000, by_epoch=False)
]

# Training settings - 100k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=200000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/multitask/endovis2017_multitask_combined_fold0'
