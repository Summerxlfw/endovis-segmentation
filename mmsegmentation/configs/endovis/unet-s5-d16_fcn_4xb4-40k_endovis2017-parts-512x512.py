_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/endovis2017_parts.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
