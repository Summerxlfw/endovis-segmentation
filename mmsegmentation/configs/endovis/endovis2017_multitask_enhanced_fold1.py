_base_ = ['./endovis2017_multitask_enhanced_fold0.py']

# Override data root and work dir for fold 1
data_root = '/home/summer/endovis/data/multitask/endovis2017_multitask_fold1'

train_dataloader = dict(
    dataset=dict(data_root=data_root)
)

val_dataloader = dict(
    dataset=dict(data_root=data_root)
)

test_dataloader = dict(
    dataset=dict(data_root=data_root)
)

work_dir = './work_dirs/multitask/endovis2017_multitask_enhanced_fold1'
