# EndoVis 医学图像分割项目

基于 MMSegmentation 框架的 EndoVis 手术器械分割项目，支持单任务和多任务学习。

## 项目概述

本项目实现了 EndoVis2017 数据集的手术器械分割任务，包括：
- **Binary Segmentation**: 二值分割（器械 vs 背景）
- **Parts Segmentation**: 器械部件分割
- **Type Segmentation**: 器械类型分割

支持单任务学习和多任务学习两种训练模式，并提供增强版多任务模型。

## 项目结构

```
.
├── archives/              # 归档文件和元数据
│   ├── ENVIRONMENT.txt    # 环境信息
│   ├── pip_freeze.txt     # Python 依赖
│   ├── mmsegmentation_diff.patch  # MMSeg 修改补丁
│   └── checkpoints_iter_40000.tsv # 权重清单
├── curated_modules/       # 精选模块
├── data/                  # 数据集目录
│   ├── raw/              # 原始数据
│   │   ├── endovis2017/
│   │   └── endovis2018/
│   ├── binary/           # 二值分割数据（4折）
│   ├── parts/            # 部件分割数据（4折）
│   ├── type/             # 类型分割数据（4折）
│   ├── official/         # 官方划分
│   └── multitask/        # 多任务数据（4折）
├── logs/                  # 训练日志
├── mmsegmentation/        # MMSegmentation 框架
│   ├── configs/endovis/  # EndoVis 配置文件
│   ├── mmseg/datasets/   # 数据集实现
│   ├── mmseg/models/     # 模型实现
│   └── evaluation/       # 评估脚本和结果
├── results/               # 结果汇总
├── scripts/               # 数据准备脚本
│   ├── prepare_official_split.py
│   ├── prepare_4fold_split_fixed.py
│   ├── prepare_multitask_dataset.py
│   └── relativize_multitask_symlinks.py
└── PROGRESS.md           # 项目进度记录
```

## 主要结果

基于 UNet (FCN-UNet-S5-D16) 模型，训练 40,000 iterations，4折交叉验证平均结果：

| 设置 | Binary mIoU / mDice | Parts mIoU / mDice | Type mIoU / mDice |
|---|---:|---:|---:|
| 单任务 (Single-task) | 92.33 / 95.44 | 47.94 / 54.93 | 20.86 / 23.68 |
| 多任务基线 (Baseline multi-task) | 91.10 / 94.68 | 42.99 / 49.48 | 19.15 / 21.45 |
| 增强多任务 (Enhanced multi-task) | 90.92 / 94.62 | 42.29 / 48.85 | 30.04 / 32.57 |

详细结果见 `results/evaluation_comparison.md`

## 环境配置

### 依赖安装

```bash
# 安装 MMSegmentation 依赖
cd mmsegmentation
pip install -r requirements.txt
pip install -e .
```

完整依赖列表见 `archives/pip_freeze.txt`

### 数据准备

1. 下载 EndoVis2017 数据集到 `data/raw/endovis2017/`
2. 运行数据准备脚本：

```bash
# 准备官方划分
python scripts/prepare_official_split.py

# 准备4折交叉验证数据
python scripts/prepare_4fold_split_fixed.py

# 准备多任务数据集
python scripts/prepare_multitask_dataset.py
```

## 训练

### 单任务训练

```bash
cd mmsegmentation

# Binary 分割
python tools/train.py configs/endovis/fcn_unet_s5-d16_binary_fold0.py

# Parts 分割
python tools/train.py configs/endovis/fcn_unet_s5-d16_parts_fold0.py

# Type 分割
python tools/train.py configs/endovis/fcn_unet_s5-d16_type_fold0.py
```

### 多任务训练

```bash
cd mmsegmentation

# 基线多任务
python tools/train.py configs/endovis/fcn_unet_s5-d16_multitask_fold0.py

# 增强多任务
python tools/train.py configs/endovis/fcn_unet_s5-d16_multitask_enhanced_fold0.py
```

## 评估

```bash
cd mmsegmentation

# 单任务评估
python evaluation/evaluate_single_task.py

# 多任务评估
python evaluation/evaluate_multitask.py

# 增强多任务评估
python evaluation/evaluate_enhanced_multitask.py
```

## 关键实现

### 数据集
- `mmsegmentation/mmseg/datasets/endovis.py` - 单任务数据集
- `mmsegmentation/mmseg/datasets/endovis_multitask.py` - 多任务数据集

### 模型
- `mmsegmentation/mmseg/models/segmentors/multitask_encoder_decoder.py` - 多任务模型
- `mmsegmentation/mmseg/models/decode_heads/fcn_*_head.py` - 解码头
- `mmsegmentation/mmseg/models/plugins/` - 增强插件

### 配置
- `mmsegmentation/configs/endovis/` - 训练配置
- `mmsegmentation/configs/_base_/datasets/endovis2017_*.py` - 数据集配置

## 权重文件

训练好的权重文件位于：
- 单任务: `mmsegmentation/work_dirs/single_task/**/iter_40000.pth`
- 多任务: `mmsegmentation/work_dirs/multitask/**/iter_40000.pth`

权重清单见 `archives/checkpoints_iter_40000.tsv`

## 归档说明

项目已创建完整归档，包含：
- 轻量记录（不含数据集）: `archives/endovis_record_20260108_193658.tar.gz`
- 完整记录（含数据集）: `archives/endovis_record_with_data_20260108_201515.tar.gz`

归档包含环境信息、代码补丁、权重清单等元数据，可完整复现实验。

## 注意事项

1. `data/multitask` 中的符号链接已转换为相对路径，确保项目可移植性
2. MMSegmentation 仓库存在未提交改动，补丁文件见 `archives/mmsegmentation_diff.patch`
3. EndoVis2018 数据集尚未纳入训练流程

## 更新日志

详见 `PROGRESS.md`

## 许可证

本项目基于 MMSegmentation 框架开发，遵循其开源许可证。
