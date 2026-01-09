# EndoVis 项目进度记录

更新时间：2026-01-08

## 1. 当前状态（已完成/可复现）

- 已完成 EndoVis2017 **单任务**与**多任务**（含增强版）4 折训练与评估闭环，并产出汇总结果与权重文件。
- 训练与评估基于 `mmsegmentation`（仓库内存在未提交改动；已导出补丁与状态信息到 `archives/`）。
- EndoVis2018 当前仅看到原始数据目录 `data/raw/endovis2018`，尚未纳入训练/评估流水线。

## 2. 数据准备

- 原始数据：
  - EndoVis2017：`data/raw/endovis2017`
  - EndoVis2018：`data/raw/endovis2018`
- 处理后数据集（已生成）：
  - 单任务（4 折）：`data/binary/`、`data/parts/`、`data/type/`
  - 官方划分：`data/official/`
  - 多任务（4 折）：`data/multitask/`
- 相关脚本（根目录）：`scripts/prepare_official_split.py`、`scripts/prepare_4fold_split_fixed.py`、`scripts/prepare_multitask_dataset.py`

## 3. 训练/评估设置（摘要）

- 模型：UNet（FCN-UNet-S5-D16）
- 训练：40,000 iterations（4 折）
- 评估：mIoU / mDice（详见 `mmsegmentation/evaluation/*.txt` 与 `results/*.md`）

## 4. 主要结果（4 折平均）

来源：`results/evaluation_comparison.md`、`mmsegmentation/evaluation/single_task_results.txt`、`mmsegmentation/evaluation/evaluation_results.txt`、`mmsegmentation/evaluation/enhanced_multitask_results.txt`

| 设置 | Binary mIoU / mDice | Parts mIoU / mDice | Type mIoU / mDice |
|---|---:|---:|---:|
| 单任务（Single-task） | 92.33 / 95.44 | 47.94 / 54.93 | 20.86 / 23.68 |
| 多任务基线（Baseline multi-task） | 91.10 / 94.68 | 42.99 / 49.48 | 19.15 / 21.45 |
| 增强多任务（Enhanced multi-task） | 90.92 / 94.62 | 42.29 / 48.85 | 30.04 / 32.57 |

## 5. 关键代码与配置产出

- 配置：`mmsegmentation/configs/endovis/`、`mmsegmentation/configs/_base_/datasets/endovis2017_*.py`
- 数据集实现：`mmsegmentation/mmseg/datasets/endovis.py`、`mmsegmentation/mmseg/datasets/endovis_multitask.py`
- 多任务模型：`mmsegmentation/mmseg/models/segmentors/multitask_encoder_decoder.py`
- 增强解码头/插件：`mmsegmentation/mmseg/models/decode_heads/fcn_*_head.py`、`mmsegmentation/mmseg/models/plugins/`
- 评估脚本：`mmsegmentation/evaluation/`

## 6. 训练产物位置

- 单任务权重：`mmsegmentation/work_dirs/single_task/**/iter_40000.pth`
- 多任务权重：`mmsegmentation/work_dirs/multitask/**/iter_40000.pth`
- 根目录训练日志：`logs/*.log`
- 结果汇总：`results/*.md`、`mmsegmentation/docs/enhanced_multitask_results.md`

## 7. 归档/打包说明

- 归档元信息与补丁：`archives/ENVIRONMENT.txt`、`archives/pip_freeze.txt`、`archives/mmsegmentation_git.txt`、`archives/mmsegmentation_diff.patch`
- 权重清单：`archives/checkpoints_iter_40000.tsv`
- 归档压缩包：
  - 轻量记录（不含数据集 `data/`）：`archives/endovis_record_20260108_193658.tar.gz`
  - 完整记录（包含数据集 `data/`）：`archives/endovis_record_with_data_20260108_201515.tar.gz`
- 符号链接说明：`data/multitask` 内存在大量图片绝对符号链接（指向 `/home/summer/endovis/...`），为保证压缩包解压到任意位置后仍可用，已将其改写为指向 `data/raw` 的相对链接。
  - 修复脚本：`scripts/relativize_multitask_symlinks.py`
  - 修复日志：`archives/relativize_multitask_symlinks_20260108_201444.log`
