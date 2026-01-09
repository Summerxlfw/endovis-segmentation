# EndoVis 2017 增强型多任务分割模型实验报告

## 实验日期
2024-12-26

---

## 一、性能对比

### 1.1 完整对比表

| 任务 | 单任务 | 基线多任务 | 增强多任务 | 增强 vs 基线 | 增强 vs 单任务 |
|------|--------|------------|------------|--------------|----------------|
| **Binary** | 92.33% | 91.10% | 90.92% | -0.18% | -1.41% |
| **Parts** | 47.94% | 42.99% | 42.29% | -0.70% | -5.65% |
| **Type** | 20.86% | 19.15% | **30.04%** | **+10.89%** | **+9.18%** |

### 1.2 各 Fold 详细结果

#### 增强多任务模型 (Enhanced Multi-task)

| Fold | Binary mIoU | Binary mDice | Parts mIoU | Parts mDice | Type mIoU | Type mDice |
|------|-------------|--------------|------------|-------------|-----------|------------|
| Fold 0 | 92.62% | 95.82% | 43.45% | 49.12% | 25.07% | 26.82% |
| Fold 1 | 90.19% | 93.56% | 48.08% | 55.03% | 44.06% | 46.36% |
| Fold 2 | 92.16% | 95.63% | 39.03% | 46.23% | 26.04% | 28.93% |
| Fold 3 | 88.70% | 93.48% | 38.62% | 45.02% | 24.97% | 28.16% |
| **平均** | **90.92%** | **94.62%** | **42.29%** | **48.85%** | **30.04%** | **32.57%** |

#### 基线多任务模型 (Baseline Multi-task)

| Fold | Binary mIoU | Parts mIoU | Type mIoU |
|------|-------------|------------|-----------|
| Fold 0 | 92.68% | 44.11% | 18.72% |
| Fold 1 | 89.31% | 48.36% | 21.72% |
| Fold 2 | 92.46% | 39.78% | 17.75% |
| Fold 3 | 89.93% | 39.70% | 18.40% |
| **平均** | **91.10%** | **42.99%** | **19.15%** |

#### 单任务模型 (Single-task)

| Fold | Binary mIoU | Parts mIoU | Type mIoU |
|------|-------------|------------|-----------|
| Fold 0 | 93.86% | 48.79% | 20.35% |
| Fold 1 | 90.68% | 52.52% | 23.66% |
| Fold 2 | 93.45% | 45.31% | 19.40% |
| Fold 3 | 91.31% | 45.12% | 20.02% |
| **平均** | **92.33%** | **47.94%** | **20.86%** |

### 1.3 总体评估

| 指标 | 单任务 | 基线多任务 | 增强多任务 |
|------|--------|------------|------------|
| mIoU 总和 | 161.13% | 153.24% | **163.25%** |
| 平均 mIoU | 53.71% | 51.08% | **54.42%** |
| 模型数量 | 3个 | 1个 | 1个 |

---

## 二、改进方法

### 2.1 整体架构

基于 UNet 编码器的多任务分割模型，为每个任务的解码头添加针对性的注意力模块：

```
                    ┌─────────────────┐
                    │  UNet Encoder   │
                    │   (共享特征)     │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ FCN + SHSA   │  │ FCN + PPA    │  │ FCN + LRSA   │
    │ (Binary)     │  │ (Parts)      │  │ (Type)       │
    └──────────────┘  └──────────────┘  └──────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
      2类输出            4类输出            8类输出
```

### 2.2 注意力模块详解

#### 2.2.1 SHSA (Single-Head Self-Attention) - Binary 任务

**来源**: `/home/summer/endovis/curated_modules/Transformer/SHSA/DEIM/SHSA.py`

**原理**:
- 将通道分为两部分：一部分进行注意力计算，另一部分保持不变
- 轻量级设计，适合边界增强

**配置**:
```python
decode_head_binary=dict(
    type='FCNSHSAHead',
    shsa_qk_dim=16,        # QK 降维维度
    shsa_residual=True,    # 残差连接
    shsa_downsample=8,     # 下采样倍数 (内存优化)
)
```

#### 2.2.2 PPA (Pyramid Pooling Attention) - Parts 任务

**来源**: `/home/summer/endovis/curated_modules/Module/hcfnet/YOLO11/hcfnet.py`

**原理**:
- 结合多尺度金字塔池化
- 包含 LocalGlobalAttention (patch_size 2和4)
- ECA 通道注意力
- 空间注意力模块

**配置**:
```python
decode_head_parts=dict(
    type='FCNPPAHead',
    ppa_filters=64,        # 中间特征维度
    ppa_downsample=8,      # 下采样倍数
)
```

#### 2.2.3 LRSA (Low-Resolution Self-Attention) - Type 任务

**来源**: `/home/summer/endovis/curated_modules/Transformer/LRSA/DEIM/LRSA.py`

**原理**:
- 多尺度池化提取全局语义特征
- 低分辨率计算注意力，降低计算量
- 适合全局语义理解任务

**配置**:
```python
decode_head_type=dict(
    type='FCNLRSAHead',
    lrsa_num_heads=4,                    # 注意力头数
    lrsa_pooled_sizes=[8, 6, 4, 2],      # 多尺度池化大小
    lrsa_q_pooled_size=8,                # Query 池化大小
    lrsa_residual=True,                  # 残差连接
    lrsa_downsample=8,                   # 下采样倍数
)
```

### 2.3 损失权重调整

针对不同任务难度设置不同损失权重：

| 任务 | 损失权重 | 原因 |
|------|----------|------|
| Binary | 1.0 | 简单任务，保持标准权重 |
| Parts | 1.5 | 中等难度，适当增加权重 |
| Type | 2.0 | 困难任务，最高权重 |

### 2.4 内存优化策略

原始注意力模块在 512×512 分辨率下会产生 262144×262144 的注意力矩阵，导致 OOM。

**解决方案**: 添加 `downsample` 参数
- 先将特征图下采样 8 倍 (512→64)
- 在低分辨率下计算注意力
- 再上采样回原始分辨率

```python
def _forward_feature(self, inputs):
    feats = super()._forward_feature(inputs)
    H, W = feats.shape[2:]

    # 下采样
    if self.downsample > 1:
        feats_down = F.adaptive_avg_pool2d(feats, (H // self.downsample, W // self.downsample))

    # 注意力计算
    attn_out = self.attention(feats_down)

    # 上采样
    if self.downsample > 1:
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

    return feats + attn_out  # 残差连接
```

---

## 三、文件结构

### 3.1 新增文件

```
mmseg/models/
├── plugins/
│   ├── __init__.py
│   ├── shsa.py          # SHSA 注意力模块
│   ├── ppa.py           # PPA 模块 (含子组件)
│   └── lrsa.py          # LRSA 注意力模块
└── decode_heads/
    ├── fcn_shsa_head.py  # FCN + SHSA 解码头
    ├── fcn_ppa_head.py   # FCN + PPA 解码头
    └── fcn_lrsa_head.py  # FCN + LRSA 解码头

configs/endovis/
├── endovis2017_multitask_enhanced_fold0.py
├── endovis2017_multitask_enhanced_fold1.py
├── endovis2017_multitask_enhanced_fold2.py
└── endovis2017_multitask_enhanced_fold3.py

evaluation/
└── evaluate_multitask_enhanced.py
```

### 3.2 模型检查点

```
work_dirs/multitask/
├── endovis2017_multitask_enhanced_fold0/iter_40000.pth
├── endovis2017_multitask_enhanced_fold1/iter_40000.pth
├── endovis2017_multitask_enhanced_fold2/iter_40000.pth
└── endovis2017_multitask_enhanced_fold3/iter_40000.pth
```

---

## 四、训练配置

```python
# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01)
)

# 学习率策略
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# 训练轮数
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
```

---

## 五、结论

1. **Type 分割显著提升**: LRSA 模块带来 +10.89% mIoU 提升，验证了全局语义理解对器械类型识别的重要性

2. **多任务整体超越单任务**: 增强多任务平均 mIoU (54.42%) > 单任务平均 mIoU (53.71%)

3. **效率提升**: 单模型实现三任务，减少训练和推理资源

4. **LRSA 效果最佳**: 相比 SHSA 和 PPA，LRSA 对其目标任务 (Type) 的提升最为显著
