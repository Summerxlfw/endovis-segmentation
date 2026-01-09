#!/usr/bin/env python
"""Evaluate combined multi-task model."""

import os
import sys
sys.path.insert(0, '/home/summer/endovis/mmsegmentation')

import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmseg.apis import init_model
import torch.nn.functional as F


def evaluate_multitask_model(config_path, checkpoint_path, data_root, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device=device)
    model.eval()

    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    binary_intersection = np.zeros(2, dtype=np.int64)
    binary_union = np.zeros(2, dtype=np.int64)
    parts_intersection = np.zeros(4, dtype=np.int64)
    parts_union = np.zeros(4, dtype=np.int64)
    type_intersection = np.zeros(8, dtype=np.int64)
    type_union = np.zeros(8, dtype=np.int64)

    with torch.no_grad():
        for img_name in val_images:
            img = cv2.imread(os.path.join(val_img_dir, img_name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            img_resized = cv2.resize(img_rgb, (1920, 1088))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
            mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std

            feats = model.backbone(img_tensor)
            binary_logits = model.decode_head_binary(feats)
            parts_logits = model.decode_head_parts(feats)
            type_logits = model.decode_head_type(feats)

            binary_pred = F.interpolate(binary_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]
            parts_pred = F.interpolate(parts_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]
            type_pred = F.interpolate(type_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]

            binary_gt = cv2.imread(os.path.join(data_root, 'annotations', 'binary', 'val', img_name), cv2.IMREAD_GRAYSCALE)
            parts_gt = cv2.imread(os.path.join(data_root, 'annotations', 'parts', 'val', img_name), cv2.IMREAD_GRAYSCALE)
            type_gt = cv2.imread(os.path.join(data_root, 'annotations', 'type', 'val', img_name), cv2.IMREAD_GRAYSCALE)

            for c in range(2):
                pred_c = (binary_pred == c)
                gt_c = (binary_gt == c)
                binary_intersection[c] += (pred_c & gt_c).sum()
                binary_union[c] += (pred_c | gt_c).sum()

            valid_parts = (parts_gt != 255)
            for c in range(4):
                pred_c = (parts_pred == c) & valid_parts
                gt_c = (parts_gt == c)
                parts_intersection[c] += (pred_c & gt_c).sum()
                parts_union[c] += (pred_c | gt_c).sum()

            valid_type = (type_gt != 255)
            for c in range(8):
                pred_c = (type_pred == c) & valid_type
                gt_c = (type_gt == c)
                type_intersection[c] += (pred_c & gt_c).sum()
                type_union[c] += (pred_c | gt_c).sum()

    binary_iou = binary_intersection / (binary_union + 1e-10) * 100
    parts_iou = parts_intersection / (parts_union + 1e-10) * 100
    type_iou = type_intersection / (type_union + 1e-10) * 100

    return {
        'binary_miou': binary_iou.mean(),
        'parts_miou': parts_iou.mean(),
        'type_miou': type_iou.mean(),
        'binary_iou': binary_iou,
        'parts_iou': parts_iou,
        'type_iou': type_iou,
    }


def main():
    print('=' * 70)
    print('综合优化模型评估 (100k iterations)')
    print('=' * 70)

    results = []
    for fold in range(4):
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_combined_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_combined_fold{fold}/iter_100000.pth'
        data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold}'
        print(f'Evaluating Fold {fold}...')
        r = evaluate_multitask_model(config, ckpt, data_root)
        results.append(r)
        print(f'  Binary={r["binary_miou"]:.2f}%, Parts={r["parts_miou"]:.2f}%, Type={r["type_miou"]:.2f}%')

    avg_binary = np.mean([r['binary_miou'] for r in results])
    avg_parts = np.mean([r['parts_miou'] for r in results])
    avg_type = np.mean([r['type_miou'] for r in results])

    print()
    print('=' * 70)
    print('综合优化模型平均结果')
    print('=' * 70)
    print(f'Binary: {avg_binary:.2f}%')
    print(f'Parts:  {avg_parts:.2f}%')
    print(f'Type:   {avg_type:.2f}%')

    # 单任务基线 (从之前验证结果)
    single_binary = 92.83
    single_parts = 51.85
    single_type = 22.57

    print()
    print('=' * 70)
    print('与单任务对比')
    print('=' * 70)
    print(f'{"任务":<10} {"单任务":<12} {"综合多任务":<12} {"差异":<10} {"结果":<10}')
    print('-' * 54)

    binary_diff = avg_binary - single_binary
    parts_diff = avg_parts - single_parts
    type_diff = avg_type - single_type

    binary_result = "✓ 超越" if binary_diff >= 0 else "✗ 未超越"
    parts_result = "✓ 超越" if parts_diff >= 0 else "✗ 未超越"
    type_result = "✓ 超越" if type_diff >= 0 else "✗ 未超越"

    print(f'{"Binary":<10} {single_binary:.2f}%{"":<6} {avg_binary:.2f}%{"":<6} {binary_diff:+.2f}%{"":<4} {binary_result}')
    print(f'{"Parts":<10} {single_parts:.2f}%{"":<6} {avg_parts:.2f}%{"":<6} {parts_diff:+.2f}%{"":<4} {parts_result}')
    print(f'{"Type":<10} {single_type:.2f}%{"":<6} {avg_type:.2f}%{"":<6} {type_diff:+.2f}%{"":<4} {type_result}')

    # 与基线多任务对比
    baseline_binary = 91.42
    baseline_parts = 45.97
    baseline_type = 19.61

    print()
    print('=' * 70)
    print('与基线多任务对比')
    print('=' * 70)
    print(f'{"任务":<10} {"基线多任务":<12} {"综合多任务":<12} {"提升":<10}')
    print('-' * 44)
    print(f'{"Binary":<10} {baseline_binary:.2f}%{"":<6} {avg_binary:.2f}%{"":<6} {avg_binary-baseline_binary:+.2f}%')
    print(f'{"Parts":<10} {baseline_parts:.2f}%{"":<6} {avg_parts:.2f}%{"":<6} {avg_parts-baseline_parts:+.2f}%')
    print(f'{"Type":<10} {baseline_type:.2f}%{"":<6} {avg_type:.2f}%{"":<6} {avg_type-baseline_type:+.2f}%')

    # 全面超越判断
    print()
    print('=' * 70)
    if binary_diff >= 0 and parts_diff >= 0 and type_diff >= 0:
        print('✓✓✓ 全面超越单任务模型！')
    else:
        exceeded = sum([binary_diff >= 0, parts_diff >= 0, type_diff >= 0])
        print(f'超越 {exceeded}/3 个任务')


if __name__ == '__main__':
    main()
