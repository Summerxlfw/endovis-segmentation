#!/usr/bin/env python
"""Complete verification of all model evaluation metrics."""

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
    """Evaluate multi-task model using accumulation method."""
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


def evaluate_singletask_model(config_path, checkpoint_path, data_root, num_classes, device='cuda:0'):
    """Evaluate single-task model using accumulation method."""
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device=device)
    model.eval()

    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

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
            logits = model.decode_head(feats)
            pred = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]

            # Single-task data has annotations in annotations/val directly
            gt = cv2.imread(os.path.join(data_root, 'annotations', 'val', img_name), cv2.IMREAD_GRAYSCALE)

            valid = (gt != 255)
            for c in range(num_classes):
                pred_c = (pred == c) & valid
                gt_c = (gt == c)
                intersection[c] += (pred_c & gt_c).sum()
                union[c] += (pred_c | gt_c).sum()

    iou = intersection / (union + 1e-10) * 100
    return {'miou': iou.mean(), 'iou': iou}


def main():
    print("=" * 70)
    print("完整验证: 所有模型评估指标 (累加法)")
    print("=" * 70)

    results = {}

    # 1. Single-task models
    print("\n" + "=" * 70)
    print("单任务模型")
    print("=" * 70)

    single_binary = []
    single_parts = []
    single_type = []

    for fold in range(4):
        # Binary
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_binary_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/single_task/binary/endovis2017_binary_fold{fold}/iter_40000.pth'
        data_root = f'/home/summer/endovis/data/binary/endovis2017_binary_fold{fold}'
        if os.path.exists(ckpt):
            r = evaluate_singletask_model(config, ckpt, data_root, 2)
            single_binary.append(r['miou'])
            print(f"  Binary Fold {fold}: {r['miou']:.2f}%")

        # Parts
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_parts_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/single_task/parts/endovis2017_parts_fold{fold}_fixed/iter_40000.pth'
        data_root = f'/home/summer/endovis/data/parts/endovis2017_parts_fold{fold}'
        if os.path.exists(ckpt):
            r = evaluate_singletask_model(config, ckpt, data_root, 4)
            single_parts.append(r['miou'])
            print(f"  Parts Fold {fold}: {r['miou']:.2f}%")

        # Type
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_type_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/single_task/type/endovis2017_type_fold{fold}_fixed/iter_40000.pth'
        data_root = f'/home/summer/endovis/data/type/endovis2017_type_fold{fold}'
        if os.path.exists(ckpt):
            r = evaluate_singletask_model(config, ckpt, data_root, 8)
            single_type.append(r['miou'])
            print(f"  Type Fold {fold}: {r['miou']:.2f}%")

    if single_binary:
        results['single_binary'] = np.mean(single_binary)
        print(f"  Binary: {results['single_binary']:.2f}%")
    if single_parts:
        results['single_parts'] = np.mean(single_parts)
        print(f"  Parts:  {results['single_parts']:.2f}%")
    if single_type:
        results['single_type'] = np.mean(single_type)
        print(f"  Type:   {results['single_type']:.2f}%")

    # 2. Baseline multi-task model
    print("\n" + "=" * 70)
    print("基线多任务模型 (40k)")
    print("=" * 70)

    baseline_results = []
    for fold in range(4):
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_fold{fold}/iter_40000.pth'
        data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold}'
        if os.path.exists(ckpt):
            print(f"  Evaluating Fold {fold}...")
            r = evaluate_multitask_model(config, ckpt, data_root)
            baseline_results.append(r)
            print(f"    Binary={r['binary_miou']:.2f}%, Parts={r['parts_miou']:.2f}%, Type={r['type_miou']:.2f}%")

    if baseline_results:
        results['baseline_binary'] = np.mean([r['binary_miou'] for r in baseline_results])
        results['baseline_parts'] = np.mean([r['parts_miou'] for r in baseline_results])
        results['baseline_type'] = np.mean([r['type_miou'] for r in baseline_results])
        print(f"\n  平均: Binary={results['baseline_binary']:.2f}%, Parts={results['baseline_parts']:.2f}%, Type={results['baseline_type']:.2f}%")

    # 3. Enhanced multi-task model
    print("\n" + "=" * 70)
    print("增强多任务模型 (40k + 注意力)")
    print("=" * 70)

    enhanced_results = []
    for fold in range(4):
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_enhanced_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_enhanced_fold{fold}/iter_40000.pth'
        data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold}'
        if os.path.exists(ckpt):
            print(f"  Evaluating Fold {fold}...")
            r = evaluate_multitask_model(config, ckpt, data_root)
            enhanced_results.append(r)
            print(f"    Binary={r['binary_miou']:.2f}%, Parts={r['parts_miou']:.2f}%, Type={r['type_miou']:.2f}%")

    if enhanced_results:
        results['enhanced_binary'] = np.mean([r['binary_miou'] for r in enhanced_results])
        results['enhanced_parts'] = np.mean([r['parts_miou'] for r in enhanced_results])
        results['enhanced_type'] = np.mean([r['type_miou'] for r in enhanced_results])
        print(f"\n  平均: Binary={results['enhanced_binary']:.2f}%, Parts={results['enhanced_parts']:.2f}%, Type={results['enhanced_type']:.2f}%")

    # 4. Weighted multi-task model
    print("\n" + "=" * 70)
    print("加权多任务模型 (80k + 类别权重)")
    print("=" * 70)

    weighted_results = []
    for fold in range(4):
        config = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_weighted_fold{fold}.py'
        ckpt = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_weighted_fold{fold}/iter_80000.pth'
        data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold}'
        if os.path.exists(ckpt):
            print(f"  Evaluating Fold {fold}...")
            r = evaluate_multitask_model(config, ckpt, data_root)
            weighted_results.append(r)
            print(f"    Binary={r['binary_miou']:.2f}%, Parts={r['parts_miou']:.2f}%, Type={r['type_miou']:.2f}%")

    if weighted_results:
        results['weighted_binary'] = np.mean([r['binary_miou'] for r in weighted_results])
        results['weighted_parts'] = np.mean([r['parts_miou'] for r in weighted_results])
        results['weighted_type'] = np.mean([r['type_miou'] for r in weighted_results])
        print(f"\n  平均: Binary={results['weighted_binary']:.2f}%, Parts={results['weighted_parts']:.2f}%, Type={results['weighted_type']:.2f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("最终对比汇总 (累加法IoU)")
    print("=" * 70)
    print(f"\n{'模型':<25} {'Binary':<12} {'Parts':<12} {'Type':<12}")
    print("-" * 61)

    if 'single_binary' in results:
        print(f"{'单任务':<25} {results.get('single_binary', 0):.2f}%{'':<6} {results.get('single_parts', 0):.2f}%{'':<6} {results.get('single_type', 0):.2f}%")
    if 'baseline_binary' in results:
        print(f"{'基线多任务 (40k)':<25} {results['baseline_binary']:.2f}%{'':<6} {results['baseline_parts']:.2f}%{'':<6} {results['baseline_type']:.2f}%")
    if 'enhanced_binary' in results:
        print(f"{'增强多任务 (40k+注意力)':<25} {results['enhanced_binary']:.2f}%{'':<6} {results['enhanced_parts']:.2f}%{'':<6} {results['enhanced_type']:.2f}%")
    if 'weighted_binary' in results:
        print(f"{'加权多任务 (80k+权重)':<25} {results['weighted_binary']:.2f}%{'':<6} {results['weighted_parts']:.2f}%{'':<6} {results['weighted_type']:.2f}%")


if __name__ == '__main__':
    main()
