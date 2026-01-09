#!/usr/bin/env python
"""Verify evaluation metrics using standard accumulation method."""

import os
import sys
sys.path.insert(0, '/home/summer/endovis/mmsegmentation')

import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmseg.apis import init_model
import torch.nn.functional as F


def evaluate_fold_accurate(fold_idx, model_type='weighted'):
    """Evaluate using accumulation method (standard IoU calculation)."""

    if model_type == 'weighted':
        config_path = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_weighted_fold{fold_idx}.py'
        checkpoint_path = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_weighted_fold{fold_idx}/iter_80000.pth'
    elif model_type == 'enhanced':
        config_path = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_enhanced_fold{fold_idx}.py'
        checkpoint_path = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_enhanced_fold{fold_idx}/iter_40000.pth'
    elif model_type == 'baseline':
        config_path = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_official_fold{fold_idx}.py'
        checkpoint_path = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_official_fold{fold_idx}/iter_40000.pth'

    data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold_idx}'

    print(f"\nEvaluating Fold {fold_idx} ({model_type})...")

    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    # Accumulators for intersection and union
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

            # Resize to model input size (1920, 1088)
            img_resized = cv2.resize(img_rgb, (1920, 1088))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).cuda()

            # Normalize
            mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
            img_tensor = (img_tensor - mean) / std

            # Forward pass
            feats = model.backbone(img_tensor)
            binary_logits = model.decode_head_binary(feats)
            parts_logits = model.decode_head_parts(feats)
            type_logits = model.decode_head_type(feats)

            # Resize predictions to original size
            binary_pred = F.interpolate(binary_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]
            parts_pred = F.interpolate(parts_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]
            type_pred = F.interpolate(type_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False).argmax(dim=1).cpu().numpy()[0]

            # Load ground truth
            binary_gt = cv2.imread(os.path.join(data_root, 'annotations', 'binary', 'val', img_name), cv2.IMREAD_GRAYSCALE)
            parts_gt = cv2.imread(os.path.join(data_root, 'annotations', 'parts', 'val', img_name), cv2.IMREAD_GRAYSCALE)
            type_gt = cv2.imread(os.path.join(data_root, 'annotations', 'type', 'val', img_name), cv2.IMREAD_GRAYSCALE)

            # Accumulate for Binary
            for c in range(2):
                pred_c = (binary_pred == c)
                gt_c = (binary_gt == c)
                binary_intersection[c] += (pred_c & gt_c).sum()
                binary_union[c] += (pred_c | gt_c).sum()

            # Accumulate for Parts (ignore 255)
            valid_parts = (parts_gt != 255)
            for c in range(4):
                pred_c = (parts_pred == c) & valid_parts
                gt_c = (parts_gt == c)
                parts_intersection[c] += (pred_c & gt_c).sum()
                parts_union[c] += (pred_c | gt_c).sum()

            # Accumulate for Type (ignore 255)
            valid_type = (type_gt != 255)
            for c in range(8):
                pred_c = (type_pred == c) & valid_type
                gt_c = (type_gt == c)
                type_intersection[c] += (pred_c & gt_c).sum()
                type_union[c] += (pred_c | gt_c).sum()

    # Compute IoU
    binary_iou = binary_intersection / (binary_union + 1e-10) * 100
    parts_iou = parts_intersection / (parts_union + 1e-10) * 100
    type_iou = type_intersection / (type_union + 1e-10) * 100

    return {
        'binary_iou': binary_iou,
        'binary_miou': binary_iou.mean(),
        'parts_iou': parts_iou,
        'parts_miou': parts_iou.mean(),
        'type_iou': type_iou,
        'type_miou': type_iou.mean(),
    }


def main():
    print("=" * 70)
    print("验证评估指标准确性 (使用累加法)")
    print("=" * 70)

    # Evaluate weighted model (all 4 folds)
    print("\n" + "=" * 70)
    print("加权多任务模型 (80k iterations)")
    print("=" * 70)

    weighted_results = []
    for fold in range(4):
        result = evaluate_fold_accurate(fold, 'weighted')
        weighted_results.append(result)
        print(f"  Fold {fold}: Binary={result['binary_miou']:.2f}%, Parts={result['parts_miou']:.2f}%, Type={result['type_miou']:.2f}%")

    # Average
    avg_binary = np.mean([r['binary_miou'] for r in weighted_results])
    avg_parts = np.mean([r['parts_miou'] for r in weighted_results])
    avg_type = np.mean([r['type_miou'] for r in weighted_results])
    print(f"\n  平均: Binary={avg_binary:.2f}%, Parts={avg_parts:.2f}%, Type={avg_type:.2f}%")

    # Evaluate enhanced model (all 4 folds)
    print("\n" + "=" * 70)
    print("增强多任务模型 (40k iterations)")
    print("=" * 70)

    enhanced_results = []
    for fold in range(4):
        result = evaluate_fold_accurate(fold, 'enhanced')
        enhanced_results.append(result)
        print(f"  Fold {fold}: Binary={result['binary_miou']:.2f}%, Parts={result['parts_miou']:.2f}%, Type={result['type_miou']:.2f}%")

    avg_binary_e = np.mean([r['binary_miou'] for r in enhanced_results])
    avg_parts_e = np.mean([r['parts_miou'] for r in enhanced_results])
    avg_type_e = np.mean([r['type_miou'] for r in enhanced_results])
    print(f"\n  平均: Binary={avg_binary_e:.2f}%, Parts={avg_parts_e:.2f}%, Type={avg_type_e:.2f}%")

    # Evaluate baseline model (all 4 folds)
    print("\n" + "=" * 70)
    print("基线多任务模型 (40k iterations)")
    print("=" * 70)

    baseline_results = []
    for fold in range(4):
        try:
            result = evaluate_fold_accurate(fold, 'baseline')
            baseline_results.append(result)
            print(f"  Fold {fold}: Binary={result['binary_miou']:.2f}%, Parts={result['parts_miou']:.2f}%, Type={result['type_miou']:.2f}%")
        except Exception as e:
            print(f"  Fold {fold}: Error - {e}")

    if baseline_results:
        avg_binary_b = np.mean([r['binary_miou'] for r in baseline_results])
        avg_parts_b = np.mean([r['parts_miou'] for r in baseline_results])
        avg_type_b = np.mean([r['type_miou'] for r in baseline_results])
        print(f"\n  平均: Binary={avg_binary_b:.2f}%, Parts={avg_parts_b:.2f}%, Type={avg_type_b:.2f}%")

    # Summary comparison
    print("\n" + "=" * 70)
    print("最终对比总结")
    print("=" * 70)
    print(f"\n{'模型':<20} {'Binary':<12} {'Parts':<12} {'Type':<12}")
    print("-" * 56)
    if baseline_results:
        print(f"{'基线多任务':<20} {avg_binary_b:.2f}%{'':<6} {avg_parts_b:.2f}%{'':<6} {avg_type_b:.2f}%")
    print(f"{'增强多任务':<20} {avg_binary_e:.2f}%{'':<6} {avg_parts_e:.2f}%{'':<6} {avg_type_e:.2f}%")
    print(f"{'加权多任务':<20} {avg_binary:.2f}%{'':<6} {avg_parts:.2f}%{'':<6} {avg_type:.2f}%")

    # Per-class analysis for weighted model
    print("\n" + "=" * 70)
    print("加权多任务模型 - 各类别IoU详情")
    print("=" * 70)

    parts_classes = ['Background', 'Shaft', 'Wrist', 'Clasper']
    type_classes = ['Background', 'Bipolar', 'Prograsp', 'LargeNeedle', 'Vessel', 'Grasping', 'Monopolar', 'Other']

    print("\nParts:")
    avg_parts_iou = np.mean([r['parts_iou'] for r in weighted_results], axis=0)
    for i, name in enumerate(parts_classes):
        print(f"  {name:<15}: {avg_parts_iou[i]:.2f}%")

    print("\nType:")
    avg_type_iou = np.mean([r['type_iou'] for r in weighted_results], axis=0)
    for i, name in enumerate(type_classes):
        print(f"  {name:<15}: {avg_type_iou[i]:.2f}%")


if __name__ == '__main__':
    main()
