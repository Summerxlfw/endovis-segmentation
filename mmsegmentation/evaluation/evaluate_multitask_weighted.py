#!/usr/bin/env python
"""Evaluate multi-task weighted loss models on EndoVis 2017 dataset."""

import os
import sys
sys.path.insert(0, '/home/summer/endovis/mmsegmentation')

import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmseg.apis import init_model
import torch.nn.functional as F

def compute_metrics(pred, gt, num_classes):
    """Compute IoU and Dice for each class."""
    pred = pred.flatten()
    gt = gt.flatten()

    # Ignore pixels with value 255
    valid = gt != 255
    pred = pred[valid]
    gt = gt[valid]

    iou_per_class = []
    dice_per_class = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()

        if union == 0:
            iou = float('nan')
            dice = float('nan')
        else:
            iou = intersection / union
            dice = 2 * intersection / (pred_cls.sum() + gt_cls.sum() + 1e-6)

        iou_per_class.append(iou)
        dice_per_class.append(dice)

    return iou_per_class, dice_per_class

def evaluate_fold(fold_idx):
    """Evaluate a single fold."""
    config_path = f'/home/summer/endovis/mmsegmentation/configs/endovis/endovis2017_multitask_weighted_fold{fold_idx}.py'
    checkpoint_path = f'/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_weighted_fold{fold_idx}/iter_80000.pth'
    data_root = f'/home/summer/endovis/data/multitask/endovis2017_multitask_fold{fold_idx}'

    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*60}")

    # Load model
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Get validation images
    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    # Annotation directories
    binary_ann_dir = os.path.join(data_root, 'annotations', 'binary', 'val')
    parts_ann_dir = os.path.join(data_root, 'annotations', 'parts', 'val')
    type_ann_dir = os.path.join(data_root, 'annotations', 'type', 'val')

    # Metrics accumulators
    binary_ious, binary_dices = [], []
    parts_ious, parts_dices = [], []
    type_ious, type_dices = [], []

    with torch.no_grad():
        for img_name in val_images:
            # Load image
            img_path = os.path.join(val_img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).cuda()

            # Normalize
            mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
            img_tensor = (img_tensor - mean) / std

            # Resize to model input size
            img_tensor = F.interpolate(img_tensor, size=(1088, 1920), mode='bilinear', align_corners=False)

            # Forward pass
            feats = model.backbone(img_tensor)

            # Get predictions for each task
            binary_logits = model.decode_head_binary(feats)
            parts_logits = model.decode_head_parts(feats)
            type_logits = model.decode_head_type(feats)

            # Resize predictions to original size
            orig_h, orig_w = img.shape[:2]
            binary_pred = F.interpolate(binary_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            parts_pred = F.interpolate(parts_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            type_pred = F.interpolate(type_logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            binary_pred = binary_pred.argmax(dim=1).cpu().numpy()[0]
            parts_pred = parts_pred.argmax(dim=1).cpu().numpy()[0]
            type_pred = type_pred.argmax(dim=1).cpu().numpy()[0]

            # Load ground truth
            binary_gt = cv2.imread(os.path.join(binary_ann_dir, img_name), cv2.IMREAD_GRAYSCALE)
            parts_gt = cv2.imread(os.path.join(parts_ann_dir, img_name), cv2.IMREAD_GRAYSCALE)
            type_gt = cv2.imread(os.path.join(type_ann_dir, img_name), cv2.IMREAD_GRAYSCALE)

            # Compute metrics
            iou, dice = compute_metrics(binary_pred, binary_gt, 2)
            binary_ious.append(iou)
            binary_dices.append(dice)

            iou, dice = compute_metrics(parts_pred, parts_gt, 4)
            parts_ious.append(iou)
            parts_dices.append(dice)

            iou, dice = compute_metrics(type_pred, type_gt, 8)
            type_ious.append(iou)
            type_dices.append(dice)

    # Aggregate results
    binary_ious = np.nanmean(binary_ious, axis=0)
    binary_dices = np.nanmean(binary_dices, axis=0)
    parts_ious = np.nanmean(parts_ious, axis=0)
    parts_dices = np.nanmean(parts_dices, axis=0)
    type_ious = np.nanmean(type_ious, axis=0)
    type_dices = np.nanmean(type_dices, axis=0)

    binary_miou = np.nanmean(binary_ious) * 100
    parts_miou = np.nanmean(parts_ious) * 100
    type_miou = np.nanmean(type_ious) * 100

    binary_mdice = np.nanmean(binary_dices) * 100
    parts_mdice = np.nanmean(parts_dices) * 100
    type_mdice = np.nanmean(type_dices) * 100

    print(f"\nFold {fold_idx} Results:")
    print(f"  Binary - mIoU: {binary_miou:.2f}%, mDice: {binary_mdice:.2f}%")
    print(f"  Parts  - mIoU: {parts_miou:.2f}%, mDice: {parts_mdice:.2f}%")
    print(f"  Type   - mIoU: {type_miou:.2f}%, mDice: {type_mdice:.2f}%")

    return {
        'binary_miou': binary_miou,
        'binary_mdice': binary_mdice,
        'parts_miou': parts_miou,
        'parts_mdice': parts_mdice,
        'type_miou': type_miou,
        'type_mdice': type_mdice,
        'binary_class_iou': (binary_ious * 100).tolist(),
        'parts_class_iou': (parts_ious * 100).tolist(),
        'type_class_iou': (type_ious * 100).tolist(),
    }

def main():
    print("="*60)
    print("EndoVis 2017 Multi-task Weighted Loss Model Evaluation")
    print("="*60)

    all_results = []
    for fold in range(4):
        results = evaluate_fold(fold)
        all_results.append(results)

    # Compute average across folds
    print("\n" + "="*60)
    print("AVERAGE RESULTS (4-fold cross-validation)")
    print("="*60)

    avg_binary_miou = np.mean([r['binary_miou'] for r in all_results])
    avg_parts_miou = np.mean([r['parts_miou'] for r in all_results])
    avg_type_miou = np.mean([r['type_miou'] for r in all_results])

    avg_binary_mdice = np.mean([r['binary_mdice'] for r in all_results])
    avg_parts_mdice = np.mean([r['parts_mdice'] for r in all_results])
    avg_type_mdice = np.mean([r['type_mdice'] for r in all_results])

    print(f"\nBinary Segmentation:")
    print(f"  mIoU:  {avg_binary_miou:.2f}%")
    print(f"  mDice: {avg_binary_mdice:.2f}%")

    print(f"\nParts Segmentation:")
    print(f"  mIoU:  {avg_parts_miou:.2f}%")
    print(f"  mDice: {avg_parts_mdice:.2f}%")

    print(f"\nType Segmentation:")
    print(f"  mIoU:  {avg_type_miou:.2f}%")
    print(f"  mDice: {avg_type_mdice:.2f}%")

    # Per-class analysis
    print("\n" + "="*60)
    print("PER-CLASS IoU ANALYSIS")
    print("="*60)

    parts_classes = ['Background', 'Shaft', 'Wrist', 'Clasper']
    type_classes = ['Background', 'Bipolar Forceps', 'Prograsp Forceps', 'Large Needle Driver',
                    'Vessel Sealer', 'Grasping Retractor', 'Monopolar Curved Scissors', 'Other']

    print("\nParts per-class IoU:")
    avg_parts_class = np.mean([r['parts_class_iou'] for r in all_results], axis=0)
    for i, name in enumerate(parts_classes):
        print(f"  {name:15s}: {avg_parts_class[i]:.2f}%")

    print("\nType per-class IoU:")
    avg_type_class = np.mean([r['type_class_iou'] for r in all_results], axis=0)
    for i, name in enumerate(type_classes):
        print(f"  {name:30s}: {avg_type_class[i]:.2f}%")

if __name__ == '__main__':
    main()
