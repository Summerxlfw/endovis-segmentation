#!/usr/bin/env python3
"""Evaluate multi-task segmentation model on EndoVis 2017 dataset."""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model
from tqdm import tqdm


def compute_iou_dice(pred, gt, num_classes, ignore_index=255):
    """Compute IoU and Dice for each class."""
    iou_per_class = []
    dice_per_class = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls) & (gt != ignore_index)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            iou = float('nan')
            dice = float('nan')
        else:
            iou = intersection / union
            dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else float('nan')

        iou_per_class.append(iou)
        dice_per_class.append(dice)

    return iou_per_class, dice_per_class


def evaluate_fold(config_path, checkpoint_path, data_root):
    """Evaluate a single fold."""
    print(f"\nEvaluating: {checkpoint_path}")

    # Load config and model
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Get validation image list
    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_binary_dir = os.path.join(data_root, 'annotations', 'binary', 'val')
    val_parts_dir = os.path.join(data_root, 'annotations', 'parts', 'val')
    val_type_dir = os.path.join(data_root, 'annotations', 'type', 'val')

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    # Results accumulators
    all_binary_iou = []
    all_binary_dice = []
    all_parts_iou = []
    all_parts_dice = []
    all_type_iou = []
    all_type_dice = []

    # Process each image
    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(val_img_dir, img_file)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        img_resized = cv2.resize(img, (1920, 1088))

        # Prepare input
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.cuda()

        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
        img_tensor = (img_tensor - mean) / std

        # Inference
        with torch.no_grad():
            # Get backbone features
            x = model.backbone(img_tensor)

            # Get predictions from each head
            pred_binary = model.decode_head_binary(x)
            pred_parts = model.decode_head_parts(x)
            pred_type = model.decode_head_type(x)

            # Upsample and get class predictions
            pred_binary = torch.nn.functional.interpolate(pred_binary, size=(1080, 1920), mode='bilinear', align_corners=False)
            pred_parts = torch.nn.functional.interpolate(pred_parts, size=(1080, 1920), mode='bilinear', align_corners=False)
            pred_type = torch.nn.functional.interpolate(pred_type, size=(1080, 1920), mode='bilinear', align_corners=False)

            pred_binary = pred_binary.argmax(dim=1).cpu().numpy()[0]
            pred_parts = pred_parts.argmax(dim=1).cpu().numpy()[0]
            pred_type = pred_type.argmax(dim=1).cpu().numpy()[0]

        # Load ground truth
        gt_binary = cv2.imread(os.path.join(val_binary_dir, img_file), cv2.IMREAD_GRAYSCALE)
        gt_parts = cv2.imread(os.path.join(val_parts_dir, img_file), cv2.IMREAD_GRAYSCALE)
        gt_type = cv2.imread(os.path.join(val_type_dir, img_file), cv2.IMREAD_GRAYSCALE)

        # Compute metrics
        binary_iou, binary_dice = compute_iou_dice(pred_binary, gt_binary, num_classes=2)
        parts_iou, parts_dice = compute_iou_dice(pred_parts, gt_parts, num_classes=4)
        type_iou, type_dice = compute_iou_dice(pred_type, gt_type, num_classes=8)

        all_binary_iou.append(binary_iou)
        all_binary_dice.append(binary_dice)
        all_parts_iou.append(parts_iou)
        all_parts_dice.append(parts_dice)
        all_type_iou.append(type_iou)
        all_type_dice.append(type_dice)

    # Compute mean metrics
    all_binary_iou = np.array(all_binary_iou)
    all_binary_dice = np.array(all_binary_dice)
    all_parts_iou = np.array(all_parts_iou)
    all_parts_dice = np.array(all_parts_dice)
    all_type_iou = np.array(all_type_iou)
    all_type_dice = np.array(all_type_dice)

    # Mean IoU (excluding NaN)
    binary_miou = np.nanmean(all_binary_iou)
    binary_mdice = np.nanmean(all_binary_dice)
    parts_miou = np.nanmean(all_parts_iou)
    parts_mdice = np.nanmean(all_parts_dice)
    type_miou = np.nanmean(all_type_iou)
    type_mdice = np.nanmean(all_type_dice)

    # Per-class IoU
    binary_class_iou = np.nanmean(all_binary_iou, axis=0)
    parts_class_iou = np.nanmean(all_parts_iou, axis=0)
    type_class_iou = np.nanmean(all_type_iou, axis=0)

    results = {
        'binary': {'mIoU': binary_miou, 'mDice': binary_mdice, 'class_iou': binary_class_iou},
        'parts': {'mIoU': parts_miou, 'mDice': parts_mdice, 'class_iou': parts_class_iou},
        'type': {'mIoU': type_miou, 'mDice': type_mdice, 'class_iou': type_class_iou}
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-task model')
    parser.add_argument('--fold', type=int, default=None, help='Fold to evaluate (0-3), or None for all')
    args = parser.parse_args()

    base_dir = '/home/summer/endovis/mmsegmentation'

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = [0, 1, 2, 3]

    all_results = []

    for fold in folds:
        config_path = f'{base_dir}/configs/endovis/endovis2017_multitask_fold{fold}.py'
        checkpoint_path = f'{base_dir}/work_dirs/endovis2017_multitask_fold{fold}/iter_40000.pth'
        data_root = f'/home/summer/endovis/endovis2017_multitask_fold{fold}'

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        results = evaluate_fold(config_path, checkpoint_path, data_root)
        all_results.append(results)

        print(f"\n=== Fold {fold} Results ===")
        print(f"Binary: mIoU={results['binary']['mIoU']*100:.2f}%, mDice={results['binary']['mDice']*100:.2f}%")
        print(f"  Class IoU: {[f'{x*100:.2f}%' for x in results['binary']['class_iou']]}")
        print(f"Parts:  mIoU={results['parts']['mIoU']*100:.2f}%, mDice={results['parts']['mDice']*100:.2f}%")
        print(f"  Class IoU: {[f'{x*100:.2f}%' for x in results['parts']['class_iou']]}")
        print(f"Type:   mIoU={results['type']['mIoU']*100:.2f}%, mDice={results['type']['mDice']*100:.2f}%")
        print(f"  Class IoU: {[f'{x*100:.2f}%' for x in results['type']['class_iou']]}")

    if len(all_results) > 1:
        print("\n" + "="*50)
        print("=== Average Results Across All Folds ===")
        print("="*50)

        # Compute averages
        avg_binary_miou = np.mean([r['binary']['mIoU'] for r in all_results])
        avg_binary_mdice = np.mean([r['binary']['mDice'] for r in all_results])
        avg_parts_miou = np.mean([r['parts']['mIoU'] for r in all_results])
        avg_parts_mdice = np.mean([r['parts']['mDice'] for r in all_results])
        avg_type_miou = np.mean([r['type']['mIoU'] for r in all_results])
        avg_type_mdice = np.mean([r['type']['mDice'] for r in all_results])

        print(f"\nBinary Segmentation:")
        print(f"  Mean mIoU:  {avg_binary_miou*100:.2f}%")
        print(f"  Mean mDice: {avg_binary_mdice*100:.2f}%")

        print(f"\nParts Segmentation:")
        print(f"  Mean mIoU:  {avg_parts_miou*100:.2f}%")
        print(f"  Mean mDice: {avg_parts_mdice*100:.2f}%")

        print(f"\nType Segmentation:")
        print(f"  Mean mIoU:  {avg_type_miou*100:.2f}%")
        print(f"  Mean mDice: {avg_type_mdice*100:.2f}%")


if __name__ == '__main__':
    main()
