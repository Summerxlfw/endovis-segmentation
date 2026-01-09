#!/usr/bin/env python3
"""Evaluate single-task segmentation models on EndoVis 2017 dataset."""

import os
import numpy as np
import torch
import cv2
from mmengine.config import Config
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


def evaluate_single_task(config_path, checkpoint_path, data_root, task_type, num_classes):
    """Evaluate a single task model."""
    print(f"\nEvaluating: {checkpoint_path}")

    # Load config and model
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Get validation image list
    val_img_dir = os.path.join(data_root, 'images', 'val')
    val_ann_dir = os.path.join(data_root, 'annotations', 'val')

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.png')])

    all_iou = []
    all_dice = []

    for img_file in tqdm(img_files, desc=f"Processing {task_type}"):
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
            x = model.backbone(img_tensor)
            pred = model.decode_head(x)
            pred = torch.nn.functional.interpolate(pred, size=(1080, 1920), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1).cpu().numpy()[0]

        # Load ground truth
        gt = cv2.imread(os.path.join(val_ann_dir, img_file), cv2.IMREAD_GRAYSCALE)

        # Compute metrics
        iou, dice = compute_iou_dice(pred, gt, num_classes=num_classes)
        all_iou.append(iou)
        all_dice.append(dice)

    all_iou = np.array(all_iou)
    all_dice = np.array(all_dice)

    miou = np.nanmean(all_iou)
    mdice = np.nanmean(all_dice)
    class_iou = np.nanmean(all_iou, axis=0)

    return {'mIoU': miou, 'mDice': mdice, 'class_iou': class_iou}


def main():
    base_dir = '/home/summer/endovis/mmsegmentation'

    tasks = {
        'binary': {'num_classes': 2, 'suffix': ''},
        'parts': {'num_classes': 4, 'suffix': '_fixed'},
        'type': {'num_classes': 8, 'suffix': '_fixed'}
    }

    all_results = {task: [] for task in tasks}

    for fold in range(4):
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print('='*50)

        for task, info in tasks.items():
            config_path = f'{base_dir}/configs/endovis/endovis2017_{task}_fold{fold}.py'
            checkpoint_path = f'{base_dir}/work_dirs/endovis2017_{task}_fold{fold}{info["suffix"]}/iter_40000.pth'
            data_root = f'/home/summer/endovis/endovis2017_{task}_fold{fold}'

            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue

            results = evaluate_single_task(config_path, checkpoint_path, data_root, task, info['num_classes'])
            all_results[task].append(results)

            print(f"{task.capitalize()}: mIoU={results['mIoU']*100:.2f}%, mDice={results['mDice']*100:.2f}%")

    # Print summary
    print("\n" + "="*60)
    print("=== Single-Task Average Results Across All Folds ===")
    print("="*60)

    for task in tasks:
        if all_results[task]:
            avg_miou = np.mean([r['mIoU'] for r in all_results[task]])
            avg_mdice = np.mean([r['mDice'] for r in all_results[task]])
            print(f"\n{task.capitalize()} Segmentation:")
            print(f"  Mean mIoU:  {avg_miou*100:.2f}%")
            print(f"  Mean mDice: {avg_mdice*100:.2f}%")


if __name__ == '__main__':
    main()
