#!/usr/bin/env python3
"""
Prepare 4-fold cross-validation datasets for EndoVis 2017 (FIXED VERSION).

Data format:
- Binary mask: instrument_name/frameXXX.png (values: 0=background, non-zero=instrument)
- Parts labels: instrument_name_Left_labels/frameXXX.png or instrument_name_Right_labels/frameXXX.png
  Values: 0=background, 10=shaft, 20=wrist, 30=claspers

Fold splits:
- Fold 0: train on seq 1,2,3,5,6,7 | val on seq 4,8
- Fold 1: train on seq 1,2,4,5,6,8 | val on seq 3,7
- Fold 2: train on seq 1,3,4,5,7,8 | val on seq 2,6
- Fold 3: train on seq 2,3,4,6,7,8 | val on seq 1,5
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 4-fold split configuration
FOLD_SPLITS = {
    0: {'train': [1, 2, 3, 5, 6, 7], 'val': [4, 8]},
    1: {'train': [1, 2, 4, 5, 6, 8], 'val': [3, 7]},
    2: {'train': [1, 3, 4, 5, 7, 8], 'val': [2, 6]},
    3: {'train': [2, 3, 4, 6, 7, 8], 'val': [1, 5]},
}

# Paths
BASE_DIR = Path('/home/summer/endovis')
SOURCE_DIR = BASE_DIR / 'endovis2017' / 'training'

# Instrument types mapping
INSTRUMENT_TYPES = {
    'Bipolar_Forceps': 1,
    'Prograsp_Forceps': 2,
    'Large_Needle_Driver': 3,
    'Vessel_Sealer': 4,
    'Grasping_Retractor': 5,
    'Monopolar_Curved_Scissors': 6,
    'Other': 7
}

# Parts value mapping (original -> target)
PARTS_VALUE_MAP = {
    0: 0,    # background -> background
    10: 1,   # shaft -> 1
    20: 2,   # wrist -> 2
    30: 3    # claspers -> 3
}


def get_instrument_dirs(gt_dir):
    """Get instrument directories (excluding *_labels directories)."""
    inst_dirs = []
    for d in gt_dir.iterdir():
        if d.is_dir() and '_labels' not in d.name:
            inst_dirs.append(d)
    return inst_dirs


def get_labels_dirs(gt_dir, inst_name):
    """Get labels directories for an instrument."""
    labels_dirs = []
    for d in gt_dir.iterdir():
        if d.is_dir() and inst_name in d.name and '_labels' in d.name:
            labels_dirs.append(d)
    return labels_dirs


def create_binary_mask(gt_dir, frame_name):
    """Create binary mask (background=0, instrument=1)."""
    mask = None
    for inst_dir in get_instrument_dirs(gt_dir):
        mask_path = inst_dir / frame_name
        if mask_path.exists():
            inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if inst_mask is not None:
                if mask is None:
                    mask = np.zeros_like(inst_mask)
                mask[inst_mask > 0] = 1
    return mask


def create_parts_mask(gt_dir, frame_name):
    """Create parts mask from *_labels directories.

    Values: 0=background, 1=shaft, 2=wrist, 3=claspers
    """
    mask = None

    for inst_dir in get_instrument_dirs(gt_dir):
        inst_name = inst_dir.name
        labels_dirs = get_labels_dirs(gt_dir, inst_name)

        for labels_dir in labels_dirs:
            labels_path = labels_dir / frame_name
            if labels_path.exists():
                labels_img = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE)
                if labels_img is not None:
                    if mask is None:
                        h, w = labels_img.shape
                        mask = np.zeros((h, w), dtype=np.uint8)

                    # Map original values (0,10,20,30) to target values (0,1,2,3)
                    for orig_val, target_val in PARTS_VALUE_MAP.items():
                        if target_val > 0:  # Skip background
                            mask[labels_img == orig_val] = target_val

    # If no labels found, try to create from binary masks
    if mask is None:
        for inst_dir in get_instrument_dirs(gt_dir):
            mask_path = inst_dir / frame_name
            if mask_path.exists():
                inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if inst_mask is not None:
                    if mask is None:
                        h, w = inst_mask.shape
                        mask = np.zeros((h, w), dtype=np.uint8)
                    # Use claspers (3) as default for instruments without parts labels
                    mask[inst_mask > 0] = 3

    return mask


def create_type_mask(gt_dir, frame_name):
    """Create type mask (background=0, each instrument type=1-7)."""
    mask = None

    for inst_dir in get_instrument_dirs(gt_dir):
        inst_name = inst_dir.name

        # Determine instrument type
        inst_type = INSTRUMENT_TYPES.get('Other', 7)
        for type_name, type_id in INSTRUMENT_TYPES.items():
            if type_name.lower() in inst_name.lower():
                inst_type = type_id
                break

        mask_path = inst_dir / frame_name
        if mask_path.exists():
            inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if inst_mask is not None:
                if mask is None:
                    h, w = inst_mask.shape
                    mask = np.zeros((h, w), dtype=np.uint8)
                mask[inst_mask > 0] = inst_type

    return mask


def prepare_fold_dataset(fold_id, task_type):
    """Prepare dataset for a specific fold and task."""
    split = FOLD_SPLITS[fold_id]
    output_dir = BASE_DIR / f'endovis2017_{task_type}_fold{fold_id}'

    print(f"\nPreparing {task_type} dataset for Fold {fold_id}")
    print(f"  Train sequences: {split['train']}")
    print(f"  Val sequences: {split['val']}")

    # Remove existing directory
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create directories
    for subset in ['train', 'val']:
        (output_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / subset).mkdir(parents=True, exist_ok=True)

    # Process each subset
    for subset in ['train', 'val']:
        sequences = split[subset]
        print(f"  Processing {subset} set ({len(sequences)} sequences)...")

        for seq_id in tqdm(sequences, desc=f"    Sequences"):
            seq_dir = SOURCE_DIR / f'instrument_dataset_{seq_id}'
            if not seq_dir.exists():
                print(f"    Warning: {seq_dir} not found, skipping...")
                continue

            images_dir = seq_dir / 'left_frames'
            gt_dir = seq_dir / 'ground_truth'

            if not images_dir.exists() or not gt_dir.exists():
                print(f"    Warning: Missing directories in {seq_dir}, skipping...")
                continue

            # Process all frames
            for img_path in sorted(images_dir.glob('*.png')):
                frame_name = img_path.name

                # Copy image
                dest_img = output_dir / 'images' / subset / f'seq{seq_id}_{frame_name}'
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

                # Create annotation based on task type
                if task_type == 'binary':
                    mask = create_binary_mask(gt_dir, frame_name)
                elif task_type == 'parts':
                    mask = create_parts_mask(gt_dir, frame_name)
                elif task_type == 'type':
                    mask = create_type_mask(gt_dir, frame_name)

                if mask is not None:
                    dest_mask = output_dir / 'annotations' / subset / f'seq{seq_id}_{frame_name}'
                    cv2.imwrite(str(dest_mask), mask)
                else:
                    # Create empty mask if no instruments in frame
                    sample_img = cv2.imread(str(img_path))
                    if sample_img is not None:
                        h, w = sample_img.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        dest_mask = output_dir / 'annotations' / subset / f'seq{seq_id}_{frame_name}'
                        cv2.imwrite(str(dest_mask), empty_mask)

    # Count files
    train_imgs = len(list((output_dir / 'images' / 'train').glob('*.png')))
    val_imgs = len(list((output_dir / 'images' / 'val').glob('*.png')))
    print(f"  Done! Train: {train_imgs} images, Val: {val_imgs} images")

    return output_dir


def verify_annotations(task_type, fold_id):
    """Verify that annotations have correct values."""
    output_dir = BASE_DIR / f'endovis2017_{task_type}_fold{fold_id}'
    ann_dir = output_dir / 'annotations' / 'val'

    all_values = set()
    for ann_file in list(ann_dir.glob('*.png'))[:50]:
        img = cv2.imread(str(ann_file), cv2.IMREAD_GRAYSCALE)
        all_values.update(np.unique(img))

    print(f"  {task_type} fold{fold_id} annotation values: {sorted(all_values)}")
    return all_values


def main():
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = ['parts', 'type']  # Default: only fix parts and type

    print("=" * 60)
    print("EndoVis 2017 4-Fold Dataset Preparation (FIXED)")
    print("=" * 60)

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print("=" * 60)

        for fold in range(4):
            prepare_fold_dataset(fold, task)
            verify_annotations(task, fold)

    print("\n" + "=" * 60)
    print("All datasets prepared successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
