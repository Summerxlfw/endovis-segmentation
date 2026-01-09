#!/usr/bin/env python3
"""
Prepare 4-fold cross-validation datasets for EndoVis 2017.

Fold splits (following academic convention):
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

# Class mappings
INSTRUMENT_TYPES = {
    'Bipolar_Forceps': 1,
    'Prograsp_Forceps': 2,
    'Large_Needle_Driver': 3,
    'Vessel_Sealer': 4,
    'Grasping_Retractor': 5,
    'Monopolar_Curved_Scissors': 6,
    'Other': 7
}

PART_CLASSES = {
    'Shaft': 1,
    'Wrist': 2,
    'Claspers': 3
}


def create_binary_mask(instruments_dir, frame_name):
    """Create binary mask (background=0, instrument=1)."""
    mask = None
    for inst_dir in instruments_dir.iterdir():
        if inst_dir.is_dir():
            mask_path = inst_dir / frame_name
            if mask_path.exists():
                inst_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if inst_mask is not None:
                    if mask is None:
                        mask = np.zeros_like(inst_mask)
                    mask[inst_mask > 0] = 1
    return mask


def create_parts_mask(instruments_dir, frame_name):
    """Create parts mask (background=0, shaft=1, wrist=2, claspers=3)."""
    mask = None
    for inst_dir in instruments_dir.iterdir():
        if inst_dir.is_dir():
            for part_name, part_id in PART_CLASSES.items():
                part_dir = inst_dir / part_name
                if part_dir.exists():
                    mask_path = part_dir / frame_name
                    if mask_path.exists():
                        part_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if part_mask is not None:
                            if mask is None:
                                h, w = part_mask.shape
                                mask = np.zeros((h, w), dtype=np.uint8)
                            mask[part_mask > 0] = part_id
    return mask


def create_type_mask(instruments_dir, frame_name):
    """Create type mask (background=0, each instrument type=1-7)."""
    mask = None
    for inst_dir in instruments_dir.iterdir():
        if inst_dir.is_dir():
            inst_name = inst_dir.name
            # Get instrument type from directory name
            inst_type = None
            for type_name, type_id in INSTRUMENT_TYPES.items():
                if type_name.lower() in inst_name.lower():
                    inst_type = type_id
                    break

            if inst_type is None:
                inst_type = INSTRUMENT_TYPES['Other']

            # Combine all parts of this instrument
            for part_dir in inst_dir.iterdir():
                if part_dir.is_dir():
                    mask_path = part_dir / frame_name
                    if mask_path.exists():
                        part_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if part_mask is not None:
                            if mask is None:
                                h, w = part_mask.shape
                                mask = np.zeros((h, w), dtype=np.uint8)
                            mask[part_mask > 0] = inst_type
    return mask


def prepare_fold_dataset(fold_id, task_type):
    """Prepare dataset for a specific fold and task."""
    split = FOLD_SPLITS[fold_id]
    output_dir = BASE_DIR / f'endovis2017_{task_type}_fold{fold_id}'

    print(f"\nPreparing {task_type} dataset for Fold {fold_id}")
    print(f"  Train sequences: {split['train']}")
    print(f"  Val sequences: {split['val']}")

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
            instruments_dir = seq_dir / 'ground_truth'

            if not images_dir.exists() or not instruments_dir.exists():
                print(f"    Warning: Missing directories in {seq_dir}, skipping...")
                continue

            # Process all frames (225 frames per sequence)
            for img_path in sorted(images_dir.glob('*.png')):
                frame_name = img_path.name

                # Copy image
                dest_img = output_dir / 'images' / subset / f'seq{seq_id}_{frame_name}'
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

                # Create annotation based on task type
                if task_type == 'binary':
                    mask = create_binary_mask(instruments_dir, frame_name)
                elif task_type == 'parts':
                    mask = create_parts_mask(instruments_dir, frame_name)
                elif task_type == 'type':
                    mask = create_type_mask(instruments_dir, frame_name)

                if mask is not None:
                    dest_mask = output_dir / 'annotations' / subset / f'seq{seq_id}_{frame_name}'
                    cv2.imwrite(str(dest_mask), mask)
                else:
                    # Create empty mask if no instruments in frame
                    if images_dir.exists():
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


def main():
    print("=" * 60)
    print("EndoVis 2017 4-Fold Cross-Validation Dataset Preparation")
    print("=" * 60)

    tasks = ['binary', 'parts', 'type']

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print("=" * 60)

        for fold in range(4):
            prepare_fold_dataset(fold, task)

    print("\n" + "=" * 60)
    print("All datasets prepared successfully!")
    print("=" * 60)

    # Summary
    print("\nDataset Summary:")
    print("-" * 60)
    for task in tasks:
        print(f"\n{task.upper()} Task:")
        for fold in range(4):
            output_dir = BASE_DIR / f'endovis2017_{task}_fold{fold}'
            train_count = len(list((output_dir / 'images' / 'train').glob('*.png')))
            val_count = len(list((output_dir / 'images' / 'val').glob('*.png')))
            print(f"  Fold {fold}: train={train_count}, val={val_count}")


if __name__ == '__main__':
    main()
