"""
Prepare EndoVis 2017 dataset with official split:
- Training: all 8 sequences from training folder (1800 frames)
- Testing: official test set from testing folder (1201 frames)
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

# Paths
TRAIN_ROOT = Path('/home/summer/endovis/endovis2017/training')
TEST_ROOT = Path('/home/summer/endovis/endovis2017/testing')
TEST_LABEL_ROOT = Path('/home/summer/endovis/endovis2017/testing_label')
OUTPUT_ROOT = Path('/home/summer/endovis')

# Instrument type mapping
INSTRUMENT_TYPES = {
    'Bipolar_Forceps': 1,
    'Prograsp_Forceps': 2,
    'Large_Needle_Driver': 3,
    'Vessel_Sealer': 4,
    'Grasping_Retractor': 5,
    'Monopolar_Curved_Scissors': 6,
    'Other': 7,
}

# Parts mapping
PARTS_MAPPING = {
    'Shaft': 1,
    'Wrist': 2,
    'Claspers': 3,
}


def prepare_binary_dataset():
    """Prepare binary segmentation dataset."""
    output_dir = OUTPUT_ROOT / 'endovis2017_binary_official'

    # Create directories
    for split in ['train', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)

    # Process training data (all 8 sequences)
    print("Processing training data (Binary)...")
    for seq_id in tqdm(range(1, 9), desc="Training sequences"):
        seq_dir = TRAIN_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'

        # Find binary mask in ground_truth
        gt_dir = seq_dir / 'ground_truth'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'train' / f'{output_name}.png')

            # Create binary mask from any instrument folder
            binary_mask = None
            for type_folder in gt_dir.iterdir():
                if type_folder.is_dir() and '_labels' not in type_folder.name:
                    mask_path = type_folder / f'{frame_name}.png'
                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path))
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        if binary_mask is None:
                            binary_mask = np.zeros_like(mask, dtype=np.uint8)
                        binary_mask[mask > 0] = 1

            if binary_mask is None:
                img = Image.open(frame_file)
                binary_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            Image.fromarray(binary_mask).save(output_dir / 'annotations' / 'train' / f'{output_name}.png')

    # Process test data
    print("Processing test data (Binary)...")
    for seq_id in tqdm(range(1, 11), desc="Test sequences"):
        seq_dir = TEST_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'
        label_dir = TEST_LABEL_ROOT / f'instrument_dataset_{seq_id}' / 'BinarySegmentation'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'test' / f'{output_name}.png')

            # Copy/convert label
            label_file = label_dir / f'{frame_name}.png'
            if label_file.exists():
                mask = np.array(Image.open(label_file))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                # Convert to binary (0 or 1)
                binary_mask = (mask > 0).astype(np.uint8)
                Image.fromarray(binary_mask).save(output_dir / 'annotations' / 'test' / f'{output_name}.png')

    train_count = len(list((output_dir / 'images' / 'train').glob('*.png')))
    test_count = len(list((output_dir / 'images' / 'test').glob('*.png')))
    print(f"Binary dataset: Train={train_count}, Test={test_count}")
    return train_count, test_count


def prepare_parts_dataset():
    """Prepare parts segmentation dataset."""
    output_dir = OUTPUT_ROOT / 'endovis2017_parts_official'

    # Create directories
    for split in ['train', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)

    # Process training data
    print("Processing training data (Parts)...")
    for seq_id in tqdm(range(1, 9), desc="Training sequences"):
        seq_dir = TRAIN_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'
        gt_dir = seq_dir / 'ground_truth'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'train' / f'{output_name}.png')

            # Create parts mask
            parts_mask = None
            for type_folder in gt_dir.iterdir():
                if type_folder.is_dir() and '_labels' in type_folder.name:
                    # This is a parts label folder
                    mask_path = type_folder / f'{frame_name}.png'
                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path))
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        if parts_mask is None:
                            parts_mask = np.zeros_like(mask, dtype=np.uint8)

                        # Map parts: 10->shaft(1), 20->wrist(2), 30->claspers(3)
                        parts_mask[mask == 10] = 1  # Shaft
                        parts_mask[mask == 20] = 2  # Wrist
                        parts_mask[mask == 30] = 3  # Claspers

            if parts_mask is None:
                img = Image.open(frame_file)
                parts_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            Image.fromarray(parts_mask).save(output_dir / 'annotations' / 'train' / f'{output_name}.png')

    # Process test data
    print("Processing test data (Parts)...")
    for seq_id in tqdm(range(1, 11), desc="Test sequences"):
        seq_dir = TEST_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'
        label_dir = TEST_LABEL_ROOT / f'instrument_dataset_{seq_id}' / 'PartsSegmentation'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'test' / f'{output_name}.png')

            # Process label
            label_file = label_dir / f'{frame_name}.png'
            if label_file.exists():
                mask = np.array(Image.open(label_file))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                # Map: 10->1, 20->2, 30->3
                parts_mask = np.zeros_like(mask, dtype=np.uint8)
                parts_mask[mask == 10] = 1
                parts_mask[mask == 20] = 2
                parts_mask[mask == 30] = 3
                Image.fromarray(parts_mask).save(output_dir / 'annotations' / 'test' / f'{output_name}.png')

    train_count = len(list((output_dir / 'images' / 'train').glob('*.png')))
    test_count = len(list((output_dir / 'images' / 'test').glob('*.png')))
    print(f"Parts dataset: Train={train_count}, Test={test_count}")
    return train_count, test_count


def prepare_type_dataset():
    """Prepare type segmentation dataset."""
    output_dir = OUTPUT_ROOT / 'endovis2017_type_official'

    # Create directories
    for split in ['train', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)

    # Process training data
    print("Processing training data (Type)...")
    for seq_id in tqdm(range(1, 9), desc="Training sequences"):
        seq_dir = TRAIN_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'
        gt_dir = seq_dir / 'ground_truth'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'train' / f'{output_name}.png')

            # Create type mask
            type_mask = None
            for type_folder in gt_dir.iterdir():
                if type_folder.is_dir() and '_labels' not in type_folder.name:
                    instrument_type = type_folder.name
                    if instrument_type not in INSTRUMENT_TYPES:
                        continue

                    class_id = INSTRUMENT_TYPES[instrument_type]
                    mask_path = type_folder / f'{frame_name}.png'

                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path))
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        if type_mask is None:
                            type_mask = np.zeros_like(mask, dtype=np.uint8)
                        type_mask[mask > 0] = class_id

            if type_mask is None:
                img = Image.open(frame_file)
                type_mask = np.zeros((img.height, img.width), dtype=np.uint8)

            Image.fromarray(type_mask).save(output_dir / 'annotations' / 'train' / f'{output_name}.png')

    # Process test data
    print("Processing test data (Type)...")
    for seq_id in tqdm(range(1, 11), desc="Test sequences"):
        seq_dir = TEST_ROOT / f'instrument_dataset_{seq_id}'
        frames_dir = seq_dir / 'left_frames'
        label_dir = TEST_LABEL_ROOT / f'instrument_dataset_{seq_id}' / 'TypeSegmentation'

        for frame_file in sorted(frames_dir.glob('*.png')):
            frame_name = frame_file.stem
            output_name = f'instrument_dataset_{seq_id}_{frame_name}'

            # Copy image
            shutil.copy(frame_file, output_dir / 'images' / 'test' / f'{output_name}.png')

            # Copy label (already in correct format 0-7)
            label_file = label_dir / f'{frame_name}.png'
            if label_file.exists():
                shutil.copy(label_file, output_dir / 'annotations' / 'test' / f'{output_name}.png')

    train_count = len(list((output_dir / 'images' / 'train').glob('*.png')))
    test_count = len(list((output_dir / 'images' / 'test').glob('*.png')))
    print(f"Type dataset: Train={train_count}, Test={test_count}")
    return train_count, test_count


if __name__ == '__main__':
    print("="*60)
    print("Preparing EndoVis 2017 datasets with official split")
    print("Training: sequences 1-8 (1800 frames)")
    print("Testing: official test set (1201 frames)")
    print("="*60)

    prepare_binary_dataset()
    print()
    prepare_parts_dataset()
    print()
    prepare_type_dataset()

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
