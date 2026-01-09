#!/usr/bin/env python3
"""Summarize 4-fold cross-validation results for EndoVis 2017."""

import os
import re
import json
import numpy as np
from pathlib import Path

WORK_DIR = Path('/home/summer/endovis/mmsegmentation/work_dirs')

TASKS = {
    'binary': {'classes': ['background', 'instrument'], 'num_classes': 2},
    'parts': {'classes': ['background', 'shaft', 'wrist', 'claspers'], 'num_classes': 4},
    'type': {'classes': ['background', 'Bipolar_Forceps', 'Prograsp_Forceps',
                         'Large_Needle_Driver', 'Vessel_Sealer', 'Grasping_Retractor',
                         'Monopolar_Curved_Scissors', 'Other'], 'num_classes': 8},
}


def parse_log_file(log_path):
    """Parse the training log to extract final validation metrics."""
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all validation results
    # Pattern: mIoU: XX.XX  mDice: XX.XX
    miou_pattern = r'mIoU:\s+([\d.]+)'
    mdice_pattern = r'mDice:\s+([\d.]+)'

    miou_matches = re.findall(miou_pattern, content)
    mdice_matches = re.findall(mdice_pattern, content)

    if miou_matches and mdice_matches:
        return {
            'mIoU': float(miou_matches[-1]),
            'mDice': float(mdice_matches[-1])
        }
    return None


def parse_json_log(json_path):
    """Parse JSON log file for detailed metrics."""
    if not json_path.exists():
        return None

    results = []
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'mIoU' in data:
                    results.append(data)
            except json.JSONDecodeError:
                continue

    return results[-1] if results else None


def main():
    print("=" * 70)
    print("EndoVis 2017 4-Fold Cross-Validation Results Summary")
    print("=" * 70)

    for task_name, task_info in TASKS.items():
        print(f"\n{'='*70}")
        print(f"Task: {task_name.upper()}")
        print("=" * 70)

        fold_miou = []
        fold_mdice = []
        fold_details = []

        for fold in range(4):
            work_path = WORK_DIR / f'endovis2017_{task_name}_fold{fold}'

            # Try to find the log file
            log_files = list(work_path.glob('*.log')) if work_path.exists() else []
            json_files = list(work_path.glob('*.json')) if work_path.exists() else []

            result = None
            if log_files:
                result = parse_log_file(log_files[0])

            json_result = None
            if json_files:
                json_result = parse_json_log(json_files[-1])

            if json_result:
                result = json_result

            if result:
                miou = result.get('mIoU', 0)
                mdice = result.get('mDice', 0)
                fold_miou.append(miou)
                fold_mdice.append(mdice)
                fold_details.append(result)
                print(f"  Fold {fold}: mIoU = {miou:.2f}%, mDice = {mdice:.2f}%")
            else:
                print(f"  Fold {fold}: Training not completed or logs not found")

        if fold_miou:
            mean_miou = np.mean(fold_miou)
            std_miou = np.std(fold_miou)
            mean_mdice = np.mean(fold_mdice)
            std_mdice = np.std(fold_mdice)

            print("-" * 50)
            print(f"  Mean mIoU:  {mean_miou:.2f}% ± {std_miou:.2f}%")
            print(f"  Mean mDice: {mean_mdice:.2f}% ± {std_mdice:.2f}%")

    print("\n" + "=" * 70)
    print("Summary Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
