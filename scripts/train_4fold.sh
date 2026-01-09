#!/bin/bash
# Train all 4-fold cross-validation models

cd /home/summer/endovis/mmsegmentation
source /home/summer/miniconda3/bin/activate mmseg

echo "=============================================="
echo "EndoVis 2017 4-Fold Cross-Validation Training"
echo "=============================================="
echo "Total models to train: 12 (3 tasks x 4 folds)"
echo ""

# Train Binary models (Fold 0-3)
echo "=========================================="
echo "BINARY TASK (4 folds)"
echo "=========================================="
for fold in 0 1 2 3; do
    echo ""
    echo "[Binary Fold $fold] Starting at: $(date)"
    python tools/train.py configs/endovis/endovis2017_binary_fold${fold}.py
    echo "[Binary Fold $fold] Completed at: $(date)"
done

# Train Parts models (Fold 0-3)
echo ""
echo "=========================================="
echo "PARTS TASK (4 folds)"
echo "=========================================="
for fold in 0 1 2 3; do
    echo ""
    echo "[Parts Fold $fold] Starting at: $(date)"
    python tools/train.py configs/endovis/endovis2017_parts_fold${fold}.py
    echo "[Parts Fold $fold] Completed at: $(date)"
done

# Train Type models (Fold 0-3)
echo ""
echo "=========================================="
echo "TYPE TASK (4 folds)"
echo "=========================================="
for fold in 0 1 2 3; do
    echo ""
    echo "[Type Fold $fold] Starting at: $(date)"
    python tools/train.py configs/endovis/endovis2017_type_fold${fold}.py
    echo "[Type Fold $fold] Completed at: $(date)"
done

echo ""
echo "=============================================="
echo "All 4-fold cross-validation training completed!"
echo "=============================================="
