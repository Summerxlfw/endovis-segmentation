#!/bin/bash
# Retrain Parts and Type tasks with fixed data

cd /home/summer/endovis/mmsegmentation
source /home/summer/miniconda3/bin/activate mmseg

echo "=============================================="
echo "EndoVis 2017 4-Fold Training (Parts & Type)"
echo "=============================================="
echo ""

# Train Parts models (Fold 0-3)
echo "=========================================="
echo "PARTS TASK (4 folds)"
echo "=========================================="
for fold in 0 1 2 3; do
    echo ""
    echo "[Parts Fold $fold] Starting at: $(date)"
    python tools/train.py configs/endovis/endovis2017_parts_fold${fold}.py --work-dir work_dirs/endovis2017_parts_fold${fold}_fixed
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
    python tools/train.py configs/endovis/endovis2017_type_fold${fold}.py --work-dir work_dirs/endovis2017_type_fold${fold}_fixed
    echo "[Type Fold $fold] Completed at: $(date)"
done

echo ""
echo "=============================================="
echo "Parts & Type 4-fold training completed!"
echo "=============================================="
