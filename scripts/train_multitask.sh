#!/bin/bash
# Train multi-task models for all 4 folds

cd /home/summer/endovis/mmsegmentation
source /home/summer/miniconda3/bin/activate mmseg

echo "=============================================="
echo "EndoVis 2017 Multi-Task 4-Fold Training"
echo "=============================================="
echo ""

for fold in 0 1 2 3; do
    echo ""
    echo "[Multi-Task Fold $fold] Starting at: $(date)"
    python tools/train.py configs/endovis/endovis2017_multitask_fold${fold}.py --work-dir work_dirs/endovis2017_multitask_fold${fold}
    echo "[Multi-Task Fold $fold] Completed at: $(date)"
done

echo ""
echo "=============================================="
echo "Multi-Task 4-fold training completed!"
echo "=============================================="
