#!/bin/bash
# Train all 4 folds of final multi-task model (120k iterations)

cd /home/summer/endovis/mmsegmentation

for fold in 0 1 2 3; do
    echo "=================================================="
    echo "Training Fold $fold"
    echo "=================================================="

    checkpoint="/home/summer/endovis/mmsegmentation/work_dirs/multitask/endovis2017_multitask_final_fold${fold}/iter_120000.pth"

    if [ -f "$checkpoint" ]; then
        echo "Fold $fold already completed, skipping..."
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 /home/summer/miniconda3/envs/mmseg/bin/python tools/train.py \
        configs/endovis/endovis2017_multitask_final_fold${fold}.py \
        2>&1 | tee work_dirs/multitask/final_fold${fold}_train.log

    echo "Fold $fold completed!"
done

echo "=================================================="
echo "All folds completed!"
echo "=================================================="

# Run evaluation
echo "Running evaluation..."
/home/summer/miniconda3/envs/mmseg/bin/python evaluation/evaluate_final.py
