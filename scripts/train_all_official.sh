#!/bin/bash
# Train all three models with official split

cd /home/summer/endovis/mmsegmentation
source /home/summer/miniconda3/bin/activate mmseg

echo "=========================================="
echo "Training EndoVis 2017 models (Official Split)"
echo "=========================================="
echo ""

# Train Binary model
echo "[1/3] Training Binary segmentation model..."
echo "Start time: $(date)"
python tools/train.py configs/endovis/endovis2017_binary_official.py
echo "Binary model completed: $(date)"
echo ""

# Train Parts model
echo "[2/3] Training Parts segmentation model..."
echo "Start time: $(date)"
python tools/train.py configs/endovis/endovis2017_parts_official.py
echo "Parts model completed: $(date)"
echo ""

# Train Type model
echo "[3/3] Training Type segmentation model..."
echo "Start time: $(date)"
python tools/train.py configs/endovis/endovis2017_type_official.py
echo "Type model completed: $(date)"
echo ""

echo "=========================================="
echo "All training completed!"
echo "=========================================="
