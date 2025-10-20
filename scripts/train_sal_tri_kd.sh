#!/bin/bash

# Dataset configuration (must be set before activating conda)
DATASET=${1:-DHF1K}  # Default to DHF1K if no argument provided

echo "activating environment"
eval "$(conda shell.bash hook)"
conda activate bi-kd

# Set dataset-specific paths and clip size
case $DATASET in
    DHF1K)
        TRAIN_DATA_PATH="/mnt/SSD/DHF1K/annotation"
        VAL_DATA_PATH="/mnt/SSD/DHF1K/val"
        CLIP_SIZE=64
        ;;
    Hollywood2)
        TRAIN_DATA_PATH="/mnt/SSD/Hollywood2/training"
        VAL_DATA_PATH="/mnt/SSD/Hollywood2/testing"
        CLIP_SIZE=32
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported datasets: DHF1K, Hollywood2"
        exit 1
        ;;
esac

echo "starting train_sal_tri_kd.py with dataset: $DATASET"
cd /home/sid/Bi-KD/src
python train_sal_tri_kd.py \
    --model_1 TMFI-Net --model_2 ViNet-S --model_3 ViNet-A \
    --train_data_path $TRAIN_DATA_PATH --val_data_path $VAL_DATA_PATH \
    --dataset $DATASET --clip_size $CLIP_SIZE

conda deactivate
