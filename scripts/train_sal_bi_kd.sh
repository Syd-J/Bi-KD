#!/bin/bash

echo "activating environment"
source /home/sid/miniconda3/bin/activate
conda activate DL

echo "starting train_sal_bi_kd.py"
cd /home/sid/Bi-KD/src
python train_sal_bi_kd.py \
    --student_name ViNet-A --teacher_name TMFI-Net \
    --train_data_path /mnt/SSD/DHF1K/annotation --val_data_path /mnt/SSD/DHF1K/val \
    --dataset DHF1K --clip_size 64

conda deactivate
