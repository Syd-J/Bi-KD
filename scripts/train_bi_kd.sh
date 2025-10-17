#!/bin/bash

echo "activating environment"
source /home/sid/miniconda3/bin/activate
conda activate DL

echo "starting train_bi_kd.py"
cd /home/sid/Bi-KD/src/Bi-KD
python train_bi_kd.py --student_name PiT-XS --teacher_name CoaT-lite-mini

conda deactivate
