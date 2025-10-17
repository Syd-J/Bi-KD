#!/bin/bash

echo "activating environment"
source /home/sid/miniconda3/bin/activate
conda activate DL

echo "starting train_quad_kd.py"
cd /home/sid/Bi-KD/src
python train_quad_kd.py --model_1 CoaT-lite-mini --model_2 PiT-XS --model_3 ResMLP-24-dist --model_4 DINOv2 --batch_size 64

conda deactivate
