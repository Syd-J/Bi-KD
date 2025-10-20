#!/bin/bash

echo "activating environment"
source /home/sid/miniconda3/bin/activate
conda activate bi-kd

echo "starting test.py"
cd /home/sid/Bi-KD/src
python test.py --student_name PiT-XS --teacher_name CoaT-lite-mini

conda deactivate
