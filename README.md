# Simplifying Knowledge Transfer in Pretrained Models (TMLR 2025)
Official PyTorch implementation | [Paper](https://openreview.net/pdf?id=eQ9AVtDaP3)

## Abstract

> Pretrained models are ubiquitous in the current deep learning landscape, offering strong
results on a broad range of tasks. Recent works have shown that models differing in various 
design choices exhibit categorically diverse generalization behavior, resulting in one
model grasping distinct data-specific insights unavailable to the other. In this paper, we
propose to leverage large publicly available model repositories as an auxiliary source of
model improvements. We introduce a data partitioning strategy where pretrained models
autonomously adopt either the role of a student, seeking knowledge, or that of a teacher,
imparting knowledge. Experiments across various tasks demonstrate the effectiveness of
our proposed approach. In image classification, we improved the performance of ViT-B
by approximately 1.4% through bidirectional knowledge transfer with ViT-T. For semantic
segmentation, our method boosted all evaluation metrics by enabling knowledge transfer
both within and across backbone architectures. In video saliency prediction, our approach
achieved a new state-of-the-art. We further extend our approach to knowledge transfer
between multiple models, leading to considerable performance improvements for all model
participants.

## Setup
### Option A — Conda (recommended for reproducibility)

Create and activate the environment (name: `bi-kd`):

```bash
conda env create -f environment.yaml
conda activate bi-kd
```

### Optiona B - pip

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

NOTE: Refer to this [repo](https://github.com/NVlabs/MambaVision) for setting up environment for MambaVision-T2

## Usage
Add new models to model.py before using them.

### Image Classification
The experiments for Bi-KD can be run using:

```bash
bash scripts/train_bi_kd.sh --student_name STUDENT --teacher_name TEACHER
```

The experiments for transferring knowledge between 3 models can be run using:

```bash
bash scripts/train_tri_kd.sh --model_1 MODEL_1 --model_2 MODEL_2 --model_3 MODEL_3
```

The experiments for transferring knowledge between 4 models can be run using:

```bash
bash scripts/train_quad_kd.sh --model_1 MODEL_1 --model_2 MODEL_2 --model_3 MODEL_3 --model_$ MODEL_4
```

For transferring knowledge between ViT models, refer to this [fork](https://github.com/Syd-J/pytorch-image-models) of this [repo](https://github.com/huggingface/pytorch-image-models)

### Semantic Segmentation
For running experiments on semantic segmentation, refer to this [fork](https://github.com/Syd-J/Mask2Former) of this [repo](https://github.com/facebookresearch/Mask2Former)

### Video Saliency Prediction
The experiments for Bi-KD on video saliency prediction can be run using:

```bash
bash scripts/train_sal_bi_kd.sh DHF1K        # for experiments on DHF1K
bash scripts/train_sal_bi_kd.sh Hollywood2   # for experiments on Hollywood2
```

The experiments for transferring knowledge between 3 models can be run using:

```bash
bash scripts/train_sal_tri_kd.sh DHF1K        # for experiments on DHF1K
bash scripts/train_sal_tri_kd.sh Hollywood2   # for experiments on Hollywood2
```

NOTE: Refer to this [repo](https://github.com/ViNet-Saliency/vinet_v2) for evaluating ViNet-A & ViNet-S, and to this [repo](https://github.com/wusonghe/TMFI-Net) for evaluating TMFI-Net

## Models
Checkpoints for MAE and video saliency models can be found [here](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/siddharth_jain_research_iiit_ac_in/EvP913lpZNRCqM7Prp4huYwBJQSc0gRqz9Knz-Y13qoqbw?e=V21u7A), move them to src/pretrained

## Citation

If your find this project useful in your research, please consider citing:

```bibtex
@article{
jain2025simplifying,
title={Simplifying Knowledge Transfer in Pretrained Models},
author={Siddharth Jain and Shyamgopal Karthik and Vineet Gandhi},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=eQ9AVtDaP3},
}
```
