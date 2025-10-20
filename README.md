# Simplifying Knowledge Transfer in Pretrained Models (TMLR 2025)
Official PyTorch implementation

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

## MODELS
Checkpoints for MAE and video saliency models can be found [here](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/siddharth_jain_research_iiit_ac_in/EvP913lpZNRCqM7Prp4huYwBJQSc0gRqz9Knz-Y13qoqbw?e=V21u7A), move them to src/pretrained
