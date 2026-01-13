# Divergence-Based Similarity Function for Multi-View Contrastive Learning

Official PyTorch implementation of **Divergence-Based Similarity Function for Multi-View Contrastive Learning**.

- Paper: https://arxiv.org/abs/2507.06560

## Overview
This repository provides code for:
- pretraining
- kNN monitoring
- linear evaluation

## Installation

### Requirements
- Python 3.9 (tested on 3.9.18)
- PyTorch (tested on 1.11.0)
- CUDA 11.7 (tested)

### Setup
```bash
conda create -n dsf-mvcl python=3.9.18 -y
conda activate dsf-mvcl
pip install -r requirements.txt
```

## Dataset
We support: ImageNet, ImageNet-100, CIFAR10, CIFAR100.

### ImageNet / ImageNet-100
We assume the standard ImageNet directory layout:
```text
/path/to/imagenet/
  train/
    n01440764/...
  val/
    n01440764/...
```

`IN100.txt` contains 100 ImageNet class WNIDs (one per line).  
You can create ImageNet-100 from ImageNet-1K using `generate_IN100.py`.

Example:
```bash
python generate_IN100.py \
  --source_folder /path/to/imagenet/train \
  --target_folder /path/to/imagenet100/train \
  --target_class IN100.txt
```

## Usage
Please check `main.sh` for full scripts.

### 1) Pretraining
```bash
python main_mocov3_multiview.py \
  --results_dir=mocov3_IN100_dsf \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=5e-3 --weight-decay=.1 \
  --epochs=100 --warmup-epochs=15 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10002' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --ns=4 --model=mocov3_dsf --seed=307 \
  /path/to/data
```

### 2) Linear Evaluation
```bash
python main_linclsv3.py \
    /path/to/data \
    --dataset=IN100 \
    --results_dir=mocov3_IN100_dsf \
    -a vit_small --lr 30 -b 256 \
    --dist-url 'tcp://localhost:10002' \
    --world-size 0 --rank 0 \
    --epochs=30 \
    --pretrained /path/to/pth.tar
```

## Notes
- Results reported in the paper use `seed=307`.
- Multi-GPU training: set `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Pretrained Checkpoints
- ViT-S (IN100, 100 epochs, seed=307): [download](https://github.com/Jeon789/DSF/releases/download/v1.0/ckpt_vits_in100_ep100_seed307.pth.tar)

## License
This project is released under the MIT License. See `LICENSE`.

## Acknowledgements
This repository is based on the codebase of `facebookresearch/moco` (https://github.com/facebookresearch/moco) under the MIT License.

Major changes:
- Implemented DSF loss (KL divergence)
- Added multi-view support (`NumSampleCropsTransform`)

## Citation
```bibtex
@article{jeon2025divergence,
  title={Divergence-Based Similarity Function for Multi-View Contrastive Learning},
  author={Jeon, Jae Hyoung and Lim, Cheolsu and Kang, Myungjoo},
  journal={arXiv preprint arXiv:2507.06560},
  year={2025}
}
```

## Authors / Contact
- Jae Hyoung Jeon (jan4021@snu.ac.kr)
- Cheolsu Lim (sky3alfory@snu.ac.kr)
- Myungjoo Kang