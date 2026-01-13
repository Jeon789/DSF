#!/bin/bash

# 1. MoCov2 + CIFAR10
# 1-1) Pretraining
python train_moco_cifar.py  \
      --results_dir=dum_mocov2 \
      --num_workers=8 --batch_size=1024 --epochs=800 --lr=0.12 \
      --dataset=CIFAR10 \
      /path/to/data

# 1-2) Linear Evaluation
python main_lincls.py \
    /path/to/data \
  --pretrained=/home/jan4021/develop/DSF_public/dum_mocov2/model_last.pth \
  --dataset=CIFAR10 --epochs=30 --schedule=[5,10,15,20] --print-freq=100 &


# 2. MoCov2_DSF + CIFAR10
# 2-1) Pretraining
python train_moco_cifar_multiview.py  \
    --results_dir=dum_mocov2_dsf \
    --num_workers=8 --batch_size=256 --epochs=250 --lr=0.06 \
    --dataset=CIFAR10 \
    --ns=4 \
    /path/to/data

# 2-1) Pretraining\
python main_lincls.py \
    /path/to/data \
  --pretrained=/home/jan4021/develop/DSF_public/dum_mocov2_dsf/model_last.pth \
  --dataset=CIFAR10 --epochs=30 --schedule=[5,10,15,20] --print-freq=100


# 3. MoCov3_DSF + ImageNet100
# 3-1) pretrain
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

# 3-2) Linear Evaluation
python main_linclsv3.py \
    /path/to/data \
    --dataset=IN100 \
    --results_dir=mocov3_IN100_dsf \
    -a vit_small --lr 30 -b 256 \
    --dist-url 'tcp://localhost:10002' \
    --world-size 0 --rank 0 \
    --epochs=30 \
    --pretrained /path/to/pth.tar