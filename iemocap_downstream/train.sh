#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

dataset=IEMOCAP
feat_path=$1

python main.py \
    dataset._name=$dataset \
    dataset.feat_path=$feat_path \
    model._name=BaseModel \
    dataset.batch_size=128 \
    optimization.epoch=100 \
    optimization.lr=5e-4 \
    dataset.eval_is_test=false \
