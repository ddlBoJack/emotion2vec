#!/bin/bash

dataset=IEMOCAP
feat_path=$1
save_dir=$2

if [ -z "$1" ]; then
    device=0
else
    device=$3
fi

python main.py \
    common.device=$device \
    dataset._name=$dataset \
    dataset.feat_path=$feat_path \
    model._name=BaseModel \
    dataset.batch_size=128 \
    optimization.epoch=100 \
    optimization.lr=5e-4 \
    dataset.eval_is_test=false \
    model.save_dir=$save_dir \

