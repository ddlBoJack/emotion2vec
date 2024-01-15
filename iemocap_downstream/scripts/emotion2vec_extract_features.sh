#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$1:$PYTHONPATH

manifest_path=$2
model_path=$3
checkpoint_path=$4
save_dir=$5

# Here we only extract the last layer
for layer in {11..11}; do 
    true_layer=$[$layer+1]
    echo "Extracting features from layer $true_layer"

    python scripts/emotion2vec_speech_features.py  \
        --data $manifest_path \
        --model $model_path \
        --split=train \
        --checkpoint=$checkpoint_path \
        --save-dir=$save_dir \
        --layer=$layer
done