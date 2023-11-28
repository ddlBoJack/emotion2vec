#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

fairseq_root=/path/to/fairseq
export PYTHONPATH=$fairseq_root:$PYTHONPATH

manifest_path=$1
checkpoint=$2
save_dir=$3

# Here we only extract the last layer
for layer in {11..11}; do 
    true_layer=$[$layer+1]
    python scripts/emotion2vec_speech_features.py  \
        $manifest_path $fairseq_root \
        --split=train --save-dir=$3 \
        --checkpoint=$checkpoint \
        --layer=$layer
done