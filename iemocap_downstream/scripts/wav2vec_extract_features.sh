#!/bin/bash
export PYTHONPATH=/path/to/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cd /path/to/this_scripts/
checkpoint=/path/to/wav2vec_small.pt
manifest=/path/to/manifest
save_dir=/path/to/save_dir

true_layer=$[$layer+1]
python wav2vec_extract_features.py  \
    $manifest \
    --split=Session_all \
    --save-dir=$save_dir \
    --checkpoint=$checkpoint \
    --layer=$layer 

