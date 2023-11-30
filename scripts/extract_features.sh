#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python extract_features.py  \
--source_file='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/scripts/test.wav' \
--target_file='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/scripts/test.npy' \
--model_dir='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/upstream' \
--checkpoint_dir='/mnt/lustre/sjtu/home/zym22/models/released/emotion2vec/emotion2vec_base.pt' \
--granularity='utterance' \
