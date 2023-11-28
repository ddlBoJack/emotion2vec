#!/bin/bash

IEMOCAP_ROOT=$1
output_path=$2

mkdir -p $output_path

for index in {1..5}; do
    cat $IEMOCAP_ROOT/Session$index/dialog/EmoEvaluation/*.txt | \
        grep Ses | cut -f2,3 | \
        awk '{if ($2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "neu" || $2 == "sad") print $0}' | \
        sed 's/\bexc\b/hap/g' > $output_path/Session${index}.emo
done

for index in {1..5}; do
    cat $output_path/Session${index}.emo >> $output_path/train.emo
    rm -rf $output_path/Session${index}.emo
done

python scripts/iemocap_manifest.py \
    --root $IEMOCAP_ROOT --dest $output_path \
    --label_path $output_path/train.emo