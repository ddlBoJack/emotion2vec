# Downstream task for emotion2vec

## Download IEMOCAP dataset
1. download from official website [IEMOCAP RELEASE](https://docs.google.com/forms/d/e/1FAIpQLScBecgI2K5bFTrXi_-05IYSSwOcqL5mX7dh57xcJV1m_NoznA/viewform?usp=sf_link)
2. download from our perserved data [Google Drive](TODO)

## Data processing and extract features
```bash
# set some Variable
IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
manifest_path=/path/to/manifest
checkpoint=/path/to/emotion2vec_ckpt
feat_path=/path/to/feats/

# 1. generate manifest and labels from raw dataset, replace iemocap_root and output_path
bash scripts/iemocap_manifest_and_labels.sh $IEMOCAP_ROOT $manifest_path

# 2. generate features from emotion2vec checkouts, replace fairseq_root
bash scripts/emotion2vec_extract_features.sh $manifest_path $checkpoint $feat_path

cp $manifest_path/train.emo $feat_path
```

## Train (5 fold)
```bash
bash train.sh ${feat_path}/train /path/to/save_dir
```