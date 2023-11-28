# Downstream task for emotion2vec

## Download IEMOCAP dataset
1. download from official website [IEMOCAP RELEASE](https://docs.google.com/forms/d/e/1FAIpQLScBecgI2K5bFTrXi_-05IYSSwOcqL5mX7dh57xcJV1m_NoznA/viewform?usp=sf_link)
2. download from our perserved data [Google Drive](TODO)

## Data processing and extract features
```bash
IEMOCAP_ROOT=/mnt/lustre/sjtu/home/zsz01/data/SER/IEMOCAP_full_release
manifest_path=/mnt/lustre/sjtu/home/zsz01/codes/emotion-recognition/manifest
checkpoint=/mnt/lustre/sjtu/home/zym22/models/emotion2vec/audio2_base_libri_cp_iemocap_meld_cmumosei_msppodcast_mead_cls1_clstype_chunk10_warmup5000_lr75e-5/checkpoint_last.pt
feat_path=/mnt/lustre/sjtu/home/zsz01/codes/emotion-recognition/feats/
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