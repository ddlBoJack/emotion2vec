# emotion2vec
Code for extracting features with emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation


## Extract Features
1. git clone repos.
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
git clone https://github.com/ddlBoJack/emotion2vec
```

2. download emotion2vec checkpoint from:
- [Google Drive](https://drive.google.com/file/d/1vzJdLTogkbhGc_ncNUc6xH2riS8oDGDI/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1-KXR6Zhl6VxxddQbKf5YJQ?pwd=1jny) (password: 1jny).

3. modify and run `emotion2vec/scripts/extract_features.sh`
