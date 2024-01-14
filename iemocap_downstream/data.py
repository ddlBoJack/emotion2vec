import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

def load_dataset(data_path, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []

    npy_data = np.load(data_path + ".npy")

    offset = 0
    skipped = 0

    if not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels

class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        labels=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample

        self.labels = labels

        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )

        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):

    data, sizes, offsets, labels = load_dataset(feature_path, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels ]
    
    num = len(labels)
    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "num": num
    } 

    return iemocap_data

def train_valid_test_iemocap_dataloader(
        data, 
        batch_size,
        test_start, 
        test_end,
        eval_is_test=False,
    ):
    feats = data['feats']
    sizes, offsets = data['sizes'], data['offsets']
    labels = data['labels']

    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_labels = labels[test_start:test_end]

    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start
    
    test_dataset = SpeechDataset(
        feats=test_feats,
        sizes=test_sizes, 
        offsets=test_offsets,
        labels=test_labels,
    )

    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)

    if eval_is_test:
        train_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
        )
        val_dataset = test_dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
    else:
        train_val_nums = data['num'] - (test_end - test_start)
        train_nums = int(0.8 * train_val_nums)
        val_nums = train_val_nums - train_nums

        train_val_dataset = SpeechDataset(
            feats=train_val_feats, 
            sizes=train_val_sizes, 
            offsets=train_val_offsets,
            labels=train_val_labels,
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator, 
                                num_workers=4, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader