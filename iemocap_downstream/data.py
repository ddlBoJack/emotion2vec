import os
import numpy as np

from torch.utils.data import DataLoader, random_split

from dataset import load_dataset, SpeechDataset

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