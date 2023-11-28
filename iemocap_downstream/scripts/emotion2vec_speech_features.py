#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile
from dataclasses import dataclass

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="extract data2vec 2.0 features for downstream tasks"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('fairseq_root', help='location of fairseq root')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for data2vec model', required=True)
    parser.add_argument('--layer', type=int, default=0, 
                        help='which layer to use. Base: 0-11. ')
    # fmt: on

    return parser


@dataclass
class Data2vec2Module:
    user_dir: str


class Data2vec2FeatureReader(object):
    def __init__(self, cp_file, layer, fairseq_root):
        os.chdir(fairseq_root)
        data2vec2_path = Data2vec2Module(
            os.path.join(fairseq_root, 'examples/data2vec'))
        fairseq.utils.import_user_module(data2vec2_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            res = self.model.extract_features(source, padding_mask=None, remove_extra_tokens=True)
            return res['x'].squeeze(0).cpu()

def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0])
                 for line in lines if len(line) > 0]

        num = len(files)
        reader = Data2vec2FeatureReader(
            args.checkpoint, args.layer, args.fairseq_root)

        def iterate():
            for fname in files:
                d2v_feats = reader.get_feats(fname)
                yield d2v_feats

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = osp.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for d2v_feats in tqdm.tqdm(iterator, total=num):
            print(len(d2v_feats), file=l_f)

            if len(d2v_feats) > 0:
                npaa.append(d2v_feats.numpy())


if __name__ == "__main__":
    main()
