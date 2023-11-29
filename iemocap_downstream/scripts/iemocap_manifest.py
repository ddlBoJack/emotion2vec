#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import os

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR", 
        default='/path/to/IEMOCAP_full_release',
        help="root directory containing audio files to index"
    )
    parser.add_argument(
        "--dest", default="/path/to/manifest", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--label_path", default="/path/to/train.emo",
    )
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    root = os.path.join(args.root, 'Session{}')
    with open(args.label_path) as rf, open(args.dest + '/train.tsv', 'w') as wf:
        print(args.root, file=wf)
        for line in rf.readlines():
            fname = line.split('\t')[0]
            session = fname[4]
            folder = fname.rsplit('_', 1)[0]
            
            fname = os.path.join(root.format(session), 'sentences/wav', folder, fname + '.wav')
            frames = soundfile.info(fname).frames
            suffix = fname.replace(args.root, '', 1).lstrip('/')
            print(suffix, frames, sep='\t', file=wf)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
