import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq

def get_parser():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', required=True)
    parser.add_argument('--target_file', help='location of target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--granularity', type=str, help='which granularity to use, frame or utterance', required=True)

    return parser

@dataclass
class UserDirModule:
    user_dir: str

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_file = args.source_file
    target_file = args.target_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity

    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    if source_file.endswith('.wav'):
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        
    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if task.cfg.normalize:
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
        try:
            feats = model.extract_features(source, padding_mask=None)
            feats = feats['x'].squeeze(0).cpu().numpy()
            if granularity == 'frame':
                feats = feats
            elif granularity == 'utterance':
                feats = np.mean(feats, axis=0)
            else:
                raise ValueError("Unknown granularity: {}".format(args.granularity))
            np.save(target_file, feats)
        except:
            Exception("Error in extracting features from {}".format(source_file))


if __name__ == '__main__':
    main()