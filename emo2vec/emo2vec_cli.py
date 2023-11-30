import argparse
from dataclasses import dataclass
import numpy as np
import os
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq


def extract_features(emo2vec_dir: str = None, checkpoint_dir: str = None, granularity: str = None, **kwargs):

    try:
        import emo2vec
        emo2vec_dir = os.path.dirname(emo2vec.__file__)
    except:
        assert emo2vec_dir is not None, "Please pip install emo2vec, or define the emo2vec_dir"

    # model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(emo2vec_dir)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.to(kwargs.get("device", "cpu"))
    
    def _extract(source_file: str = None, target_file: str = None):
    
        if source_file.endswith('.wav'):
            wav, sr = sf.read(source_file)
            channel = sf.info(source_file).channels
            assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
            assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
    
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
                raise ValueError("Unknown granularity: {}".format(granularity))
            if target_file is not None:
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                np.save(target_file, feats)
        except:
            Exception("Error in extracting features from {}".format(source_file))
        return feats
    
    return _extract
    

# @dataclass
# class UserDirModule:
#     user_dir: str

def main():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--emo2vec_dir', type=str, default="/Users/zhifu/emotion2vec/emo2vec", help='emo2vec_dir for source code', required=False)
    parser.add_argument('--checkpoint_dir', type=str, default="/Users/zhifu/Downloads/emotion2vec_base.pt", help='checkpoint for pre-trained model', required=False)
    parser.add_argument('--granularity', type=str, default="utterance", help='which granularity to use, frame or utterance',
                        required=False)
    parser.add_argument('--source_file', default="/Users/zhifu/emotion2vec/scripts/test.wav", help='location of source wav files', required=False)
    parser.add_argument('--target_file', default="/Users/zhifu/emotion2vec/scripts/test.npz", help='location of target npy files', required=False)

    args = parser.parse_args()
    print(args)

    extractor = extract_features(emo2vec_dir=args.emo2vec_dir, checkpoint_dir=args.checkpoint_dir, granularity=args.granularity)
    extractor(args.source_file, args.target_file)


if __name__ == '__main__':
    main()
    # import emotion2vec
    # emo_fn = emotion2vec("")
    # feats = emo_fn("test.wav")