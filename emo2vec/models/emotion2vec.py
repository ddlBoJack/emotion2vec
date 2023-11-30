# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .config import Data2VecMultiConfig
from .modules import AltBlock
from .audio import D2vAudioConfig, AudioEncoder

logger = logging.getLogger(__name__)


@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel):

    def __init__(self, cfg: Data2VecMultiConfig, task=None):
        super().__init__()
        self.cfg = cfg
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        mod_cfg = getattr(cfg.modalities, 'audio')
        enc = AudioEncoder(
            mod_cfg,
            cfg.embed_dim,
            make_block,
            make_layer_norm,
            cfg.layer_norm_first,
            self.alibi_biases,
            task,
        )
        self.modality_encoders['AUDIO'] = enc

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        self.apply(init_bert_params)

        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        return cls(cfg, task=task)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
        **kwargs,
    ):

        feature_extractor = self.modality_encoders['AUDIO']

        mask_seeds = None

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res
