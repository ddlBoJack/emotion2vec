from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II
from fairseq.dataclass import FairseqDataclass
from .audio import D2vAudioConfig
from .modules import Modality

@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    audio: D2vAudioConfig = D2vAudioConfig()


@dataclass
class Data2VecMultiConfig(FairseqDataclass):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    clone_batch: int = 1

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")

    skip_ema: bool = True

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 0

    decoder_group: bool = False

    adversarial_training: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use adversarial training between different corpus"},
    )
    adversarial_hidden_dim: Optional[int] = field(
        default=128,
        metadata={"help": "hidden dim for adversarial training"},
    )
    adversarial_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "weight for adversarial training"},
    )
    cls_type: Optional[str] = field(
        default="single",
        metadata={"help": "type of utterance-level loss"},
    )