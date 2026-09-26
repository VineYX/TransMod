"""
TransMod 
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """"""
    hidden_dim:    int   = 64
    pred_len:      int   = 1
    seq_len:       int   = 12
    num_gat_layers: int  = 2
    num_gat_heads:  int  = 4
    tcn_layers:    int   = 4
    tcn_kernel:    int   = 3
    top_k_bike:    int   = 5
    top_k_metro:   int   = 8
    sigma_bike:    float = 500.0
    sigma_metro:   float = 800.0
    memory_size:   int   = 128
    delta:         float = 0.3
    tau:           float = 0.07
    dropout:       float = 0.1


@dataclass
class LossConfig:
    """"""
    lambda_mmd:   float = 0.1
    lambda_nce:   float = 0.5
    lambda_geo:   float = 0.1
    lambda_div:   float = 0.05


@dataclass
class TrainConfig:
    """"""
    batch_size:    int   = 32
    num_workers:   int   = 0
    max_epochs:    int   = 100
    patience:      int   = 15
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    grad_clip:     float = 5.0
    scheduler:     str   = 'cosine'   # 'cosine' | 'plateau' | 'none'
    warmup_epochs: int   = 5
    use_amp:       bool  = True
    save_top_k:    int   = 3
    log_interval:  int   = 10


@dataclass
class TransferConfig:
    """"""
    lr:          float = 1e-4
    max_epochs:  int   = 50
    patience:    int   = 10
    batch_size:  int   = 16


@dataclass
class CityConfig:
    """"""
    name:          str
    start_ts:      str
    n_bike:        int = 0
    n_metro:       int = 0
    n_rh_zones:    int = 0
    n_zones:       int = 200
    k_neighbors:   int = 5


NYC_CONFIG = CityConfig(
    name='nyc',
    start_ts='2018-11-01 00:00:00',
    n_bike=768,
    n_metro=305,
    n_rh_zones=263,
    n_zones=200,
    k_neighbors=5,
)

CHICAGO_CONFIG = CityConfig(
    name='chicago',
    start_ts='2018-11-01 00:00:00',
    n_bike=592,
    n_metro=144,
    n_rh_zones=77,
    n_zones=100,
    k_neighbors=5,
)


@dataclass
class Config:
    """"""
    model:     ModelConfig    = field(default_factory=ModelConfig)
    loss:      LossConfig     = field(default_factory=LossConfig)
    train:     TrainConfig    = field(default_factory=TrainConfig)
    transfer:  TransferConfig = field(default_factory=TransferConfig)

    project_root: str = str(Path(__file__).parent)
    data_dir:     str = str(Path(__file__).parent / 'data' / 'processed')
    output_dir:   str = str(Path(__file__).parent / 'outputs')
    log_dir:      str = str(Path(__file__).parent / 'logs')

    exp_name:     str = 'transmod_v1'
    seed:         int = 42

    src_city:     str = 'nyc'
    tgt_city:     str = 'chicago'

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.model    = ModelConfig(**d.get('model', {}))
        cfg.loss     = LossConfig(**d.get('loss', {}))
        cfg.train    = TrainConfig(**d.get('train', {}))
        cfg.transfer = TransferConfig(**d.get('transfer', {}))
        for k in ['project_root', 'data_dir', 'output_dir', 'log_dir',
                  'exp_name', 'seed', 'src_city', 'tgt_city']:
            if k in d:
                setattr(cfg, k, d[k])
        return cfg

    def city_config(self, city: str) -> CityConfig:
        return {'nyc': NYC_CONFIG, 'chicago': CHICAGO_CONFIG}[city]
