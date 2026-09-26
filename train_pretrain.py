"""
TransMod 
 NYC + Chicago  bike & metro 
 EV_DT  Experiment/Trainer 
"""
from __future__ import annotations
import os
import sys
import math
import random
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import Config, NYC_CONFIG, CHICAGO_CONFIG
from models import TransModFramework
from data.dataset import make_multimodal_loaders
from utils.metrics import compute_metrics_tensor
from utils.logger  import TransModLogger, MetricTracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f'{n / 1e6:.2f}M'


#  Trainer                                        #

class PretrainTrainer:
    """
     TransMod :
    -  (NYC + Chicago) 
    -  + checkpoint
    - rich  + TensorBoard
    """

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.logger = TransModLogger(cfg.log_dir, cfg.exp_name)
        self.best_val_mae = float('inf')
        self.best_epoch   = 0
        self.patience_cnt = 0
        self.global_step  = 0

        self.out_dir = Path(cfg.output_dir) / cfg.exp_name / 'pretrain'
        self.out_dir.mkdir(parents=True, exist_ok=True)


    def _make_loaders(self, city: str):
        cc = self.cfg.city_config(city)
        ts = pd.Timestamp(cc.start_ts)
        return make_multimodal_loaders(
            self.cfg.data_dir, city, ts,
            seq_len=self.cfg.model.seq_len,
            pred_len=self.cfg.model.pred_len,
            batch_size=self.cfg.train.batch_size,
            k_neighbors=cc.k_neighbors,
            num_workers=self.cfg.train.num_workers,
        )


    def _build_model(self, nyc_meta: Dict, chi_meta: Dict) -> TransModFramework:
        mc = self.cfg.model
        lc = self.cfg.loss
        cc_nyc = self.cfg.city_config('nyc')
        cc_chi = self.cfg.city_config('chicago')

        n_zones = cc_nyc.n_zones

        model = TransModFramework(
            n_bike=nyc_meta['N_bike'],
            n_metro=nyc_meta['N_metro'],
            n_rh_zones=cc_nyc.n_rh_zones,
            n_zones=n_zones,
            hidden_dim=mc.hidden_dim,
            pred_len=mc.pred_len,
            num_gat_layers=mc.num_gat_layers,
            num_gat_heads=mc.num_gat_heads,
            tcn_layers=mc.tcn_layers,
            tcn_kernel=mc.tcn_kernel,
            top_k_bike=mc.top_k_bike,
            top_k_metro=mc.top_k_metro,
            sigma_bike=mc.sigma_bike,
            sigma_metro=mc.sigma_metro,
            memory_size=mc.memory_size,
            tau=mc.tau,
            delta=mc.delta,
            dropout=mc.dropout,
            lambda_mmd=lc.lambda_mmd,
            lambda_nce=lc.lambda_nce,
            lambda_geo=lc.lambda_geo,
            lambda_div=lc.lambda_div,
        ).to(self.device)

        bike_coords  = nyc_meta['bike_coords'].to(self.device)
        metro_coords = nyc_meta['metro_coords'].to(self.device)
        zone_centers = self._make_zone_centers(bike_coords, n_zones)
        model.initialize_geography(bike_coords, metro_coords, zone_centers)

        self.zone_centers = zone_centers
        self.logger.info(f': {count_params(model)}')
        return model

    @staticmethod
    def _make_zone_centers(bike_coords: torch.Tensor, n_zones: int) -> torch.Tensor:
        """"""
        N = bike_coords.size(0)
        idx = torch.linspace(0, N - 1, n_zones).long()
        return bike_coords[idx].clone()   # [n_zones, 2]


    def _step(
        self,
        model:   TransModFramework,
        batch:   Dict,
        scaler:  GradScaler,
        optimizer: optim.Optimizer,
        train: bool = True,
    ) -> Dict[str, float]:
        bike_x  = batch['bike_x'] .to(self.device)   # [B, T, N_bike]
        metro_x = batch['metro_x'].to(self.device)   # [B, T, N_metro]
        tf      = batch['tf']     .to(self.device)   # [B, T, 8]
        bike_y  = batch['bike_y'] .to(self.device)   # [B, pred_len, N_bike]
        metro_y = batch['metro_y'].to(self.device)

        with autocast(enabled=self.cfg.train.use_amp and train):
            pred, losses = model.forward_pretrain(
                bike_seq=bike_x.permute(0, 1, 2),   # already [B,T,N]
                metro_seq=metro_x.permute(0, 1, 2),
                tf=tf,
            )
            Z = pred.size(1)
            target_bike  = bike_y.mean(dim=-1, keepdim=True)   # [B, pred_len, 1]
            target_metro = metro_y.mean(dim=-1, keepdim=True)

            pred_mean = pred.mean(dim=1, keepdim=True)          # [B, 1, pred_len]
            gt_mean   = (target_bike + target_metro) / 2        # [B, pred_len, 1]
            gt_mean   = gt_mean.permute(0, 2, 1)               # [B, 1, pred_len]

            total_loss = model.compute_pretrain_loss(pred_mean, gt_mean, losses)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if self.cfg.train.use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
                optimizer.step()

        return {
            'loss': total_loss.item(),
            'mmd':  losses['mmd'].item(),
            'nce':  losses['nce'].item(),
            'geo':  losses['geo'].item(),
            'div':  losses['div'].item(),
        }


    def _run_epoch(
        self,
        model:     TransModFramework,
        loader,
        scaler:    GradScaler,
        optimizer: Optional[optim.Optimizer],
        train:     bool,
    ) -> Dict[str, float]:
        model.train() if train else model.eval()
        tracker = MetricTracker()
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                metrics = self._step(model, batch, scaler, optimizer, train)
                tracker.update(metrics)
                if train:
                    self.global_step += 1

        return tracker.mean()


    def _save_checkpoint(self, model: TransModFramework, epoch: int,
                         val_mae: float, tag: str = 'best'):
        ckpt = {
            'epoch':     epoch,
            'val_mae':   val_mae,
            'model':     model.state_dict(),
            'cfg':       self.cfg,
        }
        path = self.out_dir / f'{tag}.pt'
        torch.save(ckpt, path)
        self.logger.success(f'[OK] Checkpoint saved: {path}  (val MAE={val_mae:.4f})')


    def train(self):
        self.logger.print_banner(
            f'TransMod Pre-training  [{self.cfg.src_city.upper()}]')
        set_seed(self.cfg.seed)

        self.logger.info('...')
        tr_load, vl_load, te_load, meta_nyc = self._make_loaders('nyc')
        self.logger.info(
            f'NYC - bike:{meta_nyc["N_bike"]} metro:{meta_nyc["N_metro"]}  '
            f'train:{len(tr_load.dataset)}  val:{len(vl_load.dataset)}  '
            f'test:{len(te_load.dataset)}')

        model = self._build_model(meta_nyc, {})
        self.model = model

        tc = self.cfg.train
        optimizer = optim.AdamW(
            model.parameters(), lr=tc.lr,
            weight_decay=tc.weight_decay, amsgrad=True)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=tc.max_epochs - tc.warmup_epochs, eta_min=tc.lr * 0.01)
        scaler = GradScaler(enabled=tc.use_amp and torch.cuda.is_available())

        start_time = time.time()

        for epoch in range(1, tc.max_epochs + 1):
            if epoch <= tc.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg['lr'] = tc.lr * epoch / tc.warmup_epochs

            train_m = self._run_epoch(model, tr_load, scaler, optimizer, train=True)
            val_m   = self._run_epoch(model, vl_load, scaler, None, train=False)

            if epoch > tc.warmup_epochs:
                scheduler.step()

            self.logger.log_scalars('pretrain/train', train_m, epoch)
            self.logger.log_scalars('pretrain/val',   val_m,   epoch)

            if epoch % tc.log_interval == 0 or epoch == 1:
                self.logger.print_epoch(epoch, {**{f'tr_{k}': v for k, v in train_m.items()},
                                                 **{f'vl_{k}': v for k, v in val_m.items()}},
                                        'train')

            val_mae = val_m['loss']
            if val_mae < self.best_val_mae - 1e-4:
                self.best_val_mae = val_mae
                self.best_epoch   = epoch
                self.patience_cnt = 0
                self._save_checkpoint(model, epoch, val_mae, 'best')
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= tc.patience:
                    self.logger.warning(
                        f' at epoch {epoch} (best={self.best_epoch})')
                    break

        elapsed = time.time() - start_time
        self.logger.success(
            f'  best_epoch={self.best_epoch}  '
            f'best_val_loss={self.best_val_mae:.4f}  '
            f'={elapsed/60:.1f}min')

        self.logger.info('...')
        ckpt = torch.load(self.out_dir / 'best.pt', map_location=self.device)
        model.load_state_dict(ckpt['model'])
        test_m = self._run_epoch(model, te_load, scaler, None, train=False)
        self.logger.print_epoch(0, test_m, 'test')
        self.logger.close()
        return model


def parse_args():
    p = argparse.ArgumentParser('TransMod Pretraining')
    p.add_argument('--config', type=str, default=None, help='JSON config ')
    p.add_argument('--exp_name', type=str, default='transmod_v1')
    p.add_argument('--src_city', type=str, default='nyc')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config) if args.config else Config()

    cfg.exp_name  = args.exp_name
    cfg.src_city  = args.src_city
    cfg.seed      = args.seed
    if args.max_epochs: cfg.train.max_epochs = args.max_epochs
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr:         cfg.train.lr         = args.lr
    if args.no_amp:     cfg.train.use_amp    = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f': {device}')

    trainer = PretrainTrainer(cfg, device)
    model   = trainer.train()

    cfg.save(str(trainer.out_dir / 'config.json'))


if __name__ == '__main__':
    main()
