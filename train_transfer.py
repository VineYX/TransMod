"""
TransMod 
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import Config
from models import TransModFramework
from data.dataset import make_rh_loaders
from utils.metrics import compute_metrics_tensor
from utils.logger  import TransModLogger, MetricTracker


class TransferTrainer:
    """
    1.  checkpoint 
    2. freeze_for_transfer()
    3.  RH  prompt_net + pred_head
    """

    def __init__(self, cfg: Config, pretrain_ckpt: str, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.logger = TransModLogger(cfg.log_dir, cfg.exp_name + '_transfer')
        self.pretrain_ckpt = Path(pretrain_ckpt)

        self.best_val_mae = float('inf')
        self.best_epoch   = 0
        self.patience_cnt = 0
        self.global_step  = 0

        self.out_dir = Path(cfg.output_dir) / cfg.exp_name / 'transfer'
        self.out_dir.mkdir(parents=True, exist_ok=True)


    def _make_loaders(self, city: str):
        cc = self.cfg.city_config(city)
        ts = pd.Timestamp(cc.start_ts)
        return make_rh_loaders(
            self.cfg.data_dir, city, ts,
            seq_len=self.cfg.model.seq_len,
            pred_len=self.cfg.model.pred_len,
            batch_size=self.cfg.transfer.batch_size,
            k_neighbors=cc.k_neighbors,
            num_workers=self.cfg.train.num_workers,
        )


    def _load_pretrained(self) -> TransModFramework:
        ckpt = torch.load(self.pretrain_ckpt, map_location=self.device)
        old_cfg: Config = ckpt['cfg']
        mc = old_cfg.model
        lc = old_cfg.loss
        cc = old_cfg.city_config(old_cfg.src_city)

        model = TransModFramework(
            n_bike=old_cfg.city_config('nyc').n_bike if hasattr(old_cfg, '_nyc_n_bike')
                   else 768,
            n_metro=305,
            n_rh_zones=cc.n_rh_zones,
            n_zones=cc.n_zones,
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

        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            self.logger.warning(f'Missing keys: {missing[:5]}...')
        self.logger.success(
            f' (epoch={ckpt["epoch"]}, '
            f'val_mae={ckpt["val_mae"]:.4f})')
        return model


    def _step(
        self,
        model:     TransModFramework,
        batch:     Dict,
        rh_meta:   Dict,
        scaler:    GradScaler,
        optimizer: Optional[optim.Optimizer],
        train:     bool,
    ) -> Dict[str, float]:
        x  = batch['x'].to(self.device)    # [B, T, N_rh]
        tf = batch['tf'].to(self.device)   # [B, T, 8]
        y  = batch['y'].to(self.device)    # [B, pred_len, N_rh]

        zone_coords = rh_meta['coords'].to(self.device)  # [Z, 2]
        rh_edge     = (
            rh_meta['edge_index'].to(self.device),
            torch.ones(rh_meta['edge_index'].size(1), device=self.device),
        )

        with autocast(enabled=self.cfg.train.use_amp and train):
            pred = model.forward_transfer(x, tf, zone_coords, rh_edge)
            # pred: [B, Z, pred_len]  y: [B, pred_len, Z]
            y_perm = y.permute(0, 2, 1)   # [B, Z, pred_len]
            loss = nn.functional.huber_loss(pred, y_perm)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if self.cfg.train.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.cfg.train.grad_clip)
                optimizer.step()
            self.global_step += 1

        mean = torch.tensor(rh_meta['mean'], device=self.device)
        std  = torch.tensor(rh_meta['std'],  device=self.device)
        metrics = compute_metrics_tensor(pred, y_perm, mean, std)
        metrics['loss'] = loss.item()
        return metrics


    def _run_epoch(self, model, loader, rh_meta, scaler, optimizer, train) -> Dict:
        model.train() if train else model.eval()
        tracker = MetricTracker()
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                m = self._step(model, batch, rh_meta, scaler, optimizer, train)
                tracker.update(m)
        return tracker.mean()


    def train(self):
        self.logger.print_banner(
            f'TransMod Transfer  [{self.cfg.src_city}->{self.cfg.tgt_city}]')

        city = self.cfg.tgt_city
        tr_load, vl_load, te_load, rh_meta = self._make_loaders(city)
        self.logger.info(
            f'{city.upper()} RH - zones:{rh_meta["N_zones"]}  '
            f'train:{len(tr_load.dataset)}  val:{len(vl_load.dataset)}  '
            f'test:{len(te_load.dataset)}')

        model = self._load_pretrained()
        model.freeze_for_transfer()
        pc = model.param_count()
        self.logger.info(
            f' - total:{pc["total"]/1e6:.2f}M  '
            f'trainable:{pc["trainable"]/1e6:.2f}M  '
            f'frozen:{pc["frozen"]/1e6:.2f}M')

        tc = self.cfg.transfer
        optimizer = optim.AdamW(
            model.trainable_params(), lr=tc.lr,
            weight_decay=1e-5, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.max_epochs)
        scaler = GradScaler(
            enabled=self.cfg.train.use_amp and torch.cuda.is_available())

        start = time.time()
        for epoch in range(1, tc.max_epochs + 1):
            tr_m = self._run_epoch(model, tr_load, rh_meta, scaler, optimizer, True)
            vl_m = self._run_epoch(model, vl_load, rh_meta, scaler, None, False)
            scheduler.step()

            self.logger.log_scalars('transfer/train', tr_m, epoch)
            self.logger.log_scalars('transfer/val',   vl_m, epoch)

            if epoch % 5 == 0 or epoch == 1:
                self.logger.print_epoch(
                    epoch,
                    {**{f'tr_{k}': v for k, v in tr_m.items()},
                     **{f'vl_{k}': v for k, v in vl_m.items()}},
                    'val')

            val_mae = vl_m.get('MAE', vl_m['loss'])
            if val_mae < self.best_val_mae - 1e-4:
                self.best_val_mae = val_mae
                self.best_epoch   = epoch
                self.patience_cnt = 0
                torch.save(
                    {'epoch': epoch, 'val_mae': val_mae, 'model': model.state_dict()},
                    self.out_dir / 'best.pt')
                self.logger.success(f'[OK] Best checkpoint saved (MAE={val_mae:.4f})')
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= tc.patience:
                    self.logger.warning(
                        f' at epoch {epoch} (best={self.best_epoch})')
                    break

        elapsed = time.time() - start
        self.logger.success(
            f'  best_epoch={self.best_epoch}  '
            f'best_val_MAE={self.best_val_mae:.4f}  '
            f'={elapsed/60:.1f}min')

        ckpt = torch.load(self.out_dir / 'best.pt', map_location=self.device)
        model.load_state_dict(ckpt['model'])
        te_m = self._run_epoch(model, te_load, rh_meta, scaler, None, False)
        self.logger.print_epoch(0, te_m, 'test')
        self.logger.close()
        return model


def parse_args():
    p = argparse.ArgumentParser('TransMod Transfer')
    p.add_argument('--pretrain_ckpt', type=str, required=True,
                   help=' checkpoint  (best.pt)')
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--tgt_city', type=str, default='nyc',
                   help=': nyc | chicago')
    p.add_argument('--src_city', type=str, default='nyc')
    p.add_argument('--exp_name', type=str, default='transmod_v1')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = Config.load(args.config) if args.config else Config()
    cfg.tgt_city = args.tgt_city
    cfg.src_city = args.src_city
    cfg.exp_name = args.exp_name
    if args.max_epochs: cfg.transfer.max_epochs = args.max_epochs
    if args.lr:         cfg.transfer.lr         = args.lr

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TransferTrainer(cfg, args.pretrain_ckpt, device)
    trainer.train()


if __name__ == '__main__':
    main()
