"""
TransMod entry point.

Usage:
    python main.py pretrain --src_city nyc --exp_name transmod_nyc

    python main.py transfer \
        --pretrain_ckpt outputs/transmod_nyc/pretrain/best.pt \
        --tgt_city nyc --exp_name transmod_nyc_rh

    python main.py transfer \
        --pretrain_ckpt outputs/transmod_nyc/pretrain/best.pt \
        --src_city nyc --tgt_city chicago --exp_name transmod_nyc2chi

    python main.py evaluate \
        --ckpt outputs/transmod_nyc_rh/transfer/best.pt \
        --tgt_city nyc
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import Config


def cmd_pretrain(args):
    from train_pretrain import PretrainTrainer
    import torch, pandas as pd

    cfg = Config.load(args.config) if args.config else Config()
    cfg.exp_name  = args.exp_name
    cfg.src_city  = args.src_city
    cfg.seed      = args.seed
    if args.max_epochs: cfg.train.max_epochs = args.max_epochs
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr:         cfg.train.lr         = args.lr
    if args.no_amp:     cfg.train.use_amp    = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PretrainTrainer(cfg, device).train()


def cmd_transfer(args):
    from train_transfer import TransferTrainer
    import torch

    cfg = Config.load(args.config) if args.config else Config()
    cfg.exp_name = args.exp_name
    cfg.src_city = args.src_city
    cfg.tgt_city = args.tgt_city
    if args.max_epochs: cfg.transfer.max_epochs = args.max_epochs
    if args.lr:         cfg.transfer.lr         = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TransferTrainer(cfg, args.pretrain_ckpt, device).train()


def cmd_evaluate(args):
    """Load a trained transfer checkpoint and evaluate on the test set."""
    import torch
    import pandas as pd
    from models import TransModFramework
    from data.dataset import make_rh_loaders
    from utils.metrics import compute_metrics_tensor
    from utils.logger  import TransModLogger, MetricTracker

    cfg = Config.load(args.config) if args.config else Config()
    cfg.tgt_city = args.tgt_city

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(args.ckpt, map_location=device)
    logger = TransModLogger(cfg.log_dir, 'evaluate')

    if 'cfg' in ckpt:
        cfg = ckpt['cfg']

    cc = cfg.city_config(cfg.tgt_city)
    ts = pd.Timestamp(cc.start_ts)
    _, _, te_load, rh_meta = make_rh_loaders(
        cfg.data_dir, cfg.tgt_city, ts,
        seq_len=cfg.model.seq_len,
        pred_len=cfg.model.pred_len,
        batch_size=cfg.train.batch_size,
        k_neighbors=cc.k_neighbors,
    )

    mc = cfg.model
    lc = cfg.loss
    model = TransModFramework(
        n_bike=768, n_metro=305,
        n_rh_zones=rh_meta['N_zones'],
        n_zones=cc.n_zones,
        hidden_dim=mc.hidden_dim, pred_len=mc.pred_len,
        num_gat_layers=mc.num_gat_layers, num_gat_heads=mc.num_gat_heads,
        tcn_layers=mc.tcn_layers, tcn_kernel=mc.tcn_kernel,
        top_k_bike=mc.top_k_bike, top_k_metro=mc.top_k_metro,
        sigma_bike=mc.sigma_bike, sigma_metro=mc.sigma_metro,
        memory_size=mc.memory_size, tau=mc.tau, delta=mc.delta,
        dropout=mc.dropout,
        lambda_mmd=lc.lambda_mmd, lambda_nce=lc.lambda_nce,
        lambda_geo=lc.lambda_geo, lambda_div=lc.lambda_div,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    tracker = MetricTracker()
    mean = torch.tensor(rh_meta['mean'], device=device)
    std  = torch.tensor(rh_meta['std'],  device=device)
    zone_coords = rh_meta['coords'].to(device)

    with torch.no_grad():
        for batch in te_load:
            x  = batch['x'].to(device)
            tf = batch['tf'].to(device)
            y  = batch['y'].to(device)
            pred = model.forward_transfer(x, tf, zone_coords)
            y_perm = y.permute(0, 2, 1)
            m = compute_metrics_tensor(pred, y_perm, mean, std)
            tracker.update(m)

    results = tracker.mean()
    logger.print_epoch(0, results, 'test')
    print('\n=== Test Results ===')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
    logger.close()


def build_parser():
    p = argparse.ArgumentParser(
        'TransMod',
        description='Cross-modal Urban Mobility Forecasting via Memory Transfer')
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('pretrain', help='pre-training stage')
    sp.add_argument('--config',      type=str,  default=None)
    sp.add_argument('--exp_name',    type=str,  default='transmod_v1')
    sp.add_argument('--src_city',    type=str,  default='nyc')
    sp.add_argument('--max_epochs',  type=int,  default=None)
    sp.add_argument('--batch_size',  type=int,  default=None)
    sp.add_argument('--lr',          type=float, default=None)
    sp.add_argument('--seed',        type=int,  default=42)
    sp.add_argument('--no_amp',      action='store_true')

    sp = sub.add_parser('transfer', help='transfer learning stage')
    sp.add_argument('--pretrain_ckpt', type=str, required=True)
    sp.add_argument('--config',        type=str, default=None)
    sp.add_argument('--exp_name',      type=str, default='transmod_v1')
    sp.add_argument('--src_city',      type=str, default='nyc')
    sp.add_argument('--tgt_city',      type=str, default='chicago')
    sp.add_argument('--max_epochs',    type=int, default=None)
    sp.add_argument('--lr',            type=float, default=None)

    sp = sub.add_parser('evaluate', help='evaluate on test set')
    sp.add_argument('--ckpt',       type=str, required=True)
    sp.add_argument('--config',     type=str, default=None)
    sp.add_argument('--tgt_city',   type=str, default='nyc')

    return p


def main():
    args = build_parser().parse_args()
    {'pretrain': cmd_pretrain,
     'transfer': cmd_transfer,
     'evaluate': cmd_evaluate}[args.cmd](args)


if __name__ == '__main__':
    main()
