"""
TransMod unified training script.
Runs end-to-end on CitiBike temporal snapshots with modern training practices:
  - AdamW + parameter-group weight decay
  - Cosine LR schedule with linear warmup (epoch-level)
  - Mixed-precision (AMP) when CUDA available
  - MAE prediction loss + memory diversity regularisation
  - Early stopping on validation MAE
  - Masked MAPE (skip zero-demand stations)
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.citibike_dataset import build_datasets, collate_fn
from models.transmod_single import TransModSingle
from utils.metrics import compute_metrics


def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int,
              base_lr: float, min_lr: float = 1e-6) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def run_epoch(model, loader, edge_index, optimizer, scaler,
              lambda_div: float, device: torch.device, train: bool):
    model.train(train)
    total_mae = total_div = total_n = 0

    ctx = autocast('cuda', enabled=(device.type == 'cuda'))

    with torch.set_grad_enabled(train):
        for x_seq, target in loader:
            x_seq  = x_seq.to(device)       # [B, T, N, F]
            target = target.to(device)       # [B, N]
            ei     = edge_index.to(device)   # [2, E]

            with ctx:
                pred, div_loss = model(x_seq, ei)          # [B, N, 1]
                mae  = F.l1_loss(pred.squeeze(-1), target)
                loss = mae + lambda_div * div_loss

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            B = x_seq.size(0)
            total_mae += mae.item() * B
            total_div += div_loss.item() * B
            total_n   += B

    return total_mae / total_n, total_div / total_n


def masked_mape(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.1) -> float:
    """MAPE computed only where |target| > threshold (avoids near-zero blow-up)."""
    mask = target.abs() > threshold
    if mask.sum() == 0:
        return float('nan')
    err = ((pred[mask] - target[mask]).abs() / target[mask].abs()) * 100
    return err.mean().item()


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, edge_index, N, F_in = build_datasets(
        args.snapshots,
        window=args.window,
        k_neighbors=args.k_neighbors,
    )
    print(f"Stations: {N}  Features: {F_in}  Window: {args.window}")
    print(f"Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)

    model = TransModSingle(
        in_features    = F_in,
        hidden_dim     = args.hidden_dim,
        memory_size    = args.memory_size,
        num_gat_layers = args.gat_layers,
        num_gat_heads  = args.gat_heads,
        num_gru_layers = args.gru_layers,
        dropout        = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, args.weight_decay)
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_path    = os.path.join(args.ckpt_dir, 'best.pth')
    best_val_mae = float('inf')
    patience_ctr = 0

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    for epoch in range(args.epochs):
        lr = cosine_lr(epoch, args.epochs, args.warmup_epochs, args.lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        t0 = time.time()
        train_mae, train_div = run_epoch(
            model, train_loader, edge_index, optimizer, scaler,
            args.lambda_div, device, train=True)

        val_mae, _ = run_epoch(
            model, val_loader, edge_index, optimizer, scaler,
            args.lambda_div, device, train=False)

        print(f"Ep {epoch+1:03d}/{args.epochs}  "
              f"train_mae={train_mae:.4f}  div={train_div:.4f}  "
              f"val_mae={val_mae:.4f}  "
              f"lr={lr:.2e}  t={time.time()-t0:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'val_mae': val_mae}, best_path)
            print(f"  [OK] Best saved (val_mae={val_mae:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\n" + "=" * 60)
    print("Evaluating on test set (best checkpoint)")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state'])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for x_seq, target in test_loader:
            pred, _ = model(x_seq.to(device), edge_index.to(device))
            preds.append(pred.squeeze(-1).cpu())
            tgts.append(target.cpu())

    preds = torch.cat(preds, dim=0)   # [n_test, N]
    tgts  = torch.cat(tgts,  dim=0)   # [n_test, N]

    # De-normalise using training stats
    std_out  = train_ds.std [0, 2].item()
    mean_out = train_ds.mean[0, 2].item()
    p_dn = preds * std_out + mean_out
    t_dn = tgts  * std_out + mean_out

    mae  = F.l1_loss(p_dn, t_dn).item()
    rmse = torch.sqrt(F.mse_loss(p_dn, t_dn)).item()
    mape = masked_mape(p_dn, t_dn, threshold=0.1)

    print("=" * 60)
    print(f"Test MAE  : {mae:.4f}  (trips/station/hr)")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test MAPE : {mape:.2f}%  (masked, |target|>0.1)")
    print("=" * 60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots',     default='data/citibike_graphs/temporal_snapshots.pkl')
    p.add_argument('--ckpt_dir',      default='checkpoints')
    p.add_argument('--seed',          type=int,   default=42)
    # Data
    p.add_argument('--window',        type=int,   default=8)
    p.add_argument('--k_neighbors',   type=int,   default=5)
    p.add_argument('--batch_size',    type=int,   default=4)
    # Model
    p.add_argument('--hidden_dim',    type=int,   default=64)
    p.add_argument('--memory_size',   type=int,   default=64)
    p.add_argument('--gat_layers',    type=int,   default=2)
    p.add_argument('--gat_heads',     type=int,   default=4)
    p.add_argument('--gru_layers',    type=int,   default=2)
    p.add_argument('--dropout',       type=float, default=0.3)
    # Training
    p.add_argument('--epochs',        type=int,   default=200)
    p.add_argument('--warmup_epochs', type=int,   default=10)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-2)
    p.add_argument('--lambda_div',    type=float, default=0.05)
    p.add_argument('--patience',      type=int,   default=30)

    main(p.parse_args())
