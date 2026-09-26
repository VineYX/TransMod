from __future__ import annotations
import sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import Config, NYC_CONFIG
from data.dataset import ModalDataset, temporal_features, build_knn_edge_index
from utils.metrics import compute_metrics
from baselines import LSTMModel, GRUModel, STGCNLite, LSTMShared

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

SEQ_LEN  = 12
PRED_LEN = 1
BATCH    = 32
EPOCHS   = 15      # baseline epochs (quick CPU run; increase for full training)
PATIENCE = 5
LR       = 1e-3

DATA_DIR = ROOT / 'data' / 'processed'
START_TS = pd.Timestamp('2018-11-01')


def load_rh(city='nyc'):
    base = DATA_DIR / city
    demand = np.load(base / 'ridehailing_hourly_demand.npy').astype(np.float32)
    zones  = pd.read_csv(base / 'ridehailing_zones.csv')
    coords = zones[['lat', 'lon']].values.astype(np.float32)
    return demand, coords


def load_bike(city='nyc'):
    base   = DATA_DIR / city
    demand = np.load(base / 'bike_hourly_demand.npy').astype(np.float32)
    sdf    = pd.read_csv(base / 'bike_stations.csv')
    coords = sdf[['lat', 'lon']].values.astype(np.float32)
    return demand, coords


def make_loaders(demand, coords, batch=BATCH, k=5):
    """"""
    tr = ModalDataset(demand, coords, START_TS, SEQ_LEN, PRED_LEN, 'train', k_neighbors=k)
    vl = ModalDataset(demand, coords, START_TS, SEQ_LEN, PRED_LEN, 'val',   k_neighbors=k,
                      mean=tr.mean, std=tr.std)
    te = ModalDataset(demand, coords, START_TS, SEQ_LEN, PRED_LEN, 'test',  k_neighbors=k,
                      mean=tr.mean, std=tr.std)
    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch, num_workers=0, pin_memory=pin)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(vl, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw),
            tr.mean, tr.std, tr.edge_index, tr.coords)


def train_model(model, tr_loader, vl_loader,
                epochs=EPOCHS, lr=LR, patience=PATIENCE,
                desc='') -> float:
    opt = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4, amsgrad=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

    best_val = float('inf')
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss = 0; n = 0
        for batch in tr_loader:
            x  = batch['x'].to(DEVICE)     # [B, T, N]
            tf = batch['tf'].to(DEVICE)
            y  = batch['y'].to(DEVICE)     # [B, 1, N]
            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                pred = model(x, tf)        # [B, N, 1]
                loss = nn.functional.huber_loss(pred, y.permute(0, 2, 1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item(); n += 1
        sched.step()

        # Val
        model.eval()
        vl_loss = 0; m = 0
        with torch.no_grad():
            for batch in vl_loader:
                x  = batch['x'].to(DEVICE)
                tf = batch['tf'].to(DEVICE)
                y  = batch['y'].to(DEVICE)
                pred = model(x, tf)
                vl_loss += nn.functional.huber_loss(pred, y.permute(0, 2, 1)).item()
                m += 1

        vl_avg = vl_loss / max(m, 1)
        if vl_avg < best_val - 1e-5:
            best_val = vl_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if ep % 10 == 0:
            print(f'  [{desc}] ep={ep:3d}  tr={tr_loss/n:.4f}  vl={vl_avg:.4f}')

    if best_state:
        model.load_state_dict(best_state)
    return best_val


def evaluate_model(model, te_loader, mean, std) -> Dict[str, float]:
    model.eval()
    all_pred, all_real = [], []
    with torch.no_grad():
        for batch in te_loader:
            x   = batch['x'].to(DEVICE)
            tf  = batch['tf'].to(DEVICE)
            y   = batch['y'].cpu().numpy()  # [B, pred_len, N]
            pred = model(x, tf).cpu().numpy()  # [B, N, pred_len]
            all_pred.append(pred.transpose(0, 2, 1))  # [B, pred_len, N]
            all_real.append(y)
    pred_np = np.concatenate(all_pred, 0)
    real_np = np.concatenate(all_real, 0)
    return compute_metrics(pred_np, real_np, mean, std)


def eval_historical_avg(demand, mean, std):
    """
    HistoricalAvg: ,  '' 
    """
    T, N = demand.shape
    n_usable = T - SEQ_LEN - PRED_LEN + 1
    n_train  = int(n_usable * 0.70)
    n_val    = int(n_usable * 0.15)
    test_start = n_train + n_val

    hours_full = pd.date_range(START_TS, periods=T, freq='h')
    hourly_sum   = np.zeros((24, N)); hourly_cnt = np.zeros(24, dtype=int)
    for t in range(n_train):
        hod = hours_full[t + SEQ_LEN].hour
        hourly_sum[hod] += demand[t + SEQ_LEN]
        hourly_cnt[hod] += 1
    hourly_avg = hourly_sum / hourly_cnt.clip(1)[:, None]  # [24, N]

    preds, reals = [], []
    for t in range(test_start, n_usable):
        target_h = hours_full[t + SEQ_LEN].hour
        pred = hourly_avg[target_h]                           # [N] raw units
        real = demand[t + SEQ_LEN]                            # [N] raw units
        preds.append(pred); reals.append(real)

    pred_np = np.stack(preds)   # [n_test, N]
    real_np = np.stack(reals)
    return compute_metrics(pred_np, real_np, mean=None, std=None)


def eval_last_value(demand, mean, std):
    """"""
    T, N = demand.shape
    n_usable  = T - SEQ_LEN - PRED_LEN + 1
    n_train   = int(n_usable * 0.70)
    n_val     = int(n_usable * 0.15)
    test_start = n_train + n_val
    preds, reals = [], []
    for t in range(test_start, n_usable):
        pred = demand[t + SEQ_LEN - 1]
        real = demand[t + SEQ_LEN]
        preds.append(pred); reals.append(real)
    pred_np = np.stack(preds); real_np = np.stack(reals)
    return compute_metrics(pred_np, real_np, mean=None, std=None)


def eval_lstm_ft(rh_demand, rh_coords, bike_demand, bike_coords, mean_rh, std_rh):
    """"""
    N_bike = bike_demand.shape[1]
    N_rh   = rh_demand.shape[1]

    model = LSTMShared(N_bike, N_rh, hidden=64, n_layers=2,
                       pred_len=PRED_LEN, dropout=0.1).to(DEVICE)

    tr_b, vl_b, te_b, _, _, _, _ = make_loaders(bike_demand, bike_coords, BATCH)

    class SrcWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, tf=None): return self.m.forward_src(x)

    sw = SrcWrap(model)
    print('  [LSTM-FT] Pretrain on bike...')
    train_model(sw, tr_b, vl_b, epochs=EPOCHS, desc='LSTM-FT/src')

    model.freeze_encoder()
    tr_r, vl_r, te_r, _, _, _, _ = make_loaders(rh_demand, rh_coords, BATCH)

    class TgtWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, tf=None): return self.m.forward_tgt(x)

    tw = TgtWrap(model)
    print('  [LSTM-FT] Finetune on RH...')
    train_model(tw, tr_r, vl_r, epochs=EPOCHS // 2, lr=LR * 0.3, desc='LSTM-FT/tgt')
    return evaluate_model(tw, te_r, mean_rh, std_rh)


def eval_transmod(rh_demand, rh_coords, bike_demand, bike_coords,
                  metro_demand, metro_coords, mean_rh, std_rh):
    """
    TransMod :
    1.  bike + metro  ( bike station )
    2.  RH
    """
    from models import TransModFramework
    from data.dataset import MultiModalDataset

    N_bike  = bike_demand.shape[1]
    N_metro = metro_demand.shape[1]
    N_rh    = rh_demand.shape[1]
    n_zones = 80

    mm_tr = MultiModalDataset(bike_demand, metro_demand, bike_coords, metro_coords,
                              START_TS, SEQ_LEN, PRED_LEN, 'train')
    mm_vl = MultiModalDataset(bike_demand, metro_demand, bike_coords, metro_coords,
                              START_TS, SEQ_LEN, PRED_LEN, 'val',
                              bike_mean=mm_tr.bike_mean,  bike_std=mm_tr.bike_std,
                              metro_mean=mm_tr.metro_mean, metro_std=mm_tr.metro_std)
    pin = DEVICE.type == 'cuda'
    tr_mm = DataLoader(mm_tr, batch_size=BATCH, shuffle=True,  pin_memory=pin)
    vl_mm = DataLoader(mm_vl, batch_size=BATCH, shuffle=False, pin_memory=pin)

    model = TransModFramework(
        n_bike=N_bike, n_metro=N_metro, n_rh_zones=N_rh,
        n_zones=n_zones, hidden_dim=64, pred_len=PRED_LEN,
        num_gat_layers=2, num_gat_heads=4,
        tcn_layers=3, tcn_kernel=3,
        top_k_bike=5, top_k_metro=8,
        sigma_bike=500, sigma_metro=800,
        memory_size=64, tau=0.07, delta=0.3, dropout=0.1,
        lambda_mmd=0.1, lambda_nce=0.5, lambda_geo=0.1, lambda_div=0.05,
    ).to(DEVICE)

    b_t = torch.tensor(bike_coords, device=DEVICE)
    m_t = torch.tensor(metro_coords, device=DEVICE)
    idx = torch.linspace(0, N_bike - 1, n_zones).long()
    z_t = b_t[idx]
    model.initialize_geography(b_t, m_t, z_t)

    print('  [TransMod] Pretraining...')
    opt_pt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, amsgrad=True)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
    best_pt, best_pt_state = float('inf'), None; wait_pt = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0; n = 0
        for batch in tr_mm:
            bx  = batch['bike_x'] .to(DEVICE)   # [B,T,N_bike]
            mx  = batch['metro_x'].to(DEVICE)
            tf  = batch['tf']     .to(DEVICE)
            by  = batch['bike_y'] .to(DEVICE)   # [B,1,N_bike]
            my  = batch['metro_y'].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                pred, losses = model.forward_pretrain(bx, mx, tf)
                # bike_y: [B, pred_len, N_bike] -> permute -> [B, N_bike, pred_len]
                by_perm = by.permute(0, 2, 1)       # [B, N_bike, pred_len]
                # zone target via S^T @ H
                S = model.bike_assign()              # [N_bike, Z]
                zone_tgt = torch.einsum('nz,bnp->bzp', S, by_perm)  # [B, Z, pred_len]
                total = model.compute_pretrain_loss(pred, zone_tgt, losses)

            opt_pt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt_pt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt_pt); scaler.update()
            tr_loss += total.item(); n += 1
        sched_pt.step()

        model.eval(); vl_loss = 0; m = 0
        with torch.no_grad():
            for batch in vl_mm:
                bx = batch['bike_x'].to(DEVICE)
                mx = batch['metro_x'].to(DEVICE)
                tf = batch['tf'].to(DEVICE)
                by = batch['bike_y'].to(DEVICE)
                pred, losses = model.forward_pretrain(bx, mx, tf)
                by_perm = by.permute(0, 2, 1)
                S = model.bike_assign()
                zone_tgt = torch.einsum('nz,bnp->bzp', S, by_perm)
                vl_loss += model.compute_pretrain_loss(pred, zone_tgt, losses).item()
                m += 1
        vl_avg = vl_loss / max(m, 1)
        if vl_avg < best_pt - 1e-5:
            best_pt = vl_avg
            best_pt_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait_pt = 0
        else:
            wait_pt += 1
            if wait_pt >= PATIENCE:
                print(f'     at pretrain epoch {ep}')
                break
        if ep % 10 == 0:
            print(f'  [TransMod/pretrain] ep={ep}  tr={tr_loss/n:.4f}  vl={vl_avg:.4f}')

    if best_pt_state:
        model.load_state_dict(best_pt_state)

    model.freeze_for_transfer()
    pc = model.param_count()
    print(f'  [TransMod] Transfer  trainable={pc["trainable"]:,}')

    tr_r, vl_r, te_r, mean_rh_, std_rh_, edge_rh, coords_rh = make_loaders(
        rh_demand, rh_coords, BATCH)

    rh_t = torch.tensor(rh_coords, device=DEVICE)
    model.initialize_geography(b_t, m_t, z_t, rh_zone_centers=rh_t)

    opt_tr = optim.AdamW(model.trainable_params(), lr=LR * 0.2, weight_decay=1e-5)
    sched_tr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tr, T_max=EPOCHS // 2)

    best_tr, best_tr_state = float('inf'), None; wait_tr = 0

    for ep in range(1, EPOCHS // 2 + 1):
        model.train()
        tr_loss = 0; n = 0
        for batch in tr_r:
            x  = batch['x'].to(DEVICE)
            tf = batch['tf'].to(DEVICE)
            y  = batch['y'].to(DEVICE)  # [B, 1, N_rh]
            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                pred = model.forward_transfer(x, tf, rh_t)  # [B, N_rh, 1]
                loss = nn.functional.huber_loss(pred, y.permute(0, 2, 1))
            opt_tr.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt_tr)
            nn.utils.clip_grad_norm_(model.trainable_params(), 5.0)
            scaler.step(opt_tr); scaler.update()
            tr_loss += loss.item(); n += 1
        sched_tr.step()

        model.eval(); vl_loss = 0; m = 0
        with torch.no_grad():
            for batch in vl_r:
                x = batch['x'].to(DEVICE)
                tf = batch['tf'].to(DEVICE)
                y  = batch['y'].to(DEVICE)
                pred = model.forward_transfer(x, tf, rh_t)
                vl_loss += nn.functional.huber_loss(pred, y.permute(0, 2, 1)).item()
                m += 1
        vl_avg = vl_loss / max(m, 1)
        if vl_avg < best_tr - 1e-5:
            best_tr = vl_avg
            best_tr_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait_tr = 0
        else:
            wait_tr += 1
            if wait_tr >= PATIENCE:
                print(f'     at transfer epoch {ep}')
                break
        if ep % 5 == 0:
            print(f'  [TransMod/transfer] ep={ep}  tr={tr_loss/n:.4f}  vl={vl_avg:.4f}')

    if best_tr_state:
        model.load_state_dict(best_tr_state)

    model.eval()
    all_pred, all_real = [], []
    with torch.no_grad():
        for batch in te_r:
            x  = batch['x'].to(DEVICE)
            tf = batch['tf'].to(DEVICE)
            y  = batch['y'].cpu().numpy()
            p  = model.forward_transfer(x, tf, rh_t).cpu().numpy()
            all_pred.append(p.transpose(0, 2, 1))  # [B, 1, N_rh]
            all_real.append(y)
    p_np = np.concatenate(all_pred); r_np = np.concatenate(all_real)
    return compute_metrics(p_np, r_np, mean_rh_, std_rh_)


def main():
    print('=' * 60)
    print('  TransMod Benchmark - NYC November 2018 Ridehailing')
    print('  seq_len=12  pred_len=1  train/val/test=70/15/15')
    print('=' * 60)
    print()

    rh_demand,    rh_coords    = load_rh('nyc')
    bike_demand,  bike_coords  = load_bike('nyc')
    # Metro
    base = DATA_DIR / 'nyc'
    metro_demand = (np.load(base / 'metro_hourly_entries.npy') +
                    np.load(base / 'metro_hourly_exits.npy')) / 2
    metro_sdf    = pd.read_csv(base / 'metro_stations.csv')
    metro_coords = metro_sdf[['lat', 'lon']].values.astype(np.float32)

    T, N_rh = rh_demand.shape
    n_usable = T - SEQ_LEN - PRED_LEN + 1
    n_train  = int(n_usable * 0.70)
    train_slice = np.concatenate([rh_demand[i:i+SEQ_LEN] for i in range(n_train)], 0)
    mean_rh = train_slice.mean(0, keepdims=True).astype(np.float32)
    std_rh  = train_slice.std( 0, keepdims=True).clip(1e-5).astype(np.float32)

    tr_r, vl_r, te_r, mean_rh_, std_rh_, edge_rh, coords_rh = make_loaders(
        rh_demand, rh_coords, BATCH)

    results: Dict[str, Dict] = {}
    times: Dict[str, float]  = {}

    t0 = time.time()
    results['HistAvg'] = eval_historical_avg(rh_demand, mean_rh, std_rh)
    times['HistAvg']   = time.time() - t0
    print(f"[HistAvg]    MAE={results['HistAvg']['MAE']:.3f}  "
          f"RMSE={results['HistAvg']['RMSE']:.3f}  "
          f"MAPE={results['HistAvg']['MAPE']:.2f}%  ({times['HistAvg']:.1f}s)")

    t0 = time.time()
    results['LastVal'] = eval_last_value(rh_demand, mean_rh, std_rh)
    times['LastVal']   = time.time() - t0
    print(f"[LastVal]    MAE={results['LastVal']['MAE']:.3f}  "
          f"RMSE={results['LastVal']['RMSE']:.3f}  "
          f"MAPE={results['LastVal']['MAPE']:.2f}%  ({times['LastVal']:.1f}s)")

    print('\n[LSTM] ...')
    t0 = time.time()
    lstm = LSTMModel(N_rh, hidden=64, n_layers=2, pred_len=PRED_LEN).to(DEVICE)
    train_model(lstm, tr_r, vl_r, EPOCHS, desc='LSTM')
    results['LSTM'] = evaluate_model(lstm, te_r, mean_rh_, std_rh_)
    times['LSTM'] = time.time() - t0
    print(f"[LSTM]       MAE={results['LSTM']['MAE']:.3f}  "
          f"RMSE={results['LSTM']['RMSE']:.3f}  "
          f"MAPE={results['LSTM']['MAPE']:.2f}%  ({times['LSTM']:.1f}s)")

    print('\n[GRU] ...')
    t0 = time.time()
    gru = GRUModel(N_rh, hidden=64, n_layers=2, pred_len=PRED_LEN).to(DEVICE)
    train_model(gru, tr_r, vl_r, EPOCHS, desc='GRU')
    results['GRU'] = evaluate_model(gru, te_r, mean_rh_, std_rh_)
    times['GRU'] = time.time() - t0
    print(f"[GRU]        MAE={results['GRU']['MAE']:.3f}  "
          f"RMSE={results['GRU']['RMSE']:.3f}  "
          f"MAPE={results['GRU']['MAPE']:.2f}%  ({times['GRU']:.1f}s)")

    print('\n[STGCN] ...')
    t0 = time.time()
    stgcn = STGCNLite(N_rh, hidden=64, n_gcn=2, pred_len=PRED_LEN).to(DEVICE)
    stgcn.set_adj(edge_rh.to(DEVICE), N_rh)
    train_model(stgcn, tr_r, vl_r, EPOCHS, desc='STGCN')
    results['STGCN'] = evaluate_model(stgcn, te_r, mean_rh_, std_rh_)
    times['STGCN'] = time.time() - t0
    print(f"[STGCN]      MAE={results['STGCN']['MAE']:.3f}  "
          f"RMSE={results['STGCN']['RMSE']:.3f}  "
          f"MAPE={results['STGCN']['MAPE']:.2f}%  ({times['STGCN']:.1f}s)")

    print('\n[LSTM-FT] +...')
    t0 = time.time()
    results['LSTM-FT'] = eval_lstm_ft(
        rh_demand, rh_coords, bike_demand, bike_coords, mean_rh_, std_rh_)
    times['LSTM-FT'] = time.time() - t0
    print(f"[LSTM-FT]    MAE={results['LSTM-FT']['MAE']:.3f}  "
          f"RMSE={results['LSTM-FT']['RMSE']:.3f}  "
          f"MAPE={results['LSTM-FT']['MAPE']:.2f}%  ({times['LSTM-FT']:.1f}s)")

    print('\n[TransMod] +...')
    t0 = time.time()
    results['TransMod'] = eval_transmod(
        rh_demand, rh_coords, bike_demand, bike_coords,
        metro_demand, metro_coords, mean_rh_, std_rh_)
    times['TransMod'] = time.time() - t0
    print(f"[TransMod]   MAE={results['TransMod']['MAE']:.3f}  "
          f"RMSE={results['TransMod']['RMSE']:.3f}  "
          f"MAPE={results['TransMod']['MAPE']:.2f}%  ({times['TransMod']:.1f}s)")

    print()
    print('=' * 62)
    print(f"{'Method':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Time(s)':>9}")
    print('-' * 62)
    for name in ['HistAvg', 'LastVal', 'LSTM', 'GRU', 'STGCN', 'LSTM-FT', 'TransMod']:
        r = results[name]
        t = times[name]
        marker = ' <-' if name == 'TransMod' else ''
        print(f"{name:<12} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} {r['MAPE']:>7.2f}% "
              f"{t:>9.1f}{marker}")
    print('=' * 62)

    best_bl = min(
        {k: v for k, v in results.items() if k != 'TransMod'}.items(),
        key=lambda x: x[1]['MAE'])
    tm = results['TransMod']
    print(f"\nTransMod vs  baseline ({best_bl[0]}):")
    for metric in ['MAE', 'RMSE', 'MAPE']:
        diff = tm[metric] - best_bl[1][metric]
        pct  = diff / best_bl[1][metric] * 100
        sign = '^' if diff < 0 else 'v'
        print(f"  {metric}: {tm[metric]:.3f} vs {best_bl[1][metric]:.3f}  "
              f"{sign} {abs(pct):.1f}%")

    return results


if __name__ == '__main__':
    main()
