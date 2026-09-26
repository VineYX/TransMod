"""
: NYC / Chicago × (bike, metro, ridehailing)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def temporal_features(hours: pd.DatetimeIndex) -> np.ndarray:
    """
     sin/cos  [T, 8]
    sin/cos of: hour-of-day, day-of-week, day-of-month, week-of-year
    """
    h   = hours.hour.values
    dow = hours.dayofweek.values
    dom = hours.day.values
    woy = hours.isocalendar().week.values.astype(float)

    feats = np.stack([
        np.sin(2 * np.pi * h   / 24),  np.cos(2 * np.pi * h   / 24),
        np.sin(2 * np.pi * dow / 7),   np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * dom / 31),  np.cos(2 * np.pi * dom / 31),
        np.sin(2 * np.pi * woy / 52),  np.cos(2 * np.pi * woy / 52),
    ], axis=1).astype(np.float32)   # [T, 8]
    return feats


def build_knn_edge_index(coords: np.ndarray, k: int = 5) -> torch.Tensor:
    """
     (lat, lon)  k-NN 
    Returns edge_index [2, N*k]
    """
    coords_t = torch.tensor(coords, dtype=torch.float32)
    dist = torch.cdist(coords_t, coords_t)          # [N, N]
    dist.fill_diagonal_(float('inf'))
    _, idx = dist.topk(k, dim=1, largest=False)     # [N, k]
    N = coords_t.size(0)
    src = torch.arange(N).unsqueeze(1).expand(N, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)            # [2, N*k]


def load_station_data(city_dir: Path, modality: str):
    """
    Returns: demand [T, N], coords [N, 2], station_df
    """
    demand_files = {
        'bike':       city_dir / 'bike_hourly_demand.npy',
        'metro':      city_dir / 'metro_hourly_demand.npy',
        'metro_entries': city_dir / 'metro_hourly_entries.npy',
        'metro_exits':   city_dir / 'metro_hourly_exits.npy',
        'ridehailing':   city_dir / 'ridehailing_hourly_demand.npy',
    }
    station_files = {
        'bike':       city_dir / 'bike_stations.csv',
        'metro':      city_dir / 'metro_stations.csv',
        'metro_entries': city_dir / 'metro_stations.csv',
        'metro_exits':   city_dir / 'metro_stations.csv',
        'ridehailing':   city_dir / 'ridehailing_zones.csv',
    }

    demand = np.load(demand_files[modality]).astype(np.float32)   # [T, N]
    sdf    = pd.read_csv(station_files[modality])
    coords = sdf[['lat', 'lon']].values.astype(np.float32)        # [N, 2]
    return demand, coords, sdf


class ModalDataset(Dataset):
    """
    bike / metro / ridehailing
    : (x_seq, time_feat, target)
      x_seq    : [seq_len, N]  
      time_feat: [seq_len, 8]  
      target   : [pred_len, N] 
    : edge_index, coords, mean, std
    """

    def __init__(
        self,
        demand: np.ndarray,           # [T, N]
        coords: np.ndarray,           # [N, 2]
        start_ts: pd.Timestamp,
        seq_len:  int  = 12,
        pred_len: int  = 1,
        split:    str  = 'train',     # 'train' | 'val' | 'test'
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        k_neighbors: int  = 5,
        mean: Optional[np.ndarray] = None,
        std:  Optional[np.ndarray] = None,
    ):
        T, N = demand.shape
        hours = pd.date_range(start_ts, periods=T, freq='h')
        self.T, self.N = T, N

        self.time_feat_all = temporal_features(hours)   # [T, 8]

        n_usable = T - seq_len - pred_len + 1
        n_train  = int(n_usable * train_ratio)
        n_val    = int(n_usable * val_ratio)

        if split == 'train':
            indices = list(range(0, n_train))
        elif split == 'val':
            indices = list(range(n_train, n_train + n_val))
        else:
            indices = list(range(n_train + n_val, n_usable))
        self.indices = indices

        if mean is None or std is None:
            train_data = np.concatenate(
                [demand[i: i + seq_len] for i in range(n_train)], axis=0)
            mean = train_data.mean(axis=0, keepdims=True)  # [1, N]
            std  = train_data.std( axis=0, keepdims=True).clip(1e-5)
        self.mean = mean   # [1, N]
        self.std  = std    # [1, N]
        self.demand_norm = (demand - mean) / std   # [T, N]

        self.edge_index = build_knn_edge_index(coords, k=k_neighbors)
        self.coords     = torch.tensor(coords, dtype=torch.float32)  # [N, 2]

        self.seq_len  = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.indices[idx]
        x    = self.demand_norm[t: t + self.seq_len]                       # [seq_len, N]
        tf   = self.time_feat_all[t: t + self.seq_len]                     # [seq_len, 8]
        tgt  = self.demand_norm[t + self.seq_len: t + self.seq_len + self.pred_len]  # [pred_len, N]
        return {
            'x':    torch.tensor(x,   dtype=torch.float32),
            'tf':   torch.tensor(tf,  dtype=torch.float32),
            'y':    torch.tensor(tgt, dtype=torch.float32),
        }


class MultiModalDataset(Dataset):
    """
    : bike + metro , 
    :
      bike_x    : [seq_len, N_bike]
      metro_x   : [seq_len, N_metro]
      time_feat : [seq_len, 8]
      bike_y    : [pred_len, N_bike]   ()
      metro_y   : [pred_len, N_metro]
    : bike_edge_index, metro_edge_index, bike_coords, metro_coords
    """

    def __init__(
        self,
        bike_demand:   np.ndarray,    # [T, N_bike]
        metro_demand:  np.ndarray,    # [T, N_metro]
        bike_coords:   np.ndarray,    # [N_bike, 2]
        metro_coords:  np.ndarray,    # [N_metro, 2]
        start_ts:      pd.Timestamp,
        seq_len:   int   = 12,
        pred_len:  int   = 1,
        split:     str   = 'train',
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        k_neighbors: int   = 5,
        bike_mean:   Optional[np.ndarray] = None,
        bike_std:    Optional[np.ndarray] = None,
        metro_mean:  Optional[np.ndarray] = None,
        metro_std:   Optional[np.ndarray] = None,
    ):
        assert bike_demand.shape[0] == metro_demand.shape[0], \
            "bike  metro "
        T = bike_demand.shape[0]
        hours = pd.date_range(start_ts, periods=T, freq='h')

        self.time_feat_all = temporal_features(hours)   # [T, 8]
        self.seq_len  = seq_len
        self.pred_len = pred_len

        n_usable = T - seq_len - pred_len + 1
        n_train  = int(n_usable * train_ratio)
        n_val    = int(n_usable * val_ratio)

        if split == 'train':
            self.indices = list(range(0, n_train))
        elif split == 'val':
            self.indices = list(range(n_train, n_train + n_val))
        else:
            self.indices = list(range(n_train + n_val, n_usable))

        def _norm_params(demand, mean, std, n_train_idx, seq_len):
            if mean is None or std is None:
                td = np.concatenate([demand[i: i+seq_len] for i in range(n_train_idx)], 0)
                mean = td.mean(0, keepdims=True)
                std  = td.std (0, keepdims=True).clip(1e-5)
            return mean, std, (demand - mean) / std

        self.bike_mean,  self.bike_std,  self.bike_norm  = _norm_params(
            bike_demand,  bike_mean,  bike_std,  n_train, seq_len)
        self.metro_mean, self.metro_std, self.metro_norm = _norm_params(
            metro_demand, metro_mean, metro_std, n_train, seq_len)

        self.bike_edge_index  = build_knn_edge_index(bike_coords,  k_neighbors)
        self.metro_edge_index = build_knn_edge_index(metro_coords, k_neighbors)
        self.bike_coords  = torch.tensor(bike_coords,  dtype=torch.float32)
        self.metro_coords = torch.tensor(metro_coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t  = self.indices[idx]
        sl = self.seq_len
        pl = self.pred_len
        return {
            'bike_x':  torch.tensor(self.bike_norm [t:t+sl],       dtype=torch.float32),
            'metro_x': torch.tensor(self.metro_norm[t:t+sl],       dtype=torch.float32),
            'tf':      torch.tensor(self.time_feat_all[t:t+sl],    dtype=torch.float32),
            'bike_y':  torch.tensor(self.bike_norm [t+sl:t+sl+pl], dtype=torch.float32),
            'metro_y': torch.tensor(self.metro_norm[t+sl:t+sl+pl], dtype=torch.float32),
        }


def make_multimodal_loaders(
    processed_dir: str | Path,
    city:          str,               # 'nyc' | 'chicago'
    start_ts:      pd.Timestamp,
    seq_len:   int = 12,
    pred_len:  int = 1,
    batch_size: int = 32,
    k_neighbors: int = 5,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
     train / val / test DataLoader
     (train_loader, val_loader, test_loader, meta)
    meta edge_index
    """
    base = Path(processed_dir) / city

    if (base / 'metro_hourly_entries.npy').exists():
        metro_demand = (
            np.load(base / 'metro_hourly_entries.npy') +
            np.load(base / 'metro_hourly_exits.npy')
        ) / 2
    else:
        metro_demand = np.load(base / 'metro_hourly_demand.npy')

    bike_demand = np.load(base / 'bike_hourly_demand.npy')

    bike_sdf  = pd.read_csv(base / 'bike_stations.csv')
    metro_sdf = pd.read_csv(base / 'metro_stations.csv')
    bike_coords  = bike_sdf[['lat', 'lon']].values.astype(np.float32)
    metro_coords = metro_sdf[['lat', 'lon']].values.astype(np.float32)

    train_ds = MultiModalDataset(
        bike_demand, metro_demand, bike_coords, metro_coords,
        start_ts=start_ts, seq_len=seq_len, pred_len=pred_len,
        split='train', k_neighbors=k_neighbors)

    val_ds = MultiModalDataset(
        bike_demand, metro_demand, bike_coords, metro_coords,
        start_ts=start_ts, seq_len=seq_len, pred_len=pred_len,
        split='val', k_neighbors=k_neighbors,
        bike_mean=train_ds.bike_mean,  bike_std=train_ds.bike_std,
        metro_mean=train_ds.metro_mean, metro_std=train_ds.metro_std)

    test_ds = MultiModalDataset(
        bike_demand, metro_demand, bike_coords, metro_coords,
        start_ts=start_ts, seq_len=seq_len, pred_len=pred_len,
        split='test', k_neighbors=k_neighbors,
        bike_mean=train_ds.bike_mean,  bike_std=train_ds.bike_std,
        metro_mean=train_ds.metro_mean, metro_std=train_ds.metro_std)

    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

    meta = {
        'bike_edge_index':  train_ds.bike_edge_index,
        'metro_edge_index': train_ds.metro_edge_index,
        'bike_coords':      train_ds.bike_coords,
        'metro_coords':     train_ds.metro_coords,
        'bike_mean':        train_ds.bike_mean,
        'bike_std':         train_ds.bike_std,
        'metro_mean':       train_ds.metro_mean,
        'metro_std':        train_ds.metro_std,
        'N_bike':           bike_demand.shape[1],
        'N_metro':          metro_demand.shape[1],
    }
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        meta,
    )


def make_rh_loaders(
    processed_dir: str | Path,
    city:          str,
    start_ts:      pd.Timestamp,
    seq_len:   int = 12,
    pred_len:  int = 1,
    batch_size: int = 32,
    k_neighbors: int = 5,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
     () DataLoader
    """
    base = Path(processed_dir) / city
    rh_demand = np.load(base / 'ridehailing_hourly_demand.npy').astype(np.float32)
    rh_sdf    = pd.read_csv(base / 'ridehailing_zones.csv')
    rh_coords = rh_sdf[['lat', 'lon']].values.astype(np.float32)

    pin = torch.cuda.is_available()
    kw  = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)

    train_ds = ModalDataset(rh_demand, rh_coords, start_ts,
                            seq_len=seq_len, pred_len=pred_len, split='train',
                            k_neighbors=k_neighbors)
    val_ds  = ModalDataset(rh_demand, rh_coords, start_ts,
                           seq_len=seq_len, pred_len=pred_len, split='val',
                           k_neighbors=k_neighbors,
                           mean=train_ds.mean, std=train_ds.std)
    test_ds = ModalDataset(rh_demand, rh_coords, start_ts,
                           seq_len=seq_len, pred_len=pred_len, split='test',
                           k_neighbors=k_neighbors,
                           mean=train_ds.mean, std=train_ds.std)

    meta = {
        'edge_index': train_ds.edge_index,
        'coords':     train_ds.coords,
        'mean':       train_ds.mean,
        'std':        train_ds.std,
        'N_zones':    rh_demand.shape[1],
    }
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        meta,
    )
