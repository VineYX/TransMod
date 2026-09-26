"""
CitiBike spatio-temporal dataset for TransMod single-modal forecasting.
Loads real temporal snapshots and builds a static geographic k-NN graph.
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


def _knn_graph(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Build a directed k-NN graph from 2-D coordinates (no torch-cluster needed)."""
    dist = torch.cdist(coords, coords, p=2)          # [N, N]
    dist.fill_diagonal_(float('inf'))                 # exclude self
    _, idx = dist.topk(k, dim=1, largest=False)      # [N, k] nearest
    N = coords.size(0)
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)             # [2, N*k]


class CitiBikeDataset(Dataset):
    """
    Wraps temporal_snapshots.pkl into sliding-window sequences.

    Each sample:
        x_seq   : [T, N, F]  - normalised station features (window)
        target  : [N]        - normalised out_trips at t+1
        edge_index: [2, E]   - static geographic k-NN graph (shared)
    """

    def __init__(self, xs: torch.Tensor, edge_index: torch.Tensor,
                 window: int, indices: list,
                 mean: torch.Tensor, std: torch.Tensor):
        self.xs = xs                  # [T, N, F] normalised
        self.edge_index = edge_index  # [2, E]
        self.window = window
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x_seq = self.xs[t: t + self.window]              # [T, N, F]
        target = self.xs[t + self.window, :, 2]          # [N] normalised out_trips
        return x_seq, target


def build_datasets(snapshots_path: str,
                   window: int = 8,
                   k_neighbors: int = 5,
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.15):
    """
    Load snapshots -> normalise -> split -> return (train, val, test) datasets
    plus shared static edge_index.
    """
    with open(snapshots_path, 'rb') as f:
        snaps = pickle.load(f)

    # Stack: [T, N, F]
    xs_raw = torch.stack([s.x for s in snaps]).float()
    T, N, F = xs_raw.shape

    # Static geographic graph: k-NN on (lat, lon) - pure PyTorch, no torch-cluster
    coords = xs_raw[0, :, :2]                            # [N, 2]
    edge_index = _knn_graph(coords, k=k_neighbors)       # [2, N*k]

    # Split indices (temporal, no shuffle)
    n_usable = T - window - 1                            # last window+1 steps are target
    n_train = int(n_usable * train_ratio)
    n_val   = int(n_usable * val_ratio)

    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n_usable))

    # Compute normalisation stats on training windows only
    train_xs = torch.cat([xs_raw[i: i + window] for i in train_idx], dim=0)
    mean = train_xs.mean(dim=(0, 1), keepdim=True).squeeze(0)  # [1, F]
    std  = train_xs.std (dim=(0, 1), keepdim=True).squeeze(0).clamp(min=1e-5)

    xs = (xs_raw - mean.unsqueeze(0)) / std.unsqueeze(0)       # [T, N, F]

    def make_ds(idx):
        return CitiBikeDataset(xs, edge_index, window, idx, mean, std)

    return make_ds(train_idx), make_ds(val_idx), make_ds(test_idx), edge_index, N, F


def collate_fn(batch):
    """Stack samples into [B, T, N, F] and [B, N] tensors."""
    x_seqs, targets = zip(*batch)
    return torch.stack(x_seqs), torch.stack(targets)
