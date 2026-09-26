"""
: MAE / RMSE / MAPE ()
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Dict


def masked_mae(pred: np.ndarray, real: np.ndarray, null_val: float = 0.0) -> float:
    mask = real != null_val
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(pred[mask] - real[mask]).mean())


def masked_rmse(pred: np.ndarray, real: np.ndarray, null_val: float = 0.0) -> float:
    mask = real != null_val
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(((pred[mask] - real[mask]) ** 2).mean()))


def masked_mape(pred: np.ndarray, real: np.ndarray,
                null_val: float = 0.0, eps: float = 1.0) -> float:
    """"""
    mask = (real > eps) & (real != null_val)
    if mask.sum() == 0:
        return 0.0
    return float((np.abs(pred[mask] - real[mask]) / real[mask]).mean()) * 100


def compute_metrics(
    pred:  np.ndarray,
    real:  np.ndarray,
    mean:  np.ndarray | None = None,
    std:   np.ndarray | None = None,
) -> Dict[str, float]:
    """
     MAE / RMSE / MAPE
    pred/real: [B, Z, pred_len] or [*, Z]
    mean/std:  [1, Z] or scalar
    """
    if mean is not None and std is not None:
        # Ensure std/mean broadcast correctly against [B, Z, pred_len]:
        # reshape [1, Z] -> [1, Z, 1] to avoid [B,Z,1]*[1,Z]->[B,Z,Z]
        if np.ndim(std) == 2 and pred.ndim == 3:
            std  = std[:, :, np.newaxis]   # [1, Z, 1]
            mean = mean[:, :, np.newaxis]  # [1, Z, 1]
        pred = pred * std + mean
        real = real * std + mean

    pred = pred.reshape(-1)
    real = real.reshape(-1)

    return {
        'MAE':  masked_mae(pred, real),
        'RMSE': masked_rmse(pred, real),
        'MAPE': masked_mape(pred, real),
    }


def compute_metrics_tensor(
    pred:  torch.Tensor,
    real:  torch.Tensor,
    mean:  torch.Tensor | None = None,
    std:   torch.Tensor | None = None,
) -> Dict[str, float]:
    p = pred.detach().cpu().numpy()
    r = real.detach().cpu().numpy()
    m = mean.cpu().numpy() if mean is not None else None
    s = std .cpu().numpy() if std  is not None else None
    return compute_metrics(p, r, m, s)
