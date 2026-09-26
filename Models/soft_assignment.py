"""
Soft assignment matrix S^m: stations -> zones
Paper: TransMod Section III-C
L_geo = ||S^m - S^m_geo||_F^2  (Frobenius-norm regularisation)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftAssignment(nn.Module):
    """
    Learnable soft assignment matrix, geo-prior initialised, Frobenius-norm regularised.

    Args:
        n_stations: number of stations N
        n_zones:    number of zones Z
        top_k:      max zones each station can be assigned to (sparsity constraint)
        sigma_m:    bandwidth of the geographic Gaussian kernel (metres)
    """

    def __init__(
        self,
        n_stations: int,
        n_zones:    int,
        top_k:      int   = 5,
        sigma_m:    float = 500.0,   # unit: metres
    ):
        super().__init__()
        self.n_stations = n_stations
        self.n_zones    = n_zones
        self.top_k      = top_k
        self.sigma_m    = sigma_m

        # learnable offset θ (initialised to 0; geo prior added as bias)
        self.theta = nn.Parameter(torch.zeros(n_stations, n_zones))

        # geo prior S_geo [N, Z] and sparsity mask [N, Z] set by initialize_geography
        self.register_buffer('S_geo',   torch.zeros(n_stations, n_zones))
        self.register_buffer('top_k_mask', torch.ones(n_stations, n_zones))
        self._geo_initialized = False

    @staticmethod
    def _deg2km(lat1, lon1, lat2, lon2) -> float:
        """Approximate Euclidean distance in metres (valid for small ranges)."""
        dlat = (lat2 - lat1) * 111_000
        dlon = (lon2 - lon1) * 111_000 * np.cos(np.radians((lat1 + lat2) / 2))
        return float(np.sqrt(dlat ** 2 + dlon ** 2))

    def initialize_geography(
        self,
        station_coords: torch.Tensor,   # [N, 2] (lat, lon)
        zone_centers:   torch.Tensor,   # [Z, 2] (lat, lon)
    ):
        """
        Compute the geographic prior S_geo via a Gaussian kernel and determine the
        sparse top-K mask. Called once during Trainer construction, not during forward.
        """
        N = station_coords.size(0)
        Z = zone_centers.size(0)

        sc = station_coords.cpu().numpy()
        zc = zone_centers.cpu().numpy()

        # vectorised distance [N, Z] in metres - replaces nested Python loops
        dlat = (sc[:, 0:1] - zc[np.newaxis, :, 0]) * 111_000   # [N, Z]
        mean_lat = np.radians((sc[:, 0:1] + zc[np.newaxis, :, 0]) / 2)
        dlon = (sc[:, 1:2] - zc[np.newaxis, :, 1]) * 111_000 * np.cos(mean_lat)  # [N, Z]
        dist2 = dlat ** 2 + dlon ** 2                           # [N, Z] in m²
        S_geo_np = np.exp(-dist2 / (2 * self.sigma_m ** 2)).astype(np.float32)

        # vectorised top-K sparsification
        K = min(self.top_k, Z)
        top_idx = np.argpartition(S_geo_np, -K, axis=1)[:, -K:]   # [N, K]
        mask_np = np.zeros((N, Z), dtype=np.float32)
        np.put_along_axis(mask_np, top_idx, 1.0, axis=1)

        # row-normalise (softmax equivalent)
        S_masked = S_geo_np * mask_np
        row_sum = S_masked.sum(axis=1, keepdims=True).clip(1e-6)
        S_geo_norm = S_masked / row_sum

        dev = self.theta.device
        # if Z changed at runtime, re-register buffers and theta
        if Z != self.n_zones:
            self.n_zones = Z
            self.theta = nn.Parameter(torch.zeros(N, Z, device=dev))
        self.register_buffer('S_geo',      torch.tensor(S_geo_norm, device=dev))
        self.register_buffer('top_k_mask', torch.tensor(mask_np,    device=dev))
        self._geo_initialized = True

    def forward(self) -> torch.Tensor:
        """
        Returns soft assignment matrix S [N, Z].
        Masked softmax: non-top-K positions are filled with -inf before softmax.
        """
        # learnable offset on top of geo prior (geo prior scaled as initial bias)
        logits = self.theta + self.S_geo * 10
        # sparsify: non-mask positions -> -inf
        logits = logits * self.top_k_mask + (1 - self.top_k_mask) * (-1e9)
        S = F.softmax(logits, dim=-1)           # row softmax -> [N, Z]
        return S

    def geo_loss(self) -> torch.Tensor:
        """
        L_geo = mean_{n,z} (S_{n,z} - S_geo_{n,z})^2
        Mean form (scale-invariant w.r.t. N.Z; avoids large matrices dominating training).
        """
        S = self.forward()
        return ((S - self.S_geo) ** 2).mean()

    def aggregate_station_to_zone(
        self,
        H_station: torch.Tensor,   # [B, N, d] or [N, d]
    ) -> torch.Tensor:
        """
        H_zone = S^T @ H_station  ->  [B, Z, d] or [Z, d]
        """
        S = self.forward()   # [N, Z]
        if H_station.dim() == 2:
            return S.t() @ H_station               # [Z, d]
        else:
            return torch.einsum('nz,bnd->bzd', S, H_station)   # [B, Z, d]

    def aggregate_adj(self, A_station: torch.Tensor) -> torch.Tensor:
        """
        Map station adjacency to zone adjacency: A_zone = S^T @ A @ S  (DiffPool, TSTL Eq.8)
        A_station: [N, N]  ->  A_zone: [Z, Z]
        """
        S = self.forward()          # [N, Z]
        return S.t() @ A_station @ S   # [Z, Z]

    #  Demand correlation prior (behavioural similarity)         #

    def initialize_demand_prior(
        self,
        station_demands: np.ndarray,   # [T, N] training-period demand (unnormalised)
        alpha: float = 0.4,            # demand prior weight (geo weight = 1-alpha)
    ):
        """
        Computes temporal demand correlation between stations and zones as a supplementary prior.
        Blends with the geo prior via geometric mean: S_prior = S_geo^(1-alpha) * S_demand^alpha.
        Must be called after initialize_geography.
        """
        if not self._geo_initialized:
            raise RuntimeError("Call initialize_geography() first.")

        T, N = station_demands.shape
        Z    = self.n_zones

        S_geo_np = self.S_geo.cpu().numpy()   # [N, Z] row-normalised

        # zone demand centroid: S_geo-weighted aggregation of station demands -> [T, Z]
        zone_demand = station_demands @ S_geo_np   # [T, Z]

        # Pearson correlation: vectorised [N, Z]
        # corr(station_n, zone_z) = cov / (std_n * std_z)
        sta_mean = station_demands.mean(0, keepdims=True)    # [1, N]
        zon_mean = zone_demand.mean(0, keepdims=True)        # [1, Z]
        sta_c    = station_demands - sta_mean                # [T, N]
        zon_c    = zone_demand     - zon_mean                # [T, Z]
        cov      = (sta_c.T @ zon_c) / max(T - 1, 1)        # [N, Z]
        sta_std  = sta_c.std(0).clip(1e-6)                   # [N]
        zon_std  = zon_c.std(0).clip(1e-6)                   # [Z]
        corr     = cov / (sta_std[:, None] * zon_std[None, :])  # [N, Z]

        # keep only positive correlations, apply top-K mask
        S_demand = np.clip(corr, 0, None).astype(np.float32)
        S_demand = S_demand * self.top_k_mask.cpu().numpy()

        # row-normalise
        row_sum = S_demand.sum(1, keepdims=True).clip(1e-6)
        S_demand = S_demand / row_sum

        # geometric mean blend: S_new = S_geo^(1-alpha) * S_demand^alpha
        # equivalent to exp((1-alpha)*log(S_geo) + alpha*log(S_demand))
        log_geo  = np.log(S_geo_np.clip(1e-8))
        log_dem  = np.log(S_demand.clip(1e-8))
        S_blend  = np.exp((1 - alpha) * log_geo + alpha * log_dem)
        S_blend  = S_blend * self.top_k_mask.cpu().numpy()
        row_sum2 = S_blend.sum(1, keepdims=True).clip(1e-6)
        S_blend  = (S_blend / row_sum2).astype(np.float32)

        self.register_buffer('S_geo', torch.tensor(S_blend, device=self.theta.device))
        print(f"  [SoftAssign] demand prior blended (alpha={alpha:.2f}), "
              f"mean corr={corr[corr>0].mean():.3f}")

    #  Assignment regularisation loss (TSTL L_r)                #

    def assignment_reg_loss(self, A_station: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Lightweight proxy for TSTL L_r (assignment consistency loss).

        Instead of computing S^T A_station S (O(N²Z) = 14M ops/batch), uses:
            zone_overlap = S^T S         [Z, Z]  O(NZ²) = 946K ops
            geo_overlap  = S_geo^T S_geo [Z, Z]  fixed target, detached
        This costs 946K ops per batch, avoids the large N×N matrix,
        and lets gradients flow correctly back to S.theta.

        Semantics: "the learned zone assignment pattern should match the geo prior pattern."
        """
        S   = self.forward()                          # [N, Z]
        # learned zone co-occupancy pattern
        zone_overlap = S.t() @ S                      # [Z, Z], O(N*Z²)
        # fixed target from geo prior
        geo_overlap  = (self.S_geo.t() @ self.S_geo).detach()  # [Z, Z]
        # normalise to [0, 1]
        m1 = zone_overlap.detach().max().clamp(min=1e-6)
        m2 = geo_overlap.max().clamp(min=1e-6)
        return ((zone_overlap / m1 - geo_overlap / m2) ** 2).mean()
