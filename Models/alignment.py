"""
: MMD + InfoNCE  + 
: TransMod Section III-D
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """"""
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * bandwidth ** 2))


class MMDLoss(nn.Module):
    """
     (Multi-bandwidth Gaussian kernel)
    : H_bike [Z, d], H_metro [Z, d]  ()
    """

    def __init__(self, bandwidths=(0.5, 1.0, 2.0, 5.0)):
        super().__init__()
        self.bandwidths = bandwidths

    def forward(self, H_src: torch.Tensor, H_tgt: torch.Tensor) -> torch.Tensor:
        """
        H_src: [*, d]  H_tgt: [*, d]
        """
        src = H_src.reshape(-1, H_src.size(-1))
        tgt = H_tgt.reshape(-1, H_tgt.size(-1))
        loss = torch.tensor(0.0, device=src.device)
        for bw in self.bandwidths:
            Kss = gaussian_kernel(src, src, bw).mean()
            Ktt = gaussian_kernel(tgt, tgt, bw).mean()
            Kst = gaussian_kernel(src, tgt, bw).mean()
            loss = loss + Kss + Ktt - 2 * Kst
        return loss / len(self.bandwidths)


class InfoNCELoss(nn.Module):
    """
    : H_bike [Z, d], H_metro [Z, d]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
        """"""
        z1 = F.normalize(H1.reshape(-1, H1.size(-1)), dim=-1)
        z2 = F.normalize(H2.reshape(-1, H2.size(-1)), dim=-1)
        Z  = z1.size(0)
        sim = z1 @ z2.t() / self.tau
        labels = torch.arange(Z, device=H1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


class CrossModalAlignment(nn.Module):
    """
     MMD + InfoNCE + 

    :
        A_z = α.(A_od_bike + A_od_metro) + β.A_geo + γ.A_feat
    :
        A_od: soft_assignment  OD 
        A_geo: 
        A_feat: 
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.07):
        super().__init__()
        self.mmd_loss    = MMDLoss()
        self.infonce_loss = InfoNCELoss(temperature)

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta  = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))

        self.feat_proj = nn.Linear(hidden_dim, hidden_dim)

    def _build_geo_adj(self, zone_centers: torch.Tensor,
                       threshold_km: float = 3.0) -> torch.Tensor:
        """"""
        lat = zone_centers[:, 0] * (torch.pi / 180)
        lon = zone_centers[:, 1] * (torch.pi / 180)
        dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
        dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat.unsqueeze(1)) * torch.cos(lat.unsqueeze(0)) * \
            torch.sin(dlon / 2) ** 2
        dist_km = 6371.0 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        A_geo = (dist_km < threshold_km).float()
        A_geo.fill_diagonal_(0)
        row_sum = A_geo.sum(dim=-1, keepdim=True).clamp(min=1)
        return A_geo / row_sum

    def _build_feat_adj(self, H: torch.Tensor) -> torch.Tensor:
        """"""
        h_norm = F.normalize(self.feat_proj(H.mean(0)), dim=-1)  # [Z, d]
        A_feat = (h_norm @ h_norm.t()).clamp(min=0)
        A_feat.fill_diagonal_(0)
        row_sum = A_feat.sum(dim=-1, keepdim=True).clamp(min=1)
        return A_feat / row_sum

    def fuse_zone_graph(
        self,
        A_od_bike:   torch.Tensor,    # [Z, Z] OD from bike
        A_od_metro:  torch.Tensor,    # [Z, Z] OD from metro
        H_zone:      torch.Tensor,
        zone_centers: torch.Tensor,   # [Z, 2]
    ) -> torch.Tensor:
        """
        A_z = α.(A_od_bike + A_od_metro) / 2 + β.A_geo + γ.A_feat
        """
        alpha = torch.exp(self.log_alpha)
        beta  = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        denom = alpha + beta + gamma

        A_od   = (A_od_bike + A_od_metro) / 2    # [Z, Z]
        A_geo  = self._build_geo_adj(zone_centers).to(A_od.device)
        A_feat = self._build_feat_adj(H_zone).to(A_od.device)

        A_z = (alpha * A_od + beta * A_geo + gamma * A_feat) / denom
        return A_z   # [Z, Z]

    def forward(
        self,
        H_bike:      torch.Tensor,
        H_metro:     torch.Tensor,
        A_od_bike:   torch.Tensor,    # [Z, Z]
        A_od_metro:  torch.Tensor,    # [Z, Z]
        zone_centers: torch.Tensor,   # [Z, 2]
    ):
        """
        Returns:
            A_z:        [Z, Z]  
            mmd_loss:   scalar
            infonce_loss: scalar
        """
        H_b = H_bike.mean(0)   # [Z, d]
        H_m = H_metro.mean(0)  # [Z, d]

        mmd    = self.mmd_loss(H_b, H_m)
        nce    = self.infonce_loss(H_b, H_m)

        H_fused = (H_bike + H_metro) / 2   # [B, Z, d]
        A_z = self.fuse_zone_graph(A_od_bike, A_od_metro, H_fused, zone_centers)

        return A_z, mmd, nce
