"""
Station encoder: Gaussian uncertainty embedding + dynamic graph construction + GAT propagation.
Paper: TransMod Section III-B

Graph propagation uses SparseGATLayer - sparse multi-head graph attention with O(E.d) complexity.
No torch_geometric dependency; implemented via scatter_add over the edge list.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalFusion


#  Gaussian Uncertainty Embedding                             #

class GaussianEmbedding(nn.Module):
    """
    Maps node features to a Gaussian distribution (μ, σ) and samples via reparameterization.
    Input:  h [*, d]
    Output: z [*, d], mu [*, d], sigma [*, d]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mu_proj    = nn.Linear(dim, dim)
        self.sigma_proj = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor):
        mu    = self.mu_proj(h)
        # σ > 0 via softplus
        sigma = F.softplus(self.sigma_proj(h)) + 1e-4
        # reparameterization trick: z = μ + σ.ε,  ε ~ N(0,1)
        if self.training:
            z = mu + sigma * torch.randn_like(sigma)
        else:
            z = mu
        return z, mu, sigma


#  Dynamic Graph Constructor                                  #

class DynamicGraphConstructor(nn.Module):
    """
    Builds a dynamic station graph: static geo-proximity edges + time-decayed OD flow edges.
    Outputs edge_index [2, E] and edge_weight [E].
    """

    def __init__(
        self,
        geo_threshold_km: float = 1.0,
        od_lambda:        float = 0.1,    # temporal decay coefficient
        od_topk:          int   = 10,
    ):
        super().__init__()
        self.geo_threshold = geo_threshold_km
        self.od_lambda     = od_lambda
        self.od_topk       = od_topk

    @staticmethod
    def haversine_km(coords: torch.Tensor) -> torch.Tensor:
        """coords: [N, 2] lat/lon in degrees -> [N, N] km distance matrix"""
        lat = coords[:, 0] * (torch.pi / 180)
        lon = coords[:, 1] * (torch.pi / 180)
        dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
        dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat.unsqueeze(1)) * torch.cos(lat.unsqueeze(0)) * \
            torch.sin(dlon / 2) ** 2
        return 6371.0 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    def build_geo_graph(self, coords: torch.Tensor):
        """Static geo-proximity graph: connect stations within geo_threshold km."""
        dist = self.haversine_km(coords)   # [N, N]
        mask = (dist < self.geo_threshold) & (dist > 0)
        edge_index  = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
        edge_weight = torch.exp(-dist[mask] / self.geo_threshold)
        return edge_index, edge_weight

    def build_od_graph(self, od_history: torch.Tensor, current_t: int):
        """
        od_history: [T_hist, N, N]
        Aggregates recent OD flow with exponential temporal decay over the last od_topk steps.
        Returns edge_index [2, E] and edge_weight [E] for the top-k OD pairs.
        """
        T = od_history.size(0)
        t_range = torch.arange(T, device=od_history.device)
        decay   = torch.exp(-self.od_lambda * (current_t - t_range).clamp(min=0).float())
        od_agg  = (od_history * decay.view(T, 1, 1)).sum(0)  # [N, N]

        N = od_agg.size(0)
        val, idx = od_agg.topk(min(self.od_topk, N - 1), dim=1)
        src = torch.arange(N, device=od_agg.device).unsqueeze(1).expand_as(idx).reshape(-1)
        dst = idx.reshape(-1)
        edge_weight = val.reshape(-1)
        mask        = edge_weight > 0
        edge_index  = torch.stack([src[mask], dst[mask]], dim=0)
        edge_weight = edge_weight[mask]
        return edge_index, edge_weight

    def forward(
        self,
        coords:     torch.Tensor,               # [N, 2]
        od_history: torch.Tensor | None = None, # [T_hist, N, N]
        current_t:  int = 0,
    ):
        edge_index, edge_weight = self.build_geo_graph(coords)
        if od_history is not None and od_history.numel() > 0:
            od_ei, od_ew = self.build_od_graph(od_history, current_t)
            edge_index  = torch.cat([edge_index, od_ei],  dim=1)
            edge_weight = torch.cat([edge_weight, od_ew])
        return edge_index, edge_weight


#  Sparse GAT Layer  (O(E.d), no torch_geometric)             #

class SparseGATLayer(nn.Module):
    """
    Sparse multi-head Graph Attention Layer.

    Complexity: O(E . d)  -  same as a sparse GCN, but with per-edge learnable
    attention weights so the model can decide which neighbours matter most.

    For each edge (i -> j) and each attention head h:
        e_ij^h = LeakyReLU( a_src^h . Wh_i^h  +  a_dst^h . Wh_j^h )
        α_ij^h = softmax_{j ∈ N(i)} ( e_ij^h )        <- sparse, grouped by dst
        z_i^h  = Σ_{j} α_ij^h . Wh_j^h

    Heads are concatenated, then fused with the input via a gated residual.
    Optional edge_weight (geo / OD) is added in log-space before softmax to
    keep prior structural information while still learning attention.

    Args:
        d             : feature dimension (must be divisible by num_heads)
        num_heads     : number of attention heads
        dropout       : dropout on attention weights and output
        negative_slope: LeakyReLU slope for attention scoring
    """

    def __init__(
        self,
        d:              int,
        num_heads:      int   = 4,
        dropout:        float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert d % num_heads == 0, \
            f"hidden_dim={d} must be divisible by num_heads={num_heads}"

        self.H  = num_heads
        self.Dh = d // num_heads  # per-head feature dimension

        # Shared linear projection W : d -> d (applied before splitting into heads)
        self.W = nn.Linear(d, d, bias=False)

        # Per-head attention vectors a_src and a_dst  (shape [1, H, Dh])
        self.attn_src = nn.Parameter(torch.empty(1, num_heads, self.Dh))
        self.attn_dst = nn.Parameter(torch.empty(1, num_heads, self.Dh))
        nn.init.xavier_uniform_(self.attn_src.view(1, -1))
        nn.init.xavier_uniform_(self.attn_dst.view(1, -1))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_drop  = nn.Dropout(dropout)

        # Post-aggregation feed-forward + gated self-loop fusion
        self.W1   = nn.Linear(d, d)       # transforms aggregated neighbours
        self.W2   = nn.Linear(d, d)       # output projection
        self.gate = nn.Linear(d * 2, d)   # gate: how much neighbour vs self
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h:           torch.Tensor,              # [B, N, d]
        adj:         torch.Tensor | None = None,          # unused (kept for interface compat)
        edge_index:  torch.Tensor | None = None,          # [2, E]
        edge_weight: torch.Tensor | None = None,          # [E]  optional geo/OD weights
    ) -> torch.Tensor:
        """Returns updated node features [B, N, d]."""
        B, N, d = h.shape
        H, Dh   = self.H, self.Dh

        if edge_index is None or edge_index.numel() == 0:
            # No edges: skip graph propagation, apply feed-forward only
            out = self.drop(F.gelu(self.W2(h)))
            return self.norm(h + out)

        src, dst = edge_index[0], edge_index[1]   # each [E]
        E = src.size(0)

        Wh = self.W(h).view(B, N, H, Dh)   # [B, N, H, Dh]

        # e_ij = LeakyReLU( a_src . Wh_i  +  a_dst . Wh_j )
        e_src = (Wh[:, src, :, :] * self.attn_src).sum(-1)   # [B, E, H]
        e_dst = (Wh[:, dst, :, :] * self.attn_dst).sum(-1)   # [B, E, H]
        e = self.leaky_relu(e_src + e_dst)                    # [B, E, H]

        # Optionally bias attention by log edge weight (geo proximity / OD flow)
        # Preserves structural prior while still allowing attention to override it
        if edge_weight is not None:
            e = e + edge_weight.clamp(min=1e-9).log().view(1, -1, 1)

        # Subtract per-dst maximum for numerical stability (O(E) scatter_add).
        # We use a global max as a safe upper bound - correct normalization is
        # preserved because the shift cancels in numerator and denominator.
        e_max = e.max(dim=1, keepdim=True)[0]   # [B, 1, H]
        e_exp = torch.exp(e - e_max)             # [B, E, H]

        # Sum exp per destination node for the normalisation constant
        e_sum = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
        idx   = dst.view(1, -1, 1).expand(B, E, H)
        e_sum.scatter_add_(1, idx, e_exp)        # [B, N, H]

        # Normalised attention coefficient α_ij  (softmax over incoming edges)
        alpha = e_exp / (e_sum[:, dst, :] + 1e-16)   # [B, E, H]
        alpha = self.attn_drop(alpha)                  # [B, E, H]

        weighted = Wh[:, src, :, :] * alpha.unsqueeze(-1)   # [B, E, H, Dh]

        agg = torch.zeros(B, N, H, Dh, device=h.device, dtype=h.dtype)
        idx4 = dst.view(1, -1, 1, 1).expand(B, E, H, Dh)
        agg.scatter_add_(1, idx4, weighted)              # [B, N, H, Dh]

        agg = F.elu(agg).reshape(B, N, d)               # [B, N, d]  (concat heads)

        # gate decides how much of the aggregated message vs. the original self
        gate = torch.sigmoid(self.gate(torch.cat([h, agg], dim=-1)))   # [B, N, d]
        msg  = gate * self.W1(agg) + (1 - gate) * h

        out = self.drop(F.gelu(self.W2(msg)))
        return self.norm(h + out)


#  Deprecated layers (kept for backward compatibility)        #

class SparseGCNLayer(nn.Module):
    """
    Deprecated - superseded by SparseGATLayer.
    Fixed degree-normalised aggregation with no learnable per-edge weights.
    Kept only for loading old checkpoints.
    """

    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.W1   = nn.Linear(d, d)
        self.W2   = nn.Linear(d, d)
        self.gate = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj=None, edge_index=None, edge_weight=None):
        B, N, d = h.shape
        if edge_index is not None and edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            E = src.size(0)
            h_src = h[:, src, :]
            if edge_weight is not None:
                h_src = h_src * edge_weight.view(1, -1, 1)
            agg = torch.zeros(B, N, d, device=h.device, dtype=h.dtype)
            agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, E, d), h_src)
            w   = edge_weight if edge_weight is not None else torch.ones(E, device=h.device)
            deg = torch.zeros(N, device=h.device).scatter_add_(0, dst, w)
            agg = agg / deg.clamp(min=1).view(1, -1, 1)
            gate = torch.sigmoid(self.gate(torch.cat([h, agg], dim=-1)))
            msg  = gate * self.W1(agg) + (1 - gate) * h
        else:
            msg = h
        out = self.drop(F.gelu(self.W2(msg)))
        return self.norm(h + out)


class DenseGATLayer(nn.Module):
    """
    Deprecated - O(N².d) is prohibitively slow for N > 200.
    Kept for backward compatibility; use SparseGATLayer instead.
    """

    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % num_heads == 0
        self.heads  = num_heads
        self.d_head = d // num_heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d)

    def forward(self, h, adj=None):
        B, N, d = h.shape
        H, Dh   = self.heads, self.d_head
        q = self.Wq(h).view(B, N, H, Dh).permute(0, 2, 1, 3)
        k = self.Wk(h).view(B, N, H, Dh).permute(0, 2, 1, 3)
        v = self.Wv(h).view(B, N, H, Dh).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        if adj is not None:
            attn = attn.masked_fill((adj == 0).unsqueeze(1), -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out  = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, d)
        return self.norm(h + self.Wo(out))


#  Full Station Encoder                                       #

class StationEncoder(nn.Module):
    """
    Multi-modal station encoder.
    Pipeline:
      1. TemporalFusion (Dilated TCN): seq [B,T,N,1] + tf [B,T,8] -> h [B,N,d]
      2. Coordinate embedding concatenated -> fused [B,N,d]
      3. SparseGATLayer × num_gat_layers: O(E.d) sparse multi-head attention
      4. GaussianEmbedding -> (z, μ, σ)
    """

    def __init__(
        self,
        hidden_dim:     int   = 64,
        num_gat_layers: int   = 2,
        num_gat_heads:  int   = 4,
        tcn_layers:     int   = 4,
        tcn_kernel:     int   = 3,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # temporal encoding (demand_dim=1, time_dim=8)
        self.temporal_fusion = TemporalFusion(
            demand_dim=1, time_dim=8,
            hidden=hidden_dim, out_dim=hidden_dim,
            n_layers=tcn_layers, kernel_size=tcn_kernel, dropout=dropout)

        # coordinate embedding: lat/lon -> hidden_dim
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # fuse temporal + spatial features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # SparseGATLayer stack: O(E.d) sparse attention, no torch_geometric needed
        self.gat_layers = nn.ModuleList([
            SparseGATLayer(hidden_dim, num_gat_heads, dropout)
            for _ in range(num_gat_layers)
        ])

        self.gaussian_emb = GaussianEmbedding(hidden_dim)

    def forward(
        self,
        seq:         torch.Tensor,               # [B, T, N] demand sequence
        coords:      torch.Tensor,               # [N, 2]   lat/lon
        tf:          torch.Tensor,               # [B, T, 8] time features
        edge_index:  torch.Tensor | None = None, # [2, E]
        edge_weight: torch.Tensor | None = None, # [E]
    ):
        """
        Returns:
            z     [B, N, d]  sampled representation (reparameterized)
            mu    [B, N, d]  distribution mean
            sigma [B, N, d]  distribution std
        """
        B, T, N = seq.shape

        # 1. Temporal encoding
        seq_in = seq.unsqueeze(-1)                        # [B, T, N, 1]
        h_tcn  = self.temporal_fusion(seq_in, tf)         # [B, N, d]

        # 2. Coordinate embedding (broadcast over batch)
        h_coord = self.coord_proj(coords)                 # [N, d]
        h_coord = h_coord.unsqueeze(0).expand(B, -1, -1) # [B, N, d]

        # 3. Fuse temporal + spatial
        h = self.fusion(torch.cat([h_tcn, h_coord], dim=-1))  # [B, N, d]

        # 4. Sparse GAT propagation (learnable per-edge attention, O(E.d))
        for gat in self.gat_layers:
            h = gat(h, edge_index=edge_index, edge_weight=edge_weight)

        # 5. Gaussian uncertainty embedding
        z, mu, sigma = self.gaussian_emb(h)               # [B, N, d] each
        return z, mu, sigma
