"""
Memory-augmented transfer module.
Paper: TransMod Section III-E

Two key upgrades over the original implementation:
  1. DirectTemporalEncoder - when within-city historical demand is available, encode it
     directly (lightweight dilated TCN) and blend with memory retrieval via a learned gate.
     Cold-start falls back to pure memory (gate -> 1 for memory path).
  2. ZoneAttentionLayer - replaces the fixed-weight GCN (ZoneGNN) with full multi-head
     self-attention over zones, biased by the zone graph structure (log A_z as soft prior).
     Lets the model learn which zone interactions matter rather than hard-wiring degree normalisation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


#  Memory Pool                                               #

class MemoryPool(nn.Module):
    """
    M learnable key-value memory pairs.
    Keys index reusable spatio-temporal prototypes learned during pre-training.
    Values carry the transferable temporal knowledge retrieved at inference.

    Args:
        memory_size: M - number of memory slots
        dim:         d - memory dimension
        delta:       diversity regularisation threshold (prevents key collapse)
    """

    def __init__(self, memory_size: int = 128, dim: int = 64, delta: float = 0.3):
        super().__init__()
        self.M     = memory_size
        self.d     = dim
        self.delta = delta
        self.keys   = nn.Parameter(torch.randn(memory_size, dim) * 0.02)
        self.values = nn.Parameter(torch.randn(memory_size, dim) * 0.02)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Softmax attention retrieval.
        query: [B, Z, d]  ->  retrieved: [B, Z, d]
        """
        q = F.normalize(query,     dim=-1)      # [B, Z, d]
        k = F.normalize(self.keys, dim=-1)       # [M, d]
        attn = torch.einsum('bzd,md->bzm', q, k) / (self.d ** 0.5)
        attn = F.softmax(attn, dim=-1)           # [B, Z, M]
        return torch.einsum('bzm,md->bzd', attn, self.values)

    def diversity_loss(self) -> torch.Tensor:
        """
        L_div = mean_{i≠j} max(0, cos(k_i, k_j) - δ)
        Encourages memory keys to specialise in distinct spatio-temporal patterns.
        """
        k_norm = F.normalize(self.keys, dim=-1)  # [M, d]
        sim    = k_norm @ k_norm.t()              # [M, M]
        mask   = ~torch.eye(self.M, dtype=torch.bool, device=sim.device)
        return F.relu(sim[mask] - self.delta).mean()


#  Prompt Network (cold-start spatial query for target)      #

class PromptNetwork(nn.Module):
    """
    Maps zone spatial features (coordinates + context) to memory query vectors.
    Used in cold-start mode when no temporal history is available.
    Input:  spatial_feat [B, Z, spatial_dim]
    Output: query        [B, Z, d]
    """

    def __init__(self, spatial_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#  Priority 1a: Direct Temporal Encoder                      #

class DirectTemporalEncoder(nn.Module):
    """
    cuDNN-free flat temporal encoder for direct zone-level time series encoding.
    Replaces the previous Conv1d-based dilated TCN (which was catastrophically slow
    without cuDNN: CUDNN_STATUS_NOT_INITIALIZED on this server).

    Input:  rh_seq [B, T, Z]   - Z zone demand sequences of length T
    Output: h      [B, Z, d]
    """

    def __init__(self, hidden_dim: int, n_layers: int = 2,
                 kernel: int = 3, dropout: float = 0.1, seq_len: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, rh_seq: torch.Tensor) -> torch.Tensor:
        """rh_seq: [B, T, Z]  ->  h: [B, Z, d]"""
        B, T, Z = rh_seq.shape
        x = rh_seq.permute(0, 2, 1)          # [B, Z, T]
        x = x.reshape(B * Z, T)              # [B*Z, T]
        x = self.net(x)                       # [B*Z, d]
        return self.norm(x.reshape(B, Z, -1)) # [B, Z, d]


#  Priority 2: Zone Attention Layer                          #

class ZoneAttentionLayer(nn.Module):
    """
    Zone-level multi-head self-attention with structural bias from the zone graph.

    Replaces ZoneGNN (fixed-weight symmetric GCN) with a fully learnable attention
    mechanism so the model can discover which zone pairs actually matter for demand
    prediction, rather than relying solely on geographic adjacency.

    Design:
        score_ij = (Q_i . K_j) / √d_h  +  struct_bias(log A_z[i,j])
        α        = softmax(score, dim=zone_j)
        out_i    = Σ_j α_ij . V_j

    The structural bias term keeps log A_z as a soft prior: adjacent zones (high A_z)
    get a positive bias; non-adjacent zones get a large negative bias (log ≈ -∞ -> 0 attention).
    Unlike hard masking, this allows the model to override the prior when data demands it.

    Activity gate (from ZoneGNN): suppresses signal from sparse/inactive zones,
    which is critical for the highly sparse Chicago RH data (mean ≈ 2.35 trips/zone).

    Complexity: O(Z² . d) - feasible for Z ≤ 200 zones.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.H  = n_heads
        self.Dh = dim // n_heads

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim)

        # structural bias: log-adjacency -> per-head scalar bias [Z, Z, H]
        self.struct_bias = nn.Linear(1, n_heads, bias=False)

        # feed-forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, A_z: torch.Tensor) -> torch.Tensor:
        """
        H:   [B, Z, d]
        A_z: [Z, Z]   - zone graph adjacency (row-normalised, non-negative)
        Returns: [B, Z, d]
        """
        B, Z, d = H.shape
        H_norm = self.norm1(H)

        # Activity gate: suppress low-activity / sparse zones
        # (important for Chicago RH where many zones have near-zero demand)
        activity   = H_norm.detach().abs().mean(dim=-1, keepdim=True)      # [B, Z, 1]
        act_thresh = activity.mean(dim=1, keepdim=True).clamp(min=1e-6)    # [B, 1, 1]
        act_gate   = torch.sigmoid(5.0 * (activity / act_thresh - 0.5))   # [B, Z, 1]
        H_gated    = H_norm * act_gate                                      # [B, Z, d]

        # Multi-head projections
        Q = self.Wq(H_gated).view(B, Z, self.H, self.Dh).permute(0, 2, 1, 3)  # [B,H,Z,Dh]
        K = self.Wk(H_gated).view(B, Z, self.H, self.Dh).permute(0, 2, 1, 3)
        V = self.Wv(H_gated).view(B, Z, self.H, self.Dh).permute(0, 2, 1, 3)

        # Scaled dot-product attention scores
        score = (Q @ K.transpose(-2, -1)) / (self.Dh ** 0.5)   # [B, H, Z, Z]

        # Structural bias from zone graph: log A_z as additive prior
        # Non-adjacent zones: A_z ≈ 0  ->  log ≈ -inf  ->  near-zero attention weight
        A_log = A_z.clamp(min=1e-6).log().unsqueeze(-1)         # [Z, Z, 1]
        bias  = self.struct_bias(A_log).permute(2, 0, 1)         # [H, Z, Z]
        score = score + bias.unsqueeze(0)                         # [B, H, Z, Z]

        attn = self.drop(F.softmax(score, dim=-1))               # [B, H, Z, Z]

        # Weighted aggregation + output projection
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, Z, d)   # [B, Z, d]
        H   = H + self.Wo(out)                                    # residual 1

        # Feed-forward sublayer
        H   = H + self.ff(self.norm2(H))                         # residual 2
        return H


#  Deprecated ZoneGNN (kept for loading old checkpoints)     #

class ZoneGNN(nn.Module):
    """
    Deprecated - superseded by ZoneAttentionLayer.
    Fixed symmetric-normalised 2-layer GCN. Kept only for loading old checkpoints.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _gcn(self, A, H, W, norm):
        deg   = A.sum(dim=-1).clamp(min=1).sqrt()
        A_hat = A / (deg.unsqueeze(1) * deg.unsqueeze(0))
        activity   = H.detach().abs().mean(dim=-1, keepdim=True)
        act_thresh = activity.mean(dim=1, keepdim=True).clamp(min=1e-6)
        mask = torch.sigmoid(5.0 * (activity / act_thresh - 0.5))
        out  = torch.einsum('zk,bkd->bzd', A_hat, H * mask)
        return self.drop(F.gelu(norm(W(out))))

    def forward(self, H, A):
        h1 = self._gcn(A, H, self.W1, self.n1)
        h2 = self._gcn(A, h1, self.W2, self.n2)
        return H + h2


#  Full Memory Transfer Module                               #

class MemoryTransfer(nn.Module):
    """
    Memory-augmented transfer + zone-level prediction head.

    Pre-training:
        query = query_encoder([H_bike_zone; H_metro_zone; tf])
        mem   = MemoryPool.retrieve(query)
        H     = fuse(H_zone, mem) -> ZoneAttentionLayer(A_z) -> pred_head

    Transfer - two modes:
        (a) Within-city / cross-city with target history (rh_history is not None):
            H_mem    = fuse(H_rh_zone, prompt_net(spatial))
            H_direct = DirectTemporalEncoder(rh_history)
            H_zone   = gate * H_mem + (1 - gate) * H_direct   <- learned blend
            pred     = pred_head(ZoneAttentionLayer(H_zone, A_z_rh))

        (b) Cold-start (rh_history is None):
            H_zone   = fuse(H_rh_zone, prompt_net(spatial))   <- memory only
            pred     = pred_head(ZoneAttentionLayer(H_zone, A_z_rh))
    """

    def __init__(
        self,
        hidden_dim:  int   = 64,
        time_dim:    int   = 8,
        memory_size: int   = 128,
        delta:       float = 0.3,
        n_zones_src: int   = 200,
        dropout:     float = 0.1,
        pred_len:    int   = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        # Pre-training query encoder: [bike_d + metro_d + time_d -> d]
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transfer prompt network: spatial_dim = 2 (lat/lon) + hidden_dim (zone context)
        self.prompt_net = PromptNetwork(
            spatial_dim=hidden_dim + 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        self.memory_pool = MemoryPool(memory_size, hidden_dim, delta)

        # Gated fusion (memory + zone representation)
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Zone-level graph propagation: ZoneAttentionLayer (replaces fixed-weight GCN)
        self.zone_gnn = ZoneAttentionLayer(hidden_dim, n_heads=4, dropout=dropout)

        # Priority 1a: direct temporal encoder for within-city mode
        self.rh_temporal_enc = DirectTemporalEncoder(
            hidden_dim, n_layers=2, kernel=3, dropout=dropout)

        # Learned gate: blend memory retrieval vs direct temporal encoding
        # gate -> 1: rely on memory (cold-start); gate -> 0: rely on direct temporal
        self.temporal_fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pred_len),
        )


    def _fuse_memory(self, H_zone: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory pool and gate-fuse with zone representation.
        H_zone: [B, Z, d],  query: [B, Z, d]  ->  [B, Z, d]
        """
        mem  = self.memory_pool.retrieve(query)         # [B, Z, d]
        cat  = torch.cat([H_zone, mem], dim=-1)         # [B, Z, 2d]
        gate = self.fuse_gate(cat)                      # [B, Z, d]
        proj = self.fuse_proj(cat)                      # [B, Z, d]
        return gate * proj + (1 - gate) * H_zone


    def forward_pretrain(
        self,
        H_bike_zone:  torch.Tensor,   # [B, Z, d]
        H_metro_zone: torch.Tensor,   # [B, Z, d]
        A_z:          torch.Tensor,   # [Z, Z]
        tf:           torch.Tensor,   # [B, T, 8]
    ) -> torch.Tensor:
        """Returns pred [B, Z, pred_len]"""
        B, Z, d = H_bike_zone.shape

        # Time feature: take last timestep
        tf_last = tf[:, -1, :]                           # [B, 8]
        tf_exp  = tf_last.unsqueeze(1).expand(B, Z, -1)  # [B, Z, 8]

        cat   = torch.cat([H_bike_zone, H_metro_zone, tf_exp], dim=-1)  # [B, Z, 2d+8]
        query = self.query_encoder(cat)                  # [B, Z, d]

        # Average source modalities, then fuse with memory
        H_zone = (H_bike_zone + H_metro_zone) / 2       # [B, Z, d]
        H_zone = self._fuse_memory(H_zone, query)

        # Zone-level attention propagation
        H_zone = self.zone_gnn(H_zone, A_z)

        return self.pred_head(H_zone)                    # [B, Z, pred_len]


    def forward_transfer(
        self,
        H_rh_zone:   torch.Tensor,              # [B, Z_rh, d]  encoded zone features
        A_z_rh:      torch.Tensor,              # [Z_rh, Z_rh]  zone graph
        zone_coords: torch.Tensor,              # [Z_rh, 2]     lat/lon
        rh_history:  torch.Tensor | None = None,  # [B, T, Z_rh] raw demand (optional)
        return_aux:  bool = False,              # if True, return (pred, aux_dict)
    ) -> torch.Tensor:
        """
        Transfer stage. Two operation modes:

        Within-city / cross-city with target history (rh_history is not None):
            Memory path:  spatial prompt -> memory retrieval -> fuse with zone features
            Direct path:  rh_history -> DirectTemporalEncoder
            Final:        learned gate blends the two paths

        Cold-start (rh_history is None):
            Memory path only (same as original implementation).

        Returns pred [B, Z_rh, pred_len]
        """
        B, Z, d = H_rh_zone.shape

        coord_exp = zone_coords.unsqueeze(0).expand(B, -1, -1)  # [B, Z, 2]
        spatial   = torch.cat([H_rh_zone, coord_exp], dim=-1)   # [B, Z, d+2]
        query     = self.prompt_net(spatial)                      # [B, Z, d]
        H_mem     = self._fuse_memory(H_rh_zone, query)          # [B, Z, d]

        gate = None  # set below if direct path is active (needed for aux return)
        if rh_history is not None and rh_history.abs().max() > 1e-6:
            h_direct = self.rh_temporal_enc(rh_history)          # [B, Z, d]
            # Learned gate: decides how much to trust memory vs direct temporal encoding.
            # At cold-start initialisation gate ≈ 0.5; during fine-tuning the gate learns
            # to rely more on direct evidence when target history is informative.
            gate   = self.temporal_fusion_gate(
                torch.cat([H_mem, h_direct], dim=-1))             # [B, Z, d]
            H_zone = gate * H_mem + (1 - gate) * h_direct
        else:
            # Cold-start: no target history - rely entirely on memory
            H_zone = H_mem

        H_zone = self.zone_gnn(H_zone, A_z_rh)

        pred = self.pred_head(H_zone)                             # [B, Z, pred_len]

        if return_aux:
            # Return gate for entropy regularisation in the training loop.
            # gate is None in cold-start mode (no direct path active).
            aux = {'gate': gate if (rh_history is not None
                                    and rh_history.abs().max() > 1e-6) else None}
            return pred, aux
        return pred


    def freeze_for_transfer(self):
        """
        Full freeze: memory pool + zone attention + query encoder.
        Trainable: prompt_net, pred_head, fuse_gate, fuse_proj,
                   rh_temporal_enc, temporal_fusion_gate.
        """
        for module in [self.memory_pool, self.zone_gnn, self.query_encoder]:
            for p in module.parameters():
                p.requires_grad_(False)

    def soft_freeze_for_transfer(self):
        """
        Soft freeze (recommended): freeze only memory keys + query encoder.
        Trainable: memory values, zone_gnn, prompt_net, pred_head, fuse_*,
                   rh_temporal_enc, temporal_fusion_gate.

        Rationale:
        - Memory keys stay fixed to preserve learned knowledge index structure.
        - zone_gnn adapts its attention to target-domain zone interactions.
        - rh_temporal_enc and temporal_fusion_gate are target-domain specific;
          always trainable regardless of freeze mode.
        """
        self.memory_pool.keys.requires_grad_(False)
        for p in self.query_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def get_diversity_loss(self) -> torch.Tensor:
        return self.memory_pool.diversity_loss()
