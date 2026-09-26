"""
Dilated Temporal Convolutional Network (TCN) 
: TransMod Section III-B
:  [B, seq_len, N] ->  [B, N, d]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """
     ( + LayerNorm)
    input/output: [B, C, T]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation)
        self.norm    = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class DilatedTCN(nn.Module):
    """
     TCN
    Args:
        in_dim:     ()
        hidden:    
        out_dim:   
        n_layers:  TCN  (dilation = 2^i)
        kernel_size: 
        dropout:   dropout 
    :  x [B, T, N, in_dim]
    :  h [B, N, out_dim]   ()
    """

    def __init__(
        self,
        in_dim:     int,
        hidden:     int = 64,
        out_dim:    int = 64,
        n_layers:   int = 4,
        kernel_size: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(DilatedConvBlock(
                hidden, hidden, kernel_size, dilation, dropout))
        self.layers   = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, in_dim]
        returns h: [B, N, out_dim]
        """
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # [B*N, T, D]
        x = self.input_proj(x)                            # [B*N, T, H]
        x = x.permute(0, 2, 1)                            # [B*N, H, T]

        for layer in self.layers:
            x = layer(x)                                   # [B*N, H, T]

        x = x[:, :, -1]
        x = self.out_proj(x)                               # [B*N, out_dim]
        x = x.reshape(B, N, -1)                            # [B, N, out_dim]
        return x


class FlatTemporalEncoder(nn.Module):
    """
    cuDNN-free temporal encoder: flattens the T time steps and maps with Linear layers.
    Equivalent in expressiveness to a short TCN for seq_len=12.
    Used in place of DilatedTCN when cuDNN is unavailable (e.g. CUDNN_STATUS_NOT_INITIALIZED).

    Input:  x [B*N, T * in_dim]
    Output: h [B*N, out_dim]
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*N, flat_dim]  ->  [B*N, out_dim]"""
        return self.net(x)


class TemporalFusion(nn.Module):
    """
     +  (/ sin-cos 8)
    :
        seq: [B, T, N, in_dim]   ()
        tf:  [B, T, 8]          
    :
        h:   [B, N, out_dim]

    Uses FlatTemporalEncoder (cuDNN-free) instead of DilatedTCN because cuDNN is
    unavailable on this server (CUDNN_STATUS_NOT_INITIALIZED); Conv1d without
    cuDNN falls back to an extremely slow CUDA path (~6s/batch vs <5ms).
    """

    def __init__(self, demand_dim: int, time_dim: int = 8,
                 hidden: int = 64, out_dim: int = 64,
                 n_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1,
                 seq_len: int = 12):
        super().__init__()
        self.seq_len = seq_len
        flat_dim = seq_len * (demand_dim + time_dim)
        self.encoder = FlatTemporalEncoder(flat_dim, hidden=hidden, out_dim=out_dim,
                                           dropout=dropout)

    def forward(self, seq: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, N, demand_dim]
        tf:  [B, T, 8]
        """
        B, T, N, _ = seq.shape
        tf_exp = tf.unsqueeze(2).expand(B, T, N, -1)        # [B, T, N, 8]
        x = torch.cat([seq, tf_exp], dim=-1)                 # [B, T, N, demand_dim+8]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T * x.shape[-1])  # [B*N, T*(D+8)]
        h = self.encoder(x)                                  # [B*N, out_dim]
        return h.reshape(B, N, -1)                           # [B, N, out_dim]
