from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .station_encoder  import StationEncoder
from .soft_assignment  import SoftAssignment
from .alignment        import CrossModalAlignment
from .memory_transfer  import MemoryTransfer


class TransModFramework(nn.Module):
    """
    Full TransMod framework.

    Args:
        n_bike:      number of bike stations
        n_metro:     number of metro stations
        n_rh_zones:  number of ridehailing zones
        n_zones:     unified zone count (soft assignment target)
        hidden_dim:  hidden dimension d
        ...
    """

    def __init__(
        self,
        n_bike:       int,
        n_metro:      int,
        n_rh_zones:   int,
        n_zones:      int,
        hidden_dim:   int  = 64,
        pred_len:     int  = 1,
        num_gat_layers: int = 2,
        num_gat_heads:  int = 4,
        tcn_layers:   int  = 4,
        tcn_kernel:   int  = 3,
        top_k_bike:   int  = 5,
        top_k_metro:  int  = 8,
        sigma_bike:   float = 500.0,
        sigma_metro:  float = 800.0,
        memory_size:  int  = 128,
        tau:          float = 0.07,
        delta:        float = 0.3,
        dropout:      float = 0.1,
        # loss weights
        lambda_mmd:    float = 0.1,
        lambda_nce:    float = 0.5,
        lambda_geo:    float = 0.1,
        lambda_div:    float = 0.05,
        lambda_assign: float = 0.1,   # assignment structure regularisation (TSTL L_r)
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.lambda_mmd    = lambda_mmd
        self.lambda_nce    = lambda_nce
        self.lambda_geo    = lambda_geo
        self.lambda_div    = lambda_div
        self.lambda_assign = lambda_assign

        enc_kwargs = dict(
            hidden_dim=hidden_dim, num_gat_layers=num_gat_layers,
            num_gat_heads=num_gat_heads, tcn_layers=tcn_layers,
            tcn_kernel=tcn_kernel, dropout=dropout)

        self.bike_encoder  = StationEncoder(**enc_kwargs)
        self.metro_encoder = StationEncoder(**enc_kwargs)
        # separate target-domain encoder (instantiated independently for
        # transferable parameter isolation rather than shared backbone)
        self.rh_encoder    = StationEncoder(**enc_kwargs)

        self.bike_assign  = SoftAssignment(n_bike,  n_zones, top_k_bike,  sigma_bike)
        self.metro_assign = SoftAssignment(n_metro, n_zones, top_k_metro, sigma_metro)
        self.rh_assign    = SoftAssignment(n_rh_zones, n_zones, top_k_bike, sigma_bike)

        self.alignment = CrossModalAlignment(hidden_dim, tau)

        self.memory_transfer = MemoryTransfer(
            hidden_dim=hidden_dim, time_dim=8,
            memory_size=memory_size, delta=delta,
            dropout=dropout, pred_len=pred_len)

        # MemoryTransfer already has its own pred_head; these provide
        # bike / metro branch auxiliary losses during pre-training
        self.bike_head  = nn.Linear(hidden_dim, pred_len)
        self.metro_head = nn.Linear(hidden_dim, pred_len)

    #  Geography initialisation (call before first forward pass)      #

    def initialize_geography(
        self,
        bike_coords:   torch.Tensor,   # [N_bike,  2]
        metro_coords:  torch.Tensor,   # [N_metro, 2]
        zone_centers:  torch.Tensor,   # [Z, 2]
        rh_zone_centers: Optional[torch.Tensor] = None,  # [Z_rh, 2]
    ):
        self.bike_assign .initialize_geography(bike_coords,  zone_centers)
        self.metro_assign.initialize_geography(metro_coords, zone_centers)
        if rh_zone_centers is not None:
            self.rh_assign.initialize_geography(rh_zone_centers, zone_centers)
        # store coordinates for use by encoders
        self.register_buffer('zone_centers_buf', zone_centers)
        self.register_buffer('bike_coords_buf',  bike_coords)
        self.register_buffer('metro_coords_buf', metro_coords)
        if rh_zone_centers is not None:
            self.register_buffer('rh_coords_buf', rh_zone_centers)

    #  Pre-training forward                                           #

    @staticmethod
    def _edge_to_dense(
        edge_index: Optional[torch.Tensor],   # [2, E]
        edge_weight: Optional[torch.Tensor],  # [E]
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sparse edge_index -> dense symmetric adjacency matrix [n, n].
        Used for DiffPool: A_zone = S^T A_station S.
        """
        A = torch.zeros(n, n, device=device)
        if edge_index is not None and edge_index.numel() > 0:
            w = edge_weight if edge_weight is not None else torch.ones(
                edge_index.size(1), device=device)
            A[edge_index[0], edge_index[1]] = w
            A = (A + A.t()) / 2   # symmetrise
        A.fill_diagonal_(1.0)
        return A

    def forward_pretrain(
        self,
        bike_seq:     torch.Tensor,   # [B, T, N_bike]
        metro_seq:    torch.Tensor,   # [B, T, N_metro]
        tf:           torch.Tensor,   # [B, T, 8]
        bike_edge:    Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        metro_edge:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        A_od_bike:    Optional[torch.Tensor] = None,   # [Z, Z] zone-level OD graph
        A_od_metro:   Optional[torch.Tensor] = None,
        A_station_bike:  Optional[torch.Tensor] = None,  # [N_bike,  N_bike]  dense station graph
        A_station_metro: Optional[torch.Tensor] = None,  # [N_metro, N_metro]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            pred:   [B, Z, pred_len]  (unified zone prediction)
            losses: dict of scalar tensors

        v3 improvements:
          - DiffPool dynamic zone graph: A_z = α.(S^T A_bike S + S^T A_metro S)/2
                                               + β.A_geo + γ.A_feat
          - assignment structure regularisation L_assign (TSTL L_r)
        """
        B = bike_seq.size(0)
        N_bike  = bike_seq.size(2)
        N_metro = metro_seq.size(2)
        zone_centers = self.zone_centers_buf
        dev = bike_seq.device

        bike_ei = bike_edge[0]  if bike_edge  else None
        bike_ew = bike_edge[1]  if bike_edge  else None
        metro_ei = metro_edge[0] if metro_edge else None
        metro_ew = metro_edge[1] if metro_edge else None

        z_bike,  mu_bike,  sigma_bike  = self.bike_encoder(
            bike_seq,  self.bike_coords_buf,
            tf, bike_ei, bike_ew)
        z_metro, mu_metro, sigma_metro = self.metro_encoder(
            metro_seq, self.metro_coords_buf,
            tf, metro_ei, metro_ew)

        H_bike_zone  = self.bike_assign .aggregate_station_to_zone(z_bike)   # [B,Z,d]
        H_metro_zone = self.metro_assign.aggregate_station_to_zone(z_metro)  # [B,Z,d]

        Z = H_bike_zone.size(1)
        # use DiffPool if a dense station graph is provided; else fall back to OD graph
        if A_station_bike is not None:
            A_diff_bike  = self.bike_assign .aggregate_adj(A_station_bike.to(dev))
        else:
            A_diff_bike  = (A_od_bike  if A_od_bike  is not None
                            else torch.zeros(Z, Z, device=dev))
        if A_station_metro is not None:
            A_diff_metro = self.metro_assign.aggregate_adj(A_station_metro.to(dev))
        else:
            A_diff_metro = (A_od_metro if A_od_metro is not None
                            else torch.zeros(Z, Z, device=dev))

        A_z, mmd_loss, nce_loss = self.alignment(
            H_bike_zone, H_metro_zone,
            A_diff_bike, A_diff_metro, zone_centers)

        pred_bike  = self.bike_head (H_bike_zone)   # [B, Z, pred_len]
        pred_metro = self.metro_head(H_metro_zone)  # [B, Z, pred_len]

        pred = self.memory_transfer.forward_pretrain(
            H_bike_zone, H_metro_zone, A_z, tf)     # [B, Z, pred_len]

        geo_loss = (self.bike_assign.geo_loss() +
                    self.metro_assign.geo_loss())
        div_loss = self.memory_transfer.get_diversity_loss()

        # assignment consistency regularisation (lightweight O(NZ²) proxy for TSTL L_r)
        # compares learned vs geo-prior zone co-occupancy patterns - no N×N matrix needed
        assign_loss = (self.bike_assign.assignment_reg_loss() +
                       self.metro_assign.assignment_reg_loss())

        losses = {
            'mmd':      mmd_loss,
            'nce':      nce_loss,
            'geo':      geo_loss,
            'div':      div_loss,
            'assign':   assign_loss,
            'pred_bike':  pred_bike,    # returned for external L_pred computation
            'pred_metro': pred_metro,
        }
        return pred, losses

    def compute_pretrain_loss(
        self,
        pred:        torch.Tensor,   # [B, Z, pred_len]
        target:      torch.Tensor,   # [B, Z, pred_len]  (normalised zone mean)
        losses:      Dict,
        pred_bike:   Optional[torch.Tensor] = None,
        pred_metro:  Optional[torch.Tensor] = None,
        target_bike: Optional[torch.Tensor] = None,
        target_metro: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L = L_pred + λ_mmd.L_mmd + λ_nce.L_nce + λ_geo.L_geo
              + λ_div.L_div + λ_assign.L_assign
        """
        L = F.huber_loss(pred, target)
        L = L + self.lambda_mmd    * losses['mmd']
        L = L + self.lambda_nce    * losses['nce']
        L = L + self.lambda_geo    * losses['geo']
        L = L + self.lambda_div    * losses['div']
        L = L + self.lambda_assign * losses.get('assign', torch.tensor(0.0))
        return L

    #  Transfer forward                                               #

    def _build_geo_adj(self, zone_centers: torch.Tensor,
                       threshold_km: float = 3.0) -> torch.Tensor:
        """Haversine geo-proximity adjacency matrix [Z, Z], row-normalised."""
        lat = zone_centers[:, 0] * (torch.pi / 180)
        lon = zone_centers[:, 1] * (torch.pi / 180)
        dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
        dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat.unsqueeze(1)) * torch.cos(lat.unsqueeze(0))
             * torch.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2 * torch.atan2(torch.sqrt(a.clamp(0, 1)),
                                             torch.sqrt((1 - a).clamp(0, 1)))
        A = (dist_km < threshold_km).float()
        A.fill_diagonal_(1.0)
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1)
        return A / row_sum

    def forward_transfer(
        self,
        rh_seq:   torch.Tensor,   # [B, T, N_rh]
        tf:       torch.Tensor,   # [B, T, 8]
        rh_zone_centers: torch.Tensor,   # [Z_rh, 2]
        rh_edge:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_aux: bool = False,  # if True, return (pred, aux_dict) for extra losses
    ):
        """Returns pred [B, Z_rh, pred_len], or (pred, aux_dict) if return_aux=True."""
        rh_ei = rh_edge[0] if rh_edge else None
        rh_ew = rh_edge[1] if rh_edge else None

        # use real RH zone coordinates (not zero placeholders)
        rh_coords = getattr(self, 'rh_coords_buf',
                            torch.zeros(rh_seq.size(2), 2, device=rh_seq.device))

        z_rh, _, _ = self.rh_encoder(
            rh_seq, rh_coords, tf, rh_ei, rh_ew)   # [B, Z_rh, d]

        # zone graph: geo-proximity adjacency (replaces zero matrix for ZoneGNN propagation)
        A_z_rh = self._build_geo_adj(rh_zone_centers).to(rh_seq.device)

        # Pass raw rh_seq as rh_history so DirectTemporalEncoder can use it
        # when target demand data is available (within-city / cross-city with history).
        # forward_transfer falls back to memory-only if rh_seq is all-zero (cold-start).
        return self.memory_transfer.forward_transfer(
            z_rh, A_z_rh, rh_zone_centers,
            rh_history=rh_seq, return_aux=return_aux)

    #  Parameter freezing (transfer stage)                            #

    def freeze_for_transfer(self):
        """
        Full freeze: source encoders + soft assignment + alignment + memory pool/GNN.
        Trainable: rh_encoder + prompt_net + pred_head.
        """
        for module in [self.bike_encoder, self.metro_encoder,
                       self.bike_assign, self.metro_assign,
                       self.alignment, self.bike_head, self.metro_head]:
            for p in module.parameters():
                p.requires_grad_(False)
        self.memory_transfer.freeze_for_transfer()
        for module in [self.rh_encoder,
                       self.memory_transfer.prompt_net,
                       self.memory_transfer.pred_head,
                       self.memory_transfer.fuse_gate,
                       self.memory_transfer.fuse_proj,
                       self.memory_transfer.rh_temporal_enc,
                       self.memory_transfer.temporal_fusion_gate]:
            for p in module.parameters():
                p.requires_grad_(True)

    def soft_freeze_for_transfer(self):
        """
        Soft freeze (recommended for v3+):
        Frozen:   source encoders + soft assignment + alignment + memory keys + query_encoder
        Trainable: rh_encoder + zone_gnn + memory values + prompt_net + pred_head + fuse_*

        zone_gnn remains trainable to adapt to the target-domain zone structure
        (RH zones differ structurally from bike/metro zones).
        """
        for module in [self.bike_encoder, self.metro_encoder,
                       self.bike_assign, self.metro_assign,
                       self.alignment, self.bike_head, self.metro_head]:
            for p in module.parameters():
                p.requires_grad_(False)
        # memory_transfer: soft freeze only
        self.memory_transfer.soft_freeze_for_transfer()
        # ensure target-domain side is trainable
        for module in [self.rh_encoder,
                       self.memory_transfer.prompt_net,
                       self.memory_transfer.pred_head,
                       self.memory_transfer.fuse_gate,
                       self.memory_transfer.fuse_proj,
                       self.memory_transfer.zone_gnn,
                       # new: always trainable in transfer (target-domain specific)
                       self.memory_transfer.rh_temporal_enc,
                       self.memory_transfer.temporal_fusion_gate]:
            for p in module.parameters():
                p.requires_grad_(True)
        # memory values trainable
        self.memory_transfer.memory_pool.values.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> Dict[str, int]:
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train
        return {'total': total, 'trainable': train, 'frozen': frozen}
