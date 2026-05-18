from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_1d(positions: torch.Tensor, d: int) -> torch.Tensor:
    """
    positions: arbitrary integer-valued tensor (...,)
    returns: (..., d) float sinusoidal encoding
    """
    device = positions.device
    half = d // 2
    freq = torch.pow(
        10000.0,
        torch.arange(0, half, dtype=torch.float32, device=device) * (-2.0 / d),
    )  # [half]
    # broadcast: positions (...) x freq (half) -> (..., half)
    angles = positions.float().unsqueeze(-1) * freq  # (..., half)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., d)
    return pe


def two_dim_pe(cell_positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    cell_positions: [N, n_cells, 2]  last dim = [x_coord, y_coord]
    returns: [N, n_cells, d_model]
    """
    assert d_model % 2 == 0, "d_model must be even for 2D PE"
    x_pe = sinusoidal_1d(cell_positions[..., 0], d_model // 2)  # [N, n_cells, d//2]
    y_pe = sinusoidal_1d(cell_positions[..., 1], d_model // 2)  # [N, n_cells, d//2]
    return torch.cat([x_pe, y_pe], dim=-1)                       # [N, n_cells, d_model]


class NodeEmbedder(nn.Module):
    """
    Step 1: cell-level demand → node embedding.

    For each (batch, timestep), processes 49 cells per 50 nodes through
    a Transformer Encoder, producing one embedding vector per node.
    Node IDs are NOT used here (position-only spatial context).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        cell_positions: torch.Tensor,   # [50, 49, 2]
        n_demand_levels: int = 9,
        norm_first: bool = True,
    ):
        super().__init__()
        assert d_model % 2 == 0

        self.demand_embedding_table = nn.Embedding(n_demand_levels, d_model)

        self.day_embed     = nn.Embedding(7,  d_model)
        self.time_embed    = nn.Embedding(24, d_model)
        self.holiday_embed = nn.Embedding(2,  d_model)

        # Precompute and register 2D PE as a fixed buffer
        cell_pe = two_dim_pe(cell_positions.long(), d_model)   # [50, 49, d_model]
        self.register_buffer('cell_pe', cell_pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        cell_demands: torch.Tensor,   # [B, h, 50, 49]  long
        day:          torch.Tensor,   # [B, h]           long (0-6)
        time_idx:     torch.Tensor,   # [B, h]           long (0-23)
        holiday:      torch.Tensor,   # [B, h]           long (0-1)
    ) -> torch.Tensor:                # [B, h, 50, d_model]
        B, h, N, C = cell_demands.shape
        d = self.cell_pe.shape[-1]

        x = self.demand_embedding_table(cell_demands)  # [B, h, 50, 49, d]
        x = x + self.cell_pe                           # broadcast [50, 49, d]

        t_embed = (self.day_embed(day)
                 + self.time_embed(time_idx)
                 + self.holiday_embed(holiday))        # [B, h, d]
        x = x + t_embed[:, :, None, None, :]           # [B, h, 50, 49, d]

        x = x.reshape(B * h * N, C, d)                # [B*h*50, 49, d]
        x = self.encoder(x)                            # [B*h*50, 49, d]

        return x.mean(dim=1).reshape(B, h, N, d)       # [B, h, 50, d]
