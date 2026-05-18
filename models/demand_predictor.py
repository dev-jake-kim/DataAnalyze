from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.node_embedder import sinusoidal_1d


def temporal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard 1D sinusoidal PE. Returns [seq_len, d_model]."""
    positions = torch.arange(seq_len, device=device)
    return sinusoidal_1d(positions, d_model)  # [seq_len, d_model]


class DemandPredictor(nn.Module):
    """
    Step 2: node embedding history → next-timestep demand prediction.

    Combines time-varying node embeddings (from NodeEmbedder) with
    static node-ID embeddings, then applies a Transformer Encoder
    over the h*50 token sequence to predict demand for all 50 nodes.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        n_nodes: int = 50,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes

        self.node_id_embed = nn.Embedding(n_nodes, d_model)
        self.register_buffer('node_ids', torch.arange(n_nodes))  # [50]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, node_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_embed: [B, h, 50, d_model]  (from NodeEmbedder)
        returns:    ([B, 50] predictions, [B, 50, d_model] internal embedding)
        """
        B, h, N, d = node_embed.shape

        node_id_e = self.node_id_embed(self.node_ids)          # [50, d]
        tokens = node_embed + node_id_e[None, None]             # [B, h, 50, d]

        t_pe = temporal_pe(h, d, node_embed.device)             # [h, d]
        tokens = tokens + t_pe[None, :, None, :]                # [B, h, 50, d]

        tokens = tokens.reshape(B, h * N, d)                    # [B, h*50, d]
        out = self.encoder(tokens)                              # [B, h*50, d]

        internal = out.reshape(B, h, N, d).mean(dim=1)         # [B, 50, d]
        pred = F.relu(self.head(internal).squeeze(-1))          # [B, 50]
        return pred, internal
