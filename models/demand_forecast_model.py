from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from models.node_embedder import NodeEmbedder
from models.demand_predictor import DemandPredictor


class DemandForecastConfig(PretrainedConfig):
    model_type = "demand_forecast"

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers_1: int = 2,
        num_encoder_layers_2: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        gamma: float = 0.01,
        data_path: str = "data/processed/graph_data.json",
        norm_first: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers_1 = num_encoder_layers_1
        self.num_encoder_layers_2 = num_encoder_layers_2
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.gamma = gamma
        self.data_path = data_path
        self.norm_first = norm_first


@dataclass
class DemandForecastOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class DemandForecastModel(PreTrainedModel):
    """
    Full 2-step demand forecasting model compatible with HuggingFace Trainer.

    Step 1 (NodeEmbedder): cell-level grid demand → per-node embeddings [B, h, 50, d]
    Step 2 (DemandPredictor): node embeddings + node-ID → predicted demand [B, 50]

    Loss: MAE + gamma * mean((y + pred)^2 / (y + 1))

    Usage:
        model.save_pretrained("path/to/save")
        model = DemandForecastModel.from_pretrained("path/to/save")
    """

    config_class = DemandForecastConfig
    # No weight tying in this model (required for transformers 5.x compatibility)
    all_tied_weights_keys: dict = {}

    def __init__(self, config: DemandForecastConfig):
        super().__init__(config)

        with open(config.data_path) as f:
            graph_data = json.load(f)

        nodes = graph_data['nodes']
        cell_positions = torch.tensor(
            [[[c['x'], c['y']] for c in n['cells']] for n in nodes],
            dtype=torch.long,
        )  # [50, 49, 2]

        self.node_embedder = NodeEmbedder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers_1,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            cell_positions=cell_positions,
            norm_first=config.norm_first,
        )
        self.demand_predictor = DemandPredictor(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers_2,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            norm_first=config.norm_first,
        )

    def forward(
        self,
        cell_demands: torch.Tensor,                 # [B, h, 50, 49]  long
        labels: Optional[torch.Tensor] = None,      # [B, 50]
    ) -> DemandForecastOutput:
        node_embed = self.node_embedder(cell_demands)   # [B, h, 50, d]
        pred = self.demand_predictor(node_embed)         # [B, 50]

        loss = None
        if labels is not None:
            mae = torch.mean(torch.abs(labels - pred))
            penalty = torch.mean((labels + pred) ** 2 / (labels + 1.0))
            loss = mae + self.config.gamma * penalty

        return DemandForecastOutput(loss=loss, logits=pred)
