from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


class RAGDemandForecastConfig(PretrainedConfig):
    model_type = "rag_demand_forecast"

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


def _load_cell_positions(data_path: str) -> torch.Tensor:
    with open(data_path) as f:
        graph_data = json.load(f)
    nodes = graph_data['nodes']
    return torch.tensor(
        [[[c['x'], c['y']] for c in n['cells']] for n in nodes],
        dtype=torch.long,
    )  # [50, 49, 2]


def _build_node_embedder(config, cell_positions: torch.Tensor) -> NodeEmbedder:
    return NodeEmbedder(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_encoder_layers_1,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        cell_positions=cell_positions,
        norm_first=config.norm_first,
    )


def _build_demand_predictor(config) -> DemandPredictor:
    return DemandPredictor(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_encoder_layers_2,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        norm_first=config.norm_first,
    )


def _compute_loss(labels: torch.Tensor, pred: torch.Tensor, gamma: float) -> torch.Tensor:
    mae = torch.mean(torch.abs(labels - pred))
    penalty = torch.mean((labels + pred) ** 2 / (labels + 1.0))
    return mae + gamma * penalty


class DemandForecastModel(PreTrainedModel):
    """
    Stage 1: baseline 2-step demand forecasting model.
    NodeEmbedder includes CLS encoder; seq_vec is computed but unused in Stage 1.
    """

    config_class = DemandForecastConfig
    all_tied_weights_keys: dict = {}

    def __init__(self, config: DemandForecastConfig):
        super().__init__(config)
        cell_positions = _load_cell_positions(config.data_path)
        self.node_embedder = _build_node_embedder(config, cell_positions)
        self.demand_predictor = _build_demand_predictor(config)

    def forward(
        self,
        cell_demands: torch.Tensor,
        day:          torch.Tensor = None,
        time_idx:     torch.Tensor = None,
        holiday:      torch.Tensor = None,
        labels:       Optional[torch.Tensor] = None,
        idx:          Optional[torch.Tensor] = None,  # unused in Stage 1
        **kwargs,
    ) -> DemandForecastOutput:
        node_embed, _ = self.node_embedder(cell_demands, day, time_idx, holiday)
        pred, _ = self.demand_predictor(node_embed)

        loss = None
        if labels is not None:
            loss = _compute_loss(labels, pred, self.config.gamma)

        return DemandForecastOutput(loss=loss, logits=pred)


class RAGDemandForecastModel(PreTrainedModel):
    """
    Stage 2: RAG-augmented demand forecasting model.

    Retrieves similar past sequences via soft attention
    (Q=current seq_vec, K=DB keys, V=DB node embeddings),
    then concatenates the result with DemandPredictor's internal embedding
    before the final prediction head.

    DB is populated externally via update_db() before each training epoch.
    """

    config_class = RAGDemandForecastConfig
    all_tied_weights_keys: dict = {}

    def __init__(self, config: RAGDemandForecastConfig):
        super().__init__(config)
        cell_positions = _load_cell_positions(config.data_path)
        self.node_embedder = _build_node_embedder(config, cell_positions)
        self.demand_predictor = _build_demand_predictor(config)
        self.rag_head = nn.Linear(2 * config.d_model, 1)

        # Populated by update_db(); None until first call
        self.register_buffer('db_keys',   None)  # [T, d]
        self.register_buffer('db_values', None)  # [T, 50, d]

    @classmethod
    def init_from_stage1(
        cls,
        stage1_dir: str,
        config: RAGDemandForecastConfig,
    ) -> 'RAGDemandForecastModel':
        """Create Stage 2 model by copying Stage 1 node_embedder + demand_predictor weights."""
        stage1 = DemandForecastModel.from_pretrained(stage1_dir)
        model = cls(config)
        model.node_embedder.load_state_dict(stage1.node_embedder.state_dict())
        model.demand_predictor.load_state_dict(stage1.demand_predictor.state_dict())
        return model

    @torch.no_grad()
    def update_db(self, loader: DataLoader, device: torch.device) -> None:
        """Recompute retrieval DB using current node_embedder (called each epoch)."""
        keys_list, values_list = [], []
        self.node_embedder.eval()
        for batch in loader:
            cd  = batch['cell_demands'].to(device)
            day = batch['day'].to(device)
            ti  = batch['time_idx'].to(device)
            hol = batch['holiday'].to(device)
            node_embed, seq_vec = self.node_embedder(cd, day, ti, hol)
            keys_list.append(seq_vec.cpu())
            values_list.append(node_embed.mean(dim=1).cpu())  # mean h → [B, 50, d]
        self.node_embedder.train()
        self.db_keys   = torch.cat(keys_list,   dim=0).to(device)  # [T, d]
        self.db_values = torch.cat(values_list, dim=0).to(device)  # [T, 50, d]

    def forward(
        self,
        cell_demands: torch.Tensor,
        day:          torch.Tensor = None,
        time_idx:     torch.Tensor = None,
        holiday:      torch.Tensor = None,
        idx:          Optional[torch.Tensor] = None,  # [B] dataset indices
        labels:       Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DemandForecastOutput:
        node_embed, seq_vec = self.node_embedder(cell_demands, day, time_idx, holiday)
        pred, internal = self.demand_predictor(node_embed)  # [B,50], [B,50,d]

        if self.db_keys is not None and idx is not None:
            d = seq_vec.shape[-1]
            T = self.db_keys.shape[0]

            scores = seq_vec @ self.db_keys.T / math.sqrt(d)   # [B, T]

            # Temporal mask: exclude current and future sequences
            mask = (torch.arange(T, device=scores.device).unsqueeze(0)
                    >= idx.unsqueeze(1))                        # [B, T]
            scores = scores.masked_fill(mask, float('-inf'))

            weights = F.softmax(scores, dim=-1)                 # [B, T]
            # idx=0 처럼 과거 시퀀스가 없으면 전체 -inf → NaN; 0으로 대체해 RAG 기여 제거
            weights = torch.nan_to_num(weights, nan=0.0)

            rag_embed = torch.einsum('bt,tnd->bnd', weights, self.db_values)  # [B,50,d]

            combined = torch.cat([internal, rag_embed], dim=-1) # [B, 50, 2d]
            pred = F.relu(self.rag_head(combined).squeeze(-1))  # [B, 50]

        loss = None
        if labels is not None:
            loss = _compute_loss(labels, pred, self.config.gamma)

        return DemandForecastOutput(loss=loss, logits=pred)
