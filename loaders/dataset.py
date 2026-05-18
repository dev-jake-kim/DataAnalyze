from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class DemandDataset(Dataset):
    def __init__(self, temporal_grid_path: str, graph_data_path: str, history_len: int = 24):
        self.grid = np.load(temporal_grid_path)  # [T, 96, 78] int32

        with open(graph_data_path) as f:
            data = json.load(f)

        self.node_demands = np.array(
            [step['demand'] for step in data['x']], dtype=np.float32
        )  # [T, 50]

        self.cell_rows = np.array(
            [[c['y'] for c in n['cells']] for n in data['nodes']], dtype=np.int64
        )  # [50, 49]
        self.cell_cols = np.array(
            [[c['x'] for c in n['cells']] for n in data['nodes']], dtype=np.int64
        )  # [50, 49]

        self.history_len = history_len
        self.total = len(data['x'])

    def __len__(self) -> int:
        return self.total - self.history_len

    def __getitem__(self, idx: int) -> dict:
        window = self.grid[idx: idx + self.history_len]                    # [h, 96, 78]
        cell_demands = window[:, self.cell_rows, self.cell_cols]           # [h, 50, 49]
        label = self.node_demands[idx + self.history_len]                  # [50]
        return {
            'cell_demands': torch.tensor(cell_demands, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        'cell_demands': torch.stack([b['cell_demands'] for b in batch]),  # [B, h, 50, 49]
        'labels': torch.stack([b['labels'] for b in batch]),              # [B, 50]
    }
