from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loaders.dataset import collate_fn

log = logging.getLogger(__name__)


def test_loop(
    model: torch.nn.Module,
    test_dataset: Dataset,
    output_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)
            pred = model(
                cell_demands=batch['cell_demands'].to(device),
                day=batch['day'].to(device),
                time_idx=batch['time_idx'].to(device),
                holiday=batch['holiday'].to(device),
            ).logits
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)   # [N_test, 50]
    labels_arr = np.concatenate(all_labels, axis=0)   # [N_test, 50]

    mae = float(np.mean(np.abs(predictions - labels_arr)))
    mape = float(np.mean(np.abs(predictions - labels_arr) / (labels_arr + 1.0)) * 100.0)
    rmse = float(np.sqrt(np.mean((predictions - labels_arr) ** 2)))
    evaluater = mape + 50 * rmse

    results = {'mae': mae, 'mape': mape, 'rmse': rmse, 'evaluater': evaluater}

    out = Path(output_dir)
    np.save(out / 'test_predictions.npy', predictions)
    with open(out / 'test_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    log.info('Test metrics: %s', results)
    return results
