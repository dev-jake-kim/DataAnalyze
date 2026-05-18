from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from loaders.dataset import collate_fn

log = logging.getLogger(__name__)

# Metrics saved to CSV (evaluater excluded)
_RECORD_METRICS = ['mae', 'mape', 'rmse']

# Eval keys to skip when collecting extra metrics automatically
_SKIP_EVAL_KEYS = {'loss', 'evaluater', 'runtime', 'samples_per_second', 'steps_per_second'}


def append_run_to_csv(csv_path: Path, run_id: str, data: dict) -> None:
    """
    run_id 행이 이미 있으면 해당 행에 data 컬럼을 병합(upsert),
    없으면 새 행으로 추가. 새 컬럼은 기존 행에 NaN으로 확장.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['run'])

    # 새 컬럼 확장
    for col in data:
        if col not in df.columns:
            df[col] = float('nan')

    mask = df['run'] == run_id
    if mask.any():
        for col, val in data.items():
            df.loc[mask, col] = val
    else:
        new_row = {'run': run_id, **data}
        for col in df.columns:
            new_row.setdefault(col, float('nan'))
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    log.info('Results CSV updated → %s', csv_path)


class TrainMetricsCallback(TrainerCallback):
    """
    At the end of training:
    1. Saves a 2×2 learning-curve PNG to output_dir.
    2. Appends best-epoch val metrics (mae, mape, rmse) to
       projects/{project_name}/results.csv, adding columns if new metrics appear.
    """

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        train_logs = [e for e in state.log_history if 'loss' in e and 'eval_loss' not in e]
        eval_logs  = [e for e in state.log_history if 'eval_loss' in e]

        if not eval_logs:
            log.warning('No eval logs found – skipping TrainMetricsCallback.')
            return

        self._plot_curves(train_logs, eval_logs, args.output_dir)
        self._save_csv(eval_logs, args)

    # ------------------------------------------------------------------
    def _plot_curves(self, train_logs: list, eval_logs: list, output_dir: str) -> None:
        train_epochs = [e['epoch'] for e in train_logs]
        train_losses = [e['loss']  for e in train_logs]

        eval_epochs = [e['epoch']          for e in eval_logs]
        eval_loss   = [e.get('eval_loss')  for e in eval_logs]
        eval_mae    = [e.get('eval_mae')   for e in eval_logs]
        eval_mape   = [e.get('eval_mape')  for e in eval_logs]
        eval_rmse   = [e.get('eval_rmse')  for e in eval_logs]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Learning Curves', fontsize=14, fontweight='bold')

        _plot_ax(axes[0, 0], 'Loss',      eval_epochs, eval_loss,
                 train_epochs, train_losses)
        _plot_ax(axes[0, 1], 'MAE',       eval_epochs, eval_mae)
        _plot_ax(axes[1, 0], 'MAPE (%)',  eval_epochs, eval_mape)
        _plot_ax(axes[1, 1], 'RMSE',      eval_epochs, eval_rmse)

        plt.tight_layout()
        out_path = Path(output_dir) / 'learning_curves.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info('Saved learning curves → %s', out_path)

    # ------------------------------------------------------------------
    def _save_csv(self, eval_logs: list, args: TrainingArguments) -> None:
        # Best epoch = lowest eval_evaluater (or last epoch)
        best = min(eval_logs, key=lambda e: e.get('eval_evaluater', float('inf')))

        val_data: dict = {}
        for metric in _RECORD_METRICS:
            key = f'eval_{metric}'
            if key in best:
                val_data[metric] = best[key]

        # Future-proof: pick up any extra eval metrics automatically
        for key, val in best.items():
            if key.startswith('eval_'):
                col = key[5:]
                if col not in _SKIP_EVAL_KEYS and col not in val_data:
                    val_data[col] = val

        run_id = '/'.join(Path(args.output_dir).parts[-2:])
        csv_path = Path(args.output_dir).parent.parent / 'results.csv'
        append_run_to_csv(csv_path, run_id, val_data)


# ------------------------------------------------------------------
class RagDbUpdateCallback(TrainerCallback):
    """
    Rebuilds the RAG retrieval database at the start of each epoch by
    re-running the current node_embedder over all training samples.
    """

    def __init__(
        self,
        dataset,
        train_indices: list,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.dataset = dataset
        self.train_indices = train_indices
        self.device = device
        self.batch_size = batch_size

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is None or not hasattr(model, 'update_db'):
            return
        loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        model.update_db(loader, self.device)
        log.info('RAG DB updated at epoch %d (%d entries)',
                 state.epoch or 0, len(self.train_indices))


# ------------------------------------------------------------------
def _plot_ax(
    ax: plt.Axes,
    title: str,
    val_x: list,
    val_y: list,
    train_x: list | None = None,
    train_y: list | None = None,
) -> None:
    if train_x is not None:
        ax.plot(train_x, train_y, label='Train', color='tab:blue', linewidth=1, alpha=0.7)
    ax.plot(val_x, val_y, label='Val', color='tab:orange',
            marker='o', markersize=4, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
