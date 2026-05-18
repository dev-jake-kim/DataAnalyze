from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

log = logging.getLogger(__name__)

# Metrics saved to CSV (evaluater excluded)
_RECORD_METRICS = ['mae', 'mape', 'rmse']

# Eval keys to skip when collecting extra metrics automatically
_SKIP_EVAL_KEYS = {'loss', 'evaluater', 'runtime', 'samples_per_second', 'steps_per_second'}


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

        run_id = '/'.join(Path(args.output_dir).parts[-2:])
        new_row: dict = {'run': run_id}

        # Fixed metrics (mae, mape, rmse)
        for metric in _RECORD_METRICS:
            key = f'eval_{metric}'
            if key in best:
                new_row[metric] = best[key]

        # Future-proof: pick up any extra eval metrics automatically
        for key, val in best.items():
            if key.startswith('eval_'):
                col = key[5:]
                if col not in _SKIP_EVAL_KEYS and col not in new_row:
                    new_row[col] = val

        project_dir = Path(args.output_dir).parent.parent
        csv_path = project_dir / 'results.csv'

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for col in new_row:
                if col not in df.columns:
                    df[col] = float('nan')
        else:
            df = pd.DataFrame(columns=list(new_row.keys()))

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        log.info('Saved results → %s', csv_path)


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
