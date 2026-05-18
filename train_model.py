import torch
from torch.utils.data import Subset

import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from omegaconf import OmegaConf

from models.demand_forecast_model import (
    DemandForecastModel, DemandForecastConfig,
    RAGDemandForecastModel, RAGDemandForecastConfig,
)
from loaders.dataset import DemandDataset, collate_fn
from runners.test_loop import test_loop
from runners.training_callbacks import TrainMetricsCallback, RagDbUpdateCallback, append_run_to_csv


log = logging.getLogger(__name__)
results = []

BEST_MODEL_METRIC_ALIASES = {
    'rsme': 'rmse',
}

BEST_MODEL_METRIC_DIRECTIONS = {
    'loss': False,
    'mae': False,
    'mape': False,
    'rmse': False,
    'evaluater': False,
}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    import numpy as np

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    mae = np.mean(np.abs(predictions - labels))
    mape = np.mean(np.abs(predictions - labels) / (labels + 1.0)) * 100.0
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    evaluater = mape + 30 * mae

    return {
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'evaluater': float(evaluater)
    }


def resolve_best_model_metric(train_config):
    metric_name = str(train_config.metric_for_best_model).strip().lower()
    metric_name = BEST_MODEL_METRIC_ALIASES.get(metric_name, metric_name)

    if metric_name.startswith('eval_'):
        metric_key = metric_name[5:]
    else:
        metric_key = metric_name

    if metric_key not in BEST_MODEL_METRIC_DIRECTIONS:
        valid_metrics = ', '.join(sorted(BEST_MODEL_METRIC_DIRECTIONS))
        raise ValueError(
            f"Unsupported train.metric_for_best_model='{train_config.metric_for_best_model}'. "
            f"Use one of: {valid_metrics}"
        )

    train_config.metric_for_best_model = metric_key
    train_config.greater_is_better = BEST_MODEL_METRIC_DIRECTIONS[metric_key]
    return metric_key


def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_training_args(train_cfg: dict, output_dir: str) -> TrainingArguments:
    args = TrainingArguments(**train_cfg, output_dir=output_dir, report_to=[], log_level='info')
    resolve_best_model_metric(args)
    log.info('Best model metric: %s (greater_is_better=%s)',
             args.metric_for_best_model, args.greater_is_better)
    return args


@hydra.main(config_path="configs", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    log.info('Using device: %s', device)

    output_dir = HydraConfig.get().runtime.output_dir

    dataset = DemandDataset(
        temporal_grid_path=config.dataset.temporal_grid_path,
        graph_data_path=config.dataset.data_path,
        history_len=config.dataset.history_len,
    )

    total = dataset.total
    h = config.dataset.history_len
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train_indices = list(range(0, train_end - h))
    valid_indices = list(range(train_end, val_end - h))
    test_indices  = list(range(val_end, total - h))

    # ------------------------------------------------------------------ Stage 1
    log.info('=== Stage 1: baseline training ===')
    model_config = DemandForecastConfig(**OmegaConf.to_container(config.model))
    model = DemandForecastModel(model_config).to(device)

    stage1_args = _make_training_args(
        OmegaConf.to_container(config.stage1.train),
        str(Path(output_dir) / 'stage1'),
    )

    trainer1 = Trainer(
        model=model,
        args=stage1_args,
        train_dataset=Subset(dataset, train_indices),
        eval_dataset=Subset(dataset, valid_indices),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(**config.stage1.callbacks.early_stopping),
            TrainMetricsCallback(),
        ],
    )
    trainer1.train()

    stage1_dir = Path(output_dir) / 'stage1'
    trainer1.save_model(str(stage1_dir))
    log.info('Stage 1 model saved → %s', stage1_dir)

    # ------------------------------------------------------------------ Stage 2
    log.info('=== Stage 2: RAG training ===')
    rag_config = RAGDemandForecastConfig(**OmegaConf.to_container(config.model))
    rag_model = RAGDemandForecastModel.init_from_stage1(str(stage1_dir), rag_config).to(device)

    stage2_args = _make_training_args(
        OmegaConf.to_container(config.stage2.train),
        str(Path(output_dir) / 'stage2'),
    )

    stage2_start = int(config.stage2.get('start_idx', 0))
    stage2_train_indices = list(range(stage2_start, train_end - h))
    log.info('Stage 2 train indices: %d → %d (%d samples)',
             stage2_start, train_end - h, len(stage2_train_indices))

    trainer2 = Trainer(
        model=rag_model,
        args=stage2_args,
        train_dataset=Subset(dataset, stage2_train_indices),
        eval_dataset=Subset(dataset, valid_indices),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            RagDbUpdateCallback(dataset, train_indices, device,
                                batch_size=config.test.batch_size),  # DB는 전체 train으로 구성
            EarlyStoppingCallback(**config.stage2.callbacks.early_stopping),
            TrainMetricsCallback(),
        ],
    )
    trainer2.train()

    # ------------------------------------------------------------------ Test
    test_results = test_loop(
        rag_model, Subset(dataset, test_indices), output_dir, device,
        **config.test,
    )
    results.append(test_results)

    run_id = '/'.join(Path(output_dir).parts[-2:])
    csv_path = Path(output_dir).parent.parent / 'results.csv'
    test_data = {f'test_{k}': v for k, v in test_results.items()
                 if k in ('mae', 'mape', 'rmse')}
    append_run_to_csv(csv_path, run_id, test_data)


if __name__ == "__main__":
    run()
    log.info(results)
