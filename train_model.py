import torch
import torch.nn.functional as F
from torch.utils.data import Subset

import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from omegaconf import OmegaConf

from models.demand_forecast_model import DemandForecastModel, DemandForecastConfig
from loaders.dataset import DemandDataset, collate_fn
from runners.test_loop import test_loop
from runners.training_callbacks import TrainMetricsCallback


log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

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
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    # 결정론적 연산을 위한 설정 (필요 시)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

@hydra.main(config_path="configs", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    output_dir = HydraConfig.get().runtime.output_dir

    dataset = DemandDataset(
        temporal_grid_path=config.dataset.temporal_grid_path,
        graph_data_path=config.dataset.data_path,
        history_len=config.dataset.history_len,
    )

    model_config = DemandForecastConfig(**OmegaConf.to_container(config.model))
    model = DemandForecastModel(model_config).to(device)

    total = dataset.total
    h = config.dataset.history_len
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train_indices = list(range(0, train_end - h))
    valid_indices = list(range(train_end, val_end - h))
    test_indices = list(range(val_end, total - h))

    args = TrainingArguments(
        **config['train'],
        output_dir=output_dir,
        report_to=[],
        log_level='info'
    )

    best_model_metric = resolve_best_model_metric(args)
    log.info(
        "Best model selection metric: %s (greater_is_better=%s)",
        best_model_metric,
        args.greater_is_better
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Subset(dataset, train_indices),
        eval_dataset=Subset(dataset, valid_indices),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(**config.callbacks.early_stopping),
            TrainMetricsCallback(),
        ]
    )
    trainer.train()
    
    test_results = test_loop(model, Subset(dataset, test_indices), output_dir, device, **config.test)
    results.append(test_results)

if __name__ == "__main__":
    run()
    log.info(results)
