#!/usr/bin/env python
"""Train EfficientNetB3 brain tumor classifier."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import prepare_datasets
from src.training.trainer import train_model
from src.utils.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    logger.info("Preparing datasets...")
    splits, train_ds, val_ds, test_ds = prepare_datasets(config)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    model, history = train_model(
        config,
        train_ds,
        val_ds,
        class_weights=splits.class_weights or None,
    )

    logger.info("Evaluating on held-out test set...")
    evaluate_model(
        model,
        test_ds,
        class_names=config["data"]["class_names"],
        output_dir=config["paths"]["output_dir"],
    )

    logger.info("Training complete. Launch dashboard: python scripts/run_dashboard.py")


if __name__ == "__main__":
    main()
