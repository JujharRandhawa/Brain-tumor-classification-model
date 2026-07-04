#!/usr/bin/env python
"""Evaluate a trained model on the test split."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import prepare_datasets
from src.utils.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    model_path = Path(config["paths"]["model_dir"]) / config["paths"]["model_filename"]

    if not model_path.exists():
        logger.error("Model not found at %s. Run training first.", model_path)
        sys.exit(1)

    model = tf.keras.models.load_model(str(model_path))
    _, _, _, test_ds = prepare_datasets(config)

    metrics = evaluate_model(
        model,
        test_ds,
        class_names=config["data"]["class_names"],
        output_dir=config["paths"]["output_dir"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
