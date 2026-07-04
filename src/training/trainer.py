"""Training orchestration for EfficientNetB3 brain tumor classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import tensorflow as tf

from src.models.efficientnet_classifier import build_model, compile_model, unfreeze_base

logger = logging.getLogger(__name__)


def _build_callbacks(config: dict, model_path: Path) -> list[tf.keras.callbacks.Callback]:
    train_cfg = config["training"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
            min_lr=train_cfg["min_learning_rate"],
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def train_model(
    config: dict,
    train_ds,
    val_ds,
    class_weights: dict[int, float] | None = None,
) -> tuple[tf.keras.Model, dict]:
    """
    Two-phase training: frozen backbone, then fine-tuned top layers.
    """
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["paths"]["model_filename"]

    train_cfg = config["training"]
    history_combined: dict = {}

    # Phase 1 — train classification head
    logger.info("Phase 1: training classification head (frozen EfficientNet backbone)")
    model = build_model(config, trainable_base=False)
    compile_model(model, train_cfg["learning_rate"])

    fit_kwargs = {}
    if class_weights:
        fit_kwargs["class_weight"] = class_weights

    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(10, train_cfg["epochs"]),
        callbacks=_build_callbacks(config, model_path),
        **fit_kwargs,
    )
    history_combined.update(history_1.history)

    # Phase 2 — fine-tune
    logger.info("Phase 2: fine-tuning EfficientNet layers")
    model = tf.keras.models.load_model(str(model_path))
    model = unfreeze_base(model, train_cfg["fine_tune_at"])
    compile_model(model, train_cfg["learning_rate"] * 0.1)

    remaining_epochs = max(train_cfg["epochs"] - len(history_1.epoch), 1)
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=remaining_epochs,
        initial_epoch=len(history_1.epoch),
        callbacks=_build_callbacks(config, model_path),
        **fit_kwargs,
    )

    for key, values in history_2.history.items():
        history_combined.setdefault(key, [])
        history_combined[key].extend(values)

    model = tf.keras.models.load_model(str(model_path))

    history_path = Path(config["paths"]["output_dir"]) / config["paths"]["history_filename"]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump({k: [float(v) for v in vals] for k, vals in history_combined.items()}, handle, indent=2)

    logger.info("Best model saved to %s", model_path)
    return model, history_combined
