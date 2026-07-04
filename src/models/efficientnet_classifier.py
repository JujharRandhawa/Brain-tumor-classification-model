"""EfficientNetB3-based brain tumor classifier."""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

logger = logging.getLogger(__name__)


def build_model(config: dict, trainable_base: bool = False) -> tf.keras.Model:
    """
    Build EfficientNetB3 transfer-learning model for binary tumor detection.

    Output: sigmoid probability (1 = tumor, 0 = no tumor).
    """
    image_size = tuple(config["data"]["image_size"])
    model_cfg = config["model"]

    inputs = layers.Input(shape=(*image_size, config["data"]["channels"]), name="mri_input")

    base = EfficientNetB3(
        include_top=False,
        weights=model_cfg["weights"],
        input_tensor=inputs,
        pooling="avg",
    )
    base.trainable = trainable_base

    x = layers.Dropout(model_cfg["dropout_rate"])(base.output)
    x = layers.Dense(model_cfg["dense_units"], activation="relu", name="dense_head")(x)
    x = layers.Dropout(model_cfg["dropout_rate"])(x)
    outputs = layers.Dense(1, activation="sigmoid", name="tumor_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB3_BrainTumor")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile with Adam and binary cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_base(model: tf.keras.Model, fine_tune_at: int) -> tf.keras.Model:
    """Unfreeze top layers of EfficientNet for fine-tuning."""
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base = layer
            break
    if base is None:
        logger.warning("EfficientNet base not found; skipping fine-tune unfreeze.")
        return model

    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(int(layer.trainable) for layer in base.layers)
    logger.info("Fine-tuning enabled — %d/%d base layers trainable.", trainable, len(base.layers))
    return model


def _layer_output_rank(layer) -> int | None:
    """Return tensor rank for a layer output, if available."""
    shape = getattr(layer, "output_shape", None)
    if shape is not None:
        return len(shape)
    output = getattr(layer, "output", None)
    if output is not None and hasattr(output, "shape"):
        return len(output.shape)
    return None


def find_last_conv_layer(model: tf.keras.Model, layer_name: str | None = None) -> str:
    """Resolve the last 4D convolution layer inside EfficientNet for Grad-CAM."""
    if layer_name:
        return layer_name

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if _layer_output_rank(inner) == 4:
                    return inner.name

    for layer in reversed(model.layers):
        if _layer_output_rank(layer) == 4:
            return layer.name

    raise ValueError("Could not locate a convolutional layer for Grad-CAM.")
