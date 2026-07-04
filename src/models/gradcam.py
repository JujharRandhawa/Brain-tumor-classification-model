"""Grad-CAM visual attribution for model explainability."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.data.preprocessing import overlay_heatmap
from src.models.efficientnet_classifier import find_last_conv_layer


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Produces spatial heatmaps highlighting regions that influenced
    the tumor / no-tumor classification decision.
    """

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str | None = None):
        self.model = model
        self.last_conv_layer_name = find_last_conv_layer(model, last_conv_layer_name)
        self.grad_model = self._build_grad_model()

    def _get_backbone(self) -> tf.keras.Model | None:
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        return None

    def _apply_classification_head(self, conv_output: tf.Tensor) -> tf.Tensor:
        """Reattach trained head layers so gradients flow to conv activations."""
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_output)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                continue
            if layer.__class__.__name__ == "InputLayer":
                continue
            x = layer(x)
        return x

    def _build_grad_model(self) -> tf.keras.Model:
        """
        Build a sub-model with a single connected graph from input -> conv -> head.

        EfficientNet is nested with its own input node; we route the outer MRI input
        through a conv extractor, then rebuild the classification head on top so
        Grad-CAM gradients are non-zero in Keras 3.
        """
        model_input = self.model.input
        backbone = self._get_backbone()

        if backbone is not None:
            conv_extractor = tf.keras.Model(
                inputs=backbone.input,
                outputs=backbone.get_layer(self.last_conv_layer_name).output,
                name="conv_extractor",
            )
            conv_output = conv_extractor(model_input)
            prediction = self._apply_classification_head(conv_output)
        else:
            conv_output = self.model.get_layer(self.last_conv_layer_name).output
            prediction = self.model.output

        return tf.keras.Model(
            inputs=model_input,
            outputs=[conv_output, prediction],
            name="gradcam_model",
        )

    def compute_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Generate normalized heatmap and predicted tumor probability.

        img_array: batch of preprocessed images (1, H, W, 3).
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array, training=False)
            if pred_index is None:
                loss = predictions[:, 0]
            else:
                loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError(
                f"Could not compute gradients for layer '{self.last_conv_layer_name}'. "
                "Verify the layer is part of the model's forward graph."
            )

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap.numpy(), 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        probability = float(predictions.numpy()[0, 0])
        return heatmap.astype(np.float32), probability

    def explain(
        self,
        image_rgb: np.ndarray,
        preprocessed: np.ndarray,
        alpha: float = 0.45,
    ) -> dict:
        """Return prediction, heatmap, and OpenCV overlay."""
        heatmap, probability = self.compute_heatmap(preprocessed)
        overlay = overlay_heatmap(image_rgb, heatmap, alpha=alpha)
        label = "Tumor Detected" if probability >= 0.5 else "No Tumor"
        confidence = probability if probability >= 0.5 else 1.0 - probability

        return {
            "label": label,
            "probability": probability,
            "confidence": confidence,
            "heatmap": heatmap,
            "overlay": overlay,
            "is_tumor": probability >= 0.5,
        }
