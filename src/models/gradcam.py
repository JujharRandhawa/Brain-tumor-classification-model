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

    def _build_grad_model(self) -> tf.keras.Model:
        conv_output = self._get_conv_output_tensor()
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_output, self.model.output],
        )
        return grad_model

    def _get_conv_output_tensor(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(self.last_conv_layer_name).output
                except ValueError:
                    continue
        return self.model.get_layer(self.last_conv_layer_name).output

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
