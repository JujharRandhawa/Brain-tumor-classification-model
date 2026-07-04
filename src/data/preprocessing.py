"""Image preprocessing utilities using OpenCV and TensorFlow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_image(path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load an MRI image as RGB uint8 array."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def preprocess_for_model(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize and apply EfficientNet preprocessing."""
    if image.shape[:2] != target_size[::-1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    return preprocess_input(batch)


def normalize_display(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 for visualization."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def apply_colormap_heatmap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Convert a 2D heatmap to a colorized RGB overlay base."""
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend Grad-CAM heatmap onto the original MRI using OpenCV."""
    image = normalize_display(image)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = apply_colormap_heatmap(heatmap_resized)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def build_augmentation_layer(config: dict) -> tf.keras.Sequential:
    """Create a Keras augmentation pipeline for training."""
    aug = config["augmentation"]
    size = tuple(config["data"]["image_size"])
    layers_list = [
        tf.keras.layers.RandomRotation(aug["rotation_range"] / 360.0),
        tf.keras.layers.RandomTranslation(
            aug["height_shift_range"],
            aug["width_shift_range"],
            fill_mode=aug["fill_mode"],
        ),
        tf.keras.layers.RandomZoom(aug["zoom_range"], fill_mode=aug["fill_mode"]),
    ]
    if aug["horizontal_flip"]:
        layers_list.append(tf.keras.layers.RandomFlip("horizontal"))
    layers_list.append(tf.keras.layers.Resizing(size[0], size[1]))
    return tf.keras.Sequential(layers_list, name="augmentation")
