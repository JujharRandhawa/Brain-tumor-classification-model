"""Verify Grad-CAM works with saved model (dashboard path)."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.preprocessing import load_image, preprocess_for_model
from src.models.gradcam import GradCAM


def main() -> None:
    config = load_config()
    model_path = Path(config["paths"]["model_dir"]) / config["paths"]["model_filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    gradcam = GradCAM(model)
    print("Loaded model OK, Grad-CAM layer:", gradcam.last_conv_layer_name)

    from src.data.dataset import prepare_datasets

    splits, _, _, _ = prepare_datasets(config)
    sample_path = splits.test.iloc[0]["path"]
    size = tuple(config["data"]["image_size"])
    img = load_image(sample_path, size)
    prep = preprocess_for_model(img, size)
    result = gradcam.explain(img, prep)
    print("Prediction:", result["label"], f"{result['confidence']:.2%}")
    print("Dashboard path OK")


if __name__ == "__main__":
    main()
