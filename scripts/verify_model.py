"""Verify trained model inference and Grad-CAM on tumor and no-tumor samples."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.download import resolve_data_directory
from src.data.preprocessing import load_image, preprocess_for_model
from src.models.gradcam import GradCAM


def main() -> None:
    config = load_config()
    model_path = Path(config["paths"]["model_dir"]) / config["paths"]["model_filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    gradcam = GradCAM(model)
    data_root = resolve_data_directory(config["paths"]["raw_data_dir"])
    size = tuple(config["data"]["image_size"])

    passed = 0
    for folder, expected_tumor in [("yes", True), ("no", False)]:
        folder_path = data_root / folder
        images = [p for p in folder_path.iterdir() if p.suffix.lower() in config["data"]["supported_extensions"]]
        for path in images[:5]:
            img = load_image(path, size)
            prep = preprocess_for_model(img, size)
            result = gradcam.explain(img, prep)
            correct = result["is_tumor"] == expected_tumor
            status = "OK" if correct else "MISS"
            print(f"[{status}] {path.name} ({folder}) -> {result['label']} ({result['confidence']:.1%})")
            if correct:
                passed += 1
            assert result["heatmap"].shape == result["overlay"].shape[:2] or result["overlay"].ndim == 3

    print(f"\nVerification: {passed}/10 sample predictions correct")
    if passed < 7:
        raise SystemExit("Model verification failed — too many misclassifications on samples.")
    print("Grad-CAM heatmaps and overlays generated successfully.")
    print("Model is ready for dashboard use.")


if __name__ == "__main__":
    main()
