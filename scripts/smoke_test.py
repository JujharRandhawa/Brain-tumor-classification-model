"""Quick smoke test for dataset + Grad-CAM pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import prepare_datasets
from src.data.preprocessing import load_image, preprocess_for_model
from src.models.efficientnet_classifier import build_model, compile_model, find_last_conv_layer
from src.models.gradcam import GradCAM


def main() -> None:
    config = load_config()
    splits, train_ds, val_ds, test_ds = prepare_datasets(config)
    print("Splits:", len(splits.train), len(splits.val), len(splits.test))
    print("Class weights:", splits.class_weights)

    model = build_model(config)
    compile_model(model, 1e-4)
    print("Grad-CAM layer:", find_last_conv_layer(model))

    sample_path = splits.test.iloc[0]["path"]
    size = tuple(config["data"]["image_size"])
    img = load_image(sample_path, size)
    prep = preprocess_for_model(img, size)
    gc = GradCAM(model)
    result = gc.explain(img, prep)
    print("Sample prediction:", result["label"], f"{result['confidence']:.2%}")
    print("Pipeline OK")


if __name__ == "__main__":
    main()
