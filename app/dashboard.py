"""Streamlit dashboard for explainable brain tumor classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.preprocessing import apply_colormap_heatmap, load_image, preprocess_for_model
from src.models.gradcam import GradCAM

st.set_page_config(
    page_title="Brain Tumor XAI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5a6a7a;
        margin-bottom: 1.5rem;
    }
    .tumor-alert {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .no-tumor-alert {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2rem;
        padding: 1rem;
        background: #f1f5f9;
        border-radius: 8px;
    }
</style>
"""


@st.cache_resource
def load_model_and_gradcam(model_path: str):
    """Load trained model and Grad-CAM wrapper once per session."""
    model = tf.keras.models.load_model(model_path)
    gradcam = GradCAM(model)
    return model, gradcam


def load_metrics(output_dir: str) -> dict | None:
    metrics_path = Path(output_dir) / "evaluation_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def get_sample_image(config: dict) -> np.ndarray | None:
    try:
        from src.data.download import resolve_data_directory

        data_root = resolve_data_directory(config["paths"]["raw_data_dir"])
        sample_dirs = [data_root / "yes", data_root / "no"]
        existing = [d for d in sample_dirs if d.exists()]
        if not existing:
            return None

        import random

        folder = random.choice(existing)
        images = [
            p for p in folder.iterdir()
            if p.suffix.lower() in config["data"]["supported_extensions"]
        ]
        if not images:
            return None
        size = tuple(config["data"]["image_size"])
        return load_image(random.choice(images), size)
    except FileNotFoundError:
        return None


def render_sidebar(config: dict) -> tuple[str, float]:
    st.sidebar.title("Controls")
    st.sidebar.markdown("---")

    alpha = st.sidebar.slider(
        "Heatmap overlay intensity",
        min_value=0.1,
        max_value=0.8,
        value=float(config["model"]["gradcam_alpha"]),
        step=0.05,
    )
    mode = st.sidebar.radio(
        "Input source",
        ["Upload MRI scan", "Use sample from dataset"],
        key="input_mode",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("TensorFlow · EfficientNetB3 · Grad-CAM · OpenCV")
    return mode, alpha


def render_results(image_rgb: np.ndarray, result: dict, alpha: float) -> None:
    alert_class = "tumor-alert" if result["is_tumor"] else "no-tumor-alert"
    st.markdown(
        f'<div class="{alert_class}"><strong>{result["label"]}</strong> — '
        f"Confidence: {result['confidence']:.1%} | "
        f"Tumor probability: {result['probability']:.4f}</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original MRI")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(apply_colormap_heatmap(result["heatmap"]), use_container_width=True)
    with col3:
        st.subheader("Attribution Overlay")
        st.image(result["overlay"], use_container_width=True)

    st.markdown("#### Interpretation")
    st.markdown(
        "The **Grad-CAM heatmap** highlights brain regions that most influenced the model's "
        "classification decision. Warmer colors (red/yellow) indicate higher attribution "
        "weight — areas the network associated with tumor presence or absence."
    )


def main() -> None:
    config = load_config()
    dash_cfg = config["dashboard"]
    model_path = str(Path(config["paths"]["model_dir"]) / config["paths"]["model_filename"])

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="main-header">{dash_cfg["title"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{dash_cfg["subtitle"]}</p>', unsafe_allow_html=True)

    if not Path(model_path).exists():
        st.error(
            "Trained model not found. Please run:\n\n"
            "```\npython scripts/download_data.py\npython scripts/train.py\n```"
        )
        st.stop()

    try:
        model, gradcam = load_model_and_gradcam(model_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    mode, alpha = render_sidebar(config)

    metrics = load_metrics(config["paths"]["output_dir"])
    if metrics:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        col_m2.metric("Test AUC", f"{metrics['auc']:.3f}")
        col_m3.metric("Model", "EfficientNetB3")

    st.markdown("---")

    image_rgb = None
    size = tuple(config["data"]["image_size"])

    if mode == "Upload MRI scan":
        st.session_state.pop("sample_image", None)
        uploaded = st.file_uploader(
            "Upload a brain MRI image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )
        if uploaded:
            pil_image = Image.open(uploaded).convert("RGB")
            image_rgb = np.array(pil_image.resize(size))
    else:
        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            if st.button("Load random sample", type="primary", use_container_width=True):
                st.session_state["sample_image"] = get_sample_image(config)
        with col_hint:
            st.caption("Loads a random MRI from the yes/no training dataset.")
        image_rgb = st.session_state.get("sample_image")

    if image_rgb is None:
        st.info("Upload an MRI scan or load a sample image to generate Grad-CAM explanations.")
    else:
        try:
            preprocessed = preprocess_for_model(image_rgb, size)
            result = gradcam.explain(image_rgb, preprocessed, alpha=alpha)
            render_results(image_rgb, result, alpha)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.exception(exc)

    st.markdown(
        '<div class="disclaimer">'
        "<strong>Research &amp; Educational Use Only.</strong> This system is not a medical "
        "device and must not be used for clinical diagnosis. Future work includes integration "
        "with Clinical Decision Support Systems (CDSS) for radiologist-in-the-loop workflows."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
