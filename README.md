# Explainable Brain Tumor Characterization

**EfficientNetB3 + Grad-CAM for visual attribution in brain MRI tumor classification**

A production-ready research pipeline that classifies brain MRI scans as tumorous or non-tumorous using transfer learning, and provides **visual explanations** of model decisions through Gradient-weighted Class Activation Mapping (Grad-CAM).

> **Innovation scope:** Explainable AI (XAI) in medical imaging — bridging deep learning predictions with interpretable heatmaps for clinician trust and verification.

---

## Features

| Component | Description |
|-----------|-------------|
| **EfficientNetB3 Classifier** | ImageNet-pretrained backbone with fine-tuning for binary tumor detection |
| **Grad-CAM Attribution** | Spatial heatmaps highlighting regions that drove the classification |
| **OpenCV Visualization** | Heatmap colormap rendering and MRI overlay blending |
| **Interactive Dashboard** | Streamlit UI for upload, inference, and side-by-side XAI views |
| **Evaluation Suite** | Confusion matrix, ROC curve, precision/recall/AUC metrics |

---

## Project Structure

```
├── app/
│   └── dashboard.py          # Streamlit heatmap visualization dashboard
├── config/
│   └── config.yaml           # Hyperparameters and paths
├── scripts/
│   ├── download_data.py      # Kaggle dataset download
│   ├── train.py              # Full training pipeline
│   ├── evaluate.py           # Test-set evaluation
│   ├── run_dashboard.py      # Launch dashboard
│   ├── verify_dashboard.py   # Verify saved model + Grad-CAM
│   └── verify_model.py       # Verify sample predictions
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── download.py       # kagglehub integration
│   │   ├── dataset.py        # tf.data pipelines
│   │   └── preprocessing.py  # OpenCV + EfficientNet preprocessing
│   ├── models/
│   │   ├── efficientnet_classifier.py
│   │   └── gradcam.py        # Grad-CAM implementation
│   ├── training/
│   │   └── trainer.py        # Two-phase training
│   └── utils/
│       └── metrics.py        # Evaluation plots and reports
├── data/                     # Downloaded MRI images (gitignored)
├── models/                   # Saved `.keras` checkpoints
├── outputs/                  # Metrics, plots, training logs
└── requirements.txt
```

---

## Dataset

**Source:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (Kaggle)

| Class | Folder | Images |
|-------|--------|--------|
| Tumor | `yes/` | 155 |
| No Tumor | `no/` | 98 |

Download via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
```

---

## Pre-trained Model Weights

The trained model file (`models/efficientnetb3_brain_tumor.keras`) is **not included in this repository**. Model checkpoints, raw MRI images, and training outputs are excluded via `.gitignore` to keep the repo lightweight and avoid committing large binary files.

**To run the dashboard or any inference script, train the model locally first:**

```bash
python scripts/download_data.py
python scripts/train.py
```

Training takes roughly **30–90 minutes on CPU** (Ryzen 7 class hardware). When it finishes, weights are saved to `models/efficientnetb3_brain_tumor.keras` and evaluation metrics are written to `outputs/`.

**Verify everything works before demoing:**

```bash
python scripts/verify_dashboard.py   # saved model + Grad-CAM path
python scripts/verify_model.py       # sample predictions on yes/no images
python scripts/run_dashboard.py      # launch UI at http://localhost:8501
```

**For portfolio reviewers or quick demos:** You can publish the trained `.keras` file as a [GitHub Release](https://github.com/JujharRandhawa/Brain-tumor-classification-model/releases) asset. Downloaders should place it at:

```
models/efficientnetb3_brain_tumor.keras
```

Then they can skip training and go straight to `python scripts/run_dashboard.py`.

**Reported test performance (local run):** 84.2% accuracy, 0.954 AUC on the held-out test split.

---

## Quick Start

### 1. Environment Setup

```bash
cd "Brain Tumor classification model"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> **Note:** Kaggle download requires a configured Kaggle API credential. Run `kagglehub login` or place your `kaggle.json` in `~/.kaggle/`.

### 2. Download Data

```bash
python scripts/download_data.py
```

### 3. Train Model

```bash
python scripts/train.py
```

Training runs in two phases:
1. **Frozen backbone** — train classification head (~10 epochs)
2. **Fine-tuning** — unfreeze top EfficientNet layers

Outputs saved to `models/efficientnetb3_brain_tumor.keras` and `outputs/`.

### 4. Launch Dashboard

```bash
python scripts/run_dashboard.py
```

Open `http://localhost:8501` — upload an MRI or load a sample to view predictions and Grad-CAM heatmaps.

### 5. Evaluate (optional)

```bash
python scripts/evaluate.py
```

---

## How Grad-CAM Works

```
MRI Input → EfficientNetB3 → Tumor Probability
                  ↓
         Last Conv Layer activations
                  ×
         Gradients w.r.t. tumor class
                  ↓
         Weighted heatmap → OpenCV overlay
```

Grad-CAM produces a coarse localization map showing **where** the network looked when making its decision — critical for radiologist-in-the-loop validation in medical AI.

---

## Configuration

Edit `config/config.yaml` to adjust:

- Image size, batch size, learning rate, epochs
- Augmentation parameters
- Grad-CAM overlay alpha
- Train/val/test split ratios

---

## Future Extension: Clinical Decision Support Systems (CDSS)

This pipeline is designed as a modular XAI component for future CDSS integration:

- **REST API wrapper** around inference + Grad-CAM endpoints
- **DICOM ingestion** for PACS/RIS hospital workflows
- **HL7 FHIR** observation resources for EHR embedding
- **Radiologist feedback loop** to log agree/disagree on attributions
- **Multi-class extension** (glioma, meningioma, pituitary) with per-class Grad-CAM

---

## Disclaimer

This project is for **research and educational purposes only**. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.

---

## Tech Stack

- **TensorFlow / Keras** — deep learning framework
- **EfficientNetB3** — efficient CNN architecture
- **Grad-CAM** — gradient-based visual attribution
- **OpenCV** — image I/O and heatmap compositing
- **Streamlit** — interactive dashboard
- **kagglehub** — dataset management
- **scikit-learn** — metrics and stratified splitting

---

## License

MIT License — see dataset license on Kaggle for data usage terms.
