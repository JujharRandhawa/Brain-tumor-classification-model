"""Model evaluation and reporting utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, test_ds, class_names: list[str], output_dir: str | Path) -> dict:
    """Run evaluation on the test set and save metrics plus plots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    y_true, y_prob = _collect_predictions(model, test_ds)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    metrics_path = output / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    _plot_confusion_matrix(y_true, y_pred, class_names, output / "confusion_matrix.png")
    _plot_roc_curve(y_true, y_prob, output / "roc_curve.png")

    logger.info("Test accuracy: %.4f | AUC: %.4f", metrics["accuracy"], metrics["auc"])
    return metrics


def _collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = [], []
    for batch_x, batch_y in dataset:
        preds = model.predict(batch_x, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(batch_y.numpy().tolist())
    return np.array(y_true), np.array(y_prob)


def _plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Brain Tumor Classification")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_prob, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — EfficientNetB3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
