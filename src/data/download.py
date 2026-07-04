"""Download and locate the Kaggle brain MRI dataset."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)


def download_dataset(dataset_id: str, target_dir: str | Path) -> Path:
    """
    Download the dataset via kagglehub and copy/symlink into the project data folder.

    Returns the resolved path containing class subfolders (yes/no).
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", dataset_id)
    cache_path = Path(kagglehub.dataset_download(dataset_id))
    logger.info("Kaggle cache path: %s", cache_path)

    dataset_root = _find_dataset_root(cache_path)
    project_raw = target / "brain_mri"
    project_raw.mkdir(parents=True, exist_ok=True)

    for item in dataset_root.iterdir():
        dest = project_raw / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    logger.info("Dataset available at: %s", project_raw)
    return project_raw


def _find_dataset_root(cache_path: Path) -> Path:
    """Locate the directory that contains yes/no class folders."""
    candidates = [cache_path, *cache_path.rglob("*")]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        child_names = {child.name.lower() for child in candidate.iterdir() if child.is_dir()}
        if {"yes", "no"}.issubset(child_names):
            return candidate
    raise FileNotFoundError(
        f"Could not find 'yes' and 'no' folders under {cache_path}. "
        "Verify the Kaggle dataset structure."
    )


def resolve_data_directory(raw_data_dir: str | Path) -> Path:
    """Return an existing dataset path or raise a helpful error."""
    raw = Path(raw_data_dir)
    candidates: list[Path] = [raw / "brain_mri", raw]
    if raw.exists():
        candidates.extend(p for p in raw.rglob("*") if p.is_dir())

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        child_names = {c.name.lower() for c in candidate.iterdir() if c.is_dir()}
        if {"yes", "no"}.issubset(child_names):
            return candidate
    raise FileNotFoundError(
        f"No dataset found under {raw}. Run: python scripts/download_data.py"
    )
