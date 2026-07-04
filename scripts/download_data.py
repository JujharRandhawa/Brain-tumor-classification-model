#!/usr/bin/env python
"""Download the Kaggle brain MRI dataset into the project data directory."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.download import download_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    config = load_config()
    path = download_dataset(
        dataset_id=config["data"]["kaggle_dataset"],
        target_dir=config["paths"]["raw_data_dir"],
    )
    print(f"\nDataset ready at: {path}")
    print("Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
