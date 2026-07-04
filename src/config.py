"""Configuration loader for the brain tumor XAI project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve project-relative paths."""
    path = config_path or CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in ("data_dir", "raw_data_dir", "model_dir", "output_dir"):
        config["paths"][key] = str(PROJECT_ROOT / config["paths"][key])

    return config
