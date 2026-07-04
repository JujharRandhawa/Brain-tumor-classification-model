#!/usr/bin/env python
"""Launch the Streamlit Grad-CAM visualization dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app" / "dashboard.py"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
