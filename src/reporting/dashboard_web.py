from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.settings import PROJECT_ROOT


def run_dashboard_web(args: Any) -> None:
    try:
        __import__("streamlit")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies with: python -m pip install -e .[dev]"
        ) from exc

    app_path = Path(PROJECT_ROOT) / "src" / "reporting" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(int(args.port)),
        "--server.address",
        str(args.host),
        "--server.headless",
        "true",
    ]
    subprocess.run(command, check=True)
