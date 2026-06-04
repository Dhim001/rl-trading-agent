#!/usr/bin/env python3
"""Launch the integrated UI dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL Trading UI dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit server port")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    runner = str(python_exe) if python_exe.exists() else sys.executable

    cmd = [
        runner,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "ui_dashboard" / "app.py"),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
