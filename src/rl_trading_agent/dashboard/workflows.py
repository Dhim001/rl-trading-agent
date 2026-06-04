from __future__ import annotations

from typing import Any


# Shared workflow contracts used by both CLI scripts and UI dashboard.
WORKFLOW_DEFINITIONS: list[dict[str, Any]] = [
    {"id": "download_data", "label": "Download Data", "command": ["scripts/download_data.py"]},
    {"id": "training", "label": "Start Training", "command": ["scripts/train.py"]},
    {"id": "backtest", "label": "Run Backtest", "command": ["scripts/backtest.py"]},
    {"id": "fine_tuning", "label": "Start Fine-tuning", "command": ["scripts/tune.py"]},
    {"id": "paper_once", "label": "Paper Trade Once", "command": ["scripts/paper_trade.py", "--once"]},
    {"id": "paper_loop", "label": "Start Paper Trading Loop", "command": ["scripts/paper_trade.py"]},
]

WORKFLOW_COMMANDS: dict[str, list[str]] = {item["id"]: item["command"] for item in WORKFLOW_DEFINITIONS}
WORKFLOW_LABELS: dict[str, str] = {item["id"]: item["label"] for item in WORKFLOW_DEFINITIONS}
