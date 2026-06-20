from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui_dashboard.components.ui import render_empty_state, render_freshness_row
from ui_dashboard.services.data_service import human_size


def render(project_root: Path) -> None:
    st.subheader("Artifact Explorer")
    st.caption("Browse generated models, runs, tuning outputs, and paper/backtest files with size and recency visibility.")
    render_freshness_row(
        [
            ("Models", project_root / "models"),
            ("Runs", project_root / "runs"),
            ("Tuning", project_root / "tuning"),
            ("Results", project_root / "results"),
            ("Paper", project_root / "paper"),
        ]
    )

    rows: list[dict[str, Any]] = []
    for folder in ["models", "runs", "tuning", "results", "paper"]:
        dir_path = project_root / folder
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append(
                {
                    "path": str(path.relative_to(project_root)),
                    "size": human_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    if not rows:
        render_empty_state("No tracked artifact files found.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
