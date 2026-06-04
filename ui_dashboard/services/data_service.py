from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_equity_curve(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "equity" not in df.columns or df.empty:
        return None
    out = df.copy()
    out["step"] = range(len(out))
    return out


def summarize_equity(equity: pd.Series) -> dict[str, float]:
    if equity.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "final_equity": 0.0}
    rolling_max = equity.cummax()
    drawdown = (rolling_max - equity) / rolling_max.replace(0, pd.NA)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] else 0.0
    return {
        "total_return": total_return,
        "max_drawdown": float(drawdown.max(skipna=True) or 0.0),
        "final_equity": float(equity.iloc[-1]),
    }


def discover_tuning_trials(tuning_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for curve_path in sorted(tuning_dir.glob("trial_*/results/equity_curve.csv")):
        trial_name = curve_path.parents[1].name
        try:
            trial_num = int(trial_name.split("_")[-1])
        except ValueError:
            trial_num = -1
        curve_df = load_equity_curve(curve_path)
        if curve_df is None:
            continue
        summary = summarize_equity(curve_df["equity"])
        records.append(
            {
                "trial": trial_num,
                "path": str(curve_path),
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "final_equity": summary["final_equity"],
                "steps": len(curve_df),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("trial").reset_index(drop=True)


def format_age(path: Path) -> str:
    if not path.exists():
        return "missing"
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    age_seconds = int((datetime.now() - modified).total_seconds())
    if age_seconds < 60:
        return f"{age_seconds}s ago"
    if age_seconds < 3600:
        return f"{age_seconds // 60}m ago"
    if age_seconds < 86400:
        return f"{age_seconds // 3600}h ago"
    return f"{age_seconds // 86400}d ago"


def age_seconds(path: Path) -> int | None:
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return max(0, int((datetime.now() - modified).total_seconds()))


def age_level(path: Path) -> str:
    age = age_seconds(path)
    if age is None:
        return "missing"
    if age < 300:
        return "fresh"
    if age < 3600:
        return "stale"
    return "old"


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"
