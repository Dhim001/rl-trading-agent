from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def artifact_version(project_root: Path, rel_path: str) -> dict[str, Any] | None:
    path = project_root / rel_path
    if not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": rel_path,
        "version": f"{int(stat.st_mtime)}-{stat.st_size}-{_sha256(path)}",
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "mtime_epoch": float(stat.st_mtime),
        "size_bytes": int(stat.st_size),
    }


def _collect_models(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "models").glob("*.zip")):
        rel = str(path.relative_to(project_root))
        row = artifact_version(project_root, rel)
        if row:
            row["type"] = "model"
            rows.append(row)
    for path in sorted((project_root / "tuning").glob("trial_*/models/**/*.zip")):
        rel = str(path.relative_to(project_root))
        row = artifact_version(project_root, rel)
        if row:
            row["type"] = "model"
            rows.append(row)
    return rows


def _collect_backtests(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [project_root / "results" / "equity_curve.csv"]
    candidates.extend(sorted((project_root / "tuning").glob("trial_*/results/equity_curve.csv")))
    for path in candidates:
        if not path.exists():
            continue
        rel = str(path.relative_to(project_root))
        row = artifact_version(project_root, rel)
        if row:
            row["type"] = "backtest"
            rows.append(row)
    return rows


def build_provenance(project_root: Path) -> pd.DataFrame:
    data = artifact_version(project_root, "data/feature_stats.json")
    models = sorted(_collect_models(project_root), key=lambda r: r["mtime_epoch"])
    backtests = sorted(_collect_backtests(project_root), key=lambda r: r["mtime_epoch"])
    if not backtests:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for bt in backtests:
        bt_time = float(bt["mtime_epoch"])
        model = next((m for m in reversed(models) if float(m["mtime_epoch"]) <= bt_time), None)
        rows.append(
            {
                "data_path": data["path"] if data else "data/feature_stats.json",
                "data_version": data["version"] if data else "missing",
                "model_path": model["path"] if model else "unknown",
                "model_version": model["version"] if model else "unknown",
                "backtest_path": bt["path"],
                "backtest_version": bt["version"],
                "backtest_time": bt["mtime"],
            }
        )
    return pd.DataFrame(rows).sort_values("backtest_time", ascending=False).reset_index(drop=True)


def build_dependency_graph_dot(provenance_df: pd.DataFrame) -> str:
    if provenance_df.empty:
        return "digraph G { rankdir=LR; node [shape=box]; empty [label=\"No lineage data\"]; }"

    lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [shape=box, style=filled, fillcolor=\"#F6F8FA\"];",
    ]
    seen_nodes: set[str] = set()
    for _, row in provenance_df.iterrows():
        data_node = f"data::{row['data_version']}"
        model_node = f"model::{row['model_version']}"
        bt_node = f"backtest::{row['backtest_version']}"
        node_defs = [
            (data_node, f"Data\\n{row['data_path']}"),
            (model_node, f"Model\\n{row['model_path']}"),
            (bt_node, f"Backtest\\n{row['backtest_path']}"),
        ]
        for node_id, label in node_defs:
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            lines.append(f"\"{node_id}\" [label=\"{label}\"];")
        lines.append(f"\"{data_node}\" -> \"{model_node}\";")
        lines.append(f"\"{model_node}\" -> \"{bt_node}\";")
    lines.append("}")
    return "\n".join(lines)


def list_trial_param_files(project_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted((project_root / "tuning").glob("trial_*/params.json")):
        out[path.parents[0].name] = path
    return out


def load_trial_params(project_root: Path, trial_name: str) -> dict[str, Any] | None:
    path = project_root / "tuning" / trial_name / "params.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def diff_dicts(left: dict[str, Any], right: dict[str, Any]) -> pd.DataFrame:
    keys = sorted(set(left.keys()) | set(right.keys()))
    rows: list[dict[str, Any]] = []
    for key in keys:
        lval = left.get(key, "<missing>")
        rval = right.get(key, "<missing>")
        rows.append(
            {
                "parameter": key,
                "left": lval,
                "right": rval,
                "changed": lval != rval,
            }
        )
    return pd.DataFrame(rows)
