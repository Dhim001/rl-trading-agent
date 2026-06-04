from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_trading_agent.config import load_config
from rl_trading_agent.data.features import FEATURE_COLUMNS
from rl_trading_agent.data.pipeline import load_dataset
from rl_trading_agent.env.factory import make_trading_env
from rl_trading_agent.evaluation.metrics import save_equity_curve, summarize_backtest
from rl_trading_agent.training.trainer import split_train_test
from ui_dashboard.services.cache_service import load_rl_model_cached


def load_project_config(project_root: Path) -> dict[str, Any]:
    return load_config(project_root / "config" / "default.yaml")


def _runtime_strategy_dir(project_root: Path) -> Path:
    path = project_root / "ui_dashboard" / ".runtime" / "strategy_studio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_strategy_presets(project_root: Path) -> list[str]:
    base = _runtime_strategy_dir(project_root)
    names: list[str] = []
    for p in base.glob("*.json"):
        stem = p.stem
        # Exclude historical snapshot files like: my_strategy.v20260604_012233.json
        if ".v20" in stem:
            continue
        names.append(stem)
    return sorted(names)


def save_strategy_preset(project_root: Path, name: str, payload: dict[str, Any]) -> Path:
    base = _runtime_strategy_dir(project_root)
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip()).strip("_")
    if not safe_name:
        safe_name = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = base / f"{safe_name}.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_path = base / f"{safe_name}.v{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_strategy_preset(project_root: Path, name: str) -> dict[str, Any] | None:
    path = _runtime_strategy_dir(project_root) / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_strategy_preset_versions(project_root: Path, name: str) -> list[str]:
    base = _runtime_strategy_dir(project_root)
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.strip()).strip("_")
    if not safe_name:
        return []
    versions = sorted([p.name for p in base.glob(f"{safe_name}.v*.json")], reverse=True)
    return versions


def load_strategy_preset_version(project_root: Path, version_filename: str) -> dict[str, Any] | None:
    base = _runtime_strategy_dir(project_root)
    if not version_filename.endswith(".json"):
        return None
    path = base / version_filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_length(data: pd.DataFrame | dict[str, pd.DataFrame]) -> int:
    if isinstance(data, dict):
        symbol = next(iter(sorted(data.keys())))
        return len(data[symbol])
    return len(data)


def dataset_slice(data: pd.DataFrame | dict[str, pd.DataFrame], start: int, end: int) -> pd.DataFrame | dict[str, pd.DataFrame]:
    if isinstance(data, dict):
        return {s: df.iloc[start:end] for s, df in data.items()}
    return data.iloc[start:end]


def choose_symbol_frame(data: pd.DataFrame | dict[str, pd.DataFrame], symbol: str | None = None) -> pd.DataFrame:
    if isinstance(data, dict):
        symbols = sorted(data.keys())
        target = symbol if symbol in data else symbols[0]
        return data[target]
    return data


def run_backtest_with_overrides(
    project_root: Path,
    model_rel_path: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_project_config(project_root)
    if overrides:
        env_overrides = overrides.get("environment") or {}
        risk_overrides = overrides.get("risk") or {}
        cfg["environment"] = {**cfg["environment"], **env_overrides}
        cfg["risk"] = {**cfg["risk"], **risk_overrides}

    data, _, mode = load_dataset(cfg, project_root)
    _, test_data = split_train_test(data, cfg["training"]["train_split"])
    output_dir = (
        project_root
        / "ui_dashboard"
        / ".runtime"
        / "strategy_studio"
        / "backtests"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    model_path = project_root / model_rel_path
    metrics, equity_curve = _run_cached_model_backtest(
        model_path=model_path,
        test_data=test_data,
        mode=mode,
        env_cfg=cfg["environment"],
        risk_cfg=cfg["risk"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_equity_curve(equity_curve, output_dir / "equity_curve.csv")
    curve_path = output_dir / "equity_curve.csv"
    curve_df = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
    if not curve_df.empty and "equity" in curve_df.columns:
        curve_df["step"] = range(len(curve_df))
    return {
        "metrics": metrics,
        "equity_curve": curve_df,
        "output_dir": str(output_dir.relative_to(project_root)).replace("\\", "/"),
    }


def walk_forward_analysis(
    project_root: Path,
    model_rel_path: str,
    n_splits: int,
    train_ratio: float,
) -> pd.DataFrame:
    cfg = load_project_config(project_root)
    data, _, mode = load_dataset(cfg, project_root)
    train_data, test_data = split_train_test(data, train_ratio)

    total = dataset_length(test_data)
    if total <= n_splits:
        return pd.DataFrame()
    fold = max(1, total // n_splits)

    rows: list[dict[str, Any]] = []
    for i in range(n_splits):
        start = i * fold
        end = total if i == n_splits - 1 else min(total, (i + 1) * fold)
        fold_data = dataset_slice(test_data, start, end)
        out_dir = project_root / "ui_dashboard" / ".runtime" / "strategy_studio" / f"walk_forward_{i+1}"
        metrics, equity_curve = _run_cached_model_backtest(
            model_path=project_root / model_rel_path,
            test_data=fold_data,
            mode=mode,
            env_cfg=cfg["environment"],
            risk_cfg=cfg["risk"],
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_equity_curve(equity_curve, out_dir / "equity_curve.csv")
        rows.append(
            {
                "fold": i + 1,
                "start_idx": start,
                "end_idx": end,
                "total_return": metrics.get("total_return", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            }
        )
    return pd.DataFrame(rows)


def indicator_correlation_matrix(frame: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    cols = [c for c in indicators if c in frame.columns]
    if not cols:
        return pd.DataFrame()
    return frame[cols].corr().fillna(0.0)


def shap_feature_importance(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    needed = [c for c in FEATURE_COLUMNS if c in frame.columns]
    if not needed or "Close" not in frame.columns:
        return pd.DataFrame(), "Insufficient feature columns for SHAP analysis."

    model_frame = frame[needed + ["Close"]].dropna().copy()
    if len(model_frame) < 80:
        return pd.DataFrame(), "Not enough rows for SHAP analysis."
    model_frame["target"] = model_frame["Close"].pct_change().shift(-1)
    model_frame = model_frame.dropna()
    if model_frame.empty:
        return pd.DataFrame(), "Could not build target series for SHAP."

    X = model_frame[needed]
    y = model_frame["target"]
    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap
    except Exception:
        return pd.DataFrame(), "SHAP/scikit-learn unavailable. Install dependencies to enable this panel."

    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X, y)
    sample = X.sample(min(300, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    values = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": needed, "mean_abs_shap": values}).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index(drop=True), "ok"


def parameter_sensitivity(
    project_root: Path,
    model_rel_path: str,
    parameter: str,
    values: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in values:
        env_overrides: dict[str, Any] = {}
        risk_overrides: dict[str, Any] = {}
        if parameter in {"window_size", "transaction_cost_pct"}:
            env_overrides[parameter] = int(value) if parameter == "window_size" else float(value)
        else:
            risk_overrides[parameter] = float(value)

        result = run_backtest_with_overrides(
            project_root=project_root,
            model_rel_path=model_rel_path,
            overrides={"environment": env_overrides, "risk": risk_overrides},
        )
        metrics = result.get("metrics", {})
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "total_return": metrics.get("total_return", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _run_cached_model_backtest(
    model_path: Path,
    test_data: pd.DataFrame | dict[str, pd.DataFrame],
    mode: str,
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
) -> tuple[dict[str, float], list[float]]:
    env = make_trading_env(test_data, mode, env_cfg, risk_cfg)
    model = load_rl_model_cached(str(model_path))
    # Rebind environment for cached model instance to current backtest data.
    model.set_env(env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if mode == "multi":
            action = np.asarray(action, dtype=int)
        else:
            action = int(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    equity_curve = list(env.equity_curve)
    metrics = summarize_backtest(equity_curve)
    return metrics, equity_curve
