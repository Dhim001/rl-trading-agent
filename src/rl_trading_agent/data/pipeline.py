from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rl_trading_agent.data.features import add_technical_features, apply_normalization, normalize_features
from rl_trading_agent.data.fetcher import download_market_data
from rl_trading_agent.data.panel import build_multi_stock_panel


def is_multi_stock(cfg: dict[str, Any]) -> bool:
    symbols = cfg["data"].get("symbols")
    return bool(symbols and len(symbols) > 1)


def load_dataset(cfg: dict[str, Any], root: Path) -> tuple[Any, dict, str]:
    data_cfg = cfg["data"]
    cache_dir = root / data_cfg["cache_dir"]

    if is_multi_stock(cfg):
        symbols = data_cfg["symbols"]
        panel, stats = build_multi_stock_panel(
            symbols=symbols,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
            cache_dir=cache_dir,
        )
        return panel, stats, "multi"

    raw = download_market_data(
        symbol=data_cfg["symbol"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        cache_dir=cache_dir,
    )
    featured = add_technical_features(raw)
    normalized, stats = normalize_features(featured)
    return normalized, stats, "single"


def save_feature_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_feature_stats(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stats_for_symbol(stats: dict, symbol: str) -> dict:
    if not stats:
        return {}
    sample = next(iter(stats.values()))
    if isinstance(sample, dict) and "mean" in sample:
        return stats
    return stats.get(symbol, {})


def prepare_live_panel(
    symbols: list[str],
    lookback_days: int,
    stats: dict,
    cache_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Fetch recent history and normalize using saved training stats."""
    from datetime import datetime, timedelta

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = download_market_data(symbol, start, end, cache_dir)
        featured = add_technical_features(raw)
        symbol_stats = _stats_for_symbol(stats, symbol)
        if symbol_stats:
            panel[symbol] = apply_normalization(featured, symbol_stats)
        else:
            panel[symbol], _ = normalize_features(featured)
    return panel
