from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_trading_agent.data.features import add_technical_features, normalize_features
from rl_trading_agent.data.fetcher import download_market_data


def download_multi_stock_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/cache",
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = download_market_data(symbol, start_date, end_date, cache_dir)
    return frames


def align_stock_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common_dates = None
    for df in frames.values():
        dates = df.index
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No overlapping dates across symbols")

    common_dates = common_dates.sort_values()
    return {symbol: df.loc[common_dates].copy() for symbol, df in frames.items()}


def build_multi_stock_panel(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/cache",
) -> tuple[dict[str, pd.DataFrame], dict]:
    raw = download_multi_stock_data(symbols, start_date, end_date, cache_dir)
    aligned = align_stock_frames(raw)

    featured: dict[str, pd.DataFrame] = {}
    for symbol, df in aligned.items():
        featured[symbol] = add_technical_features(df)

    common_dates = featured[symbols[0]].index
    for symbol in symbols[1:]:
        common_dates = common_dates.intersection(featured[symbol].index)
    common_dates = common_dates.sort_values()

    normalized: dict[str, pd.DataFrame] = {}
    all_stats: dict[str, dict] = {}
    for symbol in symbols:
        sliced = featured[symbol].loc[common_dates].copy()
        norm, stats = normalize_features(sliced)
        normalized[symbol] = norm
        all_stats[symbol] = stats

    return normalized, all_stats
