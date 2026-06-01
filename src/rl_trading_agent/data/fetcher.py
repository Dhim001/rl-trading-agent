from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_market_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / f"{symbol}_{start_date}_{end_date}.csv"

    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        return df.sort_index()

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}")

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(file_path)
    return df.sort_index()
