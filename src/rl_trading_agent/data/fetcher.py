from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    valid = ~idx.isna()
    if not bool(valid.all()):
        df = df.loc[valid].copy()
        idx = idx[valid]
    df.index = idx.tz_convert("UTC").tz_localize(None).normalize()
    df.index.name = "Date"
    return df


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
        df = pd.read_csv(
            file_path,
            parse_dates=["Date"],
            index_col="Date",
            usecols=["Date", *NUMERIC_COLUMNS],
            dtype={col: "float32" for col in NUMERIC_COLUMNS},
        )
        return _normalize_date_index(df).sort_index()

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}")

    df = _normalize_date_index(df)
    df = df[NUMERIC_COLUMNS].dropna()
    df = df.astype({col: "float32" for col in NUMERIC_COLUMNS})
    df.to_csv(file_path)
    return df.sort_index()
