from __future__ import annotations

import numpy as np
import pandas as pd
import ta


FEATURE_COLUMNS = [
    "returns",
    "log_returns",
    "rsi",
    "macd",
    "macd_signal",
    "bb_width",
    "volume_change",
    "sma_ratio",
    "volatility",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)
    close = out["Close"]
    volume = out["Volume"]

    out["returns"] = close.pct_change()
    out["log_returns"] = np.log(close / close.shift(1))
    out["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20)
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    out["volume_change"] = volume.pct_change()
    sma20 = close.rolling(20).mean()
    out["sma_ratio"] = close / sma20
    out["volatility"] = out["returns"].rolling(20).std()

    out = out.dropna()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].astype("float32")
    return out


def normalize_features(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    cols = columns or FEATURE_COLUMNS
    stats: dict[str, dict[str, float]] = {}
    normalized = df.copy(deep=False)

    for col in cols:
        mean = float(normalized[col].mean())
        std = float(normalized[col].std()) or 1.0
        normalized[col] = ((normalized[col] - mean) / std).astype("float32")
        stats[col] = {"mean": mean, "std": std}

    return normalized, stats


def apply_normalization(df: pd.DataFrame, stats: dict, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or FEATURE_COLUMNS
    normalized = df.copy(deep=False)
    for col in cols:
        if col not in stats:
            continue
        mean = stats[col]["mean"]
        std = stats[col]["std"] or 1.0
        normalized[col] = ((normalized[col] - mean) / std).astype("float32")
    return normalized.dropna()
