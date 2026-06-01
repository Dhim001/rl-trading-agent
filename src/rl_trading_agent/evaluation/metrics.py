from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (rolling_max - equity_curve) / rolling_max.replace(0, np.nan)
    return float(drawdown.max())


def total_return(equity_curve: pd.Series) -> float:
    if equity_curve.iloc[0] == 0:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def summarize_backtest(equity_curve: list[float]) -> dict[str, float]:
    series = pd.Series(equity_curve)
    returns = series.pct_change().dropna()
    return {
        "total_return": total_return(series),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(series),
        "final_equity": float(series.iloc[-1]),
    }


def save_equity_curve(equity_curve: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": equity_curve}).to_csv(path, index=False)
