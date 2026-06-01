from __future__ import annotations

from typing import Any

import pandas as pd

from rl_trading_agent.env.multi_stock_env import MultiStockTradingEnv
from rl_trading_agent.env.trading_env import StockTradingEnv
from rl_trading_agent.risk.manager import RiskLimits


def make_trading_env(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    mode: str,
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
):
    risk_limits = RiskLimits(
        max_drawdown_pct=risk_cfg["max_drawdown_pct"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        position_size_pct=risk_cfg["position_size_pct"],
    )
    common = {
        "window_size": env_cfg["window_size"],
        "initial_cash": env_cfg["initial_cash"],
        "transaction_cost_pct": env_cfg["transaction_cost_pct"],
        "reward_scaling": env_cfg["reward_scaling"],
        "risk_limits": risk_limits,
    }
    if mode == "multi":
        return MultiStockTradingEnv(panel=data, **common)
    return StockTradingEnv(df=data, **common)
