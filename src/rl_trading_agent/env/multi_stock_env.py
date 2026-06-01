from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_trading_agent.data.features import FEATURE_COLUMNS
from rl_trading_agent.risk.manager import RiskLimits, RiskManager


class MultiStockTradingEnv(gym.Env):
    """
    Multi-asset trading environment.
    Action: [symbol_index, action] where action is 0=hold, 1=buy, 2=sell.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel: dict[str, pd.DataFrame],
        window_size: int = 30,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        super().__init__()
        self.symbols = sorted(panel.keys())
        self.panel = {s: panel[s].reset_index(drop=True) for s in self.symbols}
        lengths = {s: len(df) for s, df in self.panel.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError("All symbols must have the same number of aligned rows")

        self.n_symbols = len(self.symbols)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.risk = RiskManager(risk_limits or RiskLimits(), initial_cash)

        per_symbol_features = len(FEATURE_COLUMNS)
        n_features = per_symbol_features * self.n_symbols + 3 * self.n_symbols + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.n_symbols, 3])

        self.current_step = window_size
        self.cash = initial_cash
        self.shares = {s: 0 for s in self.symbols}
        self.entry_prices = {s: 0.0 for s in self.symbols}
        self.equity_curve: list[float] = []

    def _price(self, symbol: str) -> float:
        return float(self.panel[symbol].loc[self.current_step, "Close"])

    def _equity(self) -> float:
        holdings = sum(self.shares[s] * self._price(s) for s in self.symbols)
        return self.cash + holdings

    def _symbol_portfolio_features(self, symbol: str) -> np.ndarray:
        equity = self._equity()
        price = self._price(symbol)
        position_value = self.shares[symbol] * price
        cash_ratio = self.cash / max(equity, 1e-8)
        position_ratio = position_value / max(equity, 1e-8)
        unrealized = 0.0
        if self.shares[symbol] > 0 and self.entry_prices[symbol] > 0:
            unrealized = (price - self.entry_prices[symbol]) / self.entry_prices[symbol]
        return np.array([cash_ratio, position_ratio, unrealized], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        feature_rows: list[np.ndarray] = []

        for t in range(start, end):
            row: list[float] = []
            for symbol in self.symbols:
                row.extend(self.panel[symbol].iloc[t][FEATURE_COLUMNS].tolist())
            feature_rows.append(np.array(row, dtype=np.float32))

        window = np.stack(feature_rows, axis=0)
        portfolio_parts = [self._symbol_portfolio_features(symbol) for symbol in self.symbols]
        portfolio = np.concatenate(portfolio_parts)
        equity = self._equity()
        global_feats = np.array([
            self.cash / max(equity, 1e-8),
            1.0 - (self.cash / max(equity, 1e-8)),
        ], dtype=np.float32)
        tail = np.concatenate([portfolio, global_feats])
        tail_tiled = np.tile(tail, (self.window_size, 1))
        return np.concatenate([window, tail_tiled], axis=1)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = {s: 0 for s in self.symbols}
        self.entry_prices = {s: 0.0 for s in self.symbols}
        self.equity_curve = [self.initial_cash]
        self.risk = RiskManager(self.risk.limits, self.initial_cash)
        return self._get_observation(), {}

    def step(self, action: np.ndarray | list[int]):
        symbol_idx, trade_action = int(action[0]), int(action[1])
        symbol = self.symbols[symbol_idx]
        prev_equity = self._equity()
        price = self._price(symbol)
        terminated = False
        truncated = False

        if self.risk.trading_halted(prev_equity):
            trade_action = 0
        elif self.shares[symbol] > 0 and self.risk.stop_loss_triggered(self.entry_prices[symbol], price):
            trade_action = 2

        per_symbol_budget = self.cash * self.risk.limits.position_size_pct / self.n_symbols

        if trade_action == 1 and self.shares[symbol] == 0:
            shares_to_buy = int(per_symbol_budget // price) if price > 0 else 0
            if shares_to_buy > 0 and shares_to_buy * price <= self.cash:
                cost = shares_to_buy * price
                fee = cost * self.transaction_cost_pct
                self.cash -= cost + fee
                self.shares[symbol] = shares_to_buy
                self.entry_prices[symbol] = price
        elif trade_action == 2 and self.shares[symbol] > 0:
            proceeds = self.shares[symbol] * price
            fee = proceeds * self.transaction_cost_pct
            self.cash += proceeds - fee
            self.shares[symbol] = 0
            self.entry_prices[symbol] = 0.0

        self.current_step += 1
        if self.current_step >= len(self.panel[self.symbols[0]]) - 1:
            terminated = True

        equity = self._equity()
        self.risk.update_peak(equity)
        self.equity_curve.append(equity)

        reward = ((equity - prev_equity) / max(prev_equity, 1e-8)) * self.reward_scaling
        if self.risk.trading_halted(equity):
            reward -= 0.01

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "equity": equity,
            "cash": self.cash,
            "shares": dict(self.shares),
            "step": self.current_step,
            "symbol": symbol,
            "trade_action": trade_action,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(f"step={self.current_step} equity={self._equity():.2f} cash={self.cash:.2f} shares={self.shares}")
