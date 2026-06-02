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
        self._n_steps = lengths[self.symbols[0]]
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.risk = RiskManager(risk_limits or RiskLimits(), initial_cash)

        per_symbol_features = len(FEATURE_COLUMNS)
        self._feature_width = per_symbol_features * self.n_symbols
        n_features = per_symbol_features * self.n_symbols + 3 * self.n_symbols + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.n_symbols, 3])
        self._obs_buffer = np.empty(self.observation_space.shape, dtype=np.float32)
        self._zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        feature_blocks = []
        close_columns = []
        for symbol in self.symbols:
            symbol_df = self.panel[symbol]
            feature_blocks.append(symbol_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True))
            close_columns.append(symbol_df["Close"].to_numpy(dtype=np.float32, copy=True))
        self._feature_matrix = np.concatenate(feature_blocks, axis=1)
        self._close_prices = np.stack(close_columns, axis=1)

        self.current_step = window_size
        self.cash = initial_cash
        self.shares = np.zeros(self.n_symbols, dtype=np.int32)
        self.entry_prices = np.zeros(self.n_symbols, dtype=np.float32)
        self.equity_curve: list[float] = []

    def _price(self, symbol_idx: int) -> float:
        return float(self._close_prices[self.current_step, symbol_idx])

    def _equity(self) -> float:
        holdings = float(np.dot(self.shares, self._close_prices[self.current_step]))
        return self.cash + holdings

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        self._obs_buffer[:, :self._feature_width] = self._feature_matrix[start:end]
        equity = self._equity()
        denom = max(equity, 1e-8)
        prices = self._close_prices[self.current_step]
        position_values = self.shares.astype(np.float32) * prices
        position_ratios = position_values / denom
        cash_ratio = np.float32(self.cash / denom)
        unrealized = np.zeros(self.n_symbols, dtype=np.float32)
        has_position = self.shares > 0
        unrealized[has_position] = (prices[has_position] - self.entry_prices[has_position]) / self.entry_prices[has_position]

        tail = np.empty((3 * self.n_symbols + 2,), dtype=np.float32)
        tail[0::3][:self.n_symbols] = cash_ratio
        tail[1::3][:self.n_symbols] = position_ratios
        tail[2::3][:self.n_symbols] = unrealized
        tail[-2] = cash_ratio
        tail[-1] = np.float32(1.0 - cash_ratio)
        self._obs_buffer[:, self._feature_width:] = tail
        return self._obs_buffer.copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares.fill(0)
        self.entry_prices.fill(0.0)
        self.equity_curve = [self.initial_cash]
        self.risk = RiskManager(self.risk.limits, self.initial_cash)
        return self._get_observation(), {}

    def step(self, action: np.ndarray | list[int]):
        symbol_idx, trade_action = int(action[0]), int(action[1])
        symbol = self.symbols[symbol_idx]
        prev_equity = self._equity()
        price = self._price(symbol_idx)
        terminated = False
        truncated = False

        if self.risk.trading_halted(prev_equity):
            trade_action = 0
        elif self.shares[symbol_idx] > 0 and self.risk.stop_loss_triggered(float(self.entry_prices[symbol_idx]), price):
            trade_action = 2

        per_symbol_budget = self.cash * self.risk.limits.position_size_pct / self.n_symbols

        if trade_action == 1 and self.shares[symbol_idx] == 0:
            shares_to_buy = int(per_symbol_budget // price) if price > 0 else 0
            if shares_to_buy > 0 and shares_to_buy * price <= self.cash:
                cost = shares_to_buy * price
                fee = cost * self.transaction_cost_pct
                self.cash -= cost + fee
                self.shares[symbol_idx] = shares_to_buy
                self.entry_prices[symbol_idx] = price
        elif trade_action == 2 and self.shares[symbol_idx] > 0:
            proceeds = self.shares[symbol_idx] * price
            fee = proceeds * self.transaction_cost_pct
            self.cash += proceeds - fee
            self.shares[symbol_idx] = 0
            self.entry_prices[symbol_idx] = 0.0

        self.current_step += 1
        if self.current_step >= self._n_steps - 1:
            terminated = True

        equity = self._equity()
        self.risk.update_peak(equity)
        self.equity_curve.append(equity)

        reward = ((equity - prev_equity) / max(prev_equity, 1e-8)) * self.reward_scaling
        if self.risk.trading_halted(equity):
            reward -= 0.01

        obs = self._get_observation() if not terminated else self._zero_obs.copy()
        info = {
            "equity": equity,
            "cash": self.cash,
            "shares": {s: int(self.shares[i]) for i, s in enumerate(self.symbols)},
            "step": self.current_step,
            "symbol": symbol,
            "trade_action": trade_action,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        shares_by_symbol = {s: int(self.shares[i]) for i, s in enumerate(self.symbols)}
        print(
            f"step={self.current_step} equity={self._equity():.2f} "
            f"cash={self.cash:.2f} shares={shares_by_symbol}"
        )
