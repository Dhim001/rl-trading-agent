from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_trading_agent.data.features import FEATURE_COLUMNS
from rl_trading_agent.risk.manager import RiskLimits, RiskManager


class StockTradingEnv(gym.Env):
    """
    Discrete-action trading environment.
    Actions: 0=hold, 1=buy, 2=sell
    Observation: rolling window of normalized features + portfolio state
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_cash: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1.0,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.risk = RiskManager(risk_limits or RiskLimits(), initial_cash)

        n_features = len(FEATURE_COLUMNS) + 3  # cash_ratio, position_ratio, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._feature_width = len(FEATURE_COLUMNS)
        self._obs_buffer = np.empty(self.observation_space.shape, dtype=np.float32)
        self._zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.update_market_data(df)

        self.current_step = window_size
        self.cash = initial_cash
        self.shares = 0
        self.entry_price = 0.0
        self.equity_curve: list[float] = []

    def _price(self) -> float:
        return float(self._close_prices[self.current_step])

    def update_market_data(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self._feature_matrix = self.df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self._close_prices = self.df["Close"].to_numpy(dtype=np.float32, copy=True)
        self._n_steps = len(self.df)

    def _equity(self) -> float:
        return self.cash + self.shares * self._price()

    def _portfolio_features(self) -> np.ndarray:
        equity = self._equity()
        price = self._price()
        position_value = self.shares * price
        cash_ratio = self.cash / max(equity, 1e-8)
        position_ratio = position_value / max(equity, 1e-8)
        unrealized = 0.0
        if self.shares > 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price
        return np.array([cash_ratio, position_ratio, unrealized], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        self._obs_buffer[:, :self._feature_width] = self._feature_matrix[start:end]
        self._obs_buffer[:, self._feature_width:] = self._portfolio_features()
        return self._obs_buffer

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0
        self.entry_price = 0.0
        self.equity_curve = [self.initial_cash]
        self.risk = RiskManager(self.risk.limits, self.initial_cash)
        return self._get_observation(), {}

    def step(self, action: int):
        prev_equity = self._equity()
        price = self._price()
        terminated = False
        truncated = False

        if self.risk.trading_halted(prev_equity):
            action = 0
        elif self.shares > 0 and self.risk.stop_loss_triggered(self.entry_price, price):
            action = 2

        if action == 1 and self.shares == 0:
            shares_to_buy = self.risk.max_shares(self.cash, price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                fee = cost * self.transaction_cost_pct
                self.cash -= cost + fee
                self.shares = shares_to_buy
                self.entry_price = price
        elif action == 2 and self.shares > 0:
            proceeds = self.shares * price
            fee = proceeds * self.transaction_cost_pct
            self.cash += proceeds - fee
            self.shares = 0
            self.entry_price = 0.0

        self.current_step += 1
        if self.current_step >= self._n_steps - 1:
            terminated = True

        equity = self._equity()
        self.risk.update_peak(equity)
        self.equity_curve.append(equity)

        reward = ((equity - prev_equity) / max(prev_equity, 1e-8)) * self.reward_scaling
        if self.risk.trading_halted(equity):
            reward -= 0.01

        obs = self._get_observation() if not terminated else self._zero_obs
        info = {"equity": equity, "cash": self.cash, "shares": self.shares, "step": self.current_step}
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"step={self.current_step} equity={self._equity():.2f} "
            f"cash={self.cash:.2f} shares={self.shares}"
        )
