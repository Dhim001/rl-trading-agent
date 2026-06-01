from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_drawdown_pct: float = 0.15
    stop_loss_pct: float = 0.05
    position_size_pct: float = 0.95


class RiskManager:
    """Position sizing and drawdown guardrails."""

    def __init__(self, limits: RiskLimits, initial_cash: float) -> None:
        self.limits = limits
        self.initial_cash = initial_cash
        self.peak_equity = initial_cash

    def update_peak(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def trading_halted(self, equity: float) -> bool:
        return self.current_drawdown(equity) >= self.limits.max_drawdown_pct

    def max_shares(self, cash: float, price: float) -> int:
        if price <= 0:
            return 0
        budget = cash * self.limits.position_size_pct
        return int(budget // price)

    def stop_loss_triggered(self, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        loss = (entry_price - current_price) / entry_price
        return loss >= self.limits.stop_loss_pct
