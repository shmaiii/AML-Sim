"""Strategy performance tracking for AML agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyPerformance:
    """Tracks performance of one alpha strategy for one agent over time."""

    strategy_name: str = ""
    signals_generated: int = 0
    signals_acted_on: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_trades: int = 0
    cumulative_pnl: float = 0.0
    total_bp_captured: float = 0.0     # sum of (exit - entry) / entry in bp
    holding_ticks_sum: int = 0
    last_signal_time: str | None = None
    last_direction: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades <= 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.losing_trades <= 0:
            return float("inf") if self.winning_trades > 0 else 0.0
        if self.winning_trades <= 0:
            return 0.0
        return self.winning_trades / max(self.losing_trades, 1)

    @property
    def avg_holding_ticks(self) -> float:
        if self.total_trades <= 0:
            return 0.0
        return self.holding_ticks_sum / self.total_trades

    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades <= 0:
            return 0.0
        return self.cumulative_pnl / self.total_trades

    def record_signal(self, direction: float, timestamp: str | None = None) -> None:
        self.signals_generated += 1
        self.last_direction = direction
        if timestamp:
            self.last_signal_time = timestamp

    def record_acted(self) -> None:
        self.signals_acted_on += 1

    def record_trade_outcome(
        self,
        pnl: float,
        is_win: bool,
        holding_ticks: int = 0,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
    ) -> None:
        self.total_trades += 1
        self.cumulative_pnl += pnl
        self.holding_ticks_sum += holding_ticks
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        if entry_price > 0 and exit_price > 0:
            bp = abs((exit_price / entry_price) - 1.0) * 10_000
            self.total_bp_captured += bp

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "signals_generated": self.signals_generated,
            "signals_acted_on": self.signals_acted_on,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 3),
            "profit_factor": (
                round(self.profit_factor, 2) if self.profit_factor != float("inf") else "inf"
            ),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "avg_holding_ticks": round(self.avg_holding_ticks, 1),
            "last_direction": round(self.last_direction, 4),
        }


def create_performance_tracker(strategy_names: list[str]) -> dict[str, StrategyPerformance]:
    """Create a fresh performance tracker dict for a set of strategy names."""
    return {name: StrategyPerformance(strategy_name=name) for name in strategy_names}
