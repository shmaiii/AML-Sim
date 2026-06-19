"""Shared state contracts for AML agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseStrategyState:
    """Common strategy metadata updated by rule-based or LLM strategists."""

    strategy_type: str
    risk_mode: str = "normal"
    confidence: float = 1.0
    reason: str | None = None
    updated_at: str | None = None

