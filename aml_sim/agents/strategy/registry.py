"""Global registry of named alpha strategies."""

from __future__ import annotations

from typing import Type

from aml_sim.agents.strategy.alpha import AlphaStrategy


class StrategyRegistry:
    """Registry of alpha strategies available to agents.

    Strategies are registered by name and can be looked up at runtime. This
    lets the LLM slow-loop enable/disable strategies by name, and lets
    scenario YAML reference strategies without importing classes.
    """

    _strategies: dict[str, Type[AlphaStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[AlphaStrategy]) -> None:
        key = name.lower().strip()
        if not key:
            raise ValueError("Strategy name must be non-empty")
        cls._strategies[key] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Type[AlphaStrategy] | None:
        return cls._strategies.get(name.lower().strip())

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._strategies.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name.lower().strip() in cls._strategies

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies (useful for testing)."""
        cls._strategies.clear()


# ---------------------------------------------------------------------------
# Auto-register built-in strategies on first import
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Import and register every built-in alpha strategy shipped with AML-Sim."""
    from aml_sim.agents.strategy.alpha_breakout import BreakoutStrategy
    from aml_sim.agents.strategy.alpha_event_driven import EventDrivenStrategy
    from aml_sim.agents.strategy.alpha_mean_reversion import MeanReversionStrategy
    from aml_sim.agents.strategy.alpha_momentum import MomentumStrategy
    from aml_sim.agents.strategy.alpha_passive import PassiveBenchmarkStrategy
    from aml_sim.agents.strategy.alpha_volatility import VolatilityRegimeStrategy

    StrategyRegistry.register("momentum", MomentumStrategy)
    StrategyRegistry.register("mean_reversion", MeanReversionStrategy)
    StrategyRegistry.register("breakout", BreakoutStrategy)
    StrategyRegistry.register("event_driven", EventDrivenStrategy)
    StrategyRegistry.register("volatility_regime", VolatilityRegimeStrategy)
    StrategyRegistry.register("passive_benchmark", PassiveBenchmarkStrategy)


# Auto-register on module import
_register_builtins()
