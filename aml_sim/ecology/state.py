"""Hierarchical market-state propagation for global, market, and micro shocks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from aml_sim.market_state import MarketStateEngine
from aml_sim.shocks import safe_float


class ScopedMarketStateEngine:
    """Compose the existing market-state engine across explicit economic scopes."""

    def __init__(
        self,
        initial_market_state: Mapping[str, Any] | None,
        instrument_metadata: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.baseline = deepcopy(dict(initial_market_state or {}))
        self.instrument_metadata = {
            str(instrument): dict(metadata)
            for instrument, metadata in instrument_metadata.items()
        }
        self.global_engine = MarketStateEngine(self.baseline)
        self.market_engines: dict[str, MarketStateEngine] = {}
        self.instrument_engines: dict[str, MarketStateEngine] = {}

    def register(
        self,
        event_id: str,
        event: Mapping[str, Any],
        shock_payload: Mapping[str, Any],
        *,
        current_tick_id: int,
        default_duration_ticks: int,
    ) -> None:
        """Register an event only in its declared economic state scope."""
        scope = state_scope_for(event, shock_payload)
        engines: list[MarketStateEngine] = []
        if scope == "global":
            engines = [self.global_engine]
        elif scope == "market":
            for market_key in self._market_keys_for(event, shock_payload):
                engines.append(
                    self.market_engines.setdefault(
                        market_key,
                        MarketStateEngine(self.baseline),
                    )
                )
        else:
            for instrument in self._instruments_for(event, shock_payload):
                engines.append(
                    self.instrument_engines.setdefault(
                        instrument,
                        MarketStateEngine(self.baseline),
                    )
                )

        for engine in engines:
            engine.register(
                event_id,
                event,
                shock_payload,
                current_tick_id=current_tick_id,
                default_duration_ticks=default_duration_ticks,
            )

    def snapshot(self, tick_id: int | None) -> dict[str, Any]:
        """Return serializable state with scope information retained."""
        return {
            "global": self.global_engine.snapshot(tick_id),
            "markets": {
                market_key: engine.snapshot(tick_id)
                for market_key, engine in self.market_engines.items()
            },
            "instruments": {
                instrument: engine.snapshot(tick_id)
                for instrument, engine in self.instrument_engines.items()
            },
            "baseline": deepcopy(self.baseline),
            "instrument_market_keys": {
                instrument: self._market_key(metadata)
                for instrument, metadata in self.instrument_metadata.items()
            },
        }

    def _instruments_for(
        self,
        event: Mapping[str, Any],
        shock_payload: Mapping[str, Any],
    ) -> list[str]:
        configured = shock_payload.get("affected_instruments", event.get("affected_instruments", []))
        if isinstance(configured, str):
            configured = [configured]
        if isinstance(configured, (list, tuple, set)):
            return [str(instrument) for instrument in configured if str(instrument) in self.instrument_metadata]
        return list(self.instrument_metadata)

    def _market_keys_for(
        self,
        event: Mapping[str, Any],
        shock_payload: Mapping[str, Any],
    ) -> list[str]:
        configured = event.get("affected_markets", shock_payload.get("affected_asset_classes", []))
        if isinstance(configured, str):
            configured = [configured]
        if isinstance(configured, (list, tuple, set)) and configured:
            return [str(value).lower() for value in configured]
        return sorted({self._market_key(metadata) for metadata in self.instrument_metadata.values()})

    @staticmethod
    def _market_key(metadata: Mapping[str, Any]) -> str:
        return str(metadata.get("asset_class", metadata.get("symbol_type", "market"))).lower()


def state_scope_for(event: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Resolve state scope separately from the event's human-readable scope."""
    configured = event.get("state_scope", payload.get("state_scope"))
    if configured is not None:
        normalized = str(configured).strip().lower()
        if normalized in {"global", "market", "instrument"}:
            return normalized
        raise ValueError("state_scope must be global, market, or instrument")

    event_scope = str(payload.get("scope", event.get("scope", ""))).lower()
    shock_class = str(payload.get("shock_class", event.get("shock_class", ""))).lower()
    if event_scope == "micro" or shock_class == "non_systematic":
        return "instrument"
    if event_scope == "market_structure" and payload.get("affected_asset_classes"):
        return "market"
    return "global"


def effective_market_state(
    scoped_snapshot: Mapping[str, Any] | None,
    instrument: str,
) -> dict[str, Any]:
    """Merge only global plus this instrument's market and micro state."""
    if not isinstance(scoped_snapshot, Mapping):
        return {}
    global_state = scoped_snapshot.get("global", {})
    baseline = scoped_snapshot.get("baseline", {})
    if not isinstance(global_state, Mapping) or not isinstance(baseline, Mapping):
        return dict(global_state) if isinstance(global_state, Mapping) else {}

    effective = deepcopy(dict(global_state))
    market_keys = scoped_snapshot.get("instrument_market_keys", {})
    markets = scoped_snapshot.get("markets", {})
    if isinstance(market_keys, Mapping) and isinstance(markets, Mapping):
        market_state = markets.get(market_keys.get(instrument))
        if isinstance(market_state, Mapping):
            _apply_scope_delta(effective, market_state, baseline)

    instruments = scoped_snapshot.get("instruments", {})
    if isinstance(instruments, Mapping):
        instrument_state = instruments.get(instrument)
        if isinstance(instrument_state, Mapping):
            _apply_scope_delta(effective, instrument_state, baseline)
    return effective


def _apply_scope_delta(
    effective: dict[str, Any],
    scoped_state: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> None:
    for key, scoped_value in scoped_state.items():
        baseline_value = baseline.get(key)
        if not isinstance(scoped_value, (int, float)) or isinstance(scoped_value, bool):
            if scoped_value != baseline_value:
                effective[key] = deepcopy(scoped_value)
            continue
        if key == "liquidity_index":
            baseline_liquidity = max(0.01, safe_float(baseline_value, 1.0))
            ratio = float(scoped_value) / baseline_liquidity
            effective[key] = max(0.01, safe_float(effective.get(key), 1.0) * ratio)
            continue
        effective[key] = safe_float(effective.get(key), 0.0) + (
            float(scoped_value) - safe_float(baseline_value, 0.0)
        )
