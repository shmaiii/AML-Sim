"""Shock schema and effect helpers for AML simulations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from typing import Any, Mapping


EFFECT_DEFAULTS: dict[str, float] = {
    "fundamental_price_shift": 0.0,
    "order_arrival_multiplier": 1.0,
    "risk_limit_multiplier": 1.0,
    "liquidity_multiplier": 1.0,
    "volatility_multiplier": 1.0,
    "spread_multiplier": 1.0,
    "price_impact_multiplier": 1.0,
    "rate_shift_bps": 0.0,
    "yield_shift_bps": 0.0,
    "funding_spread_bps": 0.0,
    "credit_spread_bps": 0.0,
    "correlation_shift": 0.0,
    "sentiment_shift": 0.0,
    "risk_aversion_shift": 0.0,
}

EFFECT_KEYS = tuple(EFFECT_DEFAULTS.keys())

ASSET_CLASS_ALIASES = {
    "equity": "stock",
    "equities": "stock",
    "stocks": "stock",
    "option": "option",
    "options": "option",
    "future": "future",
    "futures": "future",
    "bond": "bond",
    "bonds": "bond",
    "fixed_income": "bond",
    "etfs": "etf",
    "fund": "etf",
    "funds": "etf",
}


def stable_seed(*parts: Any) -> int:
    """Return a reproducible 32-bit seed from arbitrary identity parts."""
    material = ":".join(str(part) for part in parts)
    digest = sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_instrument_metadata(
    instruments: list[str],
    exchanges_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the small cross-asset metadata map the shock agent needs."""
    metadata: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        inst_cfg = exchanges_config.get(instrument, {})
        if not isinstance(inst_cfg, Mapping):
            inst_cfg = {}
        symbol_type = str(inst_cfg.get("symbol_type", "stock")).lower()
        asset_class = normalize_asset_class(
            str(inst_cfg.get("asset_class", symbol_type)).lower()
        )
        metadata[instrument] = {
            "instrument": instrument,
            "symbol_type": symbol_type,
            "asset_class": asset_class,
            "sector": inst_cfg.get("sector"),
            "region": inst_cfg.get("region"),
            "currency": inst_cfg.get("currency", "USD"),
        }
    return metadata


def normalize_asset_class(value: str | None) -> str:
    """Normalize common asset-class labels used in scenario YAML."""
    clean = str(value or "stock").strip().lower()
    return ASSET_CLASS_ALIASES.get(clean, clean)


def build_shock_payload(
    event_id: str,
    event: Mapping[str, Any],
    clock_payload: Mapping[str, Any],
    *,
    default_duration_ticks: int,
    instrument_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    market_state: Mapping[str, Any] | None = None,
    phase: str = "active",
    trigger_type: str = "scheduled",
) -> dict[str, Any]:
    """Convert a scenario shock config into the payload agents observe."""
    event_map = dict(event)
    metadata = {
        str(key): dict(value)
        for key, value in (instrument_metadata or {}).items()
        if isinstance(value, Mapping)
    }
    current_tick = safe_int(clock_payload.get("tick_id"))
    effective_tick = event_effective_tick(event_map, current_tick)
    duration_value = safe_int(event_map.get("duration_ticks"), default_duration_ticks)
    duration_ticks = max(0, duration_value if duration_value is not None else default_duration_ticks)
    expiry_tick = (
        effective_tick + duration_ticks
        if effective_tick is not None and duration_ticks is not None
        else None
    )

    current_time = clock_payload.get("current_time")
    timestamp = current_time.isoformat() if isinstance(current_time, datetime) else current_time
    severity = clamp(safe_float(event_map.get("severity", 0.5), 0.5), 0.0, 1.0)
    direction = clamp(safe_float(event_map.get("direction", 0.0), 0.0), -1.0, 1.0)
    shock_type = str(event_map.get("shock_type", event_map.get("type", "generic_shock")))

    affected_instruments = affected_instrument_list(event_map, metadata)
    affected_asset_classes = affected_asset_class_list(event_map, metadata, affected_instruments)
    base_effects = base_effects_from_event(event_map, severity, direction)
    instrument_effects = build_instrument_effects(
        event_map,
        base_effects,
        metadata,
        affected_instruments,
    )

    payload = {
        "event_type": "AML_SHOCK",
        "shock_id": event_id,
        "shock_type": shock_type,
        "shock_class": event_map.get("shock_class")
        or event_map.get("class")
        or infer_shock_class(shock_type),
        "scope": event_map.get("scope") or event_map.get("shock_scope") or infer_scope(shock_type),
        "phase": phase,
        "trigger_type": event_map.get("trigger_type", trigger_type),
        "visibility": event_map.get("visibility", "public"),
        "information_state": event_map.get("information_state", event_map.get("visibility", "public")),
        "scheduled": bool(event_map.get("scheduled", trigger_type == "scheduled")),
        "expected_probability": clamp(
            safe_float(event_map.get("expected_probability", 1.0 if trigger_type == "scheduled" else 0.0), 0.0),
            0.0,
            1.0,
        ),
        "surprise": clamp(
            safe_float(event_map.get("surprise", 0.25 if trigger_type == "scheduled" else 1.0), 1.0),
            0.0,
            1.0,
        ),
        "timestamp": timestamp,
        "tick_id": current_tick,
        "emitted_tick_id": current_tick,
        "effective_tick_id": effective_tick,
        "expiry_tick_id": expiry_tick,
        "duration_ticks": duration_ticks,
        "severity": severity,
        "direction": direction,
        "affected_instruments": affected_instruments,
        "affected_asset_classes": affected_asset_classes,
        "instrument_effects": instrument_effects,
        "transmission": deepcopy(event_map.get("transmission", {})),
        "market_state": deepcopy(dict(market_state or {})),
        "message": event_map.get("message", event_map.get("narrative", "")),
    }
    payload.update(round_effects(base_effects))
    return payload


def event_effective_tick(
    event: Mapping[str, Any],
    fallback_tick: int | None,
) -> int | None:
    """Return the tick where an event becomes economically active."""
    tick = event.get("effective_tick", event.get("effective_tick_id"))
    if tick is None:
        tick = event.get("tick", event.get("tick_id"))
    return safe_int(tick, fallback_tick)


def affected_instrument_list(
    event: Mapping[str, Any],
    instrument_metadata: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Resolve an event's instrument universe."""
    configured = normalize_list(event.get("affected_instruments", event.get("instruments")))
    if configured:
        return configured
    asset_classes = {
        normalize_asset_class(item)
        for item in normalize_list(event.get("affected_asset_classes", event.get("asset_classes")))
    }
    if asset_classes:
        return [
            instrument
            for instrument, metadata in instrument_metadata.items()
            if normalize_asset_class(str(metadata.get("asset_class", metadata.get("symbol_type", ""))))
            in asset_classes
        ]
    return list(instrument_metadata.keys())


def affected_asset_class_list(
    event: Mapping[str, Any],
    instrument_metadata: Mapping[str, Mapping[str, Any]],
    affected_instruments: list[str],
) -> list[str]:
    """Resolve asset classes affected by a shock."""
    configured = normalize_list(event.get("affected_asset_classes", event.get("asset_classes")))
    if configured:
        return [normalize_asset_class(item) for item in configured]

    classes = []
    for instrument in affected_instruments:
        metadata = instrument_metadata.get(instrument, {})
        asset_class = normalize_asset_class(
            str(metadata.get("asset_class", metadata.get("symbol_type", "stock")))
        )
        if asset_class not in classes:
            classes.append(asset_class)
    return classes


def base_effects_from_event(
    event: Mapping[str, Any],
    severity: float,
    direction: float,
) -> dict[str, float]:
    """Compute top-level effects with backwards-compatible defaults."""
    effects = dict(EFFECT_DEFAULTS)
    nested_effects = event.get("effects", {})
    if not isinstance(nested_effects, Mapping):
        nested_effects = {}

    effects["fundamental_price_shift"] = configured_float(
        event,
        nested_effects,
        "fundamental_price_shift",
        direction * severity * safe_float(event.get("fundamental_price_shift_per_severity"), 1.0),
    )
    effects["order_arrival_multiplier"] = configured_float(
        event,
        nested_effects,
        "order_arrival_multiplier",
        1.0 + (severity * 1.25),
    )
    if has_effect_key(event, nested_effects, "risk_limit_multiplier"):
        effects["risk_limit_multiplier"] = configured_float(event, nested_effects, "risk_limit_multiplier", 1.0)
    elif direction < 0:
        effects["risk_limit_multiplier"] = 1.0 - (severity * 0.55)
    else:
        effects["risk_limit_multiplier"] = 1.0 + (severity * 0.25)
    effects["liquidity_multiplier"] = configured_float(
        event,
        nested_effects,
        "liquidity_multiplier",
        1.0 - (severity * 0.5),
    )
    effects["volatility_multiplier"] = configured_float(
        event,
        nested_effects,
        "volatility_multiplier",
        1.0 + (severity * safe_float(event.get("volatility_multiplier_per_severity"), 0.75)),
    )
    effects["spread_multiplier"] = configured_float(
        event,
        nested_effects,
        "spread_multiplier",
        1.0 + (severity * 0.5),
    )
    effects["price_impact_multiplier"] = configured_float(
        event,
        nested_effects,
        "price_impact_multiplier",
        1.0 + (severity * 0.5),
    )
    effects["sentiment_shift"] = configured_float(
        event,
        nested_effects,
        "sentiment_shift",
        direction * severity,
    )
    effects["risk_aversion_shift"] = configured_float(
        event,
        nested_effects,
        "risk_aversion_shift",
        severity if direction < 0 else -0.25 * severity,
    )

    for key in EFFECT_KEYS:
        if key in {
            "fundamental_price_shift",
            "order_arrival_multiplier",
            "risk_limit_multiplier",
            "liquidity_multiplier",
            "volatility_multiplier",
            "spread_multiplier",
            "price_impact_multiplier",
            "sentiment_shift",
            "risk_aversion_shift",
        }:
            continue
        effects[key] = configured_float(event, nested_effects, key, EFFECT_DEFAULTS[key])

    return clamp_effects(effects)


def build_instrument_effects(
    event: Mapping[str, Any],
    base_effects: Mapping[str, float],
    instrument_metadata: Mapping[str, Mapping[str, Any]],
    affected_instruments: list[str],
) -> dict[str, dict[str, float]]:
    """Build instrument-level effects from global, asset-class, and symbol overrides."""
    if not instrument_metadata:
        return {}

    asset_class_effects = event.get("asset_class_effects", event.get("cross_asset_effects", {}))
    if not isinstance(asset_class_effects, Mapping):
        asset_class_effects = {}
    per_instrument_effects = event.get("per_instrument_effects", event.get("instrument_effects", {}))
    if not isinstance(per_instrument_effects, Mapping):
        per_instrument_effects = {}

    affected = set(affected_instruments or instrument_metadata.keys())
    instrument_effects: dict[str, dict[str, float]] = {}
    for instrument, metadata in instrument_metadata.items():
        if instrument not in affected:
            continue
        effects = dict(base_effects)
        asset_class = normalize_asset_class(
            str(metadata.get("asset_class", metadata.get("symbol_type", "stock")))
        )
        symbol_type = normalize_asset_class(str(metadata.get("symbol_type", asset_class)))

        for key in (asset_class, symbol_type):
            raw_effects = asset_class_effects.get(key)
            if isinstance(raw_effects, Mapping):
                merge_effects(effects, raw_effects)

        raw_instrument_effects = per_instrument_effects.get(instrument)
        if isinstance(raw_instrument_effects, Mapping):
            merge_effects(effects, raw_instrument_effects)

        instrument_effects[instrument] = round_effects(clamp_effects(effects))
    return instrument_effects


def resolve_event_effect(
    event: Mapping[str, Any],
    instrument: str,
) -> dict[str, float]:
    """Return the best available effect map for one instrument."""
    effects = dict(EFFECT_DEFAULTS)
    merge_effects(effects, event)
    nested = event.get("effects")
    if isinstance(nested, Mapping):
        merge_effects(effects, nested)
    instrument_effects = event.get("instrument_effects")
    if isinstance(instrument_effects, Mapping):
        specific = instrument_effects.get(instrument)
        if isinstance(specific, Mapping):
            merge_effects(effects, specific)
    return clamp_effects(effects)


def infer_shock_class(shock_type: str) -> str:
    """Coarse taxonomy: systematic macro shock or idiosyncratic/micro shock."""
    clean = shock_type.lower()
    systematic_terms = (
        "rate",
        "macro",
        "inflation",
        "policy",
        "central_bank",
        "recession",
        "systematic",
    )
    if any(term in clean for term in systematic_terms):
        return "systematic"
    return "non_systematic"


def infer_scope(shock_type: str) -> str:
    """Human-readable scope for dashboards and LLM observations."""
    clean = shock_type.lower()
    if any(term in clean for term in ("rate", "macro", "policy", "inflation")):
        return "macro"
    if any(term in clean for term in ("liquidity", "funding", "credit")):
        return "market_structure"
    return "micro"


def configured_float(
    event: Mapping[str, Any],
    effects: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    """Read an effect key from top-level or nested effects config."""
    if key in event:
        return safe_float(event.get(key), default)
    if key in effects:
        return safe_float(effects.get(key), default)
    return default


def has_effect_key(event: Mapping[str, Any], effects: Mapping[str, Any], key: str) -> bool:
    return key in event or key in effects


def merge_effects(target: dict[str, float], source: Mapping[str, Any]) -> None:
    """Merge only known numeric effect keys into an effect map."""
    for key in EFFECT_KEYS:
        if key in source:
            target[key] = safe_float(source.get(key), target[key])


def clamp_effects(effects: Mapping[str, float]) -> dict[str, float]:
    """Clamp multipliers to ranges that keep toy simulations numerically sane."""
    clean = dict(effects)
    for key in (
        "order_arrival_multiplier",
        "volatility_multiplier",
        "spread_multiplier",
        "price_impact_multiplier",
    ):
        clean[key] = clamp(safe_float(clean.get(key), EFFECT_DEFAULTS[key]), 0.05, 5.0)
    for key in ("risk_limit_multiplier", "liquidity_multiplier"):
        clean[key] = clamp(safe_float(clean.get(key), EFFECT_DEFAULTS[key]), 0.05, 2.0)
    clean["correlation_shift"] = clamp(safe_float(clean.get("correlation_shift"), 0.0), -1.0, 1.0)
    clean["sentiment_shift"] = clamp(safe_float(clean.get("sentiment_shift"), 0.0), -1.0, 1.0)
    clean["risk_aversion_shift"] = clamp(safe_float(clean.get("risk_aversion_shift"), 0.0), -1.0, 1.0)
    return clean


def round_effects(effects: Mapping[str, float]) -> dict[str, float]:
    return {key: round(safe_float(effects.get(key), EFFECT_DEFAULTS[key]), 4) for key in EFFECT_KEYS}


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
