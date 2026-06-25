"""Shared serialization helpers for AML-Sim.

This module is the single source of truth for serializing agent state, strategy
state, memory events, observations, and action events to JSON-safe dicts. It
replaces the duplicated _serialize_value helpers that previously lived in
base.py, observation.py, and memory.py.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value into a JSON-safe form.

    Handles datetime, dataclass, Mapping, list, tuple, and primitive types.
    Types that are not explicitly handled (set, bytes, Decimal, UUID, etc.)
    are converted to str to avoid JSON serialization failures.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return serialize_mapping(asdict(value))
    if isinstance(value, Mapping):
        return serialize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    # Fallback: convert unhandled types to string (set, Decimal, UUID, bytes, etc.)
    try:
        return str(value)
    except Exception:
        return repr(value)


def serialize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize all values in a mapping, converting keys to strings."""
    return {str(key): serialize_value(item) for key, item in mapping.items()}
