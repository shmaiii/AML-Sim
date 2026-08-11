"""Generic runtime mechanism overrides for controlled experiments."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentPolicy:
    """Resolve optional mechanism overrides without changing normal policies.

    A mechanism absent from ``mechanism_overrides`` returns its naturally
    calculated value. This gives research scenarios one generic intervention
    point instead of adding an ablation-specific field to every agent class.
    """

    mechanism_overrides: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    frozen_strategy_fields: tuple[str, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: ExperimentPolicy | Mapping[str, Any] | None,
    ) -> ExperimentPolicy:
        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("aml_config.experiment must be a mapping")
        overrides = config.get("mechanism_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(
                "aml_config.experiment.mechanism_overrides must be a mapping"
            )
        normalized: dict[str, Mapping[str, Any]] = {}
        for mechanism, override in overrides.items():
            if not isinstance(override, Mapping):
                raise ValueError(
                    f"Experiment override '{mechanism}' must be a mapping"
                )
            normalized[str(mechanism)] = dict(override)
        frozen = config.get("frozen_strategy_fields", ())
        if not isinstance(frozen, (list, tuple)) or not all(
            isinstance(field_name, str) for field_name in frozen
        ):
            raise ValueError(
                "aml_config.experiment.frozen_strategy_fields must be a list of strings"
            )
        return cls(
            mechanism_overrides=normalized,
            frozen_strategy_fields=tuple(frozen),
        )

    def resolve_multiplier(self, mechanism: str, baseline: float) -> float:
        """Return a fixed experimental multiplier or the baseline value."""

        override = self.mechanism_overrides.get(mechanism)
        if override is None or "multiplier" not in override:
            return float(baseline)
        multiplier = float(override["multiplier"])
        if multiplier < 0:
            raise ValueError(
                f"Experiment multiplier for '{mechanism}' must be non-negative"
            )
        return multiplier

    def is_overridden(self, mechanism: str) -> bool:
        override = self.mechanism_overrides.get(mechanism)
        return override is not None and "multiplier" in override

    def resolve_value(
        self,
        mechanism: str,
        baseline: float,
        neutral: float,
    ) -> float:
        """Return the natural value or a multiple of its neutral value."""

        if not self.is_overridden(mechanism):
            return float(baseline)
        return float(neutral) * self.resolve_multiplier(mechanism, 1.0)

    def restore_frozen_fields(self, proposal: Any, current: Any) -> Any:
        """Keep selected strategy fields at their pre-proposal values."""

        if not self.frozen_strategy_fields:
            return proposal
        restored = copy(proposal)
        for field_name in self.frozen_strategy_fields:
            if hasattr(restored, field_name) and hasattr(current, field_name):
                setattr(restored, field_name, getattr(current, field_name))
        return restored
