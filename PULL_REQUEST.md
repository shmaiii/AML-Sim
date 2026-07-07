## Summary

This PR implements three interconnected architectural improvements to AML-Sim's agent system: a **robustness layer**, an **enriched behavioral profile system**, and a **pluggable alpha strategy framework**. These changes prepare each agent to be a fully LLM-controllable market participant while immediately improving simulation fidelity even without a connected LLM.

---

## Motivation

Before this PR, AML-Sim's agents had three structural weaknesses:

1. **Fragile execution** — no circuit breakers, drawdown protection, order-rate limiting, or graceful degradation on strategy-update failure. Validation only checked numeric bounds with no cross-field consistency.
2. **Dead profiles** — YAML-configured agent profiles contained behavioral traits that had no mechanical effect on trading behavior. There was no behavioral-state tracking (stress, confidence, streak).
3. **Hardcoded strategies** — only 3 alpha strategies (`momentum`, `mean_reversion`, `target_execution`) wired as if/elif branches. The LLM slow-loop could only tweak numeric fields, not propose or compose strategies.

---

## What Changed

### A. Robustness Layer (`aml_sim/agents/risk/` — new package)

| Component | File | Purpose |
|---|---|---|
| `RiskManager` | `risk/risk_manager.py` | Per-agent risk controller: max-drawdown halts, position-size limits, order-rate throttling, circuit breakers with tick-based cooldown, single-order value caps. Gates every order via `check_order()` before submission. |
| `FallbackChain` | `risk/fallback.py` | Graceful degradation: after N consecutive slow-loop failures, automatically restores a safe-strategy snapshot. Resets on first success. |

**Immediate effect (no LLM required):** agents now self-halt on excessive drawdown, reject orders when circuit-breakers trip, and recover to a known-good strategy state if the strategy-update path fails.

### B. Behavioral Profile System (`aml_sim/agents/profile/` — new package, `models/profile.py` — rewritten)

| Component | File | Purpose |
|---|---|---|
| `BehavioralTraits` (10 dimensions) | `models/profile.py` | risk_aversion, patience, conviction, adaptability, discipline, aggression, social_influence, loss_aversion, overconfidence, recency_bias — each normalized to [0, 1]. |
| `ProfileModulator` | `profile/modulator.py` | Translates traits → strategy-parameter adjustments (e.g. high risk_aversion → smaller max_position; high aggression → higher trade_probability). Traits, NOT dead metadata. |
| `BehavioralDynamics` | `profile/behavioral_dynamics.py` | Per-tick state updates: PnL win/loss streak, stress level (EMA), confidence drift, market-regime detection (trending / ranging / volatile). |
| `BehavioralState` | `profile/behavioral_dynamics.py` | Runtime emotional-state snapshot fed into both observation context (for LLM) and ProfileModulator (for mechanical execution). |
| 5 new role profiles | `models/profile.py` | `TrendFollowerProfile`, `ArbitrageurProfile`, `HedgeProfile`, `PassiveIndexProfile` + existing profiles now carry full trait defaults with sensible per-role values. |

**Immediate effect:** two retail traders with different `risk_aversion` values now produce measurably different order sizes and trade probabilities. Stress from consecutive losses mechanically tightens position limits.

### C. Pluggable Strategy Framework (`aml_sim/agents/strategy/` — substantially expanded)

| Component | File | Purpose |
|---|---|---|
| `AlphaStrategy` protocol + `AlphaContext` / `AlphaSignal` | `strategy/alpha.py` | Standardized interface for all alpha strategies. Each strategy receives a structured context and returns a typed signal (direction, strength, confidence, horizon, reason, suggested order type). |
| `StrategyRegistry` | `strategy/registry.py` | Global name→class registry. LLM can enable/disable strategies by name. 6 strategies auto-registered on import. |
| `CompositeAlphaStrategy` | `strategy/composite.py` | Blends N strategies via 4 modes: `weighted_sum`, `vote`, `unanimous`, `priority_cascade`. LLM can change blend mode and per-strategy weights at runtime. |
| `StrategyPerformance` | `strategy/performance.py` | Per-strategy tracking: signals generated, signals acted on, win rate, profit factor, cumulative PnL, avg holding ticks. Fed into observation context for LLM visibility. |

**New alpha strategies** (5 new files under `strategy/`):

| Strategy | Logic |
|---|---|
| `momentum` | Ported from hardcoded logic — follows price trends |
| `mean_reversion` | Ported — bets against short-term deviations |
| `breakout` | **New** — trades when price breaks above/below recent range |
| `volatility_regime` | **New** — adjusts exposure based on vol regime |
| `event_driven` | **New** — converts AML shock events into directional signals |
| `passive_benchmark` | **New** — control-group strategy that rarely trades |

### D. Enhanced LLM Slow-Loop Contract (`strategy/llm_slow_strategy.py`)

The LLM response contract now has three sections (previously: one flat param dict):

```json
{
  "risk_mode": "conservative",
  "confidence": 0.7,
  "reason": "Volatility spiking — reducing exposure.",
  "strategy_config": {
    "active_strategies": ["momentum", "volatility_regime"],
    "blend_mode": "priority_cascade",
    "strategy_weights": {"momentum": 0.4, "volatility_regime": 0.6}
  },
  "parameter_updates": {
    "max_position": 300,
    "entry_threshold": 0.004
  },
  "risk_overrides": {
    "max_drawdown_pct": 0.10,
    "cooldown_ticks": 6
  }
}
```

The `BaseAMLAgent.run_slow_loop()` now handles all three sections: strategy config changes are applied to the composite-strategy engine, parameter updates are validated and merged, and risk overrides are forwarded to the RiskManager.

### E. Core Agent Restructuring (`agents/base.py` + all 5 concrete agents)

`BaseAMLAgent` now orchestrates the full per-tick pipeline:

```
handle_time_tick()
  ├─ _pre_tick_health_check()       ← risk_manager.check_portfolio_health()
  ├─ fallback_chain check            ← restore safe state if degraded
  ├─ behavioral_dynamics.update()    ← track PnL streak, stress, regime
  ├─ build_observation()             ← now includes risk_status, strategy_perf, behavioral_state
  ├─ run_slow_loop()                 ← enhanced LLM response handling
  ├─ profile_modulator.modulate()    ← traits → param adjustments
  └─ run_fast_loop()                 ← risk-gated order placement
```

All concrete agents now use `profile_modulator.modulate()` in their fast loops. The institutional trader was rewritten to use `CompositeAlphaStrategy` + `StrategyRegistry` instead of hardcoded if/elif.

### F. Launcher & Scenario YAML (`launcher.py`, `scenario.py`, `scenarios/*.yaml`)

New YAML fields supported per agent:

```yaml
parameters:
  risk_limits:
    max_drawdown_pct: 0.2
    max_order_rate: 30
    max_consecutive_rejections: 6
    cooldown_ticks: 4
  behavioral_traits:
    risk_aversion: 0.55
    patience: 0.8
    conviction: 0.65
  alpha_strategies: [momentum, mean_reversion, breakout]
  blend_mode: weighted_sum
```

Deterministic random seeds are now assigned to all stochastic agents (retail, informed, liquidity taker) from `run_id + agent_id`, enabling reproducible runs.

---

## Files Changed

**Modified (13 files, +1005 −197 lines):**
`base.py`, `institutional_trader.py`, `market_maker_trader.py`, `retail_trader.py`, `informed_trader.py`, `liquidity_taker.py`, `models/profile.py`, `models/state.py`, `strategy/llm_slow_strategy.py`, `strategy/validator.py`, `context/observation.py`, `launcher.py`, `scenarios/aml_agent_infra_smoke.yaml`

**New (15 files, ~1200 lines):**
`risk/__init__.py`, `risk/risk_manager.py`, `risk/fallback.py`, `profile/__init__.py`, `profile/modulator.py`, `profile/behavioral_dynamics.py`, `strategy/alpha.py`, `strategy/registry.py`, `strategy/composite.py`, `strategy/performance.py`, `strategy/alpha_momentum.py`, `strategy/alpha_mean_reversion.py`, `strategy/alpha_breakout.py`, `strategy/alpha_volatility.py`, `strategy/alpha_event_driven.py`, `strategy/alpha_passive.py`

---

## Verification

- All 3 existing scenarios pass `--dry-run` without errors
- Full end-to-end simulation (`aml_agent_infra_smoke.yaml`) runs successfully with all 7 agent types
- Action reports now include `risk_summary`, `fallback_summary`, `behavioral_summary`, and `strategy_performance`
- No regressions: existing agent configs produce identical behavior when new fields are omitted (all new constructor parameters have safe defaults)

---

## What This Enables Next

To connect a real LLM, implement a single adapter class:

```python
class YourLLMClient:
    async def complete_json(self, context: dict) -> dict:
        # context contains observation + profile + strategies + risk status
        # return {strategy_config, parameter_updates, risk_overrides}
        ...
```

The entire pipeline — context assembly, response parsing, strategy switching, parameter validation, risk-override application, and fallback-on-failure — is already in place.
