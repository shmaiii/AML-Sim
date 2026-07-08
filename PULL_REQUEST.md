## Summary

This PR implements three interconnected architectural improvements to AML-Sim's agent system: a **robustness layer**, an **enriched behavioral profile system**, and a **pluggable alpha strategy framework**. These changes make each agent a fully LLM-controllable market participant while immediately improving simulation fidelity even with the static LLM client.

All upstream changes ( `LLM_STRATEGY_ROLE`, `_build_slow_strategist()`, `create_llm_strategist()`, `OpenAIJSONLLMClient`, `aml_config` ) are preserved and layered on top of.

---

## What Changed

### A. Robustness Layer (`aml_sim/agents/risk/` — new package)

| Component | Purpose |
|---|---|
| `RiskManager` | Per-agent risk controller: drawdown halts, position-size limits, order-rate throttling, circuit breakers with tick-based cooldown, single-order value caps. Gates every order via `check_order()` before submission. **Market orders are now properly checked** using `last_price` as a value estimate. |
| `FallbackChain` | Graceful degradation: after N consecutive slow-loop failures, automatically restores a safe-strategy snapshot. Resets on first success. |

### B. Behavioral Profile System (`aml_sim/agents/profile/` — new package, `models/profile.py` — rewritten)

| Component | Purpose |
|---|---|
| `BehavioralTraits` (10 dimensions) | risk_aversion, patience, conviction, adaptability, discipline, aggression, social_influence, loss_aversion, overconfidence, recency_bias — each normalized to [0, 1] |
| `ProfileModulator` | Translates traits into mechanical strategy-parameter adjustments (e.g., risk_aversion=0.9 → max_position reduced by 40%) |
| `BehavioralDynamics` | Per-tick state updates: PnL win/loss streak, stress level (EMA), confidence drift, market-regime detection |
| `BehavioralState` | Runtime emotional-state snapshot fed into observation context and ProfileModulator |
| 5 new role profiles | `TrendFollowerProfile`, `ArbitrageurProfile`, `HedgeProfile`, `PassiveIndexProfile` + existing profiles now carry full trait defaults |

**Verified:** A risk-averse institution (risk_aversion=0.9) gets `max_position=300` while a risk-seeking one (risk_aversion=0.1) gets `max_position=700`, from the same base of 500.

### C. Pluggable Strategy Framework (`aml_sim/agents/strategy/` — substantially expanded)

| Component | Purpose |
|---|---|
| `AlphaStrategy` protocol + `AlphaContext` / `AlphaSignal` | Standardized interface — each strategy receives structured context, returns typed signal (direction, strength, confidence, horizon, reason, suggested order type) |
| `StrategyRegistry` | Global name→class registry. LLM can enable/disable strategies by name. 6 strategies auto-registered. |
| `CompositeAlphaStrategy` | Blends N strategies via 4 modes: `weighted_sum`, `vote`, `unanimous`, `priority_cascade` |
| `StrategyPerformance` | Per-strategy tracking: win rate, profit factor, cumulative PnL, avg holding ticks |

**6 alpha strategies** registered:

| Strategy | Logic |
|---|---|
| `momentum` | Ported from hardcoded logic — follows price trends |
| `mean_reversion` | Ported — bets against short-term deviations |
| `breakout` | **New** — trades when price breaks above/below recent range |
| `volatility_regime` | **New** — adjusts exposure based on vol regime |
| `event_driven` | **New** — converts AML shock events into directional signals |
| `passive_benchmark` | **New** — control-group strategy (rarely trades) |

### D. Enhanced LLM Slow-Loop Contract

The LLM response contract now has three sections. The upstream `strategy_updates` key is preserved for backward compatibility; `strategy_config` and `risk_overrides` are layered on top:

```json
{
  "strategy_updates": {"max_position": 300, "entry_threshold": 0.004},
  "strategy_config": {
    "active_strategies": ["momentum", "volatility_regime"],
    "blend_mode": "priority_cascade",
    "strategy_weights": {"momentum": 0.4, "volatility_regime": 0.6}
  },
  "risk_overrides": {"max_drawdown_pct": 0.10, "cooldown_ticks": 6},
  "confidence": 0.7,
  "reason": "Volatility spiking — reducing exposure."
}
```

The `OpenAIJSONLLMClient` (upstream) and `StaticJSONLLMClient` both support this contract. The system prompt for OpenAI has been updated to instruct the model about the enhanced response shape.

### E. Core Agent Restructuring

`BaseAMLAgent.handle_time_tick()` now orchestrates:

```
handle_time_tick()
  ├── risk_manager.check_portfolio_health()   → HALTED? skip everything
  ├── fallback_chain check                     → restore safe state if degraded
  ├── behavioral_dynamics.update()             → track streaks, stress, regime
  ├── build_observation()                      → now includes risk/behavioral/perf
  ├── run_slow_loop()                          → apply strategy_config + risk_overrides
  └── run_fast_loop()                          → risk-gated via risk_manager.check_order()
```

All concrete agents use `profile_modulator.modulate()` in their fast loops. The institutional trader was rewritten to use `CompositeAlphaStrategy` + `StrategyRegistry` instead of hardcoded if/elif.

### F. Review Comments Addressed

| # | Issue | Fix |
|---|---|---|
| 1 | Market orders bypass `max_order_value_pct` (price=None) | `check_order()` now accepts `last_price` fallback; `place_order()` passes `self.prices[instrument]` |
| 2 | `mod_state` computed but unused in base agent | Removed unused variable. Each concrete agent calls `profile_modulator.modulate()` in its fast loop, including institutional trader |
| 3 | `snapshot()` drawdown uses stale initial-value formula | Changed to `(1 - current/peak) * 100` using `last_portfolio_value` updated by `check_portfolio_health()` |
| 4 | Int risk fields cast to float in `_apply_risk_overrides()` | `max_order_rate`, `cooldown_ticks`, `max_consecutive_rejections` now use `max(1, int(value))` |

---

## Files Changed

**Modified (13 files, ~+1000/−200 lines):**
`base.py`, `institutional_trader.py`, `market_maker_trader.py`, `retail_trader.py`, `informed_trader.py`, `liquidity_taker.py`, `models/profile.py`, `models/state.py`, `strategy/llm_slow_strategy.py`, `strategy/validator.py`, `context/observation.py`, `launcher.py`, `scenarios/aml_agent_infra_smoke.yaml`

**New (15 files, ~1200 lines):**
`risk/__init__.py`, `risk/risk_manager.py`, `risk/fallback.py`, `profile/__init__.py`, `profile/modulator.py`, `profile/behavioral_dynamics.py`, `strategy/alpha.py`, `strategy/registry.py`, `strategy/composite.py`, `strategy/performance.py`, `strategy/alpha_momentum.py`, `strategy/alpha_mean_reversion.py`, `strategy/alpha_breakout.py`, `strategy/alpha_volatility.py`, `strategy/alpha_event_driven.py`, `strategy/alpha_passive.py`

---

## Verification

### Automated (27 tests — all passing)
- Risk manager: drawdown halts, rate limits, circuit breakers, market-order value gating, snapshot accuracy
- Strategy system: 6 strategies registered, all produce valid signals, 4 blend modes, performance tracking
- Profile system: trait loading, modulation (averse=300 vs seeking=700), non-mutation, behavioral dynamics, regime detection
- LLM strategist: `create_llm_strategist()` works for all 5 roles, all static responses have full enhanced contract

### End-to-End Simulation
- Full `aml_agent_infra_smoke.yaml` run completed successfully
- **0 ERROR entries** across all 8 agent logs
- 4 trades executed on AAPL order book
- All 8 agent action reports contain `risk_summary`, `fallback_summary`, `behavioral_summary`, `strategy_performance`
- Health status: `healthy` for all active agents

### Backward Compatibility
- All 3 existing scenarios pass `--dry-run`
- Upstream patterns preserved: `LLM_STRATEGY_ROLE`, `_build_slow_strategist()`, `create_llm_strategist()`, `OpenAIJSONLLMClient`, `aml_config`
- `strategy_updates` key still accepted alongside new `strategy_config`/`risk_overrides`
- New constructor parameters have safe defaults (e.g., `RiskManager()` with permissive limits)
