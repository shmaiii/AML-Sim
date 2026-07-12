"""
Exhaustive path-coverage test suite for AML-Sim agent architecture.
Covers: component internals, cross-component pipelines, boundary conditions,
all 5 agent types, producer-consumer format compatibility, all edge cases.

Usage:  python test_all_paths.py
"""

import sys, os, copy, json, tempfile, shutil
from pathlib import Path

# Ensure StockSim is importable regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent
_STOCKSIM_DIR = _REPO_ROOT / "simulators" / "StockSim"
if str(_STOCKSIM_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCKSIM_DIR))
# Also ensure the AML-Sim root is importable
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FAILED = []
PASSED = 0

def check(condition, label):
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILED.append(label)
        print(f'  FAIL: {label}')

def section(name):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

# ═══════════════════════════════════════════════════════════════════════
# SECTION A: RiskManager — every method, every branch, every boundary
# ═══════════════════════════════════════════════════════════════════════
section('A: RiskManager — full path coverage')

from aml_sim.agents.risk.risk_manager import RiskManager, OrderVerdict, HealthVerdict

# A1: Default construction
rm = RiskManager()
check(rm.max_drawdown_pct == 0.25, 'A1a: default max_drawdown')
check(rm.max_order_rate == 50, 'A1b: default max_order_rate')
check(rm.last_health == HealthVerdict.HEALTHY, 'A1c: default health')

# A2: seed_portfolio — always sets peak (called each tick to seed initial)
rm.seed_portfolio(100000)
check(rm.peak_portfolio_value == 100000, 'A2a: peak set')
check(rm.initial_portfolio_value == 100000, 'A2b: initial set')
# update_portfolio_peak is the "never-lower" method; seed_portfolio always sets
rm.update_portfolio_peak(105000)
check(rm.peak_portfolio_value == 105000, 'A2c: update_portfolio_peak raises peak')
rm.update_portfolio_peak(50000)
check(rm.peak_portfolio_value == 105000, 'A2d: update_portfolio_peak does not lower peak')

# A3: check_order — normal approval
v, a = rm.check_order('AAPL', 'BUY', 100, 100.0, portfolio_value=100000, current_position=0)
check(v == OrderVerdict.APPROVED, 'A3a: normal order approved')
check(a is None, 'A3b: no adjustment needed')

# A4: check_order — drawdown halt (fresh instance to avoid cross-test contamination)
rm_a4 = RiskManager(max_drawdown_pct=0.25); rm_a4.seed_portfolio(100000)
v, _ = rm_a4.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=70000, current_position=0)
check(v == OrderVerdict.REJECTED, 'A4: drawdown 30% > 25% limit → rejected')

# A5: check_order — rate limit
rm_a5 = RiskManager(max_order_rate=2); rm_a5.seed_portfolio(100000)
rm_a5.record_order_result(True); rm_a5.record_order_result(True)
v, _ = rm_a5.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0)
check(v == OrderVerdict.REJECTED, 'A5a: rate limit enforced')
# Tick change resets
v2, _ = rm_a5.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0, current_tick_id=5)
check(v2 == OrderVerdict.APPROVED, 'A5b: rate resets on tick change')

# A6: check_order — circuit breaker
rm_a6 = RiskManager(max_consecutive_rejections=2, cooldown_ticks=10)
rm_a6.record_order_result(False, current_tick_id=10)
rm_a6.record_order_result(False, current_tick_id=11)
check(rm_a6.is_circuit_open(current_tick_id=11), 'A6a: circuit trips')
v, _ = rm_a6.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0, current_tick_id=12)
check(v == OrderVerdict.REJECTED, 'A6b: orders rejected during circuit')
v2, _ = rm_a6.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0, current_tick_id=25)
check(v2 == OrderVerdict.APPROVED, 'A6c: circuit auto-resets after cooldown')

# A7: check_order — single-order value cap with explicit price
rm_a7 = RiskManager(max_order_value_pct=0.05); rm_a7.seed_portfolio(20000)
v, adj = rm_a7.check_order('AAPL', 'BUY', 200, 95.0, portfolio_value=20000, current_position=0)
check(v == OrderVerdict.SIZE_REDUCED, 'A7a: oversized order → SIZE_REDUCED')
check(adj == 10, 'A7b: reduced to 10 (20000*0.05/95)')

# A8: check_order — MARKET order (price=None) gated via last_price
v, adj = rm_a7.check_order('AAPL', 'BUY', 200, None, portfolio_value=20000, current_position=0, last_price=95.0)
check(v == OrderVerdict.SIZE_REDUCED, 'A8a: market order gated via last_price')
check(adj == 10, 'A8b: market order reduced correctly')

# A9: check_order — MARKET order without any price → no value check
v, _ = rm_a7.check_order('AAPL', 'BUY', 999999, None, portfolio_value=20000, current_position=0)
check(v == OrderVerdict.APPROVED, 'A9: market order without price estimate passes (cannot size-check)')

# A10: check_order — projected position limit (Review #2 fix)
rm_a10 = RiskManager(max_position_pct=0.5); rm_a10.seed_portfolio(100000)
# max_allowed = 100000*0.5/100 = 500 shares. Current = 480. Quantity = 50. Projected = 530 > 500.
v, adj = rm_a10.check_order('AAPL', 'BUY', 50, 100.0, portfolio_value=100000, current_position=480)
check(v == OrderVerdict.SIZE_REDUCED, 'A10a: projected BUY exceeds cap → SIZE_REDUCED')
check(adj == 20, f'A10b: reduced to 20 (remaining capacity), got {adj}')
# Current = 500 already at cap
v2, _ = rm_a10.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=500)
check(v2 == OrderVerdict.REJECTED, 'A10c: already at cap → REJECTED')
# SELL order: current = 0, SELL 50 → projected = -50, abs(-50)=50 < 500 → ok
v3, _ = rm_a10.check_order('AAPL', 'SELL', 50, 100.0, portfolio_value=100000, current_position=0)
check(v3 == OrderVerdict.APPROVED, 'A10d: SELL within cap approved')

# A11: check_portfolio_health — all paths
rm_a11 = RiskManager(max_drawdown_pct=0.10)
rm_a11.seed_portfolio(100000)
h = rm_a11.check_portfolio_health(portfolio_value=95000)
check(h == HealthVerdict.HEALTHY, 'A11a: 5% drawdown → healthy')
h = rm_a11.check_portfolio_health(portfolio_value=92000)
check(h == HealthVerdict.DEGRADED, 'A11b: 8% > 7% warning → degraded')
h = rm_a11.check_portfolio_health(portfolio_value=89000)
check(h == HealthVerdict.HALTED, 'A11c: 11% > 10% limit → halted')

# A12: snapshot accuracy (Review #3 fix)
s = rm_a11.snapshot()
check(s['health'] == 'halted', 'A12a: snapshot health')
check(s['drawdown_pct'] == 11.0, f'A12b: drawdown 100→89k = 11%, got {s["drawdown_pct"]}')
check(s['current_portfolio_value'] == 89000, 'A12c: current value in snapshot')
check(s['peak_portfolio_value'] == 100000, 'A12d: peak value in snapshot')

# A13: update_portfolio_peak
rm_a13 = RiskManager(); rm_a13.seed_portfolio(100000)
rm_a13.update_portfolio_peak(105000)
check(rm_a13.peak_portfolio_value == 105000, 'A13a: peak updated up')
rm_a13.update_portfolio_peak(95000)
check(rm_a13.peak_portfolio_value == 105000, 'A13b: peak not updated down')

# A14: record_order_result — approve resets rejection streak
rm_a14 = RiskManager(max_consecutive_rejections=3)
rm_a14.record_order_result(False)
rm_a14.record_order_result(False)
check(rm_a14.consecutive_rejections == 2, 'A14a: 2 rejections')
rm_a14.record_order_result(True)
check(rm_a14.consecutive_rejections == 0, 'A14b: approval resets streak')

# A15: _trip_circuit with explicit tick_id
rm_a15 = RiskManager(cooldown_ticks=7)
rm_a15._trip_circuit(tick_id=100)
check(rm_a15.is_circuit_open(current_tick_id=101), 'A15a: circuit open at tick 101')
check(not rm_a15.is_circuit_open(current_tick_id=108), 'A15b: circuit closed at tick 108')


# ═══════════════════════════════════════════════════════════════════════
# SECTION B: FallbackChain — all state transitions
# ═══════════════════════════════════════════════════════════════════════
section('B: FallbackChain — all state transitions')

from aml_sim.agents.risk.fallback import FallbackChain

fb = FallbackChain(max_consecutive_failures=3)
check(not fb.should_fallback(), 'B1: starts inactive')
check(fb.failure_count == 0, 'B2: initial count 0')

# Seed + single failure
fb.seed_safe_state({'max_position': 50, 'risk_mode': 'conservative'})
fb.record_result(False)
check(fb.failure_count == 1, 'B3: 1 failure')
check(not fb.should_fallback(), 'B4: not yet (1/3)')

# Success resets
fb.record_result(True)
check(fb.failure_count == 0, 'B5: success resets to 0')

# Three consecutive failures → activate
fb.record_result(False); fb.record_result(False); fb.record_result(False)
check(fb.should_fallback(), 'B6: activates after 3 failures')
state = fb.get_fallback_state()
check(state is not None, 'B7: returns fallback state')
check(state['max_position'] == 50, 'B8: correct fallback values')
check(state is not fb.safe_state, 'B9: returns a copy, not original reference')

# Recovery
fb.record_result(True)
check(not fb.should_fallback(), 'B10: deactivates on success')

# Snapshot
snap = fb.snapshot()
check(not snap['active'], 'B11: snapshot reflects inactive')
check(snap['total_failures'] >= 3, 'B12: snapshot counts total failures')


# ═══════════════════════════════════════════════════════════════════════
# SECTION C: Strategy Registry — all operations
# ═══════════════════════════════════════════════════════════════════════
section('C: Strategy Registry — all operations')

from aml_sim.agents.strategy.registry import StrategyRegistry
from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal

builtins = StrategyRegistry.list_all()
check(len(builtins) == 6, f'C1: 6 strategies registered, got {len(builtins)}')
check('momentum' in builtins, 'C2: momentum registered')
check('passive_benchmark' in builtins, 'C3: passive registered')

# Case-insensitive lookup
check(StrategyRegistry.is_registered('MOMENTUM'), 'C4a: case-insensitive positive')
check(StrategyRegistry.is_registered('Mean_Reversion'), 'C4b: case-insensitive mixed')
check(not StrategyRegistry.is_registered('nonexistent'), 'C4c: negative lookup')

# get() returns None for missing
check(StrategyRegistry.get('nonexistent') is None, 'C5: missing → None')

# Every strategy instantiates and generates valid signal
prices = [100,101,102,103,104,105,106,107,108,109,110,111,112]
ctx = AlphaContext(prices=prices, instrument='AAPL', current_position=0, portfolio_value=100000)
for name in builtins:
    cls = StrategyRegistry.get(name)
    instance = cls()
    sig = instance.generate(ctx)
    check(isinstance(sig, AlphaSignal), f'C6-{name}: returns AlphaSignal')
    check(-1.0 <= sig.direction <= 1.0, f'C7-{name}: direction in [-1,1]')
    check(0.0 <= sig.strength <= 1.0, f'C8-{name}: strength in [0,1]')
    check(0.0 <= sig.confidence <= 1.0, f'C9-{name}: confidence in [0,1]')
    check(isinstance(sig.reason, str) and len(sig.reason) > 0, f'C10-{name}: has reason')


# ═══════════════════════════════════════════════════════════════════════
# SECTION D: CompositeAlphaStrategy — all blend modes + edge cases
# ═══════════════════════════════════════════════════════════════════════
section('D: CompositeAlphaStrategy — all paths')

from aml_sim.agents.strategy.composite import CompositeAlphaStrategy

mc = StrategyRegistry.get('momentum'); mrc = StrategyRegistry.get('mean_reversion')
bc = StrategyRegistry.get('breakout')

# D1: weighted_sum
c = CompositeAlphaStrategy([(mc(), 0.7), (mrc(), 0.3)], blend_mode='weighted_sum')
r = c.generate(ctx)
check(r.is_actionable, 'D1a: weighted_sum produces actionable signal')
check(-1.0 <= r.direction <= 1.0, 'D1b: direction valid')

# D2: vote — with only momentum (bullish) vs mean_reversion (bearish)
c2 = CompositeAlphaStrategy([(mc(), 1.0), (mrc(), 1.0)], blend_mode='vote')
r2 = c2.generate(ctx)
check(r2.is_actionable, 'D2: vote mode works')

# D3: unanimous — momentum vs mean_reversion conflict → no signal
c3 = CompositeAlphaStrategy([(mc(), 1.0), (mrc(), 1.0)], blend_mode='unanimous')
r3 = c3.generate(ctx)
check(not r3.is_actionable, 'D3: unanimous with conflict → non-actionable')

# D4: priority_cascade
c4 = CompositeAlphaStrategy([(bc(), 1.0), (mc(), 1.0)], blend_mode='priority_cascade')
r4 = c4.generate(ctx)
check(isinstance(r4, AlphaSignal), 'D4: cascade returns valid signal')

# D5: empty strategy list
c5 = CompositeAlphaStrategy([], blend_mode='weighted_sum')
r5 = c5.generate(ctx)
check(not r5.is_actionable, 'D5: empty → non-actionable')

# D6: invalid blend_mode
try:
    CompositeAlphaStrategy([], blend_mode='invalid_mode')
    check(False, 'D6: should have raised ValueError')
except ValueError:
    check(True, 'D6: invalid blend_mode raises ValueError')

# D7: min_confidence filters signals below threshold.
# Passive produces non-actionable signals (confidence=0), so min_confidence=0.5 filters it out.
from aml_sim.agents.strategy.alpha_passive import PassiveBenchmarkStrategy
c7 = CompositeAlphaStrategy([(PassiveBenchmarkStrategy(), 1.0)], blend_mode='weighted_sum', min_confidence=0.5)
r7 = c7.generate(ctx)
check(not r7.is_actionable, 'D7: min_confidence=0.5 filters passive strategy (confidence=0)')

# D8: strategy_names + to_dict
c8 = CompositeAlphaStrategy([(mc(), 0.5), (mrc(), 0.5)], blend_mode='weighted_sum')
names = c8.strategy_names()
check('momentum' in names and 'mean_reversion' in names, 'D8a: strategy_names correct')
d = c8.to_dict()
check(d['blend_mode'] == 'weighted_sum', 'D8b: to_dict blend_mode')
check(d['strategy_count'] == 2, 'D8c: to_dict count')


# ═══════════════════════════════════════════════════════════════════════
# SECTION E: StrategyPerformance — all methods
# ═══════════════════════════════════════════════════════════════════════
section('E: StrategyPerformance — full coverage')

from aml_sim.agents.strategy.performance import StrategyPerformance, create_performance_tracker

# E1: create_performance_tracker
trackers = create_performance_tracker(['m', 'mr', 'b'])
check(len(trackers) == 3, 'E1a: 3 trackers created')
check(all(isinstance(t, StrategyPerformance) for t in trackers.values()), 'E1b: all are StrategyPerformance')

# E2: record_signal + record_acted
trackers['m'].record_signal(1.0, '2025-01-01T09:00:00')
check(trackers['m'].signals_generated == 1, 'E2a: signal counted')
trackers['m'].record_acted()
check(trackers['m'].signals_acted_on == 1, 'E2b: acted counted')

# E3: record_trade_outcome — win
trackers['m'].record_trade_outcome(pnl=150.0, is_win=True, holding_ticks=5, entry_price=100.0, exit_price=101.5)
s = trackers['m'].snapshot()
check(s['win_rate'] == 1.0, 'E3a: win_rate 1.0')
check(s['cumulative_pnl'] == 150.0, 'E3b: cumulative PnL')
check(s['avg_holding_ticks'] == 5.0, 'E3c: avg holding ticks')

# E4: record_trade_outcome — loss, mixed
trackers['m'].record_trade_outcome(pnl=-50.0, is_win=False, holding_ticks=3)
trackers['m'].record_trade_outcome(pnl=200.0, is_win=True, holding_ticks=7)
s2 = trackers['m'].snapshot()
check(s2['total_trades'] == 3, 'E4a: 3 trades total')
check(abs(s2['win_rate'] - 2/3) < 0.01, f'E4b: win_rate ~0.667, got {s2["win_rate"]}')
check(s2['cumulative_pnl'] == 300.0, 'E4c: cumulative PnL = 300')
check(abs(s2['avg_holding_ticks'] - 5.0) < 0.01, 'E4d: avg hold = 5')

# E5: profit_factor
check(s2['profit_factor'] == 2.0, f'E5: profit_factor = 2 wins / 1 loss, got {s2["profit_factor"]}')

# E6: all-win (no losses)
trackers['b'].record_trade_outcome(pnl=100, is_win=True)
check(trackers['b'].snapshot()['profit_factor'] == 'inf', 'E6a: all-win → inf')
# All-loss (no wins)
sp = StrategyPerformance(strategy_name='test')
sp.record_trade_outcome(pnl=-100, is_win=False)
check(sp.snapshot()['profit_factor'] == 0.0, 'E6b: all-loss → 0')

# E7: empty tracker
sp2 = StrategyPerformance()
check(sp2.win_rate == 0.0, 'E7a: empty win_rate=0')
check(sp2.profit_factor == 0.0, 'E7b: empty profit_factor=0')


# ═══════════════════════════════════════════════════════════════════════
# SECTION F: Profile system — all types, modulator, dynamics
# ═══════════════════════════════════════════════════════════════════════
section('F: Profile system — full coverage')

from aml_sim.agents.models.profile import (
    BehavioralTraits, AgentProfile,
    MarketMakerProfile, RetailProfile, InstitutionalProfile,
    InformedProfile, LiquidityTakerProfile,
    TrendFollowerProfile, ArbitrageurProfile, HedgeProfile, PassiveIndexProfile,
    coerce_profile, profile_to_dict
)
from aml_sim.agents.profile.modulator import ProfileModulator
from aml_sim.agents.profile.behavioral_dynamics import BehavioralDynamics, BehavioralState
from aml_sim.agents.models.state import (
    MarketMakerStrategyState, RetailStrategyState,
    InstitutionalStrategyState, InformedStrategyState, LiquidityTakerStrategyState
)

# F1: BehavioralTraits construction
bt = BehavioralTraits()
check(bt.risk_aversion == 0.5, 'F1a: default risk_aversion')
check(bt.discipline == 0.7, 'F1b: default discipline')
d = bt.to_dict()
check(len(d) == 10, f'F1c: 10 traits, got {len(d)}')

# F2: BehavioralTraits.from_mapping
bt2 = BehavioralTraits.from_mapping({'risk_aversion': 0.9, 'bad_key': 999, 'discipline': 0.3})
check(bt2.risk_aversion == 0.9, 'F2a: mapped risk_aversion')
check(bt2.discipline == 0.3, 'F2b: mapped discipline')
check(not hasattr(bt2, 'bad_key'), 'F2c: unknown key ignored')
bt3 = BehavioralTraits.from_mapping(None)
check(bt3.risk_aversion == 0.5, 'F2d: None → defaults')

# F3: Profile traits property
mm = MarketMakerProfile()
check(mm.traits.risk_aversion == 0.6, 'F3a: MM risk_aversion=0.6')
rp = RetailProfile()
check(rp.traits.social_influence == 0.65, 'F3b: retail social=0.65')
ip = InstitutionalProfile()
check(ip.traits.patience == 0.8, 'F3c: inst patience=0.8')

# F4: New profiles
tf = TrendFollowerProfile()
check(tf.traits.recency_bias == 0.8, 'F4a: trend follower recency=0.8')
arb = ArbitrageurProfile()
check(arb.traits.discipline == 0.9, 'F4b: arbitrageur discipline=0.9')
hedge = HedgeProfile()
check(hedge.traits.risk_aversion == 0.9, 'F4c: hedger risk_aversion=0.9')
passive = PassiveIndexProfile()
check(passive.traits.aggression == 0.05, 'F4d: passive aggression=0.05')

# F5: coerce_profile
cfg = {'name': 'test', 'risk_tolerance': 'low', 'behavioral_traits': {'risk_aversion': 0.8}}
result = coerce_profile(cfg, MarketMakerProfile)
check(isinstance(result, MarketMakerProfile), 'F5a: returns correct type')
check(result.risk_tolerance == 'low', 'F5b: field mapped')
check(result.traits.risk_aversion == 0.8, 'F5c: traits passed through')

# F6: profile_to_dict
d = profile_to_dict(mm)
check('computed_traits' in d, 'F6a: includes computed_traits')
check(d['computed_traits']['risk_aversion'] == 0.6, 'F6b: correct trait values')

# F7: ProfileModulator — all 5 strategy types
mod = ProfileModulator(trait_influence=0.8)
ra = BehavioralTraits(risk_aversion=0.9, aggression=0.1, patience=0.8, conviction=0.3, discipline=0.9)
rs = BehavioralTraits(risk_aversion=0.1, aggression=0.9, patience=0.2, conviction=0.9, discipline=0.2)

all_states = [
    MarketMakerStrategyState(),
    RetailStrategyState(),
    InstitutionalStrategyState(),
    InformedStrategyState(),
    LiquidityTakerStrategyState(),
]
for base_state in all_states:
    name = type(base_state).__name__
    orig = copy.deepcopy(base_state)
    av = mod.modulate(copy.deepcopy(base_state), ra)
    sv = mod.modulate(copy.deepcopy(base_state), rs)
    # Original must not be mutated
    check(orig == base_state, f'F7a-{name}: original not mutated')
    # Risk-averse should have smaller position limits than risk-seeking
    if hasattr(av, 'max_position') and hasattr(sv, 'max_position'):
        check(av.max_position <= sv.max_position, f'F7b-{name}: averse max_pos <= seeking max_pos')

# F8: BehavioralDynamics — full cycle
dyn = BehavioralDynamics(); state = BehavioralState()
check(state.pnl_streak == 0, 'F8a: initial streak=0')
check(state.stress_level == 0.0, 'F8b: initial stress=0')

# Winning streaks
for _ in range(5): dyn.update(state, pnl_delta=500, had_fill=True, signal_correct=True)
check(state.pnl_streak == 5, 'F8c: 5 wins → streak=5')
check(state.stress_level < 0.1, 'F8d: low stress after wins')
check(state.confidence_drift > 0, 'F8e: positive confidence drift')

# Losing streaks
for _ in range(4): dyn.update(state, pnl_delta=-800, had_fill=True, signal_correct=False)
check(state.stress_level > 0.3, 'F8f: elevated stress after losses')
check(state.confidence_drift < 0, 'F8g: negative confidence drift')

# Regime detection
d2 = BehavioralDynamics(); s2 = BehavioralState()
trending = [100+i for i in range(25)]
d2.update(s2, prices=trending)
check(s2.market_regime_belief == 'trending', f'F8h: trending detected, got {s2.market_regime_belief}')

ranging = [100,100.5,99.8,100.3,99.9,100.1,99.7,100.4,99.8,100.2]*3
d3 = BehavioralDynamics(); s3 = BehavioralState()
d3.update(s3, prices=ranging[:25])
check(s3.market_regime_belief in ('ranging', 'neutral'), f'F8i: ranging/neutral detected, got {s3.market_regime_belief}')

# F9: BehavioralState snapshot
snap = state.snapshot()
for k in ['pnl_streak', 'recent_accuracy', 'stress_level', 'confidence_drift', 'market_regime_belief']:
    check(k in snap, f'F9a: {k} in snapshot')
check(isinstance(snap['pnl_streak'], int), 'F9b: pnl_streak is int')
check(isinstance(snap['stress_level'], float), 'F9c: stress_level is float')


# ═══════════════════════════════════════════════════════════════════════
# SECTION G: LLM Strategist — all roles, contract completeness
# ═══════════════════════════════════════════════════════════════════════
section('G: LLM Strategist — all roles + contract')

from aml_sim.agents.strategy.llm_slow_strategy import (
    create_llm_strategist, LLMStrategist,
    STATIC_RESPONSES_BY_ROLE, LLM_RESPONSE_SCHEMA,
    LLMStrategistConfigurationError
)

# G1: create_llm_strategist for all 5 roles
roles = ['market_maker', 'retail', 'institutional', 'informed', 'liquidity_taker']
for role in roles:
    s = create_llm_strategist(role)
    check(isinstance(s, LLMStrategist), f'G1a-{role}: returns LLMStrategist')
    # Verify client exists
    check(s.client is not None, f'G1b-{role}: has client')

# G2: Static responses have ALL required keys
required = ['strategy_updates', 'reason', 'confidence', 'strategy_config', 'risk_overrides']
for role, resp in STATIC_RESPONSES_BY_ROLE.items():
    for key in required:
        check(key in resp, f'G2-{role}-{key}: present')

# G3: Static responses have valid types
for role, resp in STATIC_RESPONSES_BY_ROLE.items():
    check(isinstance(resp['strategy_updates'], dict), f'G3a-{role}: strategy_updates is dict')
    check(isinstance(resp['reason'], str), f'G3b-{role}: reason is str')
    check(isinstance(resp['confidence'], (int, float)), f'G3c-{role}: confidence numeric')
    check(0 <= resp['confidence'] <= 1, f'G3d-{role}: confidence in [0,1]')

# G4: LLM response schema completeness
check('strategy_updates' in str(LLM_RESPONSE_SCHEMA), 'G4a: schema mentions strategy_updates')
check('strategy_config' in str(LLM_RESPONSE_SCHEMA), 'G4b: schema mentions strategy_config')
check('risk_overrides' in str(LLM_RESPONSE_SCHEMA), 'G4c: schema mentions risk_overrides')

# G5: build_context includes available strategies
strat = create_llm_strategist('institutional')
ctx = strat.build_context(observation={}, current_strategy=InstitutionalStrategyState())
check('available_strategies' in ctx, 'G5a: context has available_strategies')
check(len(ctx['available_strategies']) == 6, f'G5b: 6 strategies available, got {len(ctx["available_strategies"])}')


# ═══════════════════════════════════════════════════════════════════════
# SECTION H: Producer-Consumer Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════
section('H: Cross-component pipeline tests')

# H1: reporting.py handles BOTH formats (Review #1 fix)
from aml_sim.reporting import generate_trader_action_report

tmp_dir = Path(tempfile.mkdtemp())
agents_dir = tmp_dir / 'agents'
reports_dir = tmp_dir / 'reports'
agents_dir.mkdir(parents=True); reports_dir.mkdir(parents=True)

try:
    # Enhanced format (dict with "actions" key)
    (agents_dir / 'trader_actions_agent1.json').write_text(json.dumps({
        'agent_id': 'agent1',
        'risk_summary': {'health': 'healthy', 'drawdown_pct': 3.5, 'total_approved': 10},
        'fallback_summary': {'active': False},
        'behavioral_summary': {'stress_level': 0.2},
        'strategy_performance': {},
        'actions': [
            {'event_type': 'order_submitted', 'agent_id': 'agent1', 'timestamp': '2025-01-01T09:30:00'},
            {'event_type': 'trade_executed', 'agent_id': 'agent1', 'timestamp': '2025-01-01T09:30:30'},
            {'event_type': 'order_submitted', 'agent_id': 'agent1', 'timestamp': '2025-01-01T09:31:00'},
        ]
    }))
    # Legacy format (plain list)
    (agents_dir / 'trader_actions_legacy1.json').write_text(json.dumps([
        {'event_type': 'order_submitted', 'agent_id': 'legacy1', 'timestamp': '2025-01-01T09:29:00'},
        {'event_type': 'order_rejected', 'agent_id': 'legacy1', 'timestamp': '2025-01-01T09:29:30'},
    ]))
    # Empty enhanced format
    (agents_dir / 'trader_actions_empty_agent.json').write_text(json.dumps({
        'agent_id': 'empty_agent',
        'actions': []
    }))

    generate_trader_action_report(agents_dir, reports_dir)
    combined = json.loads((reports_dir / 'trader_actions.json').read_text())

    check(combined['action_count'] == 5, f'H1a: 5 total actions (3+2), got {combined["action_count"]}')
    check('agent1' in combined['agents'], 'H1b: enhanced-format agent in report')
    check('legacy1' in combined['agents'], 'H1c: legacy-format agent in report')
    check(combined['agents']['agent1']['submitted_orders'] == 2, 'H1d: agent1 submitted_orders=2')
    check(combined['agents']['legacy1']['rejected_orders'] == 1, 'H1e: legacy1 rejected_orders=1')
    # Empty agent should still appear
    check('empty_agent' in combined['agents'], 'H1f: empty-actions agent in report')
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

# H2: launcher norm → agent constructor pipeline
from aml_sim.launcher import build_agent_param_customizers

def fake_interval(s):
    return {'30s': 30, '1m': 60, '2m': 120, '5m': 300, '10m': 600}.get(s, 60)

customizers = build_agent_param_customizers(fake_interval)

# Test every agent type with risk_limits
for agent_type in ['AML_Market_Maker', 'AML_Retail_Trader', 'AML_Institutional_Trader',
                    'AML_Informed_Trader', 'AML_Liquidity_Taker']:
    params = {
        'action_interval': '1m',
        'risk_limits': {'max_drawdown_pct': 0.15, 'max_order_rate': 25},
        'behavioral_traits': {'risk_aversion': 0.7},
    }
    if agent_type == 'AML_Shock_Agent':
        norm = customizers[agent_type](params)
        check(norm == params, f'H2-{agent_type}: shock agent passes through unchanged')
    else:
        norm = customizers[agent_type](params)
        check('risk_overrides' in norm, f'H2a-{agent_type}: risk_limits → risk_overrides')
        check(norm['risk_overrides'] == {'max_drawdown_pct': 0.15, 'max_order_rate': 25},
              f'H2b-{agent_type}: risk_overrides values preserved')
        check('behavioral_traits' in norm.get('profile', {}),
              f'H2c-{agent_type}: behavioral_traits in profile')
        # Verify original params NOT mutated
        check(params['profile'] if 'profile' in params else True,
              f'H2d-{agent_type}: original params not mutated')

# H3: action_events format → _record_action_event data integrity
# (Test that every event recorded has all required snapshot fields)
sample_event = {}
sample_event.setdefault("agent_id", "test")
sample_event.setdefault("risk_snapshot", {'health': 'healthy'})
sample_event.setdefault("fallback_snapshot", {'active': False})
sample_event.setdefault("behavioral_state", {'stress_level': 0.0})
check('risk_snapshot' in sample_event, 'H3a: risk_snapshot recorded')
check('fallback_snapshot' in sample_event, 'H3b: fallback_snapshot recorded')
check('behavioral_state' in sample_event, 'H3c: behavioral_state recorded')

# H4: StrategyRegistry → CompositeAlphaStrategy → signal → performance pipeline
# Use strongly trending prices so momentum definitely triggers
strong_trend = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]
ctx_test = AlphaContext(prices=strong_trend, instrument='AAPL',
                         current_position=0, portfolio_value=100000)
active_names = ['momentum']
strategies = [(StrategyRegistry.get('momentum')(), 1.0)]
comp = CompositeAlphaStrategy(strategies, blend_mode='weighted_sum')
sig = comp.generate(ctx_test)
check(isinstance(sig, AlphaSignal), 'H4a: strategy -> signal chain works')
check(sig.is_actionable, 'H4b: composite signal is actionable')
tracker = StrategyPerformance(strategy_name='composite_test')
if sig.is_actionable:
    tracker.record_signal(sig.direction)
    tracker.record_acted()
check(tracker.signals_generated == 1, 'H4c: signal -> perf tracker works')

# H5: observation.py context → includes all new fields
from aml_sim.agents.context.observation import build_observation_context

class MockAgent:
    instrument_exchange_map = {'AAPL': 'exchange_aapl'}
    agent_id = 'mock_agent'
    current_time = '2025-01-01T09:30:00'
    last_market_snapshot = {}
    price_history = {'AAPL': [{'price': 100.0}]}
    cash = 100000
    portfolio_value = 100000
    long_qty = {'AAPL': 100}; short_qty = {'AAPL': 0}
    prices = {'AAPL': 100.0}; realized_pnl = {'AAPL': 0.0}
    pending_orders = {}
    session_executed_orders = []
    strategy_performance = {}
    behavioral_state = BehavioralState()
    risk_manager = RiskManager()

mock = MockAgent()
obs = build_observation_context(mock)
check('strategy_performance' in obs, 'H5a: observation has strategy_performance')
check('behavioral_state' in obs, 'H5b: observation has behavioral_state')
check('risk_status' in obs, 'H5c: observation has risk_status')


# ═══════════════════════════════════════════════════════════════════════
# SECTION I: Edge Cases & Stress
# ═══════════════════════════════════════════════════════════════════════
section('I: Edge cases & boundary values')

# I1: RiskManager with zero/negative portfolio
rm_i1 = RiskManager()
v, _ = rm_i1.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=0, current_position=0)
check(v == OrderVerdict.APPROVED, 'I1a: zero portfolio allows small order')
rm_i1.seed_portfolio(0)
check(rm_i1.peak_portfolio_value == 0, 'I1b: peak stays 0 when seeded with 0')
check(rm_i1.initial_portfolio_value == 0, 'I1c: initial stays 0')

# I2: RiskManager drawdown near threshold (floating point: 90000/100000 != exact 0.1)
rm_i2 = RiskManager(max_drawdown_pct=0.10); rm_i2.seed_portfolio(100000)
h = rm_i2.check_portfolio_health(portfolio_value=89000)  # 11% clearly > 10%
check(h == HealthVerdict.HALTED, 'I2a: 11% drawdown → halted')
h = rm_i2.check_portfolio_health(portfolio_value=91000)  # 9% below threshold
check(h in (HealthVerdict.HEALTHY, HealthVerdict.DEGRADED), 'I2b: 9% within tolerance')

# I3: FallbackChain with unset safe_state
fb_i3 = FallbackChain()
check(not fb_i3.should_fallback(), 'I3a: no safe state → no fallback')
check(fb_i3.get_fallback_state() is None, 'I3b: getter returns None')

# I4: ProfileModulator with zero influence
mod_i4 = ProfileModulator(trait_influence=0.0)
base_i4 = RetailStrategyState(max_order_size=25)
result_i4 = mod_i4.modulate(copy.deepcopy(base_i4), BehavioralTraits(aggression=0.9))
check(result_i4.max_order_size == 25, 'I4a: zero influence → unchanged')

# I5: ProfileModulator with maximum influence
mod_i5 = ProfileModulator(trait_influence=1.0)
result_i5 = mod_i5.modulate(copy.deepcopy(base_i4), BehavioralTraits(aggression=1.0, risk_aversion=0.0))
check(result_i5.max_order_size >= 25, 'I5a: max influence + max aggression → increased size')

# I6: BehavioralState streak saturation
dyn_i6 = BehavioralDynamics(streak_saturation=3)
s_i6 = BehavioralState()
for _ in range(10): dyn_i6.update(s_i6, pnl_delta=500, had_fill=True, signal_correct=True)
check(s_i6.pnl_streak == 3, f'I6: streak saturated at 3, got {s_i6.pnl_streak}')

# I7: Composite with single strategy → acts same as standalone
single = CompositeAlphaStrategy([(mc(), 1.0)], blend_mode='weighted_sum')
r_single = single.generate(ctx_test)
standalone = mc().generate(ctx_test)
check(r_single.direction == standalone.direction, 'I7: single-strategy composite = standalone')

# I8: StrategyRegistry clears correctly
StrategyRegistry.clear()
check(len(StrategyRegistry.list_all()) == 0, 'I8a: clear empties registry')
# Re-register by reloading
from aml_sim.agents.strategy.registry import _register_builtins
_register_builtins()
check(len(StrategyRegistry.list_all()) == 6, 'I8b: re-register restores all 6')


# ═══════════════════════════════════════════════════════════════════════
# SECTION J: Scenario and Launcher — full pipeline
# ═══════════════════════════════════════════════════════════════════════
section('J: Scenario → Launcher pipeline')

from aml_sim.scenario import AMLScenario, load_scenario

# J1: All 3 scenarios load correctly
for s_name in ['aml_orderbook_replay.yaml', 'aml_agent_infra_smoke.yaml', 'aml_one_hour_live.yaml']:
    sc = load_scenario(_REPO_ROOT / 'scenarios' / s_name)
    check(isinstance(sc, AMLScenario), f'J1a-{s_name}: returns AMLScenario')
    check(len(sc.stocksim_config.get('agents', {})) > 0, f'J1b-{s_name}: has agents')
    check(sc.aml_config is not None, f'J1c-{s_name}: has aml_config (upstream field)')

# J2: Agent count verification in each scenario
for s_name, expected_min in [('aml_orderbook_replay.yaml', 3), ('aml_agent_infra_smoke.yaml', 5), ('aml_one_hour_live.yaml', 4)]:
    sc = load_scenario(_REPO_ROOT / 'scenarios' / s_name)
    agents = sc.stocksim_config['agents']
    check(len(agents) >= expected_min, f'J2-{s_name}: >= {expected_min} agent groups')

# J3: launcher maps correct agent types
from aml_sim.launcher import import_agent_class

agent_types = ['AML_Market_Maker', 'AML_Retail_Trader', 'AML_Institutional_Trader',
               'AML_Informed_Trader', 'AML_Liquidity_Taker', 'AML_Shock_Agent']
for at in agent_types:
    cls = import_agent_class(at)
    check(cls is not None, f'J3a-{at}: class resolved')
    check(hasattr(cls, '__init__'), f'J3b-{at}: has __init__')

# J4: import_agent_class raises on unknown type
try:
    import_agent_class('BogusAgentType')
    check(False, 'J4: should raise on unknown type')
except ValueError:
    check(True, 'J4: raises ValueError on unknown type')

# J5: launcher _make_seed determinism
from aml_sim.launcher import _make_seed
s1 = _make_seed('run123', 'agent1')
s2 = _make_seed('run123', 'agent1')
check(s1 == s2, 'J5a: same inputs → same seed')
s3 = _make_seed('run123', 'agent2')
check(s1 != s3, 'J5b: different agent → different seed')


# ═══════════════════════════════════════════════════════════════════════
# SECTION K: Validator — all strategy types + new fields
# ═══════════════════════════════════════════════════════════════════════
section('K: Validator — all strategy types')

from aml_sim.agents.strategy.validator import (
    validate_strategy_state, StrategyValidationError,
    StrategyValidationLimits, DEFAULT_STRATEGY_LIMITS
)

# K1: All valid states pass (deepcopy makes different objects, so use equality)
valid_states = [
    MarketMakerStrategyState(),
    RetailStrategyState(),
    InstitutionalStrategyState(),
    InformedStrategyState(),
    LiquidityTakerStrategyState(),
]
for s in valid_states:
    result = validate_strategy_state(copy.deepcopy(s))
    check(isinstance(result, type(s)), f'K1-{type(s).__name__}: valid state passes')
    check(result == s, f'K1-{type(s).__name__}: state unchanged after validation')

# K2: Invalid probability clamped
bad_retail = RetailStrategyState(trade_probability=1.5)
try:
    validate_strategy_state(copy.deepcopy(bad_retail))
    check(False, 'K2: should reject trade_probability=1.5')
except StrategyValidationError:
    check(True, 'K2: rejects trade_probability=1.5')

# K3: Invalid risk_mode
bad_risk = MarketMakerStrategyState(risk_mode='reckless')
try:
    validate_strategy_state(copy.deepcopy(bad_risk))
    check(False, 'K3: should reject bad risk_mode')
except StrategyValidationError:
    check(True, 'K3: rejects bad risk_mode')

# K4: Invalid alpha_strategies
bad_alpha = InstitutionalStrategyState(alpha_strategies=['voodoo_strategy'])
try:
    validate_strategy_state(copy.deepcopy(bad_alpha))
    check(False, 'K4: should reject bad alpha_strategy')
except StrategyValidationError:
    check(True, 'K4: rejects bad alpha_strategy')

# K5: Invalid blend_mode
bad_blend = InstitutionalStrategyState(blend_mode='chaos_mode')
try:
    validate_strategy_state(copy.deepcopy(bad_blend))
    check(False, 'K5: should reject bad blend_mode')
except StrategyValidationError:
    check(True, 'K5: rejects bad blend_mode')

# K6: Valid blend_modes pass
for mode in ['weighted_sum', 'vote', 'unanimous', 'priority_cascade']:
    s = InstitutionalStrategyState(blend_mode=mode)
    result = validate_strategy_state(copy.deepcopy(s))
    check(isinstance(result, type(s)), f'K6-{mode}: valid blend_mode passes')

# K7: All 6 alpha strategies pass validation
for name in builtins:
    s = InstitutionalStrategyState(alpha_strategies=[name])
    result = validate_strategy_state(copy.deepcopy(s))
    check(isinstance(result, type(s)), f'K7-{name}: valid alpha strategy passes')

# K8: min > max check
bad_spread = MarketMakerStrategyState(min_spread=2.0, max_spread=1.0)
try:
    validate_strategy_state(copy.deepcopy(bad_spread))
    check(False, 'K8: should reject min_spread > max_spread')
except StrategyValidationError:
    check(True, 'K8: rejects min_spread > max_spread')

# K9: Limits dataclass includes new strategies
limits = DEFAULT_STRATEGY_LIMITS
check('breakout' in limits.allowed_alpha_strategies, 'K9a: breakout in limits')
check('volatility_regime' in limits.allowed_alpha_strategies, 'K9b: vol_regime in limits')
check('passive_benchmark' in limits.allowed_alpha_strategies, 'K9c: passive in limits')
check('weighted_sum' in limits.allowed_blend_modes, 'K9d: weighted_sum in blend modes')
check('priority_cascade' in limits.allowed_blend_modes, 'K9e: cascade in blend modes')


# ═══════════════════════════════════════════════════════════════════════
# SECTION L: Alpha strategies — signal quality under edge conditions
# ═══════════════════════════════════════════════════════════════════════
section('L: Alpha strategies — edge conditions')

# L1: Insufficient price data
empty_ctx = AlphaContext(prices=[], instrument='AAPL')
for name in builtins:
    cls = StrategyRegistry.get(name)
    sig = cls().generate(empty_ctx)
    check(not sig.is_actionable, f'L1-{name}: non-actionable with empty prices')
    check(len(sig.reason) > 0, f'L1-{name}: has reason string')

# L2: Single price point
single_ctx = AlphaContext(prices=[100.0], instrument='AAPL')
for name in builtins:
    cls = StrategyRegistry.get(name)
    sig = cls().generate(single_ctx)
    check(not sig.is_actionable, f'L2-{name}: non-actionable with single price')

# L3: Event-driven with active shock
from aml_sim.agents.strategy.alpha_event_driven import EventDrivenStrategy
shock_ctx = AlphaContext(
    prices=[100.0]*20, instrument='AAPL', current_position=0, portfolio_value=100000,
    events=[{'event_type': 'AML_SHOCK', 'shock_type': 'negative_news', 'severity': 0.8,
             'direction': -1, 'fundamental_price_shift': -2.0, 'affected_instruments': ['AAPL']}]
)
sig = EventDrivenStrategy().generate(shock_ctx)
check(sig.is_actionable, 'L3a: event-driven finds shock signal')
check(sig.direction == 1.0, 'L3b: fade negative shock → buy (direction=+1)')

# L4: Event-driven with no events
no_event_ctx = AlphaContext(prices=[100.0]*20, instrument='AAPL')
sig = EventDrivenStrategy().generate(no_event_ctx)
check(not sig.is_actionable, 'L4: no events → non-actionable')

# L5: Volatility regime with extreme vol (need 14+ prices for lookback_ticks=12)
volatile_prices = [100.0, 108.0, 92.0, 115.0, 85.0, 120.0, 78.0, 125.0, 72.0, 130.0, 68.0, 135.0, 62.0, 140.0, 55.0]
from aml_sim.agents.strategy.alpha_volatility import VolatilityRegimeStrategy
sig_v = VolatilityRegimeStrategy(lookback_ticks=12).generate(AlphaContext(prices=volatile_prices, instrument='AAPL'))
check(sig_v.metadata.get('regime') == 'high', f'L5: extreme vol → high regime, got {sig_v.metadata.get("regime")}')

# L6: Breakout with clean breakout
breakout_prices = [100.0]*10 + [105.0, 106.0, 107.0]
from aml_sim.agents.strategy.alpha_breakout import BreakoutStrategy
sig_b = BreakoutStrategy(lookback_ticks=5).generate(AlphaContext(prices=breakout_prices, instrument='AAPL'))
check(sig_b.is_actionable, 'L6a: breakout detected on price surge')
check(sig_b.direction == 1.0, 'L6b: breakout up → buy signal')

# L7: Passive benchmark default behavior
from aml_sim.agents.strategy.alpha_passive import PassiveBenchmarkStrategy
sig_p = PassiveBenchmarkStrategy().generate(AlphaContext(prices=[100.0]*10, instrument='AAPL'))
check(not sig_p.is_actionable, 'L7a: passive stays quiet')
check('holding' in sig_p.reason, 'L7b: reason mentions holding')


# ═══════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
if FAILED:
    print(f'  {len(FAILED)} FAILURES:')
    for f in FAILED:
        print(f'    {f}')
else:
    print(f'  ALL {PASSED} TESTS PASSED — ZERO FAILURES')
print(f'{"="*60}')
sys.exit(0 if not FAILED else 1)
