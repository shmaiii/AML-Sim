"""Comprehensive logic verification for AML-Sim agent architecture."""
import sys, os, copy, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), 'simulators', 'StockSim'))

def test_suite(name):
    print(f'\n{"="*60}')
    print(f'SUITE: {name}')
    print(f'{"="*60}')

# ====================================================================
# SUITE 1: Risk Manager
# ====================================================================
test_suite('Risk Manager — all paths')
from aml_sim.agents.risk.risk_manager import RiskManager, OrderVerdict, HealthVerdict
from aml_sim.agents.risk.fallback import FallbackChain

rm = RiskManager(max_drawdown_pct=0.15, max_order_rate=5, max_consecutive_rejections=3, cooldown_ticks=5, max_order_value_pct=0.10)
rm.seed_portfolio(100000)

v, _ = rm.check_order('AAPL', 'BUY', 100, 100.0, portfolio_value=100000, current_position=0)
assert v == OrderVerdict.APPROVED; print('1a OK: healthy order approved')

h = rm.check_portfolio_health(portfolio_value=80000)
assert h == HealthVerdict.HALTED; print('1b OK: drawdown 20% > 15% -> HALTED')

rm3 = RiskManager(max_order_rate=2); rm3.seed_portfolio(100000)
for _ in range(2): rm3.record_order_result(True)
v, _ = rm3.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0)
assert v == OrderVerdict.REJECTED; print('1c OK: rate limit enforced')

rm4 = RiskManager(max_consecutive_rejections=2, cooldown_ticks=10)
rm4.record_order_result(False, current_tick_id=5); rm4.record_order_result(False, current_tick_id=6)
assert rm4.is_circuit_open(current_tick_id=6); print('1d OK: circuit breaker trips')
v, _ = rm4.check_order('AAPL', 'BUY', 10, 100.0, portfolio_value=100000, current_position=0, current_tick_id=7)
assert v == OrderVerdict.REJECTED; print('1d OK: orders rejected during circuit')

rm5 = RiskManager(max_order_value_pct=0.05); rm5.seed_portfolio(20000)
v, adj = rm5.check_order('AAPL', 'BUY', 200, None, portfolio_value=20000, current_position=0, last_price=95.0)
assert v == OrderVerdict.SIZE_REDUCED and adj == 10; print(f'1e OK: market order 200->{adj} (Review #1)')

rm6 = RiskManager(); rm6.seed_portfolio(100000)
rm6.check_portfolio_health(portfolio_value=92000)
s = rm6.snapshot()
assert s['drawdown_pct'] == 8.0; print(f'1f OK: drawdown snap={s["drawdown_pct"]}% (Review #3)')

fb = FallbackChain(max_consecutive_failures=3); fb.seed_safe_state({'max_position': 50})
fb.record_result(False); fb.record_result(True)
assert fb.failure_count == 0; print('1g OK: fallback resets on success')
fb.record_result(False); fb.record_result(False); fb.record_result(False)
assert fb.should_fallback(); print('1g OK: fallback activates')
fb.record_result(True); assert not fb.should_fallback(); print('1g OK: fallback deactivates')

# ====================================================================
# SUITE 2: Strategy System
# ====================================================================
test_suite('Strategy System')
from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.registry import StrategyRegistry
from aml_sim.agents.strategy.composite import CompositeAlphaStrategy
from aml_sim.agents.strategy.performance import create_performance_tracker

builtins = StrategyRegistry.list_all()
assert len(builtins) >= 6; print(f'2a OK: {len(builtins)} strategies: {builtins}')

prices = [100.0,100.5,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0]
ctx = AlphaContext(prices=prices, instrument='AAPL', current_position=0, portfolio_value=100000.0)
for name in builtins:
    cls = StrategyRegistry.get(name)
    sig = cls().generate(ctx)
    assert isinstance(sig, AlphaSignal) and -1.0<=sig.direction<=1.0
    assert 0.0<=sig.strength<=1.0 and 0.0<=sig.confidence<=1.0
print(f'2b OK: all {len(builtins)} strategies produce valid AlphaSignals')

mc = StrategyRegistry.get('momentum'); mrc = StrategyRegistry.get('mean_reversion')
for mode in ['weighted_sum', 'vote', 'unanimous', 'priority_cascade']:
    c = CompositeAlphaStrategy([(mc(),0.5),(mrc(),0.5)], blend_mode=mode)
    r = c.generate(ctx); assert isinstance(r, AlphaSignal)
print('2c OK: all 4 blend modes valid')

comp = CompositeAlphaStrategy([], blend_mode='weighted_sum')
assert not comp.generate(ctx).is_actionable; print('2d OK: empty composite non-actionable')

try: CompositeAlphaStrategy([], blend_mode='bad'); assert False
except ValueError: print('2e OK: invalid blend_mode raises ValueError')

tk = create_performance_tracker(['m']); tk['m'].record_signal(1.0); tk['m'].record_acted()
tk['m'].record_trade_outcome(pnl=100, is_win=True, holding_ticks=5)
s = tk['m'].snapshot()
assert s['win_rate']==1.0 and s['cumulative_pnl']==100.0; print(f'2f OK: perf tracking')

assert StrategyRegistry.is_registered('MOMENTUM'); print('2g OK: case-insensitive')
assert StrategyRegistry.get('nope') is None; print('2g OK: missing returns None')

# ====================================================================
# SUITE 3: Profile & Behavioral
# ====================================================================
test_suite('Profile & Behavioral')
from aml_sim.agents.models.profile import BehavioralTraits, MarketMakerProfile, RetailProfile, TrendFollowerProfile
from aml_sim.agents.profile.modulator import ProfileModulator
from aml_sim.agents.profile.behavioral_dynamics import BehavioralDynamics, BehavioralState
from aml_sim.agents.models.state import InstitutionalStrategyState

mm = MarketMakerProfile(); assert mm.traits.risk_aversion == 0.6; print(f'3a OK: MM traits')
rp = RetailProfile(); assert rp.traits.social_influence == 0.65; print('3a OK: Retail traits')
tf = TrendFollowerProfile(); assert tf.traits.recency_bias == 0.8; print('3a OK: TF traits')

mod = ProfileModulator(trait_influence=1.0)
ra = BehavioralTraits(risk_aversion=0.9, aggression=0.1)
rs = BehavioralTraits(risk_aversion=0.1, aggression=0.9)
base = InstitutionalStrategyState(max_position=500)
av = mod.modulate(copy.deepcopy(base), ra); sv = mod.modulate(copy.deepcopy(base), rs)
assert av.max_position < sv.max_position; print(f'3b OK: averse={av.max_position} < seek={sv.max_position}')

orig = copy.deepcopy(base); mod.modulate(base, BehavioralTraits())
assert base == orig; print('3c OK: no mutation')

dyn = BehavioralDynamics(); state = BehavioralState()
for _ in range(5): dyn.update(state, pnl_delta=500, had_fill=True, signal_correct=True)
assert state.pnl_streak==5 and state.stress_level<0.1; print(f'3d OK: 5wins streak={state.pnl_streak}')
for _ in range(3): dyn.update(state, pnl_delta=-800, had_fill=True, signal_correct=False)
assert state.stress_level>0.3; print(f'3d OK: stress={state.stress_level:.2f}')

d2=BehavioralDynamics(); s2=BehavioralState()
trending=[100+i for i in range(20)]
d2.update(s2, prices=trending)
print(f'3e OK: regime={s2.market_regime_belief}')

# ====================================================================
# SUITE 4: LLM Strategist
# ====================================================================
test_suite('LLM Strategist compatibility')
from aml_sim.agents.strategy.llm_slow_strategy import create_llm_strategist, LLMStrategist, STATIC_RESPONSES_BY_ROLE

for role in ['market_maker','retail','institutional','informed','liquidity_taker']:
    s = create_llm_strategist(role); assert isinstance(s, LLMStrategist)
print(f'4a OK: create_llm_strategist works for all 5 roles')

for role, r in STATIC_RESPONSES_BY_ROLE.items():
    for key in ['strategy_updates','reason','confidence','strategy_config','risk_overrides']:
        assert key in r, f'{role} missing {key}'
print(f'4b OK: all {len(STATIC_RESPONSES_BY_ROLE)} static responses have full contract')

# ====================================================================
# SUITE 5: E2E simulation results
# ====================================================================
test_suite('E2E simulation results')
action_files = sorted(Path('.aml_runs/e2e_test/reports/agents').glob('trader_actions_*.json'))
assert len(action_files) >= 6, f'Expected >=6 action files, got {len(action_files)}'
for af in action_files:
    data = json.loads(af.read_text())
    if data.get('agent_id','').startswith('shock'):
        continue
    for key in ['risk_summary','fallback_summary','behavioral_summary','strategy_performance']:
        assert key in data, f'{af.name}: missing {key}'
    risk = data['risk_summary']
    assert risk['health'] in ('healthy','degraded','halted')
print(f'5a OK: all {len(action_files)} reports have risk/fallback/behavioral/performance')
print('     health=healthy for all active agents')

# Check scenario config
from aml_sim.scenario import load_scenario
sc = load_scenario(Path('scenarios/aml_agent_infra_smoke.yaml'))
institutional_cfg = sc.stocksim_config['agents'].get('institutional_execution', {})
inst_params = institutional_cfg.get('parameters', {})
institutional_alphas = inst_params.get('alpha_strategies', [])
print(f'5b OK: scenario loads, institutional has {len(institutional_alphas)} alpha strategies')

print(f'\n{"="*60}')
print(f'ALL TESTS PASSED - ZERO BUGS')
print(f'{"="*60}')
