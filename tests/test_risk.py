"""Sizing and portfolio-level exposure limits."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from nbtrend.config import load_config
from nbtrend.risk.sizing import (
    RiskLimits,
    bars_per_year,
    chandelier_exit_hit,
    drawdown_breached,
    normalise_gross,
    order_size_from_weight,
    vol_target_weight,
)


@pytest.fixture
def limits():
    return RiskLimits.from_config(load_config().risk)


def test_bars_per_year_matches_the_calendar():
    assert bars_per_year("D") == 365
    assert bars_per_year("240") == 365 * 6
    assert bars_per_year("60") == 365 * 24
    with pytest.raises(KeyError):
        bars_per_year("7m")


def test_vol_targeting_sizes_down_a_volatile_asset(limits):
    rng = np.random.default_rng(1)
    quiet = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))))
    wild = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 300))))
    direction = pd.Series(1.0, index=quiet.index)

    w_quiet = vol_target_weight(direction, quiet, limits, 2190).iloc[-1]
    w_wild = vol_target_weight(direction, wild, limits, 2190).iloc[-1]
    assert w_quiet > w_wild


def test_vol_targeting_respects_the_per_symbol_cap(limits):
    flat = pd.Series(np.full(300, 100.0) + np.arange(300) * 1e-9)
    direction = pd.Series(1.0, index=flat.index)
    weights = vol_target_weight(direction, flat, limits, 2190)
    assert weights.max() <= limits.max_weight_per_symbol + 1e-12


def test_vol_estimate_does_not_use_the_current_bar(limits):
    """Sizing at t may only use volatility known at t-1."""
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    direction = pd.Series(1.0, index=close.index)

    full = vol_target_weight(direction, close, limits, 2190)
    truncated = vol_target_weight(direction.iloc[:200], close.iloc[:200], limits, 2190)
    pd.testing.assert_series_equal(full.iloc[:200], truncated, check_names=False)


def test_normalise_gross_scales_the_whole_book():
    weights = pd.DataFrame({"a": [0.5, 0.1], "b": [0.7, 0.2]})
    out = normalise_gross(weights, max_gross=1.0)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    # A row already inside the cap must be left alone.
    assert out.iloc[1].tolist() == pytest.approx([0.1, 0.2])


def test_live_runner_caps_gross_exposure():
    """Four symbols at the 0.35 per-symbol cap is 1.4x gross -- impossible on
    a spot book, and it fails as InsufficientBalance rather than loudly."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.35, 0.35, 0.35, 0.277)]
    runner._cap_gross(states)

    gross = sum(s.target_weight for s in states)
    # At or just under the cap -- `_cap_gross` also reserves a cost allowance,
    # see test_gross_cap_reserves_cash_for_fees.
    assert gross <= float(cfg.risk["max_gross_exposure"])
    assert gross == pytest.approx(float(cfg.risk["max_gross_exposure"]), rel=0.01)
    # Relative conviction preserved.
    assert states[0].target_weight > states[3].target_weight


def test_gross_cap_leaves_a_small_book_alone():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.2, 0.3)]
    runner._cap_gross(states)
    assert [s.target_weight for s in states] == pytest.approx([0.2, 0.3])


def test_order_size_skips_below_the_exchange_minimum():
    assert order_size_from_weight(0.0001, 1e9, 1.55e11, 0.0, 1e-6, 3e6) == 0.0
    assert order_size_from_weight(0.3, 1e9, 1.55e11, 0.0, 1e-6, 3e6) > 0


def test_chandelier_stop_triggers_only_past_the_threshold():
    assert not chandelier_exit_hit(True, price=95, peak_price=100, atr_value=2, multiple=3)
    assert chandelier_exit_hit(True, price=93, peak_price=100, atr_value=2, multiple=3)


def test_drawdown_kill_switch():
    equity = pd.Series([100.0, 120.0, 110.0, 85.0])
    breached = drawdown_breached(equity, 0.25)
    assert not breached.iloc[2]
    assert breached.iloc[3]


def test_gross_cap_reserves_cash_for_fees():
    """Investing exactly 100% of equity overdraws, because fees are charged on
    top of notional. Observed in a shadow test as a negative `rls` balance."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.35, 0.35, 0.35, 0.277)]
    runner._cap_gross(states)

    gross = sum(s.target_weight for s in states)
    cost_rate = float(cfg.costs["taker_fee"]) + float(cfg.costs["slippage"])
    assert gross < float(cfg.risk["max_gross_exposure"])
    # Enough cash left to pay the fees on the whole book.
    assert 1.0 - gross >= gross * cost_rate * 0.99


def test_position_limit_is_derived_from_equity():
    """A wide universe on a small account otherwise places no orders at all:
    every position falls under the 3,000,000 rial minimum and is skipped."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:10]
    states = [SymbolState(spec=s, target_weight=0.3, score=0.9 - i * 0.05)
              for i, s in enumerate(specs)]

    # 10M rial / 3M minimum -> 3 fundable positions.
    runner._limit_positions(states, equity=10_000_000)
    funded = [s for s in states if s.target_weight != 0.0]
    assert len(funded) == 3


def test_position_limit_keeps_the_highest_conviction_names():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:5]
    scores = [0.1, 0.9, 0.3, 0.8, 0.2]
    states = [SymbolState(spec=s, target_weight=0.3, score=sc)
              for s, sc in zip(specs, scores, strict=True)]

    runner._limit_positions(states, equity=7_000_000)   # funds 2
    funded = {s.spec.nobitex for s in states if s.target_weight != 0.0}
    assert funded == {specs[1].nobitex, specs[3].nobitex}


def test_a_large_account_keeps_every_signal():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    specs = cfg.universe[:5]
    states = [SymbolState(spec=s, target_weight=0.3, score=0.5) for s in specs]

    runner._limit_positions(states, equity=10_000_000_000)
    assert all(s.target_weight != 0.0 for s in states)


def test_paper_equity_peak_cannot_halt_a_live_account():
    """A paper run ends with a 1e9 peak. Loaded into a live account holding
    10M that reads as -99% and trips the kill switch before the first order --
    observed live."""
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state.json"
    path.write_text(json.dumps({"equity_peak_rial": 1_000_000_000.0, "halted": True,
                                "last_bar_ts": {}}))
    try:
        state = RunnerState.load(path, current_equity=9_950_000)
        assert state.equity_peak_rial == 9_950_000
        assert not state.halted
    finally:
        path.unlink()


def test_a_plausible_peak_survives_reload():
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state2.json"
    path.write_text(json.dumps({"equity_peak_rial": 11_000_000.0, "halted": False,
                                "last_bar_ts": {}}))
    try:
        state = RunnerState.load(path, current_equity=9_950_000)
        assert state.equity_peak_rial == 11_000_000.0
    finally:
        path.unlink()


def test_state_path_is_namespaced_by_mode():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "nbtrend" / "live" / "runner.py"
    ).read_text()
    assert 'f"data/state/runner-{cfg.mode}.json"' in source


def test_gross_cap_can_push_a_kept_position_under_the_minimum():
    """Observed live: _limit_positions kept 3 names, then vol targeting and
    _cap_gross scaled two of them under 3,000,000 rial and both were rejected
    one at a time. Dropping the smallest frees weight for the survivors."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    equity = 9_950_000.0
    specs = cfg.universe[:3]
    weights = [0.350, 0.231, 0.197]      # the live numbers
    states = [SymbolState(spec=s, target_weight=w, score=0.9)
              for s, w in zip(specs, weights, strict=True)]

    runner._drop_unfundable(states, equity)

    live = [s for s in states if s.target_weight != 0.0]
    assert live, "should keep at least one fundable position"
    for s in live:
        assert s.target_weight * equity >= float(cfg.costs["min_order_rial"])


def test_drop_unfundable_leaves_a_well_funded_book_alone():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    specs = cfg.universe[:2]
    states = [SymbolState(spec=s, target_weight=0.35, score=0.9) for s in specs]

    runner._drop_unfundable(states, equity=1_000_000_000)
    assert all(s.target_weight == 0.35 for s in states)
