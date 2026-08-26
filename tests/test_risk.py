"""Sizing and portfolio-level exposure limits."""

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
    assert gross == pytest.approx(float(cfg.risk["max_gross_exposure"]))
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
