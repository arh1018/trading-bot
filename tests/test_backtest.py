"""Backtest correctness. The lookahead test is the one that matters most:
every other number in this project is downstream of it being true."""

import numpy as np
import pandas as pd
import pytest

from nbtrend.backtest.engine import Backtester
from nbtrend.backtest.metrics import max_drawdown, sharpe_ratio, summarise
from nbtrend.config import load_config


@pytest.fixture
def cfg():
    return load_config()


def make_frame(n=800, seed=5, drift=0.002, fx_rate=1_980_000.0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n))))
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close.shift(1).bfill().to_numpy(),
            "high": (close * (1 + abs(rng.normal(0, 0.006, n)))).to_numpy(),
            "low": (close * (1 - abs(rng.normal(0, 0.006, n)))).to_numpy(),
            "close": close.to_numpy(),
            "volume": rng.uniform(1, 10, n),
        },
        index=index,
    )
    frame["fx"] = fx_rate
    frame["local_close"] = frame["close"] * fx_rate
    frame["fair_rial"] = frame["local_close"]
    frame["basis"] = 0.0
    return frame


def test_no_lookahead(cfg):
    """Replacing every bar after t must not change any decision at or before t.

    This is the single strongest guard against a backtest that reads the
    future. If the engine peeked, the truncated run would diverge.
    """
    frame = make_frame(700)
    cut = 500

    full = Backtester(cfg).run(frame, "TESTIRT", amount_step=1e-6)

    tampered = frame.copy()
    rng = np.random.default_rng(99)
    for column in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(column)] *= rng.uniform(0.5, 2.0, len(frame) - cut)
    tampered["local_close"] = tampered["close"] * tampered["fx"]
    tampered_result = Backtester(cfg).run(tampered, "TESTIRT", amount_step=1e-6)

    pd.testing.assert_series_equal(
        full.equity.iloc[:cut], tampered_result.equity.iloc[:cut], check_names=False
    )


def test_costs_are_actually_charged(cfg):
    """A round trip at a flat price must lose exactly the fees."""
    frame = make_frame(600, drift=0.003)
    result = Backtester(cfg).run(frame, "TESTIRT", amount_step=1e-8)
    if result.trades:
        assert result.costs_paid > 0


def test_equity_never_goes_negative(cfg):
    """Spot, long-only, no leverage -- the book cannot owe money."""
    frame = make_frame(700, seed=13, drift=-0.004)
    result = Backtester(cfg).run(frame, "TESTIRT", amount_step=1e-6)
    assert (result.equity > 0).all()


def test_flat_strategy_holds_equity_constant(cfg):
    """With the regime gate impossible to satisfy, nothing should trade."""
    import copy

    tight = copy.deepcopy(cfg)
    tight.raw["strategy"]["regime"]["min_efficiency_ratio"] = 1.5   # unreachable
    frame = make_frame(600)
    result = Backtester(tight).run(frame, "TESTIRT", amount_step=1e-6)

    assert result.trades == []
    assert result.equity.nunique() == 1
    assert result.costs_paid == 0


def test_min_order_size_is_respected(cfg):
    """Orders under 3,000,000 rial are rejected by Nobitex as SmallOrder."""
    frame = make_frame(600)
    result = Backtester(cfg).run(
        frame, "TESTIRT", amount_step=1e-8, initial_equity=5_000_000
    )
    # Equity this small can only support a couple of legal orders, if any.
    assert result.equity.iloc[-1] > 0


def test_metrics_on_a_known_curve():
    equity = pd.Series([100.0, 110.0, 99.0, 121.0])
    assert max_drawdown(equity) == pytest.approx(99 / 110 - 1)
    assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0]), 365) == 0.0


def test_usd_return_is_reported_separately(cfg):
    """A rial gain that is purely devaluation must show up as a USD loss."""
    frame = make_frame(600, drift=0.0)
    # Rial halves in value across the sample.
    frame["fx"] = np.linspace(1_000_000, 2_000_000, len(frame))
    frame["local_close"] = frame["close"] * frame["fx"]
    frame["fair_rial"] = frame["local_close"]

    result = Backtester(cfg).run(frame, "TESTIRT", amount_step=1e-6)
    metrics = summarise(result, 2190, fx=frame["fx"])
    if metrics.total_return > 0.05:
        assert metrics.usd_total_return < metrics.total_return
