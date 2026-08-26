"""Indicators must be causal. A lookahead here silently inflates every backtest."""

import numpy as np
import pandas as pd
import pytest

from nbtrend.features import indicators as ind


@pytest.fixture
def ohlc():
    rng = np.random.default_rng(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 300))))
    return pd.DataFrame(
        {
            "close": close,
            "high": close * (1 + abs(rng.normal(0, 0.005, 300))),
            "low": close * (1 - abs(rng.normal(0, 0.005, 300))),
        }
    )


def test_donchian_excludes_the_current_bar(ohlc):
    """The channel at t must be formed only from bars before t, or every new
    high trivially breaks its own channel."""
    upper, _ = ind.donchian(ohlc["high"], ohlc["low"], 20)
    for i in range(50, 100):
        expected = ohlc["high"].iloc[i - 20 : i].max()
        assert upper.iloc[i] == pytest.approx(expected)


def test_atr_is_positive_and_warms_up(ohlc):
    atr = ind.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    assert atr.iloc[:13].isna().all()
    assert (atr.dropna() > 0).all()


def test_efficiency_ratio_is_one_for_a_straight_line():
    line = pd.Series(np.arange(100, dtype=float))
    er = ind.efficiency_ratio(line, 20)
    assert er.dropna().max() == pytest.approx(1.0)


def test_efficiency_ratio_is_near_zero_for_pure_chop():
    chop = pd.Series([100.0, 101.0] * 50)
    er = ind.efficiency_ratio(chop, 20)
    assert er.dropna().max() < 0.1


def test_adx_is_bounded(ohlc):
    adx = ind.adx(ohlc["high"], ohlc["low"], ohlc["close"], 14).dropna()
    assert ((adx >= 0) & (adx <= 100)).all()


def test_indicators_are_causal(ohlc):
    """Truncating the future must not change any past value."""
    full = ind.efficiency_ratio(ohlc["close"], 20)
    truncated = ind.efficiency_ratio(ohlc["close"].iloc[:200], 20)
    pd.testing.assert_series_equal(full.iloc[:200], truncated, check_names=False)
