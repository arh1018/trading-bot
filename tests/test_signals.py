"""Signal behaviour: direction, regime gating, and turnover control."""

import numpy as np
import pandas as pd
import pytest
import yaml

from nbtrend.config import PROJECT_ROOT
from nbtrend.features.signals import (
    apply_hysteresis,
    composite_score,
    donchian_trend,
    macd_trend,
    tsmom,
)


@pytest.fixture
def strategy_cfg():
    return yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())["strategy"]


def _frame(returns):
    close = pd.Series(100 * np.exp(np.cumsum(returns)))
    return pd.DataFrame(
        {
            "open": close.shift(1).bfill(),
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": np.ones(len(close)),
        }
    )


def test_macd_is_positive_in_an_uptrend():
    df = _frame(np.full(400, 0.004))
    signal = macd_trend(df["close"], [(8, 24), (16, 48), (32, 96)]).dropna()
    assert signal.iloc[-50:].mean() > 0


def test_macd_is_negative_in_a_downtrend():
    df = _frame(np.full(400, -0.004))
    signal = macd_trend(df["close"], [(8, 24), (16, 48), (32, 96)]).dropna()
    assert signal.iloc[-50:].mean() < 0


def test_macd_response_decays_when_overextended():
    """Baz et al.'s response function must fall, not saturate, at extremes --
    that is what stops the model holding full size into a blow-off top."""
    from nbtrend.features.signals import _response

    moderate = _response(pd.Series([1.4])).iloc[0]
    extreme = _response(pd.Series([5.0])).iloc[0]
    assert moderate > extreme
    assert extreme < 0.2


def test_tsmom_is_scaled_by_volatility():
    quiet = _frame(np.full(300, 0.002))
    noisy_returns = np.full(300, 0.002) + np.random.default_rng(0).normal(0, 0.05, 300)
    noisy = _frame(noisy_returns)
    q = tsmom(quiet["close"], [30, 60]).iloc[-1]
    n = tsmom(noisy["close"], [30, 60]).iloc[-1]
    assert q > n


def test_donchian_is_stateful_and_bounded():
    rng = np.random.default_rng(3)
    df = _frame(rng.normal(0.001, 0.02, 400))
    state = donchian_trend(df["high"], df["low"], df["close"], 55, 20)
    assert set(np.unique(state)).issubset({-1.0, 0.0, 1.0})


def test_hysteresis_cuts_turnover_versus_a_single_threshold():
    """A score oscillating across one level should not produce a trade a bar."""
    oscillating = pd.Series(0.3 + 0.05 * np.sin(np.arange(400) / 2))
    regime = pd.Series(True, index=oscillating.index)

    with_band = apply_hysteresis(oscillating, regime, entry=0.30, exit_=0.10, allow_short=False)
    naive = (oscillating >= 0.30).astype(float)

    assert with_band.diff().abs().sum() < naive.diff().abs().sum() / 5


def test_regime_gate_blocks_entries(strategy_cfg):
    score = pd.Series([0.9] * 100)
    closed = pd.Series([False] * 100)
    assert apply_hysteresis(score, closed, 0.3, 0.1, False).sum() == 0


def test_shorts_are_suppressed_unless_enabled():
    score = pd.Series([-0.9] * 100)
    regime = pd.Series([True] * 100)
    assert apply_hysteresis(score, regime, 0.3, 0.1, allow_short=False).min() == 0.0
    assert apply_hysteresis(score, regime, 0.3, 0.1, allow_short=True).min() == -1.0


def test_composite_score_is_bounded(strategy_cfg):
    rng = np.random.default_rng(11)
    df = _frame(rng.normal(0.001, 0.03, 600))
    out = composite_score(df, strategy_cfg)
    assert out["score"].dropna().between(-1, 1).all()
