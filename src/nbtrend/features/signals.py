"""Trend estimators and the composite score.

Why these three, and why blended
--------------------------------
Trend following has one signal and many parameterisations. Each estimator
below sees the same trend through a different lens, and each fails in a
different way, so an equal-ish blend has materially lower turnover and higher
risk-adjusted return than any single one:

* `macd_trend`   -- Baz et al. (2015), "Dissecting Investment Strategies in the
  Cross Section and Time Series". Volatility-normalised MACD across three
  timescales, squashed by an intermediate response function. Continuous, so
  it sizes conviction rather than flipping between +1/-1. This is the closest
  thing the CTA literature has to a standard trend metric.
* `tsmom`        -- Moskowitz, Ooi & Pedersen (2012). Risk-adjusted momentum:
  the return over a lookback divided by the volatility over that lookback, so
  a 10% move in a quiet asset outranks a 10% move in a wild one. Robust and
  nearly parameter-free.
* `donchian_trend` -- price-level breakout. Slow to enter but it is the only
  one of the three that anchors to actual support/resistance rather than to a
  moving average, so it catches regime changes the smoothers lag.

All three are computed on the **global USD price**, never on the local rial
price. The rial series is contaminated by USDT/IRT moves, which are driven by
Iranian macro and capital controls, not by crypto trend. Signalling on IRT
directly means half your "momentum" is a bet on rial devaluation that the
model cannot see, cannot size, and cannot exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _response(x: pd.Series) -> pd.Series:
    """Baz et al. intermediate response function.

        f(x) = x * exp(-x^2 / 4) / 0.89

    Peaks around |x| ~ 1.41 and decays after. The decay is the point: a trend
    that is already 3 standard deviations extended is more likely to mean
    revert than to continue, so conviction should *fall*, not saturate. A
    plain tanh or clip would hold full size into the blow-off top.
    """
    return x * np.exp(-(x**2) / 4.0) / 0.89


def macd_trend(
    close: pd.Series,
    pairs: list[tuple[int, int]],
    short_norm_window: int = 63,
    long_norm_window: int = 252,
) -> pd.Series:
    """Volatility-normalised multi-timescale MACD, mapped to roughly [-1, 1]."""
    signals: list[pd.Series] = []
    for fast, slow in pairs:
        macd = ind.ema_halflife(close, fast) - ind.ema_halflife(close, slow)
        # Normalise by price volatility -> comparable across assets and eras.
        price_std = close.rolling(short_norm_window, min_periods=short_norm_window // 2).std()
        q = macd / price_std.replace(0, np.nan)
        # Normalise again by its own history -> comparable across timescales.
        q_std = q.rolling(long_norm_window, min_periods=long_norm_window // 4).std()
        signals.append(_response(q / q_std.replace(0, np.nan)))

    combined = pd.concat(signals, axis=1).mean(axis=1)
    return combined.clip(-1.0, 1.0)


def tsmom(close: pd.Series, lookbacks: list[int], vol_window: int = 60) -> pd.Series:
    """Volatility-scaled time-series momentum across several lookbacks."""
    daily_vol = ind.log_returns(close).rolling(vol_window, min_periods=vol_window // 2).std()
    scores: list[pd.Series] = []
    for lb in lookbacks:
        raw = np.log(close / close.shift(lb))
        # Scale by the vol of an lb-bar move so lookbacks are comparable.
        scaled = raw / (daily_vol.replace(0, np.nan) * np.sqrt(lb))
        scores.append(np.tanh(scaled))
    return pd.concat(scores, axis=1).mean(axis=1).clip(-1.0, 1.0)


def donchian_trend(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 55, exit_window: int = 20
) -> pd.Series:
    """Stateful breakout: +1 after an upper-channel break, held until the
    faster opposite channel breaks. Returns a step function in {-1, 0, +1}.

    The asymmetric entry/exit windows are the Turtle design: enter slow to
    avoid noise, exit fast to protect the trend profit.
    """
    upper, lower = ind.donchian(high, low, window)
    exit_upper, exit_lower = ind.donchian(high, low, exit_window)

    long_entry = close > upper
    short_entry = close < lower
    long_exit = close < exit_lower
    short_exit = close > exit_upper

    state = np.zeros(len(close), dtype=float)
    current = 0.0
    le, se, lx, sx = (
        long_entry.to_numpy(), short_entry.to_numpy(),
        long_exit.to_numpy(), short_exit.to_numpy(),
    )
    for i in range(len(close)):
        if current > 0 and lx[i] or current < 0 and sx[i]:
            current = 0.0
        if current == 0.0:
            if le[i]:
                current = 1.0
            elif se[i]:
                current = -1.0
        state[i] = current

    return pd.Series(state, index=close.index)


def regime_ok(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    er_window: int = 30,
    min_er: float = 0.25,
    adx_window: int = 14,
    min_adx: float = 18.0,
) -> pd.Series:
    """True where the market is trending enough to be worth trading."""
    er = ind.efficiency_ratio(close, er_window)
    adx_ = ind.adx(high, low, close, adx_window)
    return (er >= min_er) & (adx_ >= min_adx)


def composite_score(
    df: pd.DataFrame,
    cfg: dict,
    with_components: bool = False,
) -> pd.DataFrame:
    """Blend the estimators into one score in [-1, +1].

    `df` must have open/high/low/close/volume columns on the GLOBAL price.
    Returns a frame with `score`, `regime`, and (optionally) each component.
    """
    sig_cfg = cfg["signal"]
    reg_cfg = cfg["regime"]
    close, high, low = df["close"], df["high"], df["low"]

    components: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}

    m = sig_cfg["macd"]
    components["macd"] = macd_trend(
        close,
        [tuple(p) for p in m["pairs"]],
        m["short_norm_window"],
        m["long_norm_window"],
    )
    weights["macd"] = float(m["weight"])

    t = sig_cfg["tsmom"]
    components["tsmom"] = tsmom(close, list(t["lookbacks"]), t["vol_window"])
    weights["tsmom"] = float(t["weight"])

    d = sig_cfg["donchian"]
    components["donchian"] = donchian_trend(high, low, close, d["window"], d["exit_window"])
    weights["donchian"] = float(d["weight"])

    total_w = sum(weights.values()) or 1.0
    score = sum(components[k] * (weights[k] / total_w) for k in components)

    regime = regime_ok(
        high, low, close,
        reg_cfg["efficiency_ratio_window"], reg_cfg["min_efficiency_ratio"],
        reg_cfg["adx_window"], reg_cfg["min_adx"],
    )

    out = pd.DataFrame({"score": score.clip(-1, 1), "regime": regime.fillna(False)}, index=df.index)
    if with_components:
        for k, v in components.items():
            out[f"c_{k}"] = v
    return out


def apply_hysteresis(
    score: pd.Series, regime: pd.Series, entry: float, exit_: float, allow_short: bool
) -> pd.Series:
    """Convert a continuous score into a held position direction.

    Without hysteresis a score oscillating around a single threshold produces
    a trade per bar; at ~0.55% round-trip cost on Nobitex that is the whole
    edge. The band between `exit_` and `entry` is dead space where an existing
    position is kept and a new one is not opened.
    """
    s = score.fillna(0.0).to_numpy()
    r = regime.fillna(False).to_numpy()
    out = np.zeros(len(s), dtype=float)
    current = 0.0

    for i in range(len(s)):
        if current > 0 and s[i] < exit_ or current < 0 and s[i] > -exit_:
            current = 0.0

        if current == 0.0 and r[i]:
            if s[i] >= entry:
                current = 1.0
            elif allow_short and s[i] <= -entry:
                current = -1.0
        out[i] = current

    return pd.Series(out, index=score.index)
