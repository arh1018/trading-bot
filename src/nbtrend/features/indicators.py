"""Vectorised indicators. Every function takes and returns a pandas Series
indexed the same way as its input, and is strictly causal: the value at bar
`t` uses only data up to and including `t`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: float) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def ema_halflife(series: pd.Series, halflife: float) -> pd.Series:
    """EMA parameterised by halflife, which is how Baz et al. specify the
    MACD pairs -- not the same as `span`, and mixing them up shifts every
    crossover."""
    return series.ewm(halflife=halflife, adjust=False, min_periods=1).mean()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realised_vol(close: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    """Annualised realised volatility from log returns."""
    return log_returns(close).rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(
        periods_per_year
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ATR (an EMA with alpha = 1/window, not a simple mean)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index -- trend *strength*, direction-agnostic."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    alpha = 1.0 / window
    atr_ = true_range(high, low, close).ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()


def efficiency_ratio(close: pd.Series, window: int = 30) -> pd.Series:
    """Kaufman efficiency ratio: net directional travel / total path length.

    1.0 is a straight line, 0.0 is pure chop. This is the single cheapest
    regime filter for a trend follower -- trend estimators are unbiased but
    high-variance when ER is low, and the variance is what pays the fees.
    """
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window, min_periods=window).sum()
    return direction / volatility.replace(0, np.nan)


def donchian(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Upper/lower channel, shifted so bar `t` compares against the channel
    formed by the `window` bars *before* it. Without the shift the current bar
    sets its own high and the breakout is guaranteed -- the classic lookahead
    bug in Turtle implementations."""
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    return upper, lower


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def chandelier_stop(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int, multiple: float, long: bool = True
) -> pd.Series:
    """Trailing stop level at `multiple` ATRs from the extreme since entry."""
    atr_ = atr(high, low, close, window)
    if long:
        return high.rolling(window, min_periods=window).max() - multiple * atr_
    return low.rolling(window, min_periods=window).min() + multiple * atr_
