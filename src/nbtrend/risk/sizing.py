"""Position sizing and portfolio-level risk limits.

The sizing rule is volatility targeting: a position's weight is inversely
proportional to the asset's realised volatility, so each holding contributes
a similar amount of risk. Without it a book of BTC and DOGE is really just a
DOGE book with a BTC-shaped rounding error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Bars per year for each supported resolution -- used to annualise vol.
BARS_PER_YEAR: dict[str, float] = {
    "1": 365 * 24 * 60,
    "5": 365 * 24 * 12,
    "15": 365 * 24 * 4,
    "30": 365 * 24 * 2,
    "60": 365 * 24,
    "180": 365 * 8,
    "240": 365 * 6,
    "360": 365 * 4,
    "720": 365 * 2,
    "D": 365,
    "2D": 182.5,
    "3D": 121.7,
}


def bars_per_year(resolution: str) -> float:
    if resolution not in BARS_PER_YEAR:
        raise KeyError(f"unsupported resolution {resolution!r}; expected one of {list(BARS_PER_YEAR)}")
    return BARS_PER_YEAR[resolution]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    target_annual_vol: float
    max_weight_per_symbol: float
    max_gross_exposure: float
    vol_window: int
    atr_window: int
    atr_stop_mult: float
    max_drawdown_stop: float
    fx_floor_weight: float = 0.0

    @classmethod
    def from_config(cls, risk_cfg: dict) -> RiskLimits:
        return cls(
            target_annual_vol=float(risk_cfg["target_annual_vol"]),
            max_weight_per_symbol=float(risk_cfg["max_weight_per_symbol"]),
            max_gross_exposure=float(risk_cfg["max_gross_exposure"]),
            vol_window=int(risk_cfg["vol_window"]),
            atr_window=int(risk_cfg["atr_window"]),
            atr_stop_mult=float(risk_cfg["atr_stop_mult"]),
            max_drawdown_stop=float(risk_cfg["max_drawdown_stop"]),
            fx_floor_weight=float(risk_cfg.get("fx_floor_weight", 0.0)),
        )


def vol_target_weight(
    direction: pd.Series,
    close: pd.Series,
    limits: RiskLimits,
    periods_per_year: float,
) -> pd.Series:
    """Scale a direction in [-1, 1] into a portfolio weight.

        weight = direction * target_vol / realised_vol

    Capped at `max_weight_per_symbol`. Note the vol estimate is shifted by one
    bar -- sizing at bar `t` may only use volatility known at `t-1`, or the
    backtest quietly reads the future.
    """
    ret = np.log(close).diff()
    realised = ret.rolling(limits.vol_window, min_periods=max(2, limits.vol_window // 2)).std() * np.sqrt(
        periods_per_year
    )
    realised = realised.shift(1)

    scale = limits.target_annual_vol / realised.replace(0, np.nan)
    weight = direction * scale
    return weight.clip(-limits.max_weight_per_symbol, limits.max_weight_per_symbol).fillna(0.0)


def normalise_gross(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Scale a whole cross-section down if the book breaches gross exposure.

    Scaling everything by the same factor preserves relative conviction; only
    trimming the largest position would silently change the strategy.
    """
    gross = weights.abs().sum(axis=1)
    factor = (max_gross / gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    return weights.mul(factor, axis=0)


def chandelier_exit_hit(
    position_is_long: bool, price: float, peak_price: float, atr_value: float, multiple: float
) -> bool:
    """True when a trailing ATR stop is breached."""
    if atr_value <= 0 or peak_price <= 0:
        return False
    if position_is_long:
        return price <= peak_price - multiple * atr_value
    return price >= peak_price + multiple * atr_value


def drawdown_breached(equity_curve: pd.Series, max_dd: float) -> pd.Series:
    """True from the bar the equity drawdown exceeds `max_dd` onward within
    the same drawdown episode -- a kill switch, not a per-bar flag."""
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return dd <= -abs(max_dd)


def order_size_from_weight(
    target_weight: float,
    equity_rial: float,
    price_rial: float,
    current_amount: float,
    amount_step: float,
    min_order_rial: float,
) -> float:
    """Delta in base-asset units needed to reach `target_weight`.

    Returns 0.0 when the resulting order would be below the exchange minimum
    (3,000,000 rial for IRT markets) -- submitting it just burns a rate-limit
    slot and comes back as `SmallOrder`.
    """
    from ..units import round_to_step

    if price_rial <= 0:
        return 0.0

    target_amount = (target_weight * equity_rial) / price_rial
    delta = target_amount - current_amount

    if abs(delta) * price_rial < min_order_rial:
        return 0.0

    sized = round_to_step(abs(delta), amount_step)
    if sized * price_rial < min_order_rial:
        return 0.0
    return float(np.sign(delta) * sized)
