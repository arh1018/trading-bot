"""Event-driven backtest.

Signals come from the global USD series; fills happen on the local rial
series. Equity is rial throughout, which is the honest accounting for a
Tehran-based book -- a strategy can be flat in dollar terms and still make or
lose money as the rial moves.

Lookahead discipline
--------------------
Everything here is engineered around one rule: a decision made from bar `t`
executes at the OPEN of bar `t+1`. The signal frame is shifted once, at the
top of `run`, and nothing downstream may use `t`'s close. The three places
this usually leaks in a trend backtest are (a) sizing off same-bar
volatility, (b) Donchian channels that include the current bar, and (c)
stops checked against the same close that generated the entry. All three are
handled -- see `sizing.vol_target_weight`, `indicators.donchian` and the
intrabar stop below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..features.signals import apply_hysteresis, composite_score
from ..risk import sizing
from ..risk.sizing import RiskLimits
from ..units import round_to_step

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp | None
    entry_price: float
    exit_price: float
    amount: float
    pnl_rial: float
    return_pct: float
    bars_held: int
    exit_reason: str


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    positions: pd.DataFrame
    trades: list[Trade]
    signals: pd.DataFrame
    initial_equity: float
    costs_paid: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_equity - 1.0) if len(self.equity) else 0.0


class Backtester:
    """Single-symbol backtest. `PortfolioBacktester` composes these."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.limits = RiskLimits.from_config(cfg.risk)
        self.costs = cfg.costs
        self.strategy_cfg = cfg.strategy
        self.resolution = str(cfg.data["timeframe"])
        self.periods_per_year = sizing.bars_per_year(self.resolution)

    def build_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score -> direction -> target weight, all on the global series."""
        scores = composite_score(frame, self.strategy_cfg, with_components=True)

        thresholds = self.strategy_cfg["thresholds"]
        direction = apply_hysteresis(
            scores["score"],
            scores["regime"],
            float(thresholds["entry"]),
            float(thresholds["exit"]),
            bool(self.strategy_cfg.get("allow_short", False)),
        )

        weight = sizing.vol_target_weight(
            direction, frame["close"], self.limits, self.periods_per_year
        )

        out = scores.copy()
        out["direction"] = direction
        out["target_weight"] = weight
        return out

    def run(
        self,
        frame: pd.DataFrame,
        symbol: str,
        amount_step: float = 1e-8,
        initial_equity: float | None = None,
    ) -> BacktestResult:
        """`frame` needs global o/h/l/c/v plus `local_close` (rial)."""
        equity0 = float(initial_equity or self.cfg.backtest["initial_equity_rial"])

        signals = self.build_signals(frame)
        data = frame.join(signals[["score", "regime", "direction", "target_weight"]])

        # Execute on local rial prices; fall back to the synthetic fair price
        # where Nobitex history has gaps.
        local = data.get("local_close")
        if local is None:
            raise ValueError("frame has no `local_close` column")
        exec_price = local.ffill()
        if "fair_rial" in data:
            exec_price = exec_price.fillna(data["fair_rial"])
        data = data.assign(exec_price=exec_price).dropna(subset=["exec_price"])

        # THE shift: act at t+1 on information from t.
        target_w = data["target_weight"].shift(1).fillna(0.0).to_numpy()
        prices = data["exec_price"].to_numpy()
        index = data.index

        from ..features.indicators import atr as atr_fn
        atr_series = atr_fn(data["high"], data["low"], data["close"], self.limits.atr_window)
        # ATR is in USD; convert to a fraction so it applies to the rial price.
        atr_frac = (atr_series / data["close"]).shift(1).fillna(0.0).to_numpy()

        fee = float(self.costs["taker_fee"])
        min_rebalance = float(self.cfg.execution.get("min_rebalance_weight", 0.0))
        slip = float(self.costs["slippage"])
        min_order = float(self.costs["min_order_rial"])

        n = len(data)
        equity = np.zeros(n)
        weights = np.zeros(n)
        amounts = np.zeros(n)

        cash = equity0
        amount = 0.0
        entry_price = 0.0
        entry_i = 0
        peak_price = 0.0
        costs_paid = 0.0
        trades: list[Trade] = []
        halted = False

        for i in range(n):
            price = prices[i]
            if price <= 0 or not np.isfinite(price):
                equity[i] = cash + amount * (prices[i - 1] if i else 0)
                continue

            mark = cash + amount * price

            # --- trailing stop, checked before the new target is applied ---
            if amount > 0:
                peak_price = max(peak_price, price)
                stop_distance = self.limits.atr_stop_mult * atr_frac[i] * peak_price
                if stop_distance > 0 and price <= peak_price - stop_distance:
                    proceeds = amount * price * (1 - fee - slip)
                    cost = amount * price * (fee + slip)
                    cash += proceeds
                    costs_paid += cost
                    trades.append(
                        _make_trade(symbol, index[entry_i], index[i], entry_price, price,
                                    amount, i - entry_i, "atr_stop")
                    )
                    amount = 0.0
                    peak_price = 0.0
                    mark = cash

            # --- portfolio kill switch ---
            if not halted and equity[:i].size:
                peak_equity = max(equity[:i].max(), equity0)
                if mark / peak_equity - 1.0 <= -self.limits.max_drawdown_stop:
                    halted = True
                    log.info("%s: drawdown stop hit at %s", symbol, index[i])

            desired_w = 0.0 if halted else target_w[i]
            current_w = (amount * price / mark) if mark > 0 else 0.0

            # No-trade band: ignore drift, but always act on a full exit or a
            # fresh entry -- those are the signal, not noise.
            is_state_change = (desired_w == 0.0) != (current_w == 0.0)
            if not is_state_change and abs(desired_w - current_w) < min_rebalance:
                equity[i] = cash + amount * price
                amounts[i] = amount
                weights[i] = current_w
                continue

            target_amount = desired_w * mark / price
            delta = target_amount - amount

            if abs(delta) * price >= min_order:
                sized = round_to_step(abs(delta), amount_step) * np.sign(delta)
                notional = abs(sized) * price
                if notional >= min_order:
                    cost = notional * (fee + slip)
                    if sized > 0:
                        needed = notional + cost
                        if needed <= cash:
                            cash -= needed
                            new_amount = amount + sized
                            entry_price = (
                                (entry_price * amount + price * sized) / new_amount
                                if new_amount else price
                            )
                            if amount == 0.0:
                                entry_i = i
                                peak_price = price
                            amount = new_amount
                            costs_paid += cost
                    else:
                        closing = min(abs(sized), amount)
                        if closing > 0:
                            proceeds = closing * price
                            cost = proceeds * (fee + slip)
                            cash += proceeds - cost
                            costs_paid += cost
                            amount -= closing
                            if amount <= amount_step:
                                trades.append(
                                    _make_trade(symbol, index[entry_i], index[i], entry_price,
                                                price, closing, i - entry_i, "signal")
                                )
                                amount = 0.0
                                peak_price = 0.0

            equity[i] = cash + amount * price
            amounts[i] = amount
            weights[i] = (amount * price / equity[i]) if equity[i] else 0.0

        if amount > 0:
            trades.append(
                _make_trade(symbol, index[entry_i], index[-1], entry_price, prices[-1],
                            amount, n - 1 - entry_i, "open")
            )

        return BacktestResult(
            equity=pd.Series(equity, index=index, name="equity"),
            positions=pd.DataFrame({"amount": amounts, "weight": weights}, index=index),
            trades=trades,
            signals=signals,
            initial_equity=equity0,
            costs_paid=costs_paid,
            metadata={"symbol": symbol, "resolution": self.resolution, "bars": n},
        )


def _make_trade(
    symbol: str, entry_ts, exit_ts, entry_price: float, exit_price: float,
    amount: float, bars: int, reason: str,
) -> Trade:
    pnl = (exit_price - entry_price) * amount
    return Trade(
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=entry_price,
        exit_price=exit_price,
        amount=amount,
        pnl_rial=pnl,
        return_pct=(exit_price / entry_price - 1.0) if entry_price else 0.0,
        bars_held=bars,
        exit_reason=reason,
    )
