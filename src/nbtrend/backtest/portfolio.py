"""Cross-sectional backtest: does concentrating the book help?

The single-symbol engine cannot answer this. Concentration is a question about
the CROSS SECTION -- given 90 symbols signalling at once and a fixed pot, is it
better to hold the 5 strongest or spread across all 29 the account can fund?
That needs every symbol simulated against one shared cash balance.

Two real constraints drive the answer and are modelled here:

  * `min_order_rial` (3,000,000). This is why the live book holds 29 names: at
    ~90M equity that is simply the most positions that clear the minimum. It is
    a floor on position SIZE, so it also sets an upper bound on position COUNT.
  * Costs scale with turnover, and a wider book rebalances more names per bar.

Long-only and unlevered -- the point is to test concentration WITHOUT the
borrowing and liquidation risk that margin adds.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..features.indicators import atr as atr_fn
from ..risk.sizing import RiskLimits
from .engine import Backtester
from .metrics import max_drawdown

log = logging.getLogger(__name__)

HALT_RESUME_FRACTION = 0.5
"""Match the live runner: resume once drawdown recovers to half the stop."""


class PortfolioBacktester:
    def __init__(
        self,
        cfg,
        max_positions: int | None = None,
        max_synthetic_fraction: float = 0.30,
    ):
        self.cfg = cfg
        self.limits = RiskLimits.from_config(cfg.risk)
        self.costs = cfg.costs
        self.max_positions = max_positions
        self.max_synthetic_fraction = max_synthetic_fraction
        self.excluded: dict[str, str] = {}
        self._bt = Backtester(cfg)

    def _fund_book(
        self,
        desired: dict[str, float],
        ranks: pd.Series,
        equity: float,
        min_order: float,
        max_gross: float,
    ) -> dict[str, float]:
        """The live `_limit_positions`: the most names that can EACH be funded.

        Takes the highest-conviction candidates, scales them to the gross
        limit, and shrinks the count until every survivor clears the exchange
        minimum -- so the freed weight of a dropped name is what lifts the
        others over the line. Returns weights that are all fundable, or {}.
        """
        if not desired or equity <= 0 or min_order <= 0:
            return {}

        ordered = sorted(desired, key=lambda s: ranks.get(s, np.inf))
        ceiling = int((equity * max_gross) // min_order)
        if ceiling < 1:
            return {}
        k = min(ceiling, len(ordered), self.max_positions or len(ordered))

        while k >= 1:
            chosen = ordered[:k]
            gross = sum(abs(desired[s]) for s in chosen)
            if gross <= 0:
                return {}
            scale = min(1.0, max_gross / gross)
            weights = {s: desired[s] * scale for s in chosen}
            if all(abs(w) * equity >= min_order for w in weights.values()):
                return weights
            k -= 1
        return {}

    def run(self, frames: dict[str, pd.DataFrame], initial_equity: float | None = None) -> dict:
        equity0 = float(initial_equity or self.cfg.backtest["initial_equity_rial"])
        fee = float(self.costs["taker_fee"])
        slip = float(self.costs["slippage"])
        min_order = float(self.costs["min_order_rial"])
        min_rebalance = float(self.cfg.execution.get("min_rebalance_weight", 0.0))
        max_gross = float(self.limits.max_gross_exposure)

        # Per-symbol signals and execution prices, aligned on a shared index.
        weights, scores, prices, atrs = {}, {}, {}, {}
        self.excluded: dict[str, str] = {}
        for symbol, frame in frames.items():
            if frame.empty or len(frame) < 400:
                self.excluded[symbol] = f"only {len(frame)} bars"
                continue
            local = frame.get("local_close")
            if local is None:
                self.excluded[symbol] = "no local_close"
                continue

            # Thin markets have long gaps in Nobitex history, and the
            # `fair_rial` fallback is SYNTHETIC (global USD x FX) -- a price
            # that never traded here. Filling most of a series that way means
            # backtesting fills nobody could have got, so exclude a symbol
            # that leans on it too heavily rather than quietly trusting it.
            synthetic_frac = float(local.isna().mean())
            if synthetic_frac > self.max_synthetic_fraction:
                self.excluded[symbol] = f"{synthetic_frac:.0%} synthetic prices"
                continue

            sig = self._bt.build_signals(frame)
            px = local.ffill()
            if "fair_rial" in frame:
                px = px.fillna(frame["fair_rial"])
            weights[symbol] = sig["target_weight"]
            scores[symbol] = sig["score"]
            prices[symbol] = px
            # ATR as a fraction of price, so it applies to the rial series.
            atr_series = atr_fn(
                frame["high"], frame["low"], frame["close"], self.limits.atr_window
            )
            atrs[symbol] = (atr_series / frame["close"]).shift(1).fillna(0.0)

        if not weights:
            raise ValueError("no usable symbols")

        W = pd.DataFrame(weights).sort_index()
        S = pd.DataFrame(scores).reindex_like(W)
        P = pd.DataFrame(prices).reindex_like(W).ffill()

        # Keep only bars where at least one symbol is priced.
        usable = P.notna().any(axis=1)
        W, S, P = W[usable], S[usable], P[usable]

        # THE shift: act at t+1 on information known at t.
        W = W.shift(1).fillna(0.0)
        S = S.shift(1).fillna(0.0)

        A = pd.DataFrame(atrs).reindex_like(W).fillna(0.0)
        RANK = S.abs().rank(axis=1, ascending=False, method="first")

        index = W.index
        symbols = list(W.columns)
        amounts = dict.fromkeys(symbols, 0.0)
        peaks = dict.fromkeys(symbols, 0.0)
        cash = equity0
        costs_paid = 0.0
        trades = 0
        stops_hit = 0
        halts = 0
        halted = False
        equity_curve = np.zeros(len(index))
        held_counts = np.zeros(len(index))
        peak_equity = equity0

        def _mark(px_row) -> float:
            return cash + sum(
                amt * px_row[s] for s, amt in amounts.items() if amt and np.isfinite(px_row[s])
            )

        def _sell(symbol, price, units) -> None:
            nonlocal cash, costs_paid, trades
            notional = units * price
            cost = notional * (fee + slip)
            cash += notional - cost
            costs_paid += cost
            trades += 1
            amounts[symbol] -= units
            if amounts[symbol] * price < 1.0:
                amounts[symbol] = 0.0
                peaks[symbol] = 0.0

        for i, _ts in enumerate(index):
            px_row = P.iloc[i]
            atr_row = A.iloc[i]
            mark = _mark(px_row)
            if mark <= 0:
                equity_curve[i:] = 0.0
                break

            # --- per-symbol ATR trailing stop, before any new target ---
            # The single-symbol engine and the live runner both apply this;
            # omitting it here benchmarked a more aggressive strategy than the
            # one actually running, which is not a like-for-like comparison.
            for symbol in symbols:
                held = amounts[symbol]
                price = px_row[symbol]
                if not held or not np.isfinite(price) or price <= 0:
                    continue
                peaks[symbol] = max(peaks[symbol], price)
                stop_distance = self.limits.atr_stop_mult * float(atr_row[symbol]) * peaks[symbol]
                if stop_distance > 0 and price <= peaks[symbol] - stop_distance:
                    _sell(symbol, price, held)
                    stops_hit += 1

            mark = _mark(px_row)

            # --- portfolio drawdown kill switch, with hysteresis ---
            #
            # A permanent halt is not what the live runner does, and modelling
            # it as permanent makes every configuration look terrible for the
            # same uninformative reason: the book dies on its first bad
            # stretch and sits in cash for the remaining years. The live
            # runner resumes once equity recovers clear of the stop, so the
            # backtest has to as well or it is not measuring that strategy.
            peak_equity = max(peak_equity, mark)
            drawdown = mark / peak_equity - 1.0
            if not halted and drawdown <= -self.limits.max_drawdown_stop:
                halted = True
                halts += 1
                log.info("portfolio drawdown stop hit at %s", index[i])
            elif halted and drawdown > -self.limits.max_drawdown_stop * HALT_RESUME_FRACTION:
                halted = False
                log.info("portfolio drawdown recovered at %s", index[i])

            # --- fund the book the way the live runner does ---
            #
            # Spreading gross exposure across every signal and then dropping
            # whatever lands under the 3,000,000 minimum silently discards most
            # of the book: measured at 10.7 of 17.1 signals per bar, leaving
            # ~6 positions where the live account funds 29. `_limit_positions`
            # instead takes the highest-conviction names that can EACH clear
            # the minimum and re-spreads the freed weight across them, so the
            # capital is actually deployed.
            row_w = W.iloc[i]
            desired = {
                s: float(row_w[s])
                for s in symbols
                if row_w[s] != 0.0 and np.isfinite(px_row[s]) and px_row[s] > 0
            }
            funded_w = self._fund_book(desired, RANK.iloc[i], mark, min_order, max_gross)

            # Targets are set from START-OF-BAR equity, so every symbol is
            # sized against the same number rather than against whatever the
            # earlier symbols in the loop happened to leave behind.
            targets = {}
            for symbol in symbols:
                price = px_row[symbol]
                if not np.isfinite(price) or price <= 0:
                    continue
                target_w = 0.0 if halted else funded_w.get(symbol, 0.0)
                held = amounts[symbol]
                current_w = held * price / mark if mark > 0 else 0.0
                is_state_change = (target_w == 0.0) != (current_w == 0.0)
                if not is_state_change and abs(target_w - current_w) < min_rebalance:
                    continue
                targets[symbol] = (price, target_w * mark / price - held, target_w)

            # SELLS first, then BUYS. Interleaving them lets an early buy
            # exhaust cash that a later sell was about to provide, so the book
            # silently fails to reach its target for arbitrary, ordering-
            # dependent reasons.
            for symbol, (price, delta, _tw) in targets.items():
                if delta >= 0:
                    continue
                units = min(-delta, amounts[symbol])
                if units > 0 and units * price > 0:
                    _sell(symbol, price, units)

            for symbol, (price, delta, target_w) in targets.items():
                if delta <= 0:
                    continue
                notional = delta * price
                # The exchange minimum gates ENTRIES only; an exit must always
                # be allowed or a sub-minimum holding is stranded forever.
                if target_w != 0.0 and notional < min_order:
                    continue
                cost = notional * (fee + slip)
                if notional + cost > cash:
                    continue
                cash -= notional + cost
                costs_paid += cost
                trades += 1
                if amounts[symbol] == 0.0:
                    peaks[symbol] = price
                amounts[symbol] += delta

            equity_curve[i] = _mark(px_row)
            held_counts[i] = sum(1 for amt in amounts.values() if amt)

        curve = pd.Series(equity_curve, index=index).replace(0.0, np.nan).ffill().fillna(equity0)
        rets = curve.pct_change().fillna(0.0)
        ppy = self._bt.periods_per_year
        sharpe = (
            float(rets.mean() / rets.std() * np.sqrt(ppy)) if rets.std() > 0 else 0.0
        )
        years = len(curve) / ppy if ppy else 0.0
        total = float(curve.iloc[-1] / equity0 - 1.0)
        return {
            "total_return": total,
            "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else 0.0,
            "sharpe": sharpe,
            "max_drawdown": float(max_drawdown(curve)),
            "trades": trades,
            "costs_pct": costs_paid / equity0,
            "stops_hit": stops_hit,
            "halts": halts,
            "halted": halted,
            "avg_positions": float(held_counts[held_counts > 0].mean())
            if (held_counts > 0).any()
            else 0.0,
            "equity": curve,
        }
