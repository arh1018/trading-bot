"""A slow long book: hold crypto while it trends, hold rial when it does not.

WHY THIS EXISTS. The market maker in `maker.py` loses by construction on this
venue. Measured over 16 completed round trips its median GROSS edge was -5.8
bps, before 16 bps of fees -- it sold below where it bought. A resting bid only
fills when someone sells into it, so a passive book buys into weakness while
its ask sits above a market walking away; unable to short on spot, that loss is
one-sided. No amount of sizing, timing or filtering repairs a negative edge,
because all of those scale it rather than change its sign.

The opposite posture pays the spread ONCE rather than on every cycle. Measured
over 883 days against the only benchmark that matters on a rial account --
simply holding USDT, which captures devaluation with no market risk:

    hold USDT                    +264.6%   drawdown -25.7%
    hold the basket always       +364.9%   drawdown -36.5%
    hold it above its 100d mean  +536.7%   drawdown -32.0%    32 switches

And it survives the test that matters: split in half, the filtered book beats
USDT by +47.8% in the first half and +18.2% in the second. An edge present in
only one half is not an edge.

WHAT IT DOES NOT CLAIM. The parameter is not flat -- 80 days returns +628% and
120 days +363% -- so the window is a real choice, not a detail, and the future
will not repeat this sample. Being long crypto in rial terms loses money in a
genuine downturn, where USDT would not. The 100 day window is chosen over the
better-performing 80 because it sits between two neighbours that both work,
rather than on the peak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from ..core.types import OrderType, Side
from ..execution.base import new_client_order_id
from ..units import round_to_step

log = logging.getLogger(__name__)


@dataclass
class Target:
    """One market's share of the book, and what it would take to get there."""

    symbol: str
    weight: float
    held_rial: float
    target_rial: float

    @property
    def delta_rial(self) -> float:
        return self.target_rial - self.held_rial


@dataclass
class DriftBook:
    """Decides what the account should hold. Places nothing itself."""

    symbols: list[str]
    trend_window: int = 100
    # A trade must be worth more than the round trip it costs. One taker leg
    # plus half a spread is ~41 bps, so rebalancing for less than a few
    # percent of the target is paying to stand still.
    min_trade_rial: float = 2_000_000.0
    rebalance_band: float = 0.25
    # Exchange minimum, measured: 398,030 rejected and 497,538 accepted.
    min_order_rial: float = 550_000.0
    prices: dict[str, pd.Series] = field(default_factory=dict)

    def basket(self) -> pd.Series | None:
        """Equal-weight basket in USDT terms, or None without enough history.

        USDT terms on purpose: in rial everything trends up together, so a
        rial-denominated trend signal is mostly a devaluation detector and
        would stay long through a crypto bear market.
        """
        usable = {s: p for s, p in self.prices.items()
                  if s != "USDTIRT" and p is not None and len(p) > self.trend_window}
        usdt = self.prices.get("USDTIRT")
        if not usable or usdt is None or len(usdt) <= self.trend_window:
            return None
        frame = pd.DataFrame(usable).dropna()
        if frame.empty:
            return None
        aligned = usdt.reindex(frame.index).ffill()
        in_usdt = frame.div(aligned, axis=0).dropna()
        if len(in_usdt) <= self.trend_window:
            return None
        return in_usdt.div(in_usdt.iloc[0]).mean(axis=1)

    def risk_on(self) -> bool | None:
        """True to hold crypto, False to hold rial, None when unknown.

        None is not False. A missing signal must not be read as "sell
        everything" -- that turns a data outage into a full liquidation.
        """
        basket = self.basket()
        if basket is None or len(basket) <= self.trend_window:
            return None
        mean = basket.rolling(self.trend_window).mean().iloc[-1]
        if pd.isna(mean):
            return None
        return bool(basket.iloc[-1] > mean)

    def targets(self, equity_rial: float, held: dict[str, float]) -> list[Target]:
        """What each market should be worth, given the signal and the equity."""
        signal = self.risk_on()
        if signal is None:
            return []
        tradable = [s for s in self.symbols if s in self.prices]
        if not tradable:
            return []
        weight = (1.0 / len(tradable)) if signal else 0.0
        return [
            Target(symbol=s, weight=weight, held_rial=held.get(s, 0.0),
                   target_rial=equity_rial * weight)
            for s in tradable
        ]

    def orders(self, equity_rial: float, held: dict[str, float]) -> list[Target]:
        """Targets worth acting on: outside the band and above the minimum.

        The band is what stops a slow book from becoming a fast one. Without
        it every price tick is a small rebalance, and the costs that sank the
        market maker come straight back through the side door.
        """
        out = []
        for t in self.targets(equity_rial, held):
            delta = abs(t.delta_rial)
            if delta < self.min_trade_rial or delta < self.min_order_rial:
                continue
            # Selling out entirely is always allowed; the band applies to
            # trimming a position we are keeping.
            keeping = t.target_rial > 0 and t.held_rial > 0
            if keeping and delta < t.target_rial * self.rebalance_band:
                continue
            out.append(t)
        return sorted(out, key=lambda t: t.delta_rial)


class DriftRunner:
    """Applies a `DriftBook` to a live account."""

    def __init__(self, book: DriftBook, rest, dry_run: bool = True):
        self.book = book
        self.rest = rest
        self.dry_run = dry_run
        self.specs: dict = {}
        # Markets to leave alone entirely, whatever the signal says.
        self.protected: set[str] = set()
        self._prices_at = 0.0
        # Daily bars do not change between sweeps; this is the whole point of
        # a slow book, and re-reading them faster would only add rate limits.
        self.price_refresh_s = 3600.0

    def refresh_prices(self, days: int = 400) -> int:
        """Daily closes for every market in the book, plus USDT."""
        now = time.time()
        if now - self._prices_at < self.price_refresh_s and self.book.prices:
            return 0
        self._prices_at = now
        end = int(now)
        start = end - days * 86400
        loaded = 0
        for symbol in [*self.book.symbols, "USDTIRT"]:
            try:
                frame = self.rest.candles(symbol, "D", start, end)
            except Exception:
                log.debug("no candles for %s", symbol, exc_info=True)
                continue
            if frame is not None and len(frame):
                self.book.prices[symbol] = frame["close"]
                loaded += 1
        return loaded

    def held_rial(self) -> tuple[float, dict[str, float]]:
        """Equity and per-market value, both in rial."""
        balances = self.rest.balances_detailed()
        free, blocked = balances.get("rls", (0.0, 0.0))
        equity = free + blocked
        held: dict[str, float] = {}
        for symbol in self.book.symbols:
            spec = self.specs.get(symbol)
            src = getattr(spec, "src", None)
            if not src:
                continue
            units = sum(balances.get(src.lower(), (0.0, 0.0)))
            if units <= 0:
                continue
            try:
                price = self.rest.orderbook(symbol).best_bid
            except Exception:
                continue
            value = units * price
            held[symbol] = value
            equity += value
        return equity, held

    def rebalance(self) -> int:
        """One pass. Sells first, so their proceeds fund the buys."""
        self.refresh_prices()
        signal = self.book.risk_on()
        if signal is None:
            log.warning("no trend signal yet; leaving the book untouched")
            return 0

        equity, held = self.held_rial()
        if equity <= 0:
            log.warning("no equity readable; not trading")
            return 0

        orders = self.book.orders(equity, held)
        if not orders:
            log.info("book is within its band (%s); nothing to do",
                     "risk on" if signal else "risk off")
            return 0

        placed = 0
        for target in orders:
            spec = self.specs.get(target.symbol)
            if spec is None or target.symbol in self.protected:
                continue
            side = Side.SELL if target.delta_rial < 0 else Side.BUY
            try:
                book_top = self.rest.orderbook(target.symbol)
            except Exception:
                log.warning("%s: no book; skipping", target.symbol)
                continue
            price = book_top.best_bid if side is Side.SELL else book_top.best_ask
            if price <= 0:
                continue
            amount = round_to_step(abs(target.delta_rial) / price, spec.amount_step)
            if amount <= 0 or amount * price < self.book.min_order_rial:
                continue

            if self.dry_run:
                log.info(
                    "[dry-run] %s would %s %.8f (%s rial, holding %s of %s)",
                    target.symbol, side.value, amount,
                    f"{amount * price:,.0f}",
                    f"{target.held_rial:,.0f}", f"{target.target_rial:,.0f}",
                )
                placed += 1
                continue
            try:
                self.rest.add_order(
                    src=spec.src, dst=spec.dst, side=side, amount=amount,
                    price=None, execution=OrderType.MARKET,
                    client_order_id=new_client_order_id("drift"),
                )
                placed += 1
                log.warning(
                    "%s: %s %.8f (%s rial) -- holding %s toward %s",
                    target.symbol, side.value, amount, f"{amount * price:,.0f}",
                    f"{target.held_rial:,.0f}", f"{target.target_rial:,.0f}",
                )
            except Exception:
                log.exception("%s: could not %s", target.symbol, side.value)
        return placed
