"""Market making: earn the bid-ask spread instead of paying it.

The arithmetic that makes this possible on this account: both legs REST, so a
round trip costs maker+maker -- 16 bps on IRT, 12 bps on USDT -- not the
maker+taker a spread-crossing strategy pays. Measured live, AVAXIRT quotes 33.8
bps and XRPIRT 30.2, leaving +17.8 and +14.2 bps gross. The majors (BTC 0.0,
ETH 0.5, SOL 0.2 bps) are quoted far tighter than the fee by someone else and
are not makeable.

What this design is defending against, in order of how much money it costs:

  * ADVERSE SELECTION. The defining hazard. A resting bid fills when the market
    is falling and a resting ask fills when it is rising, so the fills you get
    are selectively the ones you did not want. The quoted spread is the gross
    prize; this is what takes it back. Mitigated by quoting AROUND a reference
    mid, skewing quotes against existing inventory, and pulling quotes when the
    book moves faster than we can requote.
  * INVENTORY. One side filling leaves a position, which is a directional bet
    nobody chose. Hard-capped per symbol, and quotes skew to flatten.
  * THE ORDER RATE LIMIT. 300 placements / 10 minutes, shared with everything
    else on the account. Requoting is the entire activity here, so this is
    usually the binding constraint on how many symbols can be quoted at all.

Dry-run is the default. Nothing here places an order until it is switched off
deliberately.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..core.types import BookTop, Order, Side
from ..units import round_to_step

log = logging.getLogger(__name__)


@dataclass
class Quote:
    """One side of a two-sided quote."""

    side: Side
    price: float
    amount: float

    @property
    def notional(self) -> float:
        return self.price * self.amount


@dataclass
class SymbolBook:
    """Our own quoting state for one market."""

    symbol: str
    inventory: float = 0.0          # base units held from making, signed
    working: dict[str, Order] = field(default_factory=dict)   # coid -> order
    last_quote_ts: float = 0.0
    realized_rial: float = 0.0
    fills: int = 0


class MarketMaker:
    """Two-sided quoting with inventory skew.

    Deliberately NOT wired into `LiveRunner`: the trend strategy and this one
    disagree about what a position means -- trend wants to hold a winner, this
    wants to end the day flat -- and running both against one account has them
    fighting over the same coins and the same order budget.
    """

    def __init__(
        self,
        rest,
        specs: dict,
        symbols: list[str],
        *,
        maker_fee: float = 0.0008,
        min_edge_bps: float = 4.0,
        quote_notional_rial: float = 3_000_000.0,
        max_inventory_rial: float = 15_000_000.0,
        requote_s: float = 30.0,
        inside_touch: float = 0.9,
        min_quote_rial: float = 3_000_000.0,
        max_orders_per_window: int = 150,
        dry_run: bool = True,
    ):
        self.rest = rest
        self.specs = specs
        self.symbols = symbols
        self.maker_fee = maker_fee
        self.min_edge_bps = min_edge_bps
        self.quote_notional = quote_notional_rial
        self.max_inventory_rial = max_inventory_rial
        self.requote_s = requote_s
        # How far inside the market touch to quote for queue priority.
        # Only ever narrows toward the fee floor, never through it.
        self.inside_touch = inside_touch
        # Exchange minimum: a trimmed quote below this is rejected anyway.
        self.min_quote_rial = min_quote_rial
        self.dry_run = dry_run

        # Half the exchange budget by default, so this can never starve
        # whatever else runs on the account.
        self.max_orders_per_window = max_orders_per_window
        self._placements: list[float] = []

        self.books: dict[str, SymbolBook] = {s: SymbolBook(symbol=s) for s in symbols}
        # Wallet snapshot; None means 'unknown, fall back to tracked fills'.
        self._balances: dict[str, float] | None = None

    # -- economics ---------------------------------------------------------
    def breakeven_bps(self) -> float:
        """Both legs rest, so the cost is the maker fee TWICE."""
        return 2 * self.maker_fee * 10_000

    def edge_bps(self, book: BookTop) -> float:
        """Gross bps left after fees if both of our quotes fill at the touch."""
        mid = book.mid
        if mid <= 0 or book.best_ask <= book.best_bid:
            return 0.0
        spread_bps = (book.best_ask - book.best_bid) / mid * 10_000
        return spread_bps - self.breakeven_bps()

    def is_worth_quoting(self, book: BookTop) -> bool:
        """Require a margin above breakeven, not merely clearing it.

        A market exactly at breakeven is a coin flip that pays nothing and
        carries all of the inventory risk, so `min_edge_bps` is the buffer that
        keeps us out of those.
        """
        return self.edge_bps(book) >= self.min_edge_bps

    # -- quoting -----------------------------------------------------------
    def inventory_skew(self, symbol: str, mid: float) -> float:
        """How far to shift quotes, as a fraction of the half-spread.

        Long inventory pushes both quotes DOWN (cheaper to sell, less eager to
        buy); short pushes them up. Returns -1..+1 where +1 means "maximally
        long, lean hard on selling". Without this the maker accumulates on the
        losing side of a trend and the spread capture never covers it.
        """
        if self.max_inventory_rial <= 0 or mid <= 0:
            return 0.0
        held_rial = self.held_rial(symbol, mid)
        return max(-1.0, min(1.0, held_rial / self.max_inventory_rial))

    def held_rial(self, symbol: str, mid: float) -> float:
        """Rial value of what we ACTUALLY hold in this market.

        The single source of truth for both the inventory cap and the skew.
        Both used to read `books[symbol].inventory`, which counts only fills
        THIS PROCESS observed -- so a restart, a missed fill, or a position
        bought by anything else is invisible. Live, PROM reached 14,803,985
        rial against a 9,000,000 cap (64% over) because the cap was measured
        against a number that did not describe the account.

        The wallet is authoritative when we have a snapshot; tracked fills are
        only the fallback for the no-broker path.
        """
        if self._balances is not None:
            return self.available_base(symbol) * mid
        return self.books[symbol].inventory * mid

    def quoted_edge_bps(self, bid: float, ask: float) -> float:
        """Net bps our OWN pair earns if both legs fill. The number that
        actually decides profitability -- the market's spread is only the
        opportunity, these prices are the trade."""
        mid = (bid + ask) / 2
        if mid <= 0 or ask <= bid:
            return 0.0
        return (ask - bid) / mid * 10_000 - self.breakeven_bps()

    def make_quotes(self, symbol: str, book: BookTop) -> list[Quote]:
        """Two-sided quote, skewed against inventory, or [] if not worth it."""
        if not self.is_worth_quoting(book):
            return []

        spec = self.specs[symbol]
        mid = book.mid
        half = (book.best_ask - book.best_bid) / 2.0
        skew = self.inventory_skew(symbol, mid)

        # THE QUOTED SPREAD MUST CLEAR THE FEE -- not just the market's.
        #
        # The earlier version priced at a fixed 0.9 of the market half-spread
        # and skewed by narrowing one side, neither of which referenced the
        # fee. Two ways that lost money:
        #   * A 21.7 bps market quoted at 0.9 becomes 19.5 bps against a 16 bps
        #     breakeven -- the gate approved 5.7 bps of edge and the prices
        #     delivered 3.5.
        #   * Skew narrowed a single leg: at skew 0.5 the ask moved to 0.45 of
        #     the half-spread, so a 22.5 bps market quoted ~10 bps on that side
        #     -- inside the 16 bps fee, losing on every fill.
        #
        # So the half-width is floored at what the fee demands, and skew SHIFTS
        # the pair up or down without ever closing it.
        # Solve for the half-width exactly, because skew moves the denominator
        # too. We need (ask-bid)/quoted_mid >= target, and with half-width w and
        # centre c = mid - skew*w that is 2w/c >= target, so
        #     w >= (target*mid/2) / (1 + target*skew/2)
        # Using mid as the denominator instead leaves the pair a hair inside the
        # floor whenever skew shifts the centre up -- 7.997 bps against an 8.0
        # bps floor, which is a loss on a knife edge rather than a margin.
        target = (self.breakeven_bps() + self.min_edge_bps) / 1e4
        floor_half = (target * mid / 2.0) / (1.0 + target * skew / 2.0)
        half_width = max(half * self.inside_touch, floor_half)

        # Skew displaces the centre; it no longer touches the width.
        centre = mid - skew * half_width
        bid_px = centre - half_width
        ask_px = centre + half_width
        if ask_px <= bid_px:
            return []

        amount = round_to_step(self.quote_notional / mid, spec.amount_step)
        if amount <= 0:
            return []

        quotes: list[Quote] = []
        # Suppress the side that would push inventory further past the cap.
        # Measured from the WALLET, not from fills this process happened to
        # see -- that is what let PROM run 64% over the cap.
        held_rial = self.held_rial(symbol, mid)
        # Room for the WHOLE quote, not just "not yet at the cap": a 3,000,000
        # quote posted against an 8,900,000 position breaches a 9,000,000 cap
        # the moment it fills, which is how a cap gets exceeded even when it is
        # read correctly.
        room = self.max_inventory_rial - held_rial
        if room >= amount * bid_px:
            quotes.append(Quote(Side.BUY, bid_px, amount))
        elif room > 0:
            trimmed = round_to_step(room / bid_px, spec.amount_step)
            if trimmed > 0 and trimmed * bid_px >= self.min_quote_rial:
                quotes.append(Quote(Side.BUY, bid_px, trimmed))

        # SPOT CANNOT SELL WHAT IT DOES NOT HOLD. Inventory here counts only
        # our own fills, so on a freshly funded account it is 0 and every ask
        # is rejected "Order Validation Failed" -- half the order budget spent
        # on errors, every cycle. Cap the ask at the units actually available,
        # which also means a two-sided quote only appears once the bid side has
        # bought something to sell back.
        sellable = round_to_step(
            min(amount, max(0.0, self.available_base(symbol))), spec.amount_step
        )
        if held_rial > -self.max_inventory_rial and sellable > 0:
            quotes.append(Quote(Side.SELL, ask_px, sellable))
        return quotes

    def available_base(self, symbol: str) -> float:
        """Base-currency units on hand, from the wallet snapshot if we have one.

        Falls back to tracked inventory when no snapshot has been supplied, so
        the pure-logic path stays testable without a broker.
        """
        if self._balances is None:
            return self.books[symbol].inventory
        base = symbol.lower()
        for suffix in ("irt", "usdt"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return float(self._balances.get(base, 0.0))

    def refresh_balances(self) -> None:
        """Snapshot wallet balances so asks are sized against real holdings."""
        try:
            self._balances = self.rest.wallets()
        except Exception:
            log.warning("could not read balances; sizing asks from tracked inventory",
                        exc_info=True)
            self._balances = None

    # -- rate budget -------------------------------------------------------
    def _budget_left(self) -> int:
        now = time.time()
        self._placements = [t for t in self._placements if now - t < 600]
        return self.max_orders_per_window - len(self._placements)

    def can_place(self, n: int = 1) -> bool:
        return self._budget_left() >= n

    def note_placement(self, n: int = 1) -> None:
        now = time.time()
        self._placements.extend([now] * n)

    def max_quotable_symbols(self) -> float:
        """How many symbols this budget supports, two-sided, at `requote_s`."""
        per_symbol_per_window = 2 * (600.0 / self.requote_s)
        return self.max_orders_per_window / per_symbol_per_window

    # -- fills -------------------------------------------------------------
    def on_fill(self, symbol: str, side: Side, amount: float, price: float) -> None:
        """Update inventory and realized cash from one fill.

        Realized rial is signed cash flow net of the maker fee: a sell adds
        proceeds, a buy subtracts cost. Profit is this plus the mark-to-market
        of whatever inventory is left -- the spread is only earned once BOTH
        sides have traded, so cash flow alone flatters an unbalanced book.
        """
        b = self.books[symbol]
        fee = price * amount * self.maker_fee
        if side is Side.BUY:
            b.inventory += amount
            b.realized_rial -= price * amount + fee
        else:
            b.inventory -= amount
            b.realized_rial += price * amount - fee
        b.fills += 1

    def pnl_rial(self, symbol: str, mid: float) -> float:
        """Realized cash plus inventory marked to the current mid."""
        b = self.books[symbol]
        return b.realized_rial + b.inventory * mid

    def total_pnl_rial(self, mids: dict[str, float]) -> float:
        return sum(
            self.pnl_rial(s, mids[s]) for s in self.books if s in mids
        )


class MakerRunner:
    """Quote loop. Dry-run by default; places nothing until told otherwise.

    Deliberately synchronous and simple: the whole strategy is "read the book,
    replace the quotes", and at a 30s requote there is nothing to gain from
    concurrency that would justify the extra failure modes.
    """

    def __init__(self, cfg, maker: MarketMaker, rest):
        self.cfg = cfg
        self.mm = maker
        self.rest = rest

    def cancel_working(self, symbol: str) -> int:
        """Pull our quotes before posting new ones.

        Requoting without cancelling leaves the old pair in the book, so the
        exposure doubles every cycle. Cancels by clientOrderId, which is the
        only handle that reliably works on this exchange.
        """
        book = self.mm.books[symbol]
        cancelled = 0
        for coid in list(book.working):
            if self.mm.dry_run:
                book.working.pop(coid, None)
                cancelled += 1
                continue
            try:
                if self.rest.cancel_order(client_order_id=coid):
                    cancelled += 1
                book.working.pop(coid, None)
            except Exception:
                log.warning("could not cancel quote %s on %s", coid, symbol, exc_info=True)
                book.working.pop(coid, None)
        return cancelled

    def sweep(self, symbols: list[str]) -> None:
        """One requote pass over every symbol.

        Balances are refreshed ONCE here rather than per symbol: the wallet
        endpoint is rate limited and shared, and a single snapshot also keeps
        every symbol's ask sized against the same view.
        """
        self.mm.refresh_balances()
        for sym in symbols:
            self.quote_once(sym)

    def quote_once(self, symbol: str) -> list[Quote]:
        """One requote cycle for one symbol. Returns the quotes posted."""
        try:
            top = self.rest.orderbook(symbol)
        except Exception:
            log.warning("no book for %s; leaving quotes pulled", symbol, exc_info=True)
            self.cancel_working(symbol)
            return []

        quotes = self.mm.make_quotes(symbol, top)
        edge = self.mm.edge_bps(top)

        self.cancel_working(symbol)
        if not quotes:
            log.info(
                "%s: edge %.1f bps below the %.1f bps floor -- not quoting",
                symbol, edge, self.mm.min_edge_bps,
            )
            return []

        if not self.mm.can_place(len(quotes)):
            log.warning("%s: order budget exhausted; skipping this requote", symbol)
            return []

        spec = self.mm.specs[symbol]
        posted: list[Quote] = []
        for q in quotes:
            if self.mm.dry_run:
                log.info(
                    "[dry-run] %s would post %s %.8f @ %s (edge %.1f bps)",
                    symbol, q.side.value, q.amount, f"{q.price:,.0f}", edge,
                )
                posted.append(q)
                continue
            try:
                from ..execution.base import new_client_order_id

                coid = new_client_order_id("mkr")
                order = self.rest.add_order(
                    src=spec.src, dst=spec.dst, side=q.side,
                    amount=q.amount, price=q.price, client_order_id=coid,
                )
                self.mm.books[symbol].working[coid] = order
                self.mm.note_placement(1)
                posted.append(q)
            except Exception:
                # Never retried: a duplicate quote is real extra exposure.
                log.exception("%s: could not post %s quote", symbol, q.side.value)
        return posted
