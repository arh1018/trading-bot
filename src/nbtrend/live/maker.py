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
import math
import time
from dataclasses import dataclass, field

from ..core.types import BookTop, Order, Side
from ..units import round_to_step

log = logging.getLogger(__name__)


def _is_missing_order(exc: Exception) -> bool:
    """True when the exchange says the order does not exist.

    A resting quote that filled is gone by the time we requote, so cancelling
    it 404s. That is success, not failure.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    return "not found" in str(exc).lower()


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
        min_quote_rial: float = 550_000.0,
        min_cash_rial: float = 0.0,
        max_basis: float = 0.02,
        fair_weight: float = 0.5,
        max_orders_per_window: int = 240,
        dry_run: bool = True,
    ):
        self.rest = rest
        self.specs = specs
        self.symbols = symbols
        self.maker_fee = maker_fee
        self.min_edge_bps = min_edge_bps
        self.quote_notional = quote_notional_rial
        # SIZE THE QUOTE TO THE MARKET, not to a single global constant.
        #
        # A flat size is wrong in both directions at once: 1,500,000 is a
        # rounding error in ZEC or BTC, which turn over trillions of rial a
        # day, and it is a large share of the book in SUPER, which turns over
        # 2.7 billion. The ladder is (24h rial volume -> rial per quote); the
        # largest tier whose threshold the market clears wins, and anything
        # below the smallest threshold keeps the base size.
        self.notional_tiers: list[tuple[float, float]] = [
            (500e9, 12_000_000.0),
            (100e9, 8_000_000.0),
            (20e9, 3_000_000.0),
        ]
        # symbol -> 24h rial volume, refreshed on its own slow clock.
        self._volumes: dict[str, float] = {}
        self.max_inventory_rial = max_inventory_rial
        self.requote_s = requote_s
        # How far inside the market touch to quote for queue priority.
        # Only ever narrows toward the fee floor, never through it.
        self.inside_touch = inside_touch
        # Exchange minimum: a trimmed quote below this is rejected anyway.
        # MEASURED, not assumed -- 398,030 rejected and 497,538 accepted on
        # JASMYIRT. The caller passes costs.min_order_rial; this default only
        # covers the pure-logic tests. It read 3,000,000 for a long time, which
        # silently stranded every position worth less than that.
        self.min_quote_rial = min_quote_rial
        # Rial the book must keep unspent. PER-SYMBOL caps do not bound the
        # PORTFOLIO: 12 symbols x 9,000,000 permits 108,000,000 of buying
        # against an account holding 18,962,531 of cash. Bids fill and asks
        # do not, so without this the maker converts the whole account into
        # inventory -- observed live, 77,200,000 of cash to 208,250 in 17
        # minutes across 12 coins.
        self.min_cash_rial = min_cash_rial
        # How far the local book may sit from global fair before we stand
        # aside. The trend runner uses 5%; a maker is far more exposed to
        # this because it rests orders rather than crossing, so 2%.
        self.max_basis = max_basis
        # How far to pull quotes toward global fair. 0 = pure local book,
        # 1 = ignore the local book entirely. Both extremes are wrong.
        self.fair_weight = fair_weight
        self.dry_run = dry_run

        # Nobitex allows 300 placements per 10 minutes. This was 150 to leave
        # room for the trend runner sharing the account -- that runner is gone,
        # and the stale reservation silently strangled the maker: 12 symbols
        # two-sided at a 60s requote needs 240, so the allowance was spent in
        # the first sweeps and then refused everything, 2,120 times, with ZERO
        # successful placements. Held-inventory asks were refused along with
        # the rest, which is why stranded coins never sold.
        #
        # 240 leaves 60 for manual intervention and cancels.
        self.max_orders_per_window = max_orders_per_window
        self._placements: list[float] = []

        self.books: dict[str, SymbolBook] = {s: SymbolBook(symbol=s) for s in symbols}
        # Wallet snapshot; None means 'unknown, fall back to tracked fills'.
        self._balances: dict[str, float] | None = None
        # Rial committed to bids during the current sweep.
        self._committed_rial = 0.0
        # symbol -> global fair value in rial.
        self._fair: dict[str, float] = {}

    def book_for(self, symbol: str) -> SymbolBook:
        """Per-symbol state, created on demand.

        `books` is built from the CONFIGURED symbols, but the runner also
        quotes markets discovered from the wallet so a position can never be
        orphaned. Those have no entry, and indexing raised KeyError: 'BTCIRT'
        on startup -- the orphan fix and this state map disagreed about which
        symbols exist.
        """
        book = self.books.get(symbol)
        if book is None:
            book = self.books[symbol] = SymbolBook(symbol=symbol)
        return book

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

    def notional_for(self, symbol: str) -> float:
        """Rial to quote in this market, from its 24h volume.

        Falls back to the configured base size when the volume is unknown --
        a market we have no stats for is not evidence of a deep one.
        """
        vol = self._volumes.get(symbol)
        if not vol:
            return self.quote_notional
        for threshold, size in self.notional_tiers:
            if vol >= threshold:
                return size
        return self.quote_notional

    def cash_room(self) -> float:
        """Rial available to spend on bids, above the reserve floor.

        Read from the wallet each sweep. Returns a large number when no
        snapshot exists so the pure-logic path is unconstrained by it.
        """
        # A snapshot with no rial entry means UNKNOWN, not empty. Reading it as
        # zero blocks every bid, which silently turns the maker into a
        # sell-only process -- a failure that looks exactly like working.
        if self._balances is None or "rls" not in self._balances:
            return float("inf")
        # Subtract what THIS sweep has already committed. The wallet snapshot
        # is up to 30s old, so twelve symbols quoting in quick succession all
        # saw the same pre-commitment cash and each concluded there was room:
        # 48,981,598 fell to 7,118,000, straight through an 8,000,000 floor.
        # The floor bounded one bid, never a sweep of twelve.
        cash = float(self._balances["rls"]) - self._committed_rial
        return max(0.0, cash - self.min_cash_rial)

    def commit_cash(self, rial: float) -> None:
        """Record rial spent this sweep, before the wallet reflects it."""
        self._committed_rial += max(0.0, rial)

    def reset_commitments(self) -> None:
        """Called at the start of each sweep, once balances are refreshed."""
        self._committed_rial = 0.0

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

    def fair_rial(self, symbol: str) -> float | None:
        """What the asset is worth globally, converted to rial.

        `global_usd x USDT/IRT`. The local book is only one venue's opinion; a
        quote priced purely off it is blind to the asset having moved on every
        other exchange. That is a second source of adverse selection on top of
        the queue: our bid rests at a stale rial price while the world reprices
        the coin, and the fill we get is precisely the one we did not want.
        """
        ref = self._fair.get(symbol)
        return ref if ref and ref > 0 else None

    def set_fair(self, symbol: str, global_usd: float, fx_rial_per_usdt: float,
                 multiplier: int = 1) -> None:
        """Record the global reference price for a market."""
        if global_usd > 0 and fx_rial_per_usdt > 0:
            # MULTIPLY. 1K_SHIBIRT quotes a THOUSAND shib, so its fair price is
            # 1000x the per-unit global price, not a thousandth. Dividing gave
            # 1M_PEPEIRT a fair value of 0 and a basis of 99,568,862,847,924%.
            # Reuses the canonical helper rather than restating the arithmetic.
            from ..data.fx import fair_rial_price

            self._fair[symbol] = fair_rial_price(
                global_usd, fx_rial_per_usdt, max(1, multiplier)
            )

    def basis(self, symbol: str, book: BookTop) -> float | None:
        """Local mid versus global fair, as a fraction. + means locally rich."""
        fair = self.fair_rial(symbol)
        if not fair or book.mid <= 0:
            return None
        return book.mid / fair - 1.0

    def make_quotes(self, symbol: str, book: BookTop) -> list[Quote]:
        """Two-sided quote, skewed against inventory, or [] if not worth it."""
        if not self.is_worth_quoting(book):
            return []

        # REFUSE a market dislocated from global pricing. A local book trading
        # far from `global_usd x fx` is usually mid-repricing, and quoting into
        # that means buying just before the local price catches down (or
        # selling before it catches up). The spread looks the same; the fill is
        # systematically bad.
        drift = self.basis(symbol, book)
        if drift is not None and abs(drift) > self.max_basis:
            log.info(
                "%s: local mid is %+.2f%% from global fair (limit %.2f%%) -- not quoting",
                symbol, drift * 100, self.max_basis * 100,
            )
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

        # JOIN THE TOUCH BY ONE TICK, not by a fixed fraction of the spread.
        #
        # `inside_touch` 0.9 gave away a tenth of every spread for nothing: the
        # only thing that buys queue priority is being one tick better than the
        # current best, and a tick is almost always far smaller than 10% of the
        # spread. Measured live: XLM 36.9 -> 40.5 bps, PYTH 61.5 -> 66.7,
        # 1M_PEPE 38.0 -> 42.0, all still in front of the book.
        #
        # A FIXED RIAL OFFSET CANNOT WORK HERE, which is why this uses the tick
        # and not a constant. Ten rial is 0.02 bps on 1M_PEPE and 16.2 bps on
        # 1K_SHIB; on 1K_SHIB, whose spread is 0.8 bps, stepping both sides in
        # by 10 rial inverts the quote to -15.4 bps -- a bid above our own ask.
        # The tick is the exchange's own unit of "one better" and scales with
        # the price, so it is the right one.
        # Step in by a tick where the market defines one. With no tick the
        # step has no natural size, so fall back to the fraction -- quoting AT
        # the touch is not inside it, and skew then pushes a side back out.
        tick_for_touch = float(getattr(spec, "price_step", 0) or 0)
        touch_half = half - tick_for_touch if tick_for_touch > 0 else half * self.inside_touch
        if touch_half <= 0:
            # The spread is one tick wide or less. There is no room to improve
            # on the touch, so fall through to the fee floor rather than
            # crossing (TNSR and 1K_SHIB both land here at times).
            touch_half = half * self.inside_touch

        # The floor still wins whenever joining the touch would quote inside
        # the fee -- this widens the quote, it never narrows it through
        # breakeven.
        half_width = max(touch_half, floor_half)

        # Anchor the centre toward GLOBAL fair, not purely the local mid.
        #
        # The local mid is one venue's opinion and lags the world. Pricing off
        # it alone means our bid rests where the coin used to be worth, so the
        # counterparty picking us off is the one who already saw the move --
        # the same adverse selection that made realized P&L negative, arriving
        # through price rather than through queue position.
        #
        # Blended rather than replaced: the fair value is itself an estimate
        # (a global feed and an FX rate, both with their own staleness), and
        # quoting purely off it would ignore the book we actually trade in.
        anchor = mid
        fair = self.fair_rial(symbol)
        if fair:
            anchor = mid * (1.0 - self.fair_weight) + fair * self.fair_weight

        # Skew displaces the centre; it no longer touches the width.
        centre = anchor - skew * half_width
        bid_px = centre - half_width
        ask_px = centre + half_width
        if ask_px <= bid_px:
            return []

        # IMPROVE THE TOUCH WHERE THE SPREAD PAYS FOR IT.
        #
        # Skew displaces the centre, so on a book we are long the bid falls
        # BEHIND the best bid -- RAYIRT quoted 1,817,800 against a 1,818,890
        # best bid, 5.9 bps back, on a 357 bps spread. Sitting behind the touch
        # earns no queue at all: the order only fills once everything in front
        # of it does, which on a wide book means only when the price runs
        # through us. That is adverse selection bought at full price.
        #
        # A wide spread has room to lean on inventory AND stay in front, so
        # step each side to one tick inside the touch whenever doing so leaves
        # the pair still clearing the fee. Skew keeps its say: it decides how
        # much of the remaining width each side gets, and when the book is too
        # thin to afford both, the floor below wins and the lean stands.
        if tick_for_touch > 0:
            want_bid = book.best_bid + tick_for_touch
            want_ask = book.best_ask - tick_for_touch

            # CLAIM THE WHOLE SPREAD, not just enough of it to be in front.
            #
            # An earlier version only ever tightened toward the mid, which left
            # money on the table on a wide book: RAYIRT, 143 bps of spread,
            # asked 1,882,200 when one tick inside the best ask was 1,886,000 --
            # already in front either way, and 20 bps better for the same queue
            # position. Skew had pulled the centre down and nothing pulled the
            # far side back out.
            #
            # So each side moves TO one tick inside the touch, in or out. Skew
            # still governs which side gets suppressed entirely when inventory
            # runs to the cap, and the fee floor below still governs whether
            # the pair may be quoted at all -- this only decides where a side
            # sits once both have been allowed.
            new_bid = min(max(bid_px, want_bid), want_bid)
            new_ask = max(min(ask_px, want_ask), want_ask)

            # Keep it only if the pair still clears breakeven. On a book too
            # thin to pay for both sides at the touch, the skewed prices stand.
            if new_ask - new_bid >= target * ((new_ask + new_bid) / 2.0):
                bid_px, ask_px = new_bid, new_ask

        # ROUND PRICES TO THE TICK. Amounts were always stepped; prices never
        # were, so a market with price_step 10 got 78,445.8497 and rejected
        # every quote -- 76 "Order Validation Failed" on TNSRIRT alone. Round
        # the bid DOWN and the ask UP: rounding either toward the mid would
        # narrow the pair back through the fee floor we just enforced.
        tick = float(getattr(spec, "price_step", 0) or 0)
        if tick > 0:
            bid_px = math.floor(bid_px / tick) * tick
            ask_px = math.ceil(ask_px / tick) * tick
            if ask_px <= bid_px:
                return []

            # RE-ASSERT THE TOUCH AFTER ROUNDING. The book's best price is not
            # necessarily tick-aligned -- RAYIRT showed a 1,856,510 best bid on
            # a 100 tick -- so flooring our intended 1,856,610 landed on
            # 1,856,500, one tick BEHIND the very touch we had just stepped in
            # front of. Round outward to the next aligned price that still
            # improves, and only when the pair goes on clearing the fee.
            if bid_px <= book.best_bid < bid_px + tick:
                lifted = math.floor(book.best_bid / tick) * tick + tick
                if ask_px - lifted >= target * ((ask_px + lifted) / 2.0):
                    bid_px = lifted
            if ask_px - tick < book.best_ask <= ask_px:
                lowered = math.ceil(book.best_ask / tick) * tick - tick
                if lowered - bid_px >= target * ((lowered + bid_px) / 2.0):
                    ask_px = lowered
            if ask_px <= bid_px:
                return []

        # Size from the QUOTED price, not the mid, and round the amount UP to
        # the step. Sizing off the mid then stepping the amount down left
        # notionals hovering either side of the 3,000,000 minimum -- a
        # 2,985,259 bid on ZROIRT was rejected while the ask beside it passed
        # at 3,002,160. Add a small buffer so tick and step rounding cannot
        # drop a quote back under the floor.
        # Sizing off bid_px (not the mid) is what fixes the original defect:
        # a quote sized at the mid is worth LESS at the bid, which is how a
        # ZROIRT bid landed at 2,985,259 under a 3,000,000 minimum.
        step = spec.amount_step or 1e-8
        amount = math.ceil(self.notional_for(symbol) / bid_px / step) * step
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
        room = min(
            self.max_inventory_rial - held_rial,
            self.cash_room(),
        )
        # One rial of tolerance. Rounding the amount up to its step can put
        # the notional fractions of a rial over an exactly-equal room -- 0.005
        # rial cost an entire bid here -- and sub-rial precision is meaningless
        # for money.
        if room + 1.0 >= amount * bid_px:
            quotes.append(Quote(Side.BUY, bid_px, amount))
        elif room > 0:
            # Rounding the size UP to clear the exchange minimum fights the cap
            # ceiling: 6,000,000 held against a 9,000,000 cap leaves exactly
            # 3,000,000 of room, and a quote rounded up to 3,173,567 exceeded
            # it and was suppressed entirely. Round the trim DOWN to the step
            # (it must fit the room) and only quote it if it still clears the
            # minimum -- otherwise there is genuinely no legal bid to make.
            trimmed = round_to_step(room / bid_px, spec.amount_step)
            if trimmed > 0 and trimmed * bid_px >= self.min_quote_rial:
                quotes.append(Quote(Side.BUY, bid_px, trimmed))
            elif room >= self.min_quote_rial:
                # The step is coarse enough that rounding down lost the
                # minimum; the room is there, so nudge one step back up.
                stepped = trimmed + (spec.amount_step or 0)
                if stepped * bid_px <= room:
                    quotes.append(Quote(Side.BUY, bid_px, stepped))

        # SPOT CANNOT SELL WHAT IT DOES NOT HOLD. Inventory here counts only
        # our own fills, so on a freshly funded account it is 0 and every ask
        # is rejected "Order Validation Failed" -- half the order budget spent
        # on errors, every cycle. Cap the ask at the units actually available,
        # which also means a two-sided quote only appears once the bid side has
        # bought something to sell back.
        sellable = round_to_step(
            min(amount, max(0.0, self.available_base(symbol))), spec.amount_step
        )
        # The ask must clear the exchange minimum too. Capping it at what we
        # HOLD is not enough: holding 0.10045944 HOLO produced a 13,845 rial
        # ask against the minimum, rejected "Order Validation Failed" on every
        # sweep -- 47 times in under two minutes. The bid was always checked;
        # the ask never was.
        #
        # SELL THE WHOLE FRAGMENT. Quoting a fixed size leaves a remainder
        # behind on every partial round trip, and a remainder under the
        # minimum can never be sold again -- 33,486,842 rial accumulated that
        # way across 17 coins, more than half the account. If what remains
        # after a normal-sized ask would be unsellable, offer the lot instead.
        free = max(0.0, self.available_base(symbol))
        remainder = free - sellable
        if 0 < remainder * ask_px < self.min_quote_rial:
            sellable = round_to_step(free, spec.amount_step)

        if (
            held_rial > -self.max_inventory_rial
            and sellable > 0
            and sellable * ask_px >= self.min_quote_rial
        ):
            quotes.append(Quote(Side.SELL, ask_px, sellable))
        return quotes

    def available_base(self, symbol: str) -> float:
        """Base-currency units on hand, from the wallet snapshot if we have one.

        Falls back to tracked inventory when no snapshot has been supplied, so
        the pure-logic path stays testable without a broker.
        """
        if self._balances is None:
            return self.book_for(symbol).inventory
        base = symbol.lower()
        for suffix in ("irt", "usdt"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return float(self._balances.get(base, 0.0))

    def _own_blocked(self, currency: str, blocked: float) -> float:
        """How much of `blocked` is locked in quotes WE placed and will cancel.

        We track our working orders per symbol, so if we have any live quote in
        that market the blocked units are ours; otherwise they belong to an
        order we did not place and must not be counted as sellable.
        """
        symbol = currency.upper() + "IRT"
        book = self.books.get(symbol)
        return blocked if (book and book.working) else 0.0

    def refresh_balances(self) -> None:
        """Snapshot wallet balances so asks are sized against real holdings.

        Uses TOTAL balance, not `activeBalance`. `activeBalance` excludes units
        blocked in working orders -- including our own resting quotes -- so a
        symbol whose inventory is committed to a quote reads as empty and gets
        no ask at all. ONEIRT held 1796.48944 units with an activeBalance of
        0.08944 (148 rial, under the 0.1 amount step), so `sellable` rounded to
        zero and it silently quoted bid-only while looking like a 0% ask fill.

        Quotes are cancelled before requoting, so the blocked units are about
        to be free; sizing against the total is the correct view.
        """
        try:
            detailed = self.rest.balances_detailed()
            # Free units, PLUS the ones locked in our own quotes -- those are
            # released by `cancel_working` moments before we repost. Blocked
            # units we did NOT place stay excluded: selling them is what got
            # 240 "Order Validation Failed" on TNSRIRT/ZROIRT/JASMYIRT.
            self._balances = {
                cur: free + self._own_blocked(cur, blocked)
                for cur, (free, blocked) in detailed.items()
            }
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
        b = self.book_for(symbol)
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
        b = self.book_for(symbol)
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

    def __init__(self, cfg, maker: MarketMaker, rest, requote_tolerance_bps: float = 3.0):
        self.cfg = cfg
        self.mm = maker
        self.rest = rest
        # How far a quote may drift before it is worth reposting.
        self.requote_tolerance_bps = requote_tolerance_bps
        self._last_quotes: dict[str, list[Quote]] = {}
        self._balances_at = 0.0
        # Wallet reads are rate limited; decouple them from the quote loop.
        self.balance_refresh_s = 30.0
        self._fair_at = 0.0
        # Global prices move on the world's clock, not our quote loop's.
        self.fair_refresh_s = 60.0
        self._volumes_at = 0.0
        # A day's volume barely moves between sweeps, and this costs one REST
        # call per market, so it runs far more slowly than the quote loop.
        self.volume_refresh_s = 900.0
        # Markets the bot must never quote, whatever the wallet says. Anything
        # held here belongs to someone else.
        self.protected: set[str] = set()
        # clientOrderId -> symbol. The order list gives display names and a
        # null id, so this is the only way to know which market an order is.
        self._coid_symbol: dict[str, str] = {}
        self._orders_cache: list[dict] = []
        self._orders_at = 0.0
        # Shared across symbols; invalidated whenever we place or cancel.
        #
        # 5s was too aggressive once quoting became event driven -- the socket
        # fires continuously, so even a shared listing every 5s tripped the
        # limiter. But the TTL must also never EXCEED the requote cooldown: at
        # a 60s cooldown a 20s cache was stale by the time a symbol came round
        # again, so `_live_orders` returned nothing, `cancel_working` had
        # nothing to cancel, and the repost stacked a second quote on top of
        # the first -- Ethena, Jasmy and SuperVerse all ended up doubled.
        #
        # Half the cooldown: fresh enough to see our own resting orders, slow
        # enough not to hammer the endpoint.
        self.orders_cache_s = 20.0

    def invalidate_orders_cache(self) -> None:
        """Force the next `_live_orders` to re-read from the exchange.

        A TTL cannot express what is actually needed: the listing must be
        fresh AT THE MOMENT a symbol requotes. With a 60s cooldown and a 20s
        TTL it was always stale by then, so `_live_orders` returned nothing,
        `cancel_working` had nothing to cancel, and the repost stacked a second
        quote on the first -- Ethena, Jasmy and SuperVerse all doubled.
        Refreshing once per sweep is one extra call, shared by every symbol.
        """
        self._orders_at = 0.0

    def _unchanged(self, symbol: str, quotes: list[Quote]) -> bool:
        """True when the live quotes already match what we want to post.

        Compared with a tolerance in BPS, not exact equality: a quote one rial
        away is not worth surrendering queue position and two order slots for.
        """
        book = self.mm.book_for(symbol)
        if not book.working or len(book.working) != len(quotes):
            return False
        prev = self._last_quotes.get(symbol)
        if not prev or len(prev) != len(quotes):
            return False
        for a, b in zip(sorted(prev, key=lambda q: q.side.value),
                        sorted(quotes, key=lambda q: q.side.value), strict=True):
            if a.side is not b.side or a.price <= 0:
                return False
            if abs(a.price - b.price) / a.price * 10_000 > self.requote_tolerance_bps:
                return False
            if a.amount <= 0 or abs(a.amount - b.amount) / a.amount > 0.02:
                return False

        # LOSING THE TOUCH IS A CHANGE, however small the price move was.
        #
        # Everything above compares our new quote to our OLD one, so a book
        # that moves out from under a resting order looks like "no change" and
        # the order is left where it is. RAYIRT rested a bid 10 rial behind the
        # best bid -- 0.05 bps, far inside the 3 bps tolerance, and therefore
        # never reposted -- while the book had moved on. Behind the touch the
        # order earns no queue at all: it fills only once everything in front
        # of it does, which is exactly when we do not want it to.
        #
        # So the comparison has to include the book, not just our own history.
        # AT the touch is not IN FRONT of it. `<` and `>` let a quote resting
        # exactly on the best price pass as unchanged, so RAYIRT sat at
        # 1,886,100 -- equal to the best ask, joining the back of that price
        # level rather than leading the book -- while the code wanted
        # 1,882,200. Queue priority at the same price goes to whoever was
        # there first, and that was not us.
        top = self.latest_book(symbol)
        if top is not None:
            for q in quotes:
                if q.side is Side.BUY and top.best_bid and q.price <= top.best_bid:
                    return False
                if q.side is Side.SELL and top.best_ask and q.price >= top.best_ask:
                    return False
        return True

    def latest_book(self, symbol: str):
        """Freshest book, if a socket is feeding us one. REST otherwise."""
        return None

    def _live_orders(self, symbol: str) -> list[dict]:
        """Our orders actually resting on the exchange for this market.

        Matched by clientOrderId, recorded when we post. The order list cannot
        be matched on the symbol: it returns DISPLAY names ("Kamino Finance",
        "Harmony"), not currency codes, and `"id": null` besides -- the
        clientOrderId is the only reliable handle this exchange gives us.

        Returns [] on a read failure, which makes the caller cancel and repost:
        wasteful, but it can never stack.
        """
        # ONE listing per sweep, shared across symbols and cached briefly.
        #
        # This used to fetch per symbol, which was tolerable at a 60s poll and
        # is not once quoting is event driven: the socket fires far more often,
        # and calling it per symbol per event earned 13 x 400 and 7 x 429 in
        # under two minutes, so nothing could be placed at all.
        now = time.time()
        if now - self._orders_at >= self.orders_cache_s:
            try:
                raw = self.rest._get("/market/orders/list", status="open", details=2)
                self._orders_cache = raw.get("orders", [])
                self._orders_at = now
            except Exception:
                log.warning("could not list live orders", exc_info=True)
                return []
        live = [
            o for o in self._orders_cache
            if self._coid_symbol.get(str(o.get("clientOrderId") or "")) == symbol
        ]
        # Orders placed since the cache was taken are live even though the
        # listing predates them. Without this a symbol quoted twice inside the
        # 20s cache window cannot see its own first order and posts a second --
        # Harmony and Raydium both ended up doubled.
        known = {str(o.get("clientOrderId") or "") for o in live}
        for coid in self.mm.book_for(symbol).working:
            if coid not in known:
                live.append({"clientOrderId": coid})
        return live

    def cancel_working(self, symbol: str, live: list[dict] | None = None) -> int:
        """Pull our quotes before posting new ones.

        Requoting without cancelling leaves the old pair in the book, so the
        exposure doubles every cycle. Cancels by clientOrderId, which is the
        only handle that reliably works on this exchange.
        """
        book = self.mm.book_for(symbol)
        cancelled = 0
        # Cancel what the EXCHANGE says is resting, plus anything we think we
        # posted -- the union, so neither view can leave an order behind.
        coids = list(book.working)
        for o in live or []:
            c = str(o.get("clientOrderId") or "")
            if c and c not in coids:
                coids.append(c)
        for coid in coids:
            if self.mm.dry_run:
                book.working.pop(coid, None)
                cancelled += 1
                continue
            try:
                if self.rest.cancel_order(client_order_id=coid):
                    cancelled += 1
            except Exception as exc:
                # A 404 means the order is GONE -- it filled, or was already
                # cancelled. For a resting quote that is the normal outcome,
                # not a failure: it is what earning the spread looks like.
                # Logging it as an error with a traceback produced 71 scary
                # warnings that hid the real problem elsewhere.
                if _is_missing_order(exc):
                    log.debug("quote %s on %s already gone (filled or cancelled)",
                              coid, symbol)
                else:
                    log.warning("could not cancel quote %s on %s", coid, symbol,
                                exc_info=True)
            finally:
                book.working.pop(coid, None)
        if cancelled:
            self._orders_at = 0.0
        return cancelled

    def held_symbols(self, min_rial: float = 50_000.0) -> list[str]:
        """Every market we hold inventory in, whether or not we quote it.

        Selling a position down must NEVER depend on that market still being
        attractive to ENTER. Cutting the symbol list from 12 to 6 stranded
        39,884,121 rial -- 65% of holdings -- in six markets with nothing
        quoting them: a directional bet on six alt-coins that nobody chose.

        Derived from the wallet, so a config change cannot orphan a position
        again.
        """
        balances = self.mm._balances or {}
        out = []
        for cur, units in balances.items():
            if cur == "rls" or not units:
                continue
            symbol = cur.upper() + "IRT"
            if symbol not in self.mm.specs:
                continue
            # NEVER quote a market the bot did not put us into. Holdings are
            # discovered from the wallet so a position cannot be orphaned, but
            # the same mechanism would happily start making a market in
            # something the ACCOUNT OWNER is trading by hand -- 478,875,002
            # rial of PAXG sat in this wallet, placed by a person, and the
            # maker had no way to know it was not its own.
            if symbol in self.protected:
                log.debug("%s is protected; not quoting it", symbol)
                continue
            try:
                mid = self.rest.orderbook(symbol).mid
            except Exception:
                continue
            if units * mid >= min_rial:
                out.append(symbol)
        return out

    def refresh_volumes(self, specs: dict) -> int:
        """Update each market's 24h rial volume, which sets its quote size.

        On its own slow clock: a day's volume does not move meaningfully
        between sweeps, and this is one REST call per market -- the endpoint
        rejects a batched srcCurrency list with a 400.

        A market that fails to report keeps whatever volume it had, and a
        market that has never reported is simply absent, which `notional_for`
        reads as "unknown" and answers with the base size. Guessing large from
        missing data would size a quote off nothing.
        """
        now = time.time()
        if now - self._volumes_at < self.volume_refresh_s:
            return 0
        self._volumes_at = now

        updated = 0
        for symbol in self.mm.symbols:
            spec = specs.get(symbol)
            src = getattr(spec, "src", None)
            if not src:
                continue
            try:
                # `market_stats` already unwraps the "stats" envelope and
                # returns {"btc-rls": {...}}. Unwrapping it a second time
                # yields {} for every market, which silently sizes the whole
                # book at the base notional -- the tiers would simply never
                # engage, and nothing would look wrong.
                stats = self.rest.market_stats(src.lower())
            except Exception:
                continue
            row = stats.get(f"{src.lower()}-rls")
            if not row:
                continue
            try:
                vol = float(row.get("volumeDst") or 0.0)
            except (TypeError, ValueError):
                continue
            if vol > 0:
                self.mm._volumes[symbol] = vol
                updated += 1
        if updated:
            sizes = {s: self.mm.notional_for(s) for s in sorted(self.mm._volumes)}
            log.info(
                "quote sizes by volume: %s",
                ", ".join(f"{s}={v/1e6:.1f}M" for s, v in sizes.items()),
            )
        return updated

    def refresh_fair_values(self, specs: dict) -> int:
        """Update global fair values: global USD price x USDT/IRT.

        Refreshed on its own slow clock -- global prices move on the world's
        timescale, not our quote loop's -- and via KuCoin's all-tickers
        endpoint, which is ONE request for every symbol rather than one per
        symbol for a single number.
        """
        now = time.time()
        if now - self._fair_at < self.fair_refresh_s:
            return 0
        self._fair_at = now

        try:
            fx = self.rest.orderbook("USDTIRT").mid
        except Exception:
            log.warning("no USDT/IRT rate; quoting off the local book alone", exc_info=True)
            return 0
        if fx <= 0:
            return 0

        from ..data.kucoin import KuCoinFeed, all_tickers

        try:
            tickers = all_tickers()
        except Exception:
            log.warning("no global tickers; quoting off the local book alone", exc_info=True)
            return 0

        mapper = KuCoinFeed()
        updated = 0
        for symbol, spec in specs.items():
            tv = getattr(spec, "tradingview", None)
            if not tv:
                continue
            price = tickers.get(mapper._to_kucoin_symbol(tv))
            if not price:
                continue
            self.mm.set_fair(symbol, price, fx, getattr(spec, "multiplier", 1))
            updated += 1
        if updated:
            log.debug("refreshed %d global fair values at fx %.0f", updated, fx)
        return updated

    def sweep(self, symbols: list[str]) -> None:
        """One requote pass over every symbol.

        Balances are refreshed ONCE here rather than per symbol: the wallet
        endpoint is rate limited and shared, and a single snapshot also keeps
        every symbol's ask sized against the same view.
        """
        # Balances are refreshed on their OWN clock, not the loop's. The wallet
        # endpoint is rate limited and already returned 429s at a 60s cycle;
        # at 10s it would be read 6x as often for data that barely changes.
        # Fills still invalidate it immediately, so this cannot hide a fill.
        now = time.time()
        if now - self._balances_at >= self.balance_refresh_s:
            self.mm.refresh_balances()
            self._balances_at = now
        # Commitments are per sweep: the snapshot above already includes any
        # fills from previous ones.
        self.mm.reset_commitments()
        # And the order listing must be fresh for THIS sweep, or a repost
        # cannot see the quote it is meant to replace.
        self.invalidate_orders_cache()
        # Global reference prices, on their own slow clock.
        self.refresh_fair_values(self.mm.specs)
        # And each market's quote size, on a slower one still.
        self.refresh_volumes(self.mm.specs)

        # Quote the configured markets PLUS anything we hold. A held position
        # with no ask is a directional bet nobody chose, and it persists until
        # something sells it.
        held = self.held_symbols()
        orphans = [s for s in held if s not in symbols]
        if orphans:
            log.warning(
                "holding %d market(s) outside the quote list: %s -- quoting asks "
                "so they can be sold down",
                len(orphans), ", ".join(orphans),
            )
        for sym in [*symbols, *orphans]:
            self.quote_once(sym)

    def quote_once(self, symbol: str) -> list[Quote]:
        """One requote cycle for one symbol. Returns the quotes posted."""
        top = self.latest_book(symbol)
        if top is None:
            try:
                top = self.rest.orderbook(symbol)
            except Exception:
                log.warning("no book for %s; leaving quotes pulled", symbol, exc_info=True)
                self.cancel_working(symbol)
                return []

        quotes = self.mm.make_quotes(symbol, top)
        edge = self.mm.edge_bps(top)

        # DO NOT REPLACE A QUOTE THAT HAS NOT MOVED.
        #
        # Cancel-and-repost is the entire cost of requoting: 2 placements per
        # symbol per sweep against a 300-per-10-minutes limit. At a 10s requote
        # on 6 symbols that is 720 -- 2.4x over, so most quotes get refused and
        # the book updates LESS often than at 30s.
        #
        # But a quote only needs replacing when the price actually changed.
        # Skipping unchanged ones decouples how often we LOOK from how often we
        # SPEND, which is what makes a fast loop affordable. It also preserves
        # queue position, which a needless cancel throws away.
        # Reconcile against the EXCHANGE, not our own bookkeeping.
        #
        # `book.working` is what we believe we posted; the exchange is what is
        # actually resting. They drift -- a cancel that 404s pops the entry
        # whether or not an order was really pulled, and a skipped requote
        # leaves entries we then compare against. That drift stacked three
        # KMNOIRT bids at 51,750 / 52,080 / 52,210 simultaneously.
        #
        # Counting live orders per symbol makes stacking impossible to express:
        # if the exchange already holds our pair and it has not moved, keep it;
        # otherwise cancel whatever is actually there before posting.
        live = self._live_orders(symbol)
        if quotes and len(live) == len(quotes) and self._unchanged(symbol, quotes):
            log.debug("%s: quotes unchanged; keeping queue position", symbol)
            return quotes

        self.cancel_working(symbol, live=live)
        if not quotes:
            # Say WHY. This used to blame the edge unconditionally and print
            # "edge 44.1 bps below the 8.0 bps floor" -- self-contradictory,
            # and it sent the investigation to the wrong place while the real
            # cause (no sellable inventory) went unnoticed.
            drift = self.mm.basis(symbol, top)
            if drift is not None and abs(drift) > self.mm.max_basis:
                # Checked FIRST: a dislocated market is refused before the
                # inventory question is even asked, so blaming inventory here
                # sent an investigation chasing a phantom stranding bug while
                # ONEIRT sat +8.74% from global fair, working as designed.
                reason = (
                    f"local mid is {drift * 100:+.2f}% from global fair "
                    f"(limit {self.mm.max_basis * 100:.1f}%)"
                )
            elif edge < self.mm.min_edge_bps:
                reason = f"edge {edge:.1f} bps under the {self.mm.min_edge_bps:.1f} bps floor"
            elif self.mm.cash_room() <= 0:
                reason = "no cash above the reserve, and nothing sellable"
            else:
                free = self.mm.available_base(symbol)
                # `top` is the BookTop for this sweep. `book_for` returns our
                # own SymbolBook -- quoting state, no prices on it at all.
                held = free * top.mid if top.mid else 0.0
                if 0 < held < self.mm.min_quote_rial:
                    # Name the real cause. The old wording blamed "locked in
                    # working orders" for what was nearly always a holding
                    # sitting under the ask minimum, and cost an investigation.
                    reason = (
                        f"edge {edge:.1f} bps is fine, but the {held:,.0f} rial held "
                        f"is under the {self.mm.min_quote_rial:,.0f} ask minimum "
                        f"-- cannot offer it"
                    )
                else:
                    reason = (
                        f"edge {edge:.1f} bps is fine, but nothing to quote: free base "
                        f"{free:.8f} (inventory may be locked in working orders)"
                    )
            log.info("%s: not quoting -- %s", symbol, reason)
            return []

        if not self.mm.can_place(len(quotes)):
            # Loud, and says what to change. Quoted at a rate the budget cannot
            # sustain, the maker stops trading entirely while every log line
            # still reads like an ordinary skip.
            log.error(
                "%s: ORDER BUDGET EXHAUSTED (%d/%d used in the last 10min). "
                "Quoting %d symbols two-sided every %.0fs needs %.0f placements; "
                "reduce symbols or lengthen --requote.",
                symbol, len(self.mm._placements), self.mm.max_orders_per_window,
                len(self.mm.symbols), self.mm.requote_s,
                len(self.mm.symbols) * 2 * (600.0 / max(1.0, self.mm.requote_s)),
            )
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
                self.mm.book_for(symbol).working[coid] = order
                self._coid_symbol[coid] = symbol
                self._orders_at = 0.0
                if q.side is Side.BUY:
                    self.mm.commit_cash(q.notional)
                self.mm.note_placement(1)
                posted.append(q)
            except Exception:
                # Never retried: a duplicate quote is real extra exposure.
                log.exception("%s: could not post %s quote", symbol, q.side.value)
        self._last_quotes[symbol] = posted
        # A newly posted quote may fill at any moment, so the cached balance is
        # now suspect: shorten its life rather than trusting it a full window.
        self._balances_at = min(self._balances_at, time.time() - self.balance_refresh_s / 2)
        return posted


class SocketMakerRunner(MakerRunner):
    """Event-driven quoting: react to book updates, not a polling clock.

    Why this matters for P&L rather than tidiness. Measured over 24h of REST
    polling, realized round trips were -80,276 rial across 7 markets with only
    1 profitable. That is adverse selection: a quote priced off a book we read
    seconds ago gets filled by someone reacting to the book as it is NOW, so
    our bid fills into a falling market and our ask into a rising one. The
    quoted spread was never the problem; the staleness was.

    A websocket removes the polling interval from that equation -- quotes are
    repriced when the book actually moves, not when a timer says so.

    What does NOT change is the exchange's 300-placements-per-10-minutes limit.
    Reacting to every tick would blow through it instantly, so the same
    machinery still applies: an unchanged quote is not reposted, and a
    per-symbol cooldown bounds how often any one market can be requoted. The
    socket improves WHEN we act, not how much we are allowed to act.
    """

    def __init__(self, cfg, maker: MarketMaker, rest, ws,
                 requote_tolerance_bps: float = 3.0,
                 min_requote_gap_s: float = 2.0):
        super().__init__(cfg, maker, rest, requote_tolerance_bps=requote_tolerance_bps)
        self.ws = ws
        # Floor on how often a single symbol may be requoted, whatever the
        # book does. Without it a fast-moving market would eat the whole
        # placement budget on its own and starve every other symbol.
        self.min_requote_gap_s = min_requote_gap_s
        self._last_requote: dict[str, float] = {}
        needed = len(maker.symbols) * 2 * (600.0 / max(1.0, min_requote_gap_s))
        if needed > maker.max_orders_per_window:
            log.error(
                "quoting %d symbols every %.0fs needs %.0f placements/10min but the "
                "budget is %d -- the allowance will be spent in the first sweeps and "
                "every later quote refused, including asks on held inventory. "
                "Raise --requote to at least %.0fs or cut symbols.",
                len(maker.symbols), min_requote_gap_s, needed,
                maker.max_orders_per_window,
                len(maker.symbols) * 2 * 600.0 / maker.max_orders_per_window,
            )
        self._books: dict[str, BookTop] = {}
        self._dirty: set[str] = set()

    def on_book(self, symbol: str, top: BookTop) -> None:
        """Websocket callback. Cheap on purpose -- it runs on the event loop.

        Only marks the symbol dirty; the actual quoting is done by the worker,
        because placing orders is blocking I/O and must never run here. Doing
        REST calls in a websocket handler is what starved Centrifugo's 25s
        ping and got the trend bot disconnected for 'no pong'.
        """
        self._books[symbol] = top
        self._dirty.add(symbol)

    def latest_book(self, symbol: str) -> BookTop | None:
        return self._books.get(symbol)

    def due(self, symbol: str, now: float | None = None) -> bool:
        """Is this symbol past its cooldown and in need of a look?

        Dirty means the book moved. But a symbol with orders resting on the
        exchange needs a look whether or not a tick arrived: our quote can stop
        being valid without the book moving under it at all -- the spread
        compresses past the edge gate, inventory fills up, the basis drifts --
        and in every one of those cases `make_quotes` returns nothing, so the
        old order should be CANCELLED rather than left to rest.

        Waiting on `_dirty` alone left exactly that: every symbol failed the
        edge gate as spreads compressed, no ticks were arriving on the quiet
        ones, and stale quotes sat at prices the maker would no longer post --
        1M_PEPE, RAY, SUPER, TNSR and PYTH all resting behind the touch with
        nothing scheduled to revisit them.
        """
        now = now if now is not None else time.time()
        if now - self._last_requote.get(symbol, 0.0) < self.min_requote_gap_s:
            return False
        if symbol in self._dirty:
            return True
        # Nothing resting means nothing to correct; wait for a real tick.
        return bool(self.mm.book_for(symbol).working)

    def take_due(self, now: float | None = None) -> list[str]:
        """Symbols ready to requote, clearing their dirty flag.

        Considers every symbol we quote, not just the dirty ones: `due` also
        returns True for a symbol with orders resting on the exchange, which
        must be revisited even when no tick has arrived to mark it dirty.
        Iterating `_dirty` alone made that check unreachable.
        """
        now = now if now is not None else time.time()
        candidates = set(self._dirty) | set(self.mm.books)
        ready = [s for s in sorted(candidates) if self.due(s, now)]
        for s in ready:
            self._dirty.discard(s)
            self._last_requote[s] = now
        return ready
