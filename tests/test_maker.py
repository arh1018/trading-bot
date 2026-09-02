"""Market maker.

The tests are the economics: quoting a market whose spread does not clear the
fee loses on every round trip, and letting inventory run turns a spread-capture
strategy into an unintended directional bet.
"""

from __future__ import annotations

import pytest

from nbtrend.core.types import BookTop, Side
from nbtrend.live.maker import MarketMaker


class _Spec:
    def __init__(self, amount_step=1e-8):
        self.amount_step = amount_step
        self.src, self.dst = "avax", "rls"


def _book(bid, ask):
    return BookTop(symbol="AVAXIRT", best_bid=bid, best_ask=ask,
                   last_trade=(bid + ask) / 2, ts_ms=0)


def _mm(**kw):
    kw.setdefault("maker_fee", 0.0008)
    kw.setdefault("dry_run", True)
    return MarketMaker(None, {"AVAXIRT": _Spec()}, ["AVAXIRT"], **kw)


# -- the fee arithmetic -----------------------------------------------------
def test_breakeven_is_the_maker_fee_twice_not_maker_plus_taker():
    """Both legs REST, so a round trip pays maker on each. Using maker+taker
    (what a spread-crossing strategy pays) would overstate the hurdle and
    reject markets that are actually profitable -- IRT breakeven is 16 bps,
    not 18."""
    assert _mm(maker_fee=0.0008).breakeven_bps() == pytest.approx(16.0)
    assert _mm(maker_fee=0.0006).breakeven_bps() == pytest.approx(12.0)


def test_edge_is_spread_minus_breakeven():
    # AVAXIRT measured live at 33.8 bps against a 16 bps breakeven.
    mm = _mm()
    mid = 1_000_000.0
    half = mid * 33.8 / 10_000 / 2
    assert mm.edge_bps(_book(mid - half, mid + half)) == pytest.approx(17.8, abs=0.2)


@pytest.mark.parametrize(
    "spread_bps,expected",
    [(0.0, False), (0.5, False), (16.0, False), (19.9, False), (20.1, True), (33.8, True)],
)
def test_only_markets_clearing_breakeven_plus_margin_are_quoted(spread_bps, expected):
    """BTCIRT quotes 0.0 bps and SOLIRT 0.2 -- tighter than the fee, because
    someone else is already making them. Quoting there loses on every fill."""
    mm = _mm(min_edge_bps=4.0)
    mid = 1_000_000.0
    half = mid * spread_bps / 10_000 / 2
    assert mm.is_worth_quoting(_book(mid - half, mid + half)) is expected


def test_a_crossed_or_empty_book_is_never_quoted():
    mm = _mm()
    assert mm.edge_bps(_book(100.0, 100.0)) == 0.0
    assert not mm.is_worth_quoting(_book(101.0, 100.0))


# -- inventory --------------------------------------------------------------
def test_quotes_skew_down_when_long_to_shed_inventory():
    """Without skew the maker keeps buying into a falling market -- adverse
    selection accumulates on one side and the spread never covers it."""
    mm = _mm(max_inventory_rial=10_000_000.0)
    mm._balances = {"avax": 2.0}                 # 2,000,000 rial: stock to sell, under the cap
    mid = 1_000_000.0
    half = mid * 40 / 10_000 / 2
    book = _book(mid - half, mid + half)

    mm._balances = {"avax": 0.5}                 # near flat
    flat = mm.make_quotes("AVAXIRT", book)
    mm._balances = {"avax": 5.0}                 # 5,000,000 rial long
    long_ = mm.make_quotes("AVAXIRT", book)

    flat_bid = next(q for q in flat if q.side is Side.BUY).price
    long_bid = next(q for q in long_ if q.side is Side.BUY).price
    flat_ask = next(q for q in flat if q.side is Side.SELL).price
    long_ask = next(q for q in long_ if q.side is Side.SELL).price

    # Long inventory pushes BOTH quotes down: buy less eagerly, sell more.
    assert long_bid < flat_bid, "a long book must bid less eagerly"
    assert long_ask < flat_ask, "and must offer to sell closer to the mid"


def test_skew_is_signed_correctly_in_both_directions():
    """Tested on `inventory_skew` directly, because a SHORT book cannot be
    reached through `make_quotes` on spot: a wallet balance is never negative,
    so no ask is produced and there is no pair to compare. The sign convention
    still has to be right -- inverting it makes the maker buy harder the longer
    it already is.
    """
    mm = _mm(max_inventory_rial=10_000_000.0)
    mid = 1_000_000.0

    mm._balances = {"avax": 5.0}                 # 5,000,000 rial long
    assert mm.inventory_skew("AVAXIRT", mid) == pytest.approx(0.5)

    mm._balances = {"avax": 0.0}
    assert mm.inventory_skew("AVAXIRT", mid) == pytest.approx(0.0)

    # Negative only exists on the tracked-fill fallback path.
    mm._balances = None
    mm.books["AVAXIRT"].inventory = -5.0
    assert mm.inventory_skew("AVAXIRT", mid) == pytest.approx(-0.5)

    # And it is clamped, so an outsized position cannot invert the quote.
    mm.books["AVAXIRT"].inventory = -500.0
    assert mm.inventory_skew("AVAXIRT", mid) == pytest.approx(-1.0)


def test_the_buy_side_is_suppressed_at_the_long_inventory_cap():
    """The cap has to be a hard stop, not a lean: past it, buying more is
    simply taking a bigger directional position."""
    mm = _mm(max_inventory_rial=5_000_000.0)
    mm._balances = {"avax": 6.0}                 # 6,000,000 rial, over the cap
    mid = 1_000_000.0
    half = mid * 40 / 10_000 / 2          # 6,000,000 rial, over the cap
    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY not in sides
    assert Side.SELL in sides


def test_the_sell_side_is_suppressed_at_the_short_inventory_cap():
    # No wallet snapshot: a spot balance cannot be negative, so the short case
    # only exists on the tracked-inventory fallback path.
    mm = _mm(max_inventory_rial=5_000_000.0)
    mid = 1_000_000.0
    half = mid * 40 / 10_000 / 2
    mm.books["AVAXIRT"].inventory = -6.0
    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.SELL not in sides
    assert Side.BUY in sides


def test_quotes_stay_inside_the_touch_and_uncrossed():
    # Huge cap so skew is ~0: this isolates the quote GEOMETRY. Skew shifting
    # the pair outside the touch is correct behaviour and is covered separately.
    mm = _mm(max_inventory_rial=10_000_000_000.0)
    mm._balances = {"avax": 2.0}
    mid, half = 1_000_000.0, 2_000.0
    book = _book(mid - half, mid + half)
    qs = mm.make_quotes("AVAXIRT", book)
    bid = next(q for q in qs if q.side is Side.BUY).price
    ask = next(q for q in qs if q.side is Side.SELL).price
    assert bid < ask, "our own quotes must never cross"
    assert bid >= book.best_bid and ask <= book.best_ask


# -- rate budget ------------------------------------------------------------
def test_order_budget_is_capped_below_the_exchange_limit():
    """300/10min is shared with everything else on the account; requoting is
    this strategy's entire activity, so an uncapped maker starves the rest."""
    mm = _mm(max_orders_per_window=150)
    assert mm.can_place(150)
    mm.note_placement(150)
    assert not mm.can_place(1)


def test_budget_frees_up_as_the_window_slides():
    import time as _t

    mm = _mm(max_orders_per_window=10)
    mm._placements = [_t.time() - 601] * 10      # all older than the window
    assert mm.can_place(10)


def test_quotable_symbol_count_follows_the_requote_rate():
    """The constraint that decides how many markets can be quoted at all."""
    assert _mm(max_orders_per_window=150, requote_s=30).max_quotable_symbols() == pytest.approx(3.75)
    assert _mm(max_orders_per_window=150, requote_s=60).max_quotable_symbols() == pytest.approx(7.5)
    assert _mm(max_orders_per_window=150, requote_s=10).max_quotable_symbols() == pytest.approx(1.25)


# -- P&L --------------------------------------------------------------------
def test_a_completed_round_trip_earns_the_spread_minus_both_fees():
    """The whole thesis, in one assertion."""
    mm = _mm(maker_fee=0.0008)
    mm.on_fill("AVAXIRT", Side.BUY, 3.0, 1_000_000.0)
    mm.on_fill("AVAXIRT", Side.SELL, 3.0, 1_003_380.0)   # +33.8 bps

    gross = 3.0 * (1_003_380.0 - 1_000_000.0)
    fees = 0.0008 * 3.0 * (1_000_000.0 + 1_003_380.0)
    assert mm.books["AVAXIRT"].inventory == pytest.approx(0.0)
    assert mm.pnl_rial("AVAXIRT", 1_003_380.0) == pytest.approx(gross - fees)
    assert mm.pnl_rial("AVAXIRT", 1_003_380.0) > 0, "33.8 bps must beat 16 bps of fees"


def test_a_round_trip_inside_breakeven_loses_money():
    """A 10 bps capture against a 16 bps fee is a loss, however many times it
    is repeated -- this is why the majors are excluded."""
    mm = _mm(maker_fee=0.0008)
    mm.on_fill("AVAXIRT", Side.BUY, 3.0, 1_000_000.0)
    mm.on_fill("AVAXIRT", Side.SELL, 3.0, 1_001_000.0)   # +10 bps
    assert mm.pnl_rial("AVAXIRT", 1_001_000.0) < 0


def test_unbalanced_fills_are_marked_to_market_not_counted_as_profit():
    """One-sided fills are the adverse-selection case: cash flow looks fine
    while the position is the whole risk. P&L must include the mark."""
    mm = _mm()
    mm.on_fill("AVAXIRT", Side.BUY, 3.0, 1_000_000.0)
    assert mm.books["AVAXIRT"].realized_rial < 0          # cash went out
    assert mm.pnl_rial("AVAXIRT", 1_000_000.0) == pytest.approx(-0.0008 * 3_000_000.0)
    # And a fall in the mid is a real loss, not a flat book.
    assert mm.pnl_rial("AVAXIRT", 950_000.0) < mm.pnl_rial("AVAXIRT", 1_000_000.0)


# -- spot reality: you cannot sell what you do not hold ----------------------
def test_no_ask_is_quoted_with_zero_holdings():
    """On a freshly funded account inventory is 0, so every ask comes back
    "Order Validation Failed" -- half the order budget burned on errors each
    cycle. Observed live right after liquidating to rial."""
    mm = _mm()
    mm._balances = {"avax": 0.0}
    mid, half = 1_000_000.0, 2_000.0
    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.SELL not in sides
    assert Side.BUY in sides, "the bid still works -- that is how inventory is bootstrapped"


def test_the_ask_is_capped_at_the_units_actually_held():
    mm = _mm(quote_notional_rial=3_000_000.0)
    mm._balances = {"avax": 1.0}          # only 1 unit, quote wants 3
    mid, half = 1_000_000.0, 2_000.0
    ask = next(q for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
               if q.side is Side.SELL)
    assert ask.amount == pytest.approx(1.0)


def test_a_full_two_sided_quote_returns_once_holdings_exist():
    mm = _mm(quote_notional_rial=3_000_000.0)
    mm._balances = {"avax": 5.0}          # 5,000,000 rial: sellable, under cap
    mid, half = 1_000_000.0, 2_000.0
    qs = mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
    assert {q.side for q in qs} == {Side.BUY, Side.SELL}
    assert next(q for q in qs if q.side is Side.SELL).amount == pytest.approx(3.0)


def test_without_a_snapshot_it_falls_back_to_tracked_inventory():
    """Keeps the pure-logic path usable when no wallet read has happened."""
    mm = _mm()
    assert mm._balances is None
    mm.books["AVAXIRT"].inventory = 4.0
    assert mm.available_base("AVAXIRT") == pytest.approx(4.0)


# -- the quoted spread must clear the fee, not just the market's -------------
def test_our_own_quoted_pair_always_clears_breakeven_plus_margin():
    """The bug this pins: pricing at a fixed 0.9 of the market half-spread
    referenced the MARKET spread and never the fee. A 21.7 bps market became
    19.5 bps quoted, against a 16 bps breakeven -- the gate approved 5.7 bps of
    edge and the prices delivered 3.5."""
    mm = _mm(min_edge_bps=8.0, maker_fee=0.0008)     # breakeven 16, floor 24
    mm._balances = {"avax": 2.0}
    mid = 1_000_000.0

    for spread_bps in (21.7, 24.0, 30.2, 33.8, 60.0, 128.2):
        half = mid * spread_bps / 10_000 / 2
        qs = mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
        if not qs:
            continue
        bid = next(q for q in qs if q.side is Side.BUY).price
        ask = next(q for q in qs if q.side is Side.SELL).price
        assert mm.quoted_edge_bps(bid, ask) >= mm.min_edge_bps - 1e-6, (
            f"at a {spread_bps} bps market our pair earns only "
            f"{mm.quoted_edge_bps(bid, ask):.1f} bps"
        )


@pytest.mark.parametrize("inventory", [-9.0, -5.0, -1.0, 0.0, 1.0, 5.0, 9.0])
def test_skew_shifts_the_pair_without_ever_narrowing_it(inventory):
    """Skew used to narrow ONE leg: at skew 0.5 the ask moved to 0.45 of the
    half-spread, so a 22.5 bps market was quoted ~10 bps on that side -- inside
    the 16 bps fee, losing on every fill. Skew must move the centre only."""
    mm = _mm(min_edge_bps=8.0, max_inventory_rial=9_000_000.0)
    mm._balances = {"avax": 2.0}
    mm.books["AVAXIRT"].inventory = inventory
    mid = 1_000_000.0
    half = mid * 25.0 / 10_000 / 2

    qs = mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
    bid = next((q.price for q in qs if q.side is Side.BUY), None)
    ask = next((q.price for q in qs if q.side is Side.SELL), None)
    if bid is None or ask is None:
        return          # a suppressed side at the cap is fine
    assert mm.quoted_edge_bps(bid, ask) >= mm.min_edge_bps - 1e-6


def test_a_thin_market_is_not_narrowed_to_the_queue_priority_width():
    """When the market is barely over breakeven, queue priority is not worth
    quoting through the fee: the 0.9x step-inside is overridden by the floor,
    so we quote 24 bps into a 25 bps market rather than 22.5."""
    mm = _mm(min_edge_bps=8.0, maker_fee=0.0008)     # needs >= 24 bps
    mm._balances = {"avax": 2.0}
    mid = 1_000_000.0
    half = mid * 25.0 / 10_000 / 2                   # market only 25 bps
    book = _book(mid - half, mid + half)

    qs = mm.make_quotes("AVAXIRT", book)
    bid = next(q for q in qs if q.side is Side.BUY).price
    ask = next(q for q in qs if q.side is Side.SELL).price
    assert (ask - bid) > (book.best_ask - book.best_bid) * 0.9, "floor beats 0.9x"
    assert (ask - bid) < (book.best_ask - book.best_bid), "still inside the touch"
    assert mm.quoted_edge_bps(bid, ask) >= 8.0 - 1e-6


def test_a_wide_market_is_still_quoted_inside_the_touch_for_queue_priority():
    """Where there is room, stepping inside the touch still earns the queue."""
    mm = _mm(min_edge_bps=8.0, max_inventory_rial=10_000_000_000.0)
    mm._balances = {"avax": 2.0}
    mid = 1_000_000.0
    half = mid * 128.2 / 10_000 / 2                  # HOLOIRT, measured live
    book = _book(mid - half, mid + half)

    qs = mm.make_quotes("AVAXIRT", book)
    bid = next(q for q in qs if q.side is Side.BUY).price
    ask = next(q for q in qs if q.side is Side.SELL).price
    assert bid > book.best_bid and ask < book.best_ask, "should improve the touch"
    assert mm.quoted_edge_bps(bid, ask) > 90.0


# -- the inventory cap must read the WALLET ---------------------------------
def test_the_cap_is_measured_from_the_wallet_not_tracked_fills():
    """The live failure: PROM reached 14,803,985 rial against a 9,000,000 cap
    (64% over). The cap read `books[symbol].inventory`, which counts only fills
    THIS process observed -- so a restart, a missed fill, or anything bought
    elsewhere is invisible to it."""
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0)
    mid, half = 1_000_000.0, 20_000.0

    # Tracked inventory says flat; the wallet says we are already over.
    mm.books["AVAXIRT"].inventory = 0.0
    mm._balances = {"avax": 14.8}                # 14,800,000 rial

    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY not in sides, "must not buy more while over the cap"
    assert Side.SELL in sides, "and must keep offering to shed it"


def test_a_quote_that_would_breach_the_cap_is_trimmed_not_posted_whole():
    """Checking "not yet at the cap" is not enough: a 3,000,000 quote posted
    against an 8,000,000 position breaches a 9,000,000 cap the moment it fills."""
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0,
             min_quote_rial=500_000.0)
    mid, half = 1_000_000.0, 20_000.0
    mm._balances = {"avax": 8.0}                 # 8,000,000 rial, 1,000,000 of room

    buy = next((q for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
                if q.side is Side.BUY), None)
    assert buy is not None
    assert buy.notional <= 9_000_000.0 - 8_000_000.0 + 1.0, (
        f"quote of {buy.notional:,.0f} would breach the cap"
    )


def test_a_trim_below_the_exchange_minimum_is_dropped_entirely():
    """A trimmed quote under min_order_rial is rejected by the exchange, so
    posting it only burns order budget."""
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0,
             min_quote_rial=3_000_000.0)
    mid, half = 1_000_000.0, 20_000.0
    mm._balances = {"avax": 8.5}                 # only 500,000 of room

    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY not in sides


# -- the portfolio-level cash floor -----------------------------------------
def test_bids_stop_when_cash_hits_the_reserve():
    """Per-symbol caps do not bound the PORTFOLIO. 12 symbols x 9,000,000
    permits 108,000,000 of buying against an account holding 18,962,531 of
    cash -- and since bids fill while asks do not, the maker converted
    77,200,000 of cash into 208,250 across 12 coins in 17 minutes."""
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0,
             min_cash_rial=10_000_000.0)
    mid, half = 1_000_000.0, 20_000.0
    mm._balances = {"avax": 1.0, "rls": 10_500_000.0}   # only 500k above the floor

    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY not in sides, "must not spend into the reserve"
    assert Side.SELL in sides, "selling to rebuild cash is still allowed"


def test_bids_resume_with_cash_well_above_the_reserve():
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0,
             min_cash_rial=10_000_000.0)
    mid, half = 1_000_000.0, 20_000.0
    mm._balances = {"avax": 1.0, "rls": 30_000_000.0}
    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY in sides


def test_cash_floor_is_inert_without_a_wallet_snapshot():
    """Keeps the pure-logic path testable without a broker."""
    mm = _mm(min_cash_rial=10_000_000.0)
    assert mm.cash_room() == float("inf")


def test_a_snapshot_without_a_rial_entry_does_not_block_bidding():
    """Missing means unknown, not empty. Reading it as zero blocks every bid
    and silently turns the maker into a sell-only process."""
    mm = _mm(min_cash_rial=10_000_000.0)
    mm._balances = {"avax": 1.0}          # no "rls" key at all
    assert mm.cash_room() == float("inf")


# -- orphaned inventory -----------------------------------------------------
def _runner(mm, orderbooks):
    from nbtrend.live.maker import MakerRunner

    class _REST:
        def orderbook(self, symbol):
            return orderbooks[symbol]

    return MakerRunner(None, mm, _REST())


def test_held_markets_are_quoted_even_when_not_in_the_symbol_list():
    """Cutting the symbol list from 12 to 6 stranded 39,884,121 rial -- 65% of
    holdings -- in markets with nothing quoting them. Selling a position down
    must not depend on that market still being worth entering."""
    from nbtrend.live.maker import MarketMaker

    specs = {"AVAXIRT": _Spec(), "SUIIRT": _Spec()}
    mm = MarketMaker(None, specs, ["AVAXIRT"], maker_fee=0.0008, dry_run=True)
    mm._balances = {"avax": 1.0, "sui": 6.77, "rls": 20_000_000.0}

    books = {
        "AVAXIRT": _book(998_000.0, 1_002_000.0),
        "SUIIRT": _book(1_549_000.0, 1_551_000.0),
    }
    runner = _runner(mm, books)
    held = runner.held_symbols()
    assert "SUIIRT" in held, "a held market must be visible regardless of the config"


def test_dust_is_not_promoted_into_a_quoted_market():
    """zec/btc dust of ~150,000 rial is not worth an order slot."""
    from nbtrend.live.maker import MarketMaker

    specs = {"BTCIRT": _Spec()}
    mm = MarketMaker(None, specs, [], maker_fee=0.0008, dry_run=True)
    mm._balances = {"btc": 0.000001}
    books = {"BTCIRT": _book(164_000_000_000.0, 165_000_000_000.0)}
    assert _runner(mm, books).held_symbols(min_rial=1_000_000.0) == []


def test_a_coin_outside_the_universe_is_ignored():
    """No spec means no amount_step, so it cannot be quoted safely."""
    from nbtrend.live.maker import MarketMaker

    mm = MarketMaker(None, {"AVAXIRT": _Spec()}, ["AVAXIRT"], dry_run=True)
    mm._balances = {"weirdcoin": 1000.0}
    assert _runner(mm, {}).held_symbols() == []
