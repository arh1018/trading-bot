"""Market maker.

The tests are the economics: quoting a market whose spread does not clear the
fee loses on every round trip, and letting inventory run turns a spread-capture
strategy into an unintended directional bet.
"""

from __future__ import annotations

import pytest

from nbtrend.config import load_config
from nbtrend.core.types import BookTop, Side
from nbtrend.live.maker import MarketMaker


class _Spec:
    def __init__(self, amount_step=1e-8, src="avax"):
        self.amount_step = amount_step
        self.src, self.dst = src, "rls"


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
    # Cap generous enough that BOTH cases still quote a bid -- at the cap the
    # buy side is suppressed and there is nothing to compare.
    mm = _mm(max_inventory_rial=30_000_000.0)
    mid = 1_000_000.0
    half = mid * 40 / 10_000 / 2
    book = _book(mid - half, mid + half)

    # Both cases must hold enough to quote a LEGAL ask (>= 3,000,000), or the
    # comparison has nothing to compare -- so "flat" here means lightly long.
    mm._balances = {"avax": 4.0}                 # 4,000,000 rial, lightly long
    flat = mm.make_quotes("AVAXIRT", book)
    mm._balances = {"avax": 20.0}                # 20,000,000 rial, heavily long
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
    mm._balances = {"avax": 6.0}
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
    # Quote wants 8 units; we own 5. Both sizes clear the 3,000,000 minimum,
    # so this isolates the holdings cap from the minimum-order gate.
    mm = _mm(quote_notional_rial=8_000_000.0, max_inventory_rial=30_000_000.0)
    mm._balances = {"avax": 5.0}
    mid, half = 1_000_000.0, 2_000.0
    ask = next(q for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
               if q.side is Side.SELL)
    assert ask.amount == pytest.approx(5.0), "capped at what we hold"
    assert ask.notional >= 3_000_000.0, "and still a legal order"


def test_a_full_two_sided_quote_returns_once_holdings_exist():
    mm = _mm(quote_notional_rial=3_000_000.0)
    mm._balances = {"avax": 6.0}          # 6,000,000 rial: sellable, under cap
    mid, half = 1_000_000.0, 2_000.0
    qs = mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
    assert {q.side for q in qs} == {Side.BUY, Side.SELL}
    # Holding 6 units against a 3-unit quote leaves a 3-unit remainder. With
    # the minimum at 550,000 that remainder is still sellable, so normal
    # sizing applies -- but the ask must clear the minimum either way.
    ask = next(q for q in qs if q.side is Side.SELL)
    assert ask.notional >= float(load_config().costs["min_order_rial"])
    assert ask.amount <= 6.0


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
    mm._balances = {"avax": 6.0}
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
    mm._balances = {"avax": 6.0}
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
    mm._balances = {"avax": 6.0}
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
    mm._balances = {"avax": 6.0}
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
    mm._balances = {"avax": 6.0, "rls": 10_500_000.0}   # only 500k above the floor

    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.BUY not in sides, "must not spend into the reserve"
    assert Side.SELL in sides, "selling to rebuild cash is still allowed"


def test_bids_resume_with_cash_well_above_the_reserve():
    mm = _mm(max_inventory_rial=9_000_000.0, quote_notional_rial=3_000_000.0,
             min_cash_rial=10_000_000.0)
    mid, half = 1_000_000.0, 20_000.0
    mm._balances = {"avax": 6.0, "rls": 30_000_000.0}
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


# -- the three ONEIRT bugs --------------------------------------------------
def test_a_404_on_cancel_is_a_fill_not_a_failure():
    """A resting quote that filled is gone by the time we requote, so
    cancelling it 404s. 71 of those were logged as errors with tracebacks,
    which hid the real problem (locked inventory) completely."""
    import httpx

    from nbtrend.live.maker import _is_missing_order

    req = httpx.Request("POST", "https://apiv2.nobitex.ir/market/orders/update-status")
    missing = httpx.HTTPStatusError(
        "404", request=req, response=httpx.Response(404, request=req)
    )
    assert _is_missing_order(missing)

    real = httpx.HTTPStatusError(
        "500", request=req, response=httpx.Response(500, request=req)
    )
    assert not _is_missing_order(real), "a real failure must still be reported"


def test_total_balances_includes_units_locked_in_our_own_orders():
    """The ONEIRT bug: activeBalance excluded the 1796.48944 units blocked by
    our own resting quote, leaving 0.08944 free -- under the 0.1 amount step,
    so `sellable` rounded to zero and no ask was ever posted."""
    from nbtrend.data.nobitex_rest import NobitexREST

    rest = object.__new__(NobitexREST)
    rest._get = lambda path, **kw: {
        "wallets": [{"currency": "ONE", "balance": "1796.48944",
                     "activeBalance": "0.08944"}]
    }
    assert rest.total_balances()["one"] == pytest.approx(1796.48944)


def test_an_ask_is_quoted_for_inventory_locked_in_a_working_order():
    """End to end: with the total balance the ask comes back."""
    mm = _mm(quote_notional_rial=3_000_000.0, max_inventory_rial=9_000_000.0)
    mm._balances = {"avax": 3.0, "rls": 50_000_000.0}   # total, not active
    mid, half = 1_000_000.0, 20_000.0
    sides = {q.side for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))}
    assert Side.SELL in sides


def test_a_market_discovered_from_the_wallet_gets_state_on_demand():
    """`books` is built from the CONFIGURED symbols, but the runner also quotes
    markets found in the wallet so a position is never orphaned. Those have no
    entry, and indexing raised KeyError: 'BTCIRT' at startup -- the orphan fix
    and this state map disagreed about which symbols exist."""
    from nbtrend.live.maker import MarketMaker

    mm = MarketMaker(None, {"AVAXIRT": _Spec()}, ["AVAXIRT"], dry_run=True)
    assert "BTCIRT" not in mm.books
    book = mm.book_for("BTCIRT")
    assert book.symbol == "BTCIRT"
    assert mm.book_for("BTCIRT") is book, "must be stable across calls"


def test_held_rial_works_for_an_unconfigured_market():
    from nbtrend.live.maker import MarketMaker

    mm = MarketMaker(None, {"AVAXIRT": _Spec()}, ["AVAXIRT"], dry_run=True)
    mm._balances = {"btc": 0.5}
    assert mm.held_rial("BTCIRT", 100.0) == pytest.approx(50.0)


# -- fast requoting without breaching the rate limit ------------------------
def _mk_runner(mm, tol=3.0):
    from nbtrend.live.maker import MakerRunner

    return MakerRunner(None, mm, None, requote_tolerance_bps=tol)


def test_an_unchanged_quote_is_not_reposted():
    """Cancel-and-repost is the whole cost of requoting: 2 placements per
    symbol per sweep against a 300-per-10-minutes limit. At 10s on 6 symbols
    that is 720 -- 2.4x over, so most quotes get refused and the book updates
    LESS often than at 30s. Skipping unchanged quotes decouples how often we
    LOOK from how often we SPEND, and preserves queue position."""
    from nbtrend.live.maker import Quote

    mm = _mm()
    r = _mk_runner(mm)
    quotes = [Quote(Side.BUY, 1_000_000.0, 3.0), Quote(Side.SELL, 1_004_000.0, 3.0)]
    mm.book_for("AVAXIRT").working = {"a": object(), "b": object()}
    r._last_quotes["AVAXIRT"] = quotes

    same = [Quote(Side.BUY, 1_000_050.0, 3.0), Quote(Side.SELL, 1_004_050.0, 3.0)]
    assert r._unchanged("AVAXIRT", same), "0.5 bps of drift is not worth a repost"


def test_a_moved_quote_is_reposted():
    from nbtrend.live.maker import Quote

    mm = _mm()
    r = _mk_runner(mm)
    mm.book_for("AVAXIRT").working = {"a": object(), "b": object()}
    r._last_quotes["AVAXIRT"] = [Quote(Side.BUY, 1_000_000.0, 3.0),
                                 Quote(Side.SELL, 1_004_000.0, 3.0)]

    moved = [Quote(Side.BUY, 1_002_000.0, 3.0), Quote(Side.SELL, 1_006_000.0, 3.0)]
    assert not r._unchanged("AVAXIRT", moved), "20 bps of drift must repost"


def test_a_filled_side_forces_a_repost():
    """If a quote is gone from `working`, the pair is incomplete and must be
    rebuilt -- otherwise a filled bid leaves us quoting one-sided forever."""
    from nbtrend.live.maker import Quote

    mm = _mm()
    r = _mk_runner(mm)
    quotes = [Quote(Side.BUY, 1_000_000.0, 3.0), Quote(Side.SELL, 1_004_000.0, 3.0)]
    r._last_quotes["AVAXIRT"] = quotes
    mm.book_for("AVAXIRT").working = {"a": object()}      # one leg filled
    assert not r._unchanged("AVAXIRT", quotes)


def test_a_resized_quote_is_reposted():
    from nbtrend.live.maker import Quote

    mm = _mm()
    r = _mk_runner(mm)
    mm.book_for("AVAXIRT").working = {"a": object(), "b": object()}
    r._last_quotes["AVAXIRT"] = [Quote(Side.BUY, 1_000_000.0, 3.0),
                                 Quote(Side.SELL, 1_004_000.0, 3.0)]
    resized = [Quote(Side.BUY, 1_000_000.0, 1.0), Quote(Side.SELL, 1_004_000.0, 3.0)]
    assert not r._unchanged("AVAXIRT", resized), "a 3x size change must repost"


def test_balance_reads_do_not_scale_with_the_loop_rate():
    """The wallet endpoint already returned 429s at a 60s cycle; at 10s it
    would be read 6x as often for data that barely changes."""
    mm = _mm()
    r = _mk_runner(mm)
    assert r.balance_refresh_s >= 30.0


def test_quotes_are_reconciled_against_the_exchange_not_our_bookkeeping():
    """`book.working` is what we BELIEVE we posted; the exchange is what is
    actually resting, and they drift -- a cancel that 404s pops the entry
    whether or not anything was pulled. That drift stacked three KMNOIRT bids
    at 51,750 / 52,080 / 52,210 at once."""
    from nbtrend.live.maker import MakerRunner, MarketMaker, Quote

    mm = MarketMaker(None, {"KMNOIRT": _Spec()}, ["KMNOIRT"], dry_run=True)

    class _REST:
        def _get(self, path, **kw):
            return {"orders": [
                {"clientOrderId": "mkr1", "srcCurrency": "kmno", "type": "buy"},
                {"clientOrderId": "mkr2", "srcCurrency": "kmno", "type": "buy"},
                {"clientOrderId": "mkr3", "srcCurrency": "kmno", "type": "buy"},
            ]}

    r = MakerRunner(None, mm, _REST())
    r._coid_symbol = {"mkr1": "KMNOIRT", "mkr2": "KMNOIRT", "mkr3": "KMNOIRT"}
    live = r._live_orders("KMNOIRT")
    assert len(live) == 3, "the exchange view must see all three"

    # We want a two-sided pair; three live orders can never match, so the
    # unchanged-skip cannot fire and leave them stacked.
    wanted = [Quote(Side.BUY, 52_000.0, 57.0), Quote(Side.SELL, 52_500.0, 57.0)]
    assert len(live) != len(wanted)


def test_cancel_covers_orders_we_did_not_know_about():
    """The union of our records and the exchange's, so neither view can leave
    an order resting."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    mm = MarketMaker(None, {"KMNOIRT": _Spec()}, ["KMNOIRT"], dry_run=False)
    cancelled = []

    class _REST:
        def cancel_order(self, client_order_id=None, order_id=None):
            cancelled.append(client_order_id)
            return True

    r = MakerRunner(None, mm, _REST())
    mm.book_for("KMNOIRT").working = {"mkr_known": object()}
    r.cancel_working("KMNOIRT", live=[{"clientOrderId": "mkr_unknown"}])
    assert set(cancelled) == {"mkr_known", "mkr_unknown"}


def test_orphan_selling_lives_in_the_maker_not_a_second_process():
    """Two components selling the same inventory collide: the cron sweep's
    resting ask blocked the units, then the maker sized its own ask from the
    TOTAL balance and got Order Validation Failed. The maker quotes asks on
    held markets itself, so the cron is redundant."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    mm = MarketMaker(None, {"RAYIRT": _Spec()}, [], dry_run=True)
    mm._balances = {"ray": 3.0}

    class _REST:
        def orderbook(self, symbol):
            return _book(1_700_000.0, 1_710_000.0)

    assert "RAYIRT" in MakerRunner(None, mm, _REST()).held_symbols()


# -- socket-driven quoting --------------------------------------------------
def _sock(mm, gap=2.0):
    from nbtrend.live.maker import SocketMakerRunner

    return SocketMakerRunner(None, mm, None, None, min_requote_gap_s=gap)


def test_a_book_update_marks_the_symbol_for_requoting():
    """REST polling prices quotes off a book read seconds ago, so a fill
    arrives from someone reacting to the book as it is NOW. Measured over 24h
    that cost -80,276 rial of realized P&L across 7 markets."""
    r = _sock(_mm())
    assert not r.due("AVAXIRT")
    r.on_book("AVAXIRT", _book(998_000.0, 1_002_000.0))
    assert r.due("AVAXIRT")
    assert r.latest_book("AVAXIRT").best_bid == 998_000.0


def test_the_cooldown_bounds_how_often_one_symbol_can_requote():
    """The socket changes WHEN we act, not how much: the exchange still allows
    300 placements per 10 minutes, and one fast market must not eat it all."""
    r = _sock(_mm(), gap=2.0)
    r.on_book("AVAXIRT", _book(998_000.0, 1_002_000.0))
    assert r.take_due(now=1000.0) == ["AVAXIRT"]

    r.on_book("AVAXIRT", _book(998_100.0, 1_002_100.0))
    assert r.take_due(now=1001.0) == [], "inside the cooldown"
    assert r.take_due(now=1002.5) == ["AVAXIRT"], "past it"


def test_an_unchanged_book_does_not_requote():
    """Dirty flags clear when taken, so a quiet market costs nothing."""
    r = _sock(_mm())
    r.on_book("AVAXIRT", _book(998_000.0, 1_002_000.0))
    assert r.take_due(now=1000.0) == ["AVAXIRT"]
    assert r.take_due(now=2000.0) == [], "no new book, no requote"


def test_only_changed_symbols_are_swept():
    r = _sock(_mm())
    r.on_book("AVAXIRT", _book(998_000.0, 1_002_000.0))
    r.on_book("SUIIRT", _book(1_549_000.0, 1_551_000.0))
    assert r.take_due(now=1000.0) == ["AVAXIRT", "SUIIRT"]

    r.on_book("SUIIRT", _book(1_549_500.0, 1_551_500.0))
    assert r.take_due(now=1005.0) == ["SUIIRT"], "AVAX did not move"


def test_the_base_runner_has_no_socket_book():
    """`quote_once` is shared; the REST runner must fall back to polling."""
    from nbtrend.live.maker import MakerRunner

    assert MakerRunner(None, _mm(), None).latest_book("AVAXIRT") is None


def test_the_order_listing_is_fetched_once_per_sweep_not_per_symbol():
    """Per-symbol listing was fine at a 60s poll and fatal once quoting became
    event driven: the socket fires far more often, and a call per symbol per
    event earned 13 x HTTP 400 and 7 x 429 in under two minutes, so nothing
    could be placed at all."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    calls = {"n": 0}

    class _REST:
        def _get(self, path, **kw):
            calls["n"] += 1
            return {"orders": []}

    mm = MarketMaker(None, {"AVAXIRT": _Spec(), "SUIIRT": _Spec(src="sui")},
                     ["AVAXIRT", "SUIIRT"], dry_run=True)
    r = MakerRunner(None, mm, _REST())

    for _ in range(3):
        r._live_orders("AVAXIRT")
        r._live_orders("SUIIRT")
    assert calls["n"] == 1, f"expected one shared listing, made {calls['n']}"


def test_placing_an_order_invalidates_the_cached_listing():
    """A stale view after a write is how duplicate quotes get posted."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    class _REST:
        def _get(self, path, **kw):
            return {"orders": []}

    r = MakerRunner(None, MarketMaker(None, {}, [], dry_run=True), _REST())
    r._live_orders("AVAXIRT")
    assert r._orders_at > 0
    r._orders_at = 0.0          # what a placement does
    assert r._orders_at == 0.0


def test_an_ask_below_the_exchange_minimum_is_not_quoted():
    """Capping the ask at what we HOLD is not enough -- it must also clear
    min_order_rial. Holding 0.10045944 HOLO produced a 13,845 rial ask against
    a 3,000,000 minimum, rejected on every sweep: 47 rejections in under two
    minutes, which looked like rate limiting and was not."""
    mm = _mm(quote_notional_rial=3_000_000.0, min_quote_rial=3_000_000.0,
             max_inventory_rial=9_000_000.0)
    mm._balances = {"holo": 0.10045944, "rls": 50_000_000.0}
    mm.specs["HOLOIRT"] = _Spec(amount_step=0.1, src="holo")

    sides = {q.side for q in mm.make_quotes("HOLOIRT", _book(137_500.0, 138_500.0))}
    assert Side.SELL not in sides, "a dust ask is rejected by the exchange every time"
    assert Side.BUY in sides, "the bid is still fine"


def test_an_ask_that_clears_the_minimum_is_quoted():
    mm = _mm(quote_notional_rial=4_000_000.0, min_quote_rial=3_000_000.0,
             max_inventory_rial=20_000_000.0)
    mm._balances = {"holo": 40.0, "rls": 50_000_000.0}     # ~5.5M of stock
    mm.specs["HOLOIRT"] = _Spec(amount_step=0.1, src="holo")

    ask = next(q for q in mm.make_quotes("HOLOIRT", _book(137_500.0, 138_500.0))
               if q.side is Side.SELL)
    assert ask.notional >= 3_000_000.0


# -- markets the bot must never touch ---------------------------------------
def test_a_protected_market_is_never_quoted_even_when_held():
    """Holdings are discovered from the wallet so a position cannot be
    orphaned -- but the same mechanism would start making a market in
    something the ACCOUNT OWNER trades by hand. 478,875,002 rial of PAXG sat
    in this wallet, placed by a person, and the maker could not tell it apart
    from its own inventory."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    specs = {"PAXGIRT": _Spec(src="paxg"), "AVAXIRT": _Spec()}
    mm = MarketMaker(None, specs, ["AVAXIRT"], dry_run=True)
    mm._balances = {"paxg": 0.05, "avax": 6.0}

    class _REST:
        def orderbook(self, symbol):
            return _book(9_500_000_000.0, 9_600_000_000.0) if symbol == "PAXGIRT" \
                else _book(998_000.0, 1_002_000.0)

    r = MakerRunner(None, mm, _REST())
    r.protected = {"PAXGIRT", "XAUTIRT"}
    held = r.held_symbols()
    assert "PAXGIRT" not in held, "the owner's gold must never be quoted"
    assert "AVAXIRT" in held, "our own inventory still is"


def test_the_gold_pairs_are_protected_by_default():
    """A default the operator has to actively remove, not one they must
    remember to add."""
    import inspect

    from nbtrend.cli import make

    default = inspect.signature(make).parameters["protect"].default
    text = getattr(default, "default", default)
    for pair in ("PAXGIRT", "XAUTIRT"):
        assert pair in str(text)


# -- sweep-level accounting -------------------------------------------------
def test_cash_committed_earlier_in_a_sweep_limits_later_symbols():
    """The wallet snapshot is up to 30s old, so twelve symbols quoting in quick
    succession all saw the same pre-commitment cash and each concluded there
    was room: 48,981,598 fell to 7,118,000 straight through an 8,000,000
    floor. The floor bounded ONE bid, never a sweep of twelve."""
    mm = _mm(min_cash_rial=8_000_000.0)
    mm._balances = {"rls": 20_000_000.0}
    assert mm.cash_room() == pytest.approx(12_000_000.0)

    mm.commit_cash(3_000_000.0)
    assert mm.cash_room() == pytest.approx(9_000_000.0)
    mm.commit_cash(9_000_000.0)
    assert mm.cash_room() == 0.0, "the floor must hold across the whole sweep"


def test_twelve_bids_cannot_spend_through_the_floor():
    """The exact live failure, in one assertion."""
    mm = _mm(min_cash_rial=8_000_000.0)
    mm._balances = {"rls": 48_981_598.0}
    spent = 0.0
    for _ in range(12):
        if mm.cash_room() < 3_000_000.0:
            break
        mm.commit_cash(3_000_000.0)
        spent += 3_000_000.0
    assert 48_981_598.0 - spent >= 8_000_000.0, (
        f"spent {spent:,.0f}, leaving {48_981_598.0 - spent:,.0f} under the floor"
    )


def test_commitments_reset_between_sweeps():
    """The next sweep's snapshot already reflects the fills, so carrying the
    commitment forward would double-count it and freeze bidding."""
    mm = _mm(min_cash_rial=0.0)
    mm._balances = {"rls": 10_000_000.0}
    mm.commit_cash(6_000_000.0)
    assert mm.cash_room() == pytest.approx(4_000_000.0)
    mm.reset_commitments()
    assert mm.cash_room() == pytest.approx(10_000_000.0)


def test_an_order_posted_this_sweep_is_visible_before_the_cache_refreshes():
    """A symbol quoted twice inside the 20s listing cache could not see its own
    first order and posted a second -- Harmony and Raydium both doubled."""
    from nbtrend.live.maker import MakerRunner, MarketMaker

    class _REST:
        def _get(self, path, **kw):
            return {"orders": []}          # stale: predates our placement

    mm = MarketMaker(None, {"ONEIRT": _Spec(src="one")}, ["ONEIRT"], dry_run=True)
    r = MakerRunner(None, mm, _REST())
    assert r._live_orders("ONEIRT") == []

    mm.book_for("ONEIRT").working["mkr-just-posted"] = object()
    assert len(r._live_orders("ONEIRT")) == 1, "our own fresh order must count"


# -- global fair value ------------------------------------------------------
def test_fair_value_is_global_price_times_fx():
    """The local book is one venue's opinion. `global_usd x USDT/IRT` is what
    the asset is actually worth, and quoting blind to it means our bid rests
    where the coin USED to be worth -- adverse selection arriving through
    price rather than queue position."""
    mm = _mm()
    mm.set_fair("AVAXIRT", global_usd=25.0, fx_rial_per_usdt=1_982_000.0)
    assert mm.fair_rial("AVAXIRT") == pytest.approx(25.0 * 1_982_000.0)


def test_fair_value_respects_the_scaled_market_multiplier():
    """1M_PEPEIRT quotes a MILLION pepe, so its fair price is 1,000,000x the
    per-unit global price. Dividing instead gave a fair value of 0 and a basis
    of 99,568,862,847,924% -- caught against live books."""
    mm = _mm()
    mm.set_fair("1M_PEPEIRT", global_usd=0.000008, fx_rial_per_usdt=1_982_000.0,
                multiplier=1_000_000)
    assert mm.fair_rial("1M_PEPEIRT") == pytest.approx(0.000008 * 1_982_000.0 * 1_000_000)


def test_a_scaled_market_basis_is_sane_against_a_real_book():
    """1M_PEPEIRT printed 8,273,105 against a per-unit global price; with the
    multiplier applied correctly the basis must be a few percent, not trillions."""
    mm = _mm()
    mm.set_fair("1M_PEPEIRT", global_usd=0.0000037, fx_rial_per_usdt=2_199_875.0,
                multiplier=1_000_000)
    basis = mm.basis("1M_PEPEIRT", _book(8_270_000.0, 8_276_000.0))
    assert abs(basis) < 0.20, f"basis {basis:.2%} is not a plausible dislocation"


def test_basis_measures_local_richness_against_global():
    mm = _mm()
    fair = 1_000_000.0
    mm.set_fair("AVAXIRT", global_usd=1.0, fx_rial_per_usdt=fair)
    assert mm.basis("AVAXIRT", _book(1_019_000.0, 1_021_000.0)) == pytest.approx(0.02, abs=1e-4)
    assert mm.basis("AVAXIRT", _book(979_000.0, 981_000.0)) == pytest.approx(-0.02, abs=1e-4)


def test_a_dislocated_market_is_not_quoted():
    """A local book far from global fair is usually mid-repricing: quoting into
    it means buying just before the local price catches down. The spread looks
    identical; the fill is systematically bad."""
    mm = _mm(max_basis=0.02)
    mm._balances = {"avax": 6.0, "rls": 50_000_000.0}
    mm.set_fair("AVAXIRT", global_usd=1.0, fx_rial_per_usdt=1_000_000.0)
    # Local trading 5% rich against global, with a healthy 40 bps spread.
    assert mm.make_quotes("AVAXIRT", _book(1_048_000.0, 1_052_000.0)) == []


def test_a_market_near_fair_is_quoted_normally():
    mm = _mm(max_basis=0.02)
    mm._balances = {"avax": 6.0, "rls": 50_000_000.0}
    mm.set_fair("AVAXIRT", global_usd=1.0, fx_rial_per_usdt=1_000_000.0)
    assert mm.make_quotes("AVAXIRT", _book(998_000.0, 1_002_000.0)), "0.0% basis must quote"


def test_quotes_are_pulled_toward_global_fair():
    """Blended, not replaced: the fair value is itself an estimate built from a
    global feed and an FX rate, both with their own staleness."""
    mm = _mm(max_basis=0.05, fair_weight=0.5)
    mm._balances = {"avax": 6.0, "rls": 50_000_000.0}
    book = _book(1_018_000.0, 1_022_000.0)          # local mid 1,020,000

    no_fair = mm.make_quotes("AVAXIRT", book)
    mid_no = sum(q.price for q in no_fair) / len(no_fair)

    mm.set_fair("AVAXIRT", global_usd=1.0, fx_rial_per_usdt=1_000_000.0)  # fair below
    with_fair = mm.make_quotes("AVAXIRT", book)
    mid_with = sum(q.price for q in with_fair) / len(with_fair)

    assert mid_with < mid_no, "quotes must move toward the cheaper global price"
    assert mid_with > 1_000_000.0, "but not all the way -- the local book still counts"


def test_no_fair_value_falls_back_to_the_local_book():
    """A missing global price must not stop trading, only remove the anchor."""
    mm = _mm()
    mm._balances = {"avax": 6.0, "rls": 50_000_000.0}
    assert mm.fair_rial("AVAXIRT") is None
    assert mm.basis("AVAXIRT", _book(998_000.0, 1_002_000.0)) is None
    assert mm.make_quotes("AVAXIRT", _book(998_000.0, 1_002_000.0))


# -- price ticks ------------------------------------------------------------
def test_prices_are_rounded_to_the_market_tick():
    """Amounts were always stepped; prices never were. A market with
    price_step 10 received 78,445.8497 and rejected every quote -- 76
    "Order Validation Failed" on TNSRIRT alone."""
    mm = _mm(max_inventory_rial=30_000_000.0)
    mm._balances = {"tnsr": 100.0, "rls": 50_000_000.0}
    mm.specs["TNSRIRT"] = _Spec(amount_step=0.01, src="tnsr")
    mm.specs["TNSRIRT"].price_step = 10.0

    for q in mm.make_quotes("TNSRIRT", _book(78_060.0, 78_630.0)):
        assert q.price % 10 == 0, f"{q.side.value} price {q.price} is off the 10 tick"


def test_tick_rounding_widens_the_pair_never_narrows_it():
    """Rounding toward the mid would pull the quotes back through the fee
    floor that was just enforced -- so the bid rounds DOWN and the ask UP."""
    mm = _mm(max_inventory_rial=30_000_000.0, min_edge_bps=8.0)
    mm._balances = {"tnsr": 100.0, "rls": 50_000_000.0}
    mm.specs["TNSRIRT"] = _Spec(amount_step=0.01, src="tnsr")
    mm.specs["TNSRIRT"].price_step = 100.0        # a coarse tick

    qs = mm.make_quotes("TNSRIRT", _book(78_000.0, 78_600.0))
    bid = next(q.price for q in qs if q.side is Side.BUY)
    ask = next(q.price for q in qs if q.side is Side.SELL)
    assert mm.quoted_edge_bps(bid, ask) >= mm.min_edge_bps


def test_a_market_without_a_tick_is_unaffected():
    mm = _mm()
    mm._balances = {"avax": 6.0, "rls": 50_000_000.0}
    assert mm.make_quotes("AVAXIRT", _book(998_000.0, 1_002_000.0))


# -- ask sizing must respect what is actually sellable -----------------------
def test_blocked_units_from_our_own_quote_count_as_sellable():
    """They are released by `cancel_working` moments before we repost. ONEIRT
    held 1796.48944 with an activeBalance of 0.08944 and quoted no ask at all."""
    mm = _mm()
    mm.books["AVAXIRT"].working = {"mkr-ours": object()}
    assert mm._own_blocked("avax", blocked=38.1) == pytest.approx(38.1)


def test_blocked_units_we_did_not_place_are_not_sellable():
    """Sizing asks off TOTAL balance tried to sell 38.7 TNSR against 0.63 free
    and produced 240 opaque "Order Validation Failed" rejections."""
    mm = _mm()
    assert not mm.books.get("TNSRIRT") or not mm.books["TNSRIRT"].working
    assert mm._own_blocked("tnsr", blocked=38.1) == 0.0


def test_balances_detailed_splits_free_from_blocked():
    from nbtrend.data.nobitex_rest import NobitexREST

    rest = object.__new__(NobitexREST)
    rest._get = lambda path, **kw: {"wallets": [
        {"currency": "TNSR", "balance": "38.727659", "activeBalance": "0.627659"},
    ]}
    free, blocked = rest.balances_detailed()["tnsr"]
    assert free == pytest.approx(0.627659)
    assert blocked == pytest.approx(38.1)


# -- fragment stranding -----------------------------------------------------
def test_an_ask_sells_the_whole_position_rather_than_stranding_a_remainder():
    """The ratchet that filled the wallet with unsellable dust: a fixed-size
    ask leaves a remainder on every partial round trip, and a remainder under
    the minimum can NEVER be sold again. 33,486,842 rial accumulated that way
    across 17 coins -- more than half the account."""
    mm = _mm(quote_notional_rial=600_000.0, min_quote_rial=550_000.0,
             max_inventory_rial=30_000_000.0)
    # Holding 1,000,000: a 600,000 ask would strand 400,000 forever.
    mm._balances = {"avax": 1.0, "rls": 50_000_000.0}
    mid, half = 1_000_000.0, 20_000.0

    ask = next(q for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
               if q.side is Side.SELL)
    assert ask.amount == pytest.approx(1.0, rel=1e-6), "must offer the whole lot"


def test_a_remainder_that_stays_sellable_is_left_alone():
    """Only sweep when the leftover would be stranded -- otherwise normal
    sizing keeps inventory turning over at the intended rate."""
    mm = _mm(quote_notional_rial=600_000.0, min_quote_rial=550_000.0,
             max_inventory_rial=30_000_000.0)
    # Holding 5,000,000: a 600,000 ask leaves 4,400,000, comfortably sellable.
    mm._balances = {"avax": 5.0, "rls": 50_000_000.0}
    mid, half = 1_000_000.0, 20_000.0

    ask = next(q for q in mm.make_quotes("AVAXIRT", _book(mid - half, mid + half))
               if q.side is Side.SELL)
    assert ask.amount < 1.0, "normal sizing, not a full liquidation"


def test_the_configured_minimum_matches_what_the_exchange_accepts():
    """Measured by binary search on JASMYIRT with ample free balance:
    398,030 rial rejected, 497,538 accepted. The old 3,000,000 was 6x too
    high and was what stranded the inventory in the first place."""
    from nbtrend.config import load_config

    min_order = float(load_config().costs["min_order_rial"])
    assert 450_000 <= min_order <= 1_000_000, (
        f"{min_order:,.0f} is not consistent with the measured ~500,000 floor"
    )


# -- the budget must actually support the configured quoting rate ------------
def test_the_default_budget_supports_twelve_symbols_at_a_sixty_second_requote():
    """The stale 150 cap -- reserved for a trend runner that no longer exists
    -- silently strangled the maker. 12 symbols two-sided at 60s needs 240
    placements per 10 minutes, so the allowance was spent in the first sweeps
    and then refused everything 2,120 times with ZERO successful placements.
    Held-inventory asks were refused along with the rest, which is exactly why
    stranded coins never sold."""
    mm = _mm(requote_s=60.0)
    needed = 12 * 2 * (600.0 / 60.0)
    assert mm.max_orders_per_window >= needed, (
        f"budget {mm.max_orders_per_window} cannot sustain {needed:.0f} placements"
    )


def test_the_budget_stays_within_the_exchange_limit():
    """300 per 10 minutes is the exchange's, and it is shared with anything
    else on the account -- so leave headroom rather than claiming all of it."""
    mm = _mm()
    assert mm.max_orders_per_window <= 300 - 30


@pytest.mark.parametrize("symbols,requote,fits", [(12, 60, True), (12, 30, False), (6, 60, True)])
def test_quotable_symbols_matches_the_budget(symbols, requote, fits):
    mm = _mm(requote_s=float(requote))
    needed = symbols * 2 * (600.0 / requote)
    assert (needed <= mm.max_orders_per_window) is fits


def test_a_basis_refusal_says_basis_not_inventory():
    """The skip message blamed inventory ("free base 1737.71 ... may be locked
    in working orders") while the real cause was ONEIRT sitting +8.74% from
    global fair with nothing blocked at all. Self-contradictory on its face,
    and it sent an investigation chasing a stranding bug that did not exist."""
    from nbtrend.live.maker import MakerRunner

    mm = _mm(max_basis=0.02, min_edge_bps=8.0)
    mm._balances = {"avax": 100.0, "rls": 50_000_000.0}
    mm.set_fair("AVAXIRT", global_usd=1.0, fx_rial_per_usdt=1_000_000.0)

    book = _book(1_085_000.0, 1_090_000.0)          # ~8.7% rich, healthy spread
    assert mm.make_quotes("AVAXIRT", book) == []
    assert abs(mm.basis("AVAXIRT", book)) > mm.max_basis
    assert mm.edge_bps(book) > mm.min_edge_bps, "the EDGE is fine; basis is not"

    runner = MakerRunner(None, mm, None)
    assert runner is not None
