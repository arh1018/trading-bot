"""The slow long book: signal, sizing, and the ways it could lose money."""

from __future__ import annotations

import pandas as pd
import pytest

from nbtrend.live.drift import DriftBook, DriftRunner


class _Spec:
    def __init__(self, src="btc", dst="rls", amount_step=1e-8):
        self.src = src
        self.dst = dst
        self.amount_step = amount_step
        self.price_step = 10.0


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def _book(rising: bool, n=200, window=100):
    """A book whose basket is above or below its own mean."""
    if rising:
        coin = [100.0 + i for i in range(n)]
    else:
        coin = [100.0 + n] * (n // 2) + [100.0 + n - i * 2 for i in range(n - n // 2)]
    b = DriftBook(symbols=["BTCIRT"], trend_window=window)
    b.prices = {"BTCIRT": _series(coin), "USDTIRT": _series([1.0] * n)}
    return b


def test_a_rising_basket_is_risk_on():
    assert _book(rising=True).risk_on() is True


def test_a_falling_basket_is_risk_off():
    assert _book(rising=False).risk_on() is False


def test_an_unknown_signal_is_not_a_sell_signal():
    """None must never be read as False.

    A data outage that reads as "risk off" liquidates the whole book at market,
    paying taker on every leg, for no reason at all. The distinction between
    "the trend is down" and "I cannot see the trend" has to survive.
    """
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    assert book.risk_on() is None                     # no prices at all

    book.prices = {"BTCIRT": _series([100.0] * 20), "USDTIRT": _series([1.0] * 20)}
    assert book.risk_on() is None, "20 days cannot fill a 100 day window"

    # And nothing is traded on an absent signal.
    assert book.targets(equity_rial=60_000_000.0, held={}) == []
    assert book.orders(equity_rial=60_000_000.0, held={}) == []


def test_risk_off_targets_zero_in_every_market():
    book = _book(rising=False)
    targets = book.targets(equity_rial=60_000_000.0, held={"BTCIRT": 30_000_000.0})
    assert [t.target_rial for t in targets] == [0.0]
    assert targets[0].delta_rial == -30_000_000.0


def test_risk_on_spreads_equally_across_the_book():
    book = DriftBook(symbols=["BTCIRT", "ETHIRT"], trend_window=100)
    rising = [100.0 + i for i in range(200)]
    book.prices = {"BTCIRT": _series(rising), "ETHIRT": _series(rising),
                   "USDTIRT": _series([1.0] * 200)}
    targets = book.targets(equity_rial=60_000_000.0, held={})
    assert [t.target_rial for t in targets] == [30_000_000.0, 30_000_000.0]


def test_a_small_drift_does_not_trigger_a_trade():
    """The band is what keeps a slow book slow.

    Without it, every tick is a small rebalance and the costs that sank the
    market maker return through the side door: one taker leg plus half a
    spread is ~41 bps, so churning a position for a 1% drift loses money.
    """
    book = _book(rising=True)
    book.min_trade_rial = 2_000_000.0
    # Target 60,000,000; holding 59,000,000. A 1,000,000 delta is under the
    # minimum trade AND inside the band.
    assert book.orders(60_000_000.0, {"BTCIRT": 59_000_000.0}) == []
    # A 40% gap is worth closing.
    assert len(book.orders(60_000_000.0, {"BTCIRT": 36_000_000.0})) == 1


def test_an_exit_is_never_blocked_by_the_band():
    """Trimming a keeper is optional; leaving a market is not."""
    book = _book(rising=False)
    orders = book.orders(60_000_000.0, {"BTCIRT": 3_000_000.0})
    assert len(orders) == 1
    assert orders[0].target_rial == 0.0


def test_orders_below_the_exchange_minimum_are_skipped():
    """A 550,000 rial floor, measured against the live book."""
    book = _book(rising=True)
    book.min_trade_rial = 100_000.0
    assert book.orders(1_000_000.0, {"BTCIRT": 700_000.0}) == []


def test_sells_are_placed_before_buys():
    """Their proceeds are what funds the buys.

    Buying first spends rial the account may not have until the sells settle,
    and the rejections land on the leg that most needed to happen.
    """
    book = DriftBook(symbols=["AIRT", "BIRT"], trend_window=100)
    rising = [100.0 + i for i in range(200)]
    book.prices = {"AIRT": _series(rising), "BIRT": _series(rising),
                   "USDTIRT": _series([1.0] * 200)}
    orders = book.orders(60_000_000.0, {"AIRT": 0.0, "BIRT": 60_000_000.0})
    assert orders[0].delta_rial < 0 < orders[-1].delta_rial


def test_the_signal_is_measured_in_usdt_not_rial():
    """In rial everything trends up together, because the rial falls.

    A rial-denominated trend signal is mostly a devaluation detector: it would
    have stayed long through a crypto bear market simply because the currency
    was collapsing faster than the asset.
    """
    n = 200
    # Crypto flat in dollars; rial halving against USDT. In rial the price
    # doubles and looks like a roaring uptrend. In USDT it is a flat line.
    usdt = _series([1.0 + i * 0.01 for i in range(n)])
    coin_rial = _series([100.0 * (1.0 + i * 0.01) for i in range(n)])
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    book.prices = {"BTCIRT": coin_rial, "USDTIRT": usdt}

    basket = book.basket()
    assert basket is not None
    assert basket.max() - basket.min() < 1e-6, (
        "the basket must be flat: this asset did nothing in dollar terms"
    )


class _REST:
    def __init__(self, rising=True, rls=60_000_000.0, units=0.0, extra=None):
        self.rising = rising
        self.rls = rls
        self.units = units
        self.extra = extra or {}
        self.orders: list[dict] = []

    def candles(self, symbol, resolution, start, end):
        n = 200
        vals = ([100.0 + i for i in range(n)] if self.rising
                else [200.0 - i for i in range(n)])
        if symbol == "USDTIRT":
            vals = [1.0] * n
        return pd.DataFrame({"close": vals},
                            index=pd.date_range("2024-01-01", periods=n, freq="D"))

    def balances_detailed(self):
        out = {"rls": (self.rls, 0.0), "btc": (self.units, 0.0)}
        out.update(self.extra)
        return out

    def orderbook(self, symbol):
        from nbtrend.core.types import BookTop
        return BookTop(symbol=symbol, best_bid=1_000_000.0, best_ask=1_001_000.0,
                       last_trade=1_000_500.0, ts_ms=0)

    def add_order(self, **kw):
        self.orders.append(kw)
        return object()


def test_a_dry_run_places_nothing():
    rest = _REST(rising=True)
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    runner = DriftRunner(book, rest, dry_run=True)
    runner.specs = {"BTCIRT": _Spec()}
    assert runner.rebalance() == 1
    assert rest.orders == [], "dry run must not touch the exchange"


def test_a_live_run_buys_into_a_rising_basket():
    rest = _REST(rising=True)
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    runner = DriftRunner(book, rest, dry_run=False)
    runner.specs = {"BTCIRT": _Spec()}
    assert runner.rebalance() == 1
    assert len(rest.orders) == 1
    from nbtrend.core.types import Side
    assert rest.orders[0]["side"] is Side.BUY


def test_a_protected_market_is_never_traded():
    rest = _REST(rising=True)
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    runner = DriftRunner(book, rest, dry_run=False)
    runner.specs = {"BTCIRT": _Spec()}
    runner.protected = {"BTCIRT"}
    runner.rebalance()
    assert rest.orders == []


@pytest.mark.parametrize("window", [20, 50, 100, 200])
def test_the_window_is_a_parameter_not_a_constant(window):
    """80d returned +628% and 120d +363% over the same 883 days.

    The window is a real choice, not a detail, so it must stay configurable
    rather than hardening into a magic number somewhere in the code.
    """
    book = DriftBook(symbols=["BTCIRT"], trend_window=window)
    assert book.trend_window == window


class _StatsREST:
    """Volumes and spreads for the liquidity screen."""

    def __init__(self, rows):
        # rows: {src: (volume, bid, ask)}
        self.rows = rows

    def market_stats(self, src, dst="rls"):
        row = self.rows.get(src)
        if row is None:
            return {}
        vol, bid, ask = row
        return {f"{src}-rls": {"volumeDst": str(vol),
                               "bestBuy": str(bid), "bestSell": str(ask)}}


def test_the_book_is_the_head_of_the_liquidity_ranking():
    """Liquidity rank is the signal, not breadth.

    Measured over 504 days: ranks 1-5 returned +930.9 percent, ranks 6-15
    +36.7, ranks 16-30 +78.6. Random five-name baskets returned a median of
    +72.5, no better than random twenty -- so concentration only pays when it
    concentrates on the ranking.
    """
    from nbtrend.live.drift import rank_by_liquidity

    specs = {f"{n}IRT": _Spec(src=n.lower()) for n in ("AAA", "BBB", "CCC", "DDD")}
    rest = _StatsREST({
        "aaa": (900e9, 100.0, 101.0),
        "bbb": (100e9, 100.0, 101.0),
        "ccc": (500e9, 100.0, 101.0),
        "ddd": (700e9, 100.0, 101.0),
    })
    assert rank_by_liquidity(rest, specs, top_n=3) == ["AAAIRT", "DDDIRT", "CCCIRT"]


def test_an_illiquid_or_wide_market_never_enters_the_book():
    """A position we cannot exit is not an investment."""
    from nbtrend.live.drift import rank_by_liquidity

    specs = {f"{n}IRT": _Spec(src=n.lower()) for n in ("THIN", "WIDE", "GOOD")}
    rest = _StatsREST({
        "thin": (1e9, 100.0, 101.0),        # under the volume floor
        "wide": (900e9, 100.0, 140.0),      # 3,400 bps spread
        "good": (500e9, 100.0, 101.0),
    })
    assert rank_by_liquidity(rest, specs, top_n=10) == ["GOODIRT"]


def test_a_market_with_no_stats_is_skipped_not_guessed():
    """Missing data must not put a name in the book by default."""
    from nbtrend.live.drift import rank_by_liquidity

    specs = {"AAAIRT": _Spec(src="aaa"), "BBBIRT": _Spec(src="bbb")}
    rest = _StatsREST({"aaa": (900e9, 100.0, 101.0)})     # bbb absent
    assert rank_by_liquidity(rest, specs, top_n=10) == ["AAAIRT"]


def test_usdt_is_the_benchmark_and_never_a_holding():
    """USDT is the most traded rial market, so volume ranking selects it.

    Live, the first auto-selected book was "USDTIRT, ZECIRT, BTCIRT, ..." --
    it would have bought the benchmark as a position. In USDT terms that name
    is a permanently flat line, so it corrupts the trend signal AND spends the
    account on the thing the strategy exists to beat.
    """
    from nbtrend.live.drift import rank_by_liquidity

    specs = {"USDTIRT": _Spec(src="usdt"), "USDCIRT": _Spec(src="usdc"),
             "BTCIRT": _Spec(src="btc")}
    rest = _StatsREST({
        "usdt": (9_000e9, 100.0, 101.0),     # by far the most traded
        "usdc": (8_000e9, 100.0, 101.0),
        "btc": (500e9, 100.0, 101.0),
    })
    assert rank_by_liquidity(rest, specs, top_n=5) == ["BTCIRT"]


def test_a_buy_is_bounded_by_cash_actually_available():
    """Targets are computed against equity; equity is mostly coins.

    Live, ZEC was asked to buy 3,154,700 rial of stock against 408,425 rial of
    free cash and the exchange rejected it -- twice, on consecutive passes.
    Sells go first so their proceeds fund the buys, but settlement is not
    instant, so the buy has to be sized against the cash that exists now.
    """
    from nbtrend.core.types import Side

    rest = _REST(rising=True, rls=1_000_000.0)
    book = DriftBook(symbols=["BTCIRT"], trend_window=100)
    book.min_trade_rial = 500_000.0
    runner = DriftRunner(book, rest, dry_run=False)
    runner.specs = {"BTCIRT": _Spec()}

    runner.rebalance()
    assert len(rest.orders) == 1
    order = rest.orders[0]
    assert order["side"] is Side.BUY
    # ask is 1,001,000 in the fake book; the order must fit inside 1,000,000.
    assert order["amount"] * 1_001_000.0 <= 1_000_000.0 + 1.0, (
        f"bought {order['amount'] * 1_001_000.0:,.0f} rial against 1,000,000 free"
    )


def test_buys_across_a_whole_pass_cannot_exceed_the_cash_balance():
    """The pass as a whole is bounded, not just each order in isolation.

    The market maker had the same defect at sweep level: twelve symbols read
    one stale snapshot, each concluded there was room, and 48,981,598 rial
    fell to 7,118,000 straight through an 8,000,000 floor. Here the running
    balance is decremented as each buy is placed.
    """
    from nbtrend.core.types import Side

    # Equity is 10,000,000 -- mostly coins -- but only 1,600,000 is cash, so
    # each 5,000,000 target far exceeds what can actually be spent.
    # Equity ~10,000,000, of which only 1,600,000 is cash. Both markets are
    # far under their 5,000,000 target, so both want more than the cash allows.
    rest = _REST(rising=True, rls=1_600_000.0,
                 extra={"a": (4.2, 0.0), "b": (0.0, 0.0)})
    book = DriftBook(symbols=["AIRT", "BIRT"], trend_window=100)
    book.min_trade_rial = 500_000.0
    runner = DriftRunner(book, rest, dry_run=False)
    runner.specs = {"AIRT": _Spec(src="a"), "BIRT": _Spec(src="b")}

    runner.rebalance()
    assert len(rest.orders) >= 1, "at least one buy should have been attempted"
    spent = sum(o["amount"] * 1_001_000.0 for o in rest.orders
                if o["side"] is Side.BUY)
    assert spent <= 1_600_000.0 + 1.0, (
        f"spent {spent:,.0f} rial against 1,600,000 free"
    )
