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
    def __init__(self, rising=True, rls=60_000_000.0, units=0.0):
        self.rising = rising
        self.rls = rls
        self.units = units
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
        return {"rls": (self.rls, 0.0), "btc": (self.units, 0.0)}

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
