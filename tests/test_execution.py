"""Fill simulation and order-price construction."""

import pytest

from nbtrend.core.types import BookTop, Side
from nbtrend.execution.base import limit_price_through_touch
from nbtrend.execution.paper import _split_symbol, _walk_book


@pytest.fixture
def book():
    return BookTop(symbol="BTCIRT", best_bid=155_000_000_000,
                   best_ask=155_100_000_000, last_trade=155_050_000_000, ts_ms=0)


def test_walking_the_book_reports_real_slippage():
    """Size that eats three levels must fill at the VWAP of those levels."""
    levels = [(100.0, 1.0), (101.0, 2.0), (102.0, 5.0)]
    filled, avg = _walk_book(levels, 3.0, None, Side.BUY)
    assert filled == 3.0
    assert avg == pytest.approx((100 * 1 + 101 * 2) / 3)
    assert avg > 100.0   # not the touch


def test_limit_price_stops_the_walk():
    levels = [(100.0, 1.0), (101.0, 2.0)]
    filled, avg = _walk_book(levels, 3.0, 100.5, Side.BUY)
    assert filled == 1.0
    assert avg == 100.0


def test_partial_fill_when_the_book_is_thin():
    filled, _ = _walk_book([(100.0, 0.5)], 10.0, None, Side.BUY)
    assert filled == 0.5


def test_empty_book_fills_nothing():
    assert _walk_book([], 1.0, None, Side.BUY) == (0.0, 0.0)


def test_buy_limit_never_lands_below_the_ask(book):
    """Rounding down a buy price can put it under the ask and never fill."""
    price = limit_price_through_touch(book, Side.BUY, offset_bps=5, price_step=10)
    assert price >= book.best_ask


def test_sell_limit_never_lands_above_the_bid(book):
    price = limit_price_through_touch(book, Side.SELL, offset_bps=5, price_step=10)
    assert price <= book.best_bid


def test_limit_prices_respect_the_tick(book):
    for side in (Side.BUY, Side.SELL):
        price = limit_price_through_touch(book, side, 5, 10)
        assert price % 10 == 0


def test_symbol_splitting():
    assert _split_symbol("BTCIRT") == ("btc", "rls")
    assert _split_symbol("ETHUSDT") == ("eth", "usdt")
    with pytest.raises(ValueError):
        _split_symbol("BTCEUR")


def test_order_rate_guard_blocks_past_the_limit():
    from nbtrend.data.nobitex_rest import NobitexError
    from nbtrend.execution.nobitex import _OrderRateGuard

    guard = _OrderRateGuard(limit=3, window_s=600)
    for _ in range(3):
        guard.check()
    with pytest.raises(NobitexError, match="rate limit"):
        guard.check()


def test_no_printf_comma_formats_in_log_calls():
    """`%,.0f` is f-string syntax, not printf. logging uses %-formatting, so a
    comma-separated format there raises ValueError at emit time -- inside the
    trading loop, where it is worst."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "nbtrend"
    offenders = [
        f"{path.relative_to(root)}:{i}"
        for path in root.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if "%," in line
    ]
    assert not offenders, f"invalid printf comma format in: {offenders}"
