"""KuCoin global feed.

Exists because Binance and TradingView are both unreachable from the Iranian
VM this deploys to. The dangerous part is not connectivity but the payload
shape: KuCoin transposes close and high relative to Binance, so a positional
copy of the Binance parser produces candles whose high sits below their low.
Nothing raises -- ATR and every stop built on it just quietly go wrong.
"""

from __future__ import annotations

import pytest

from nbtrend.data.kucoin import KuCoinFeed, _to_frame

# Real rows from api.kucoin.com, newest first:
# [time, open, close, high, low, volume, turnover]
SAMPLE = [
    ["1788177600", "78305.8", "77886.3", "78532", "77703.4", "386.76", "30193438.8"],
    ["1788163200", "78199.2", "78305.9", "78796.1", "77989.1", "358.30", "28098897.9"],
    ["1788148800", "77756.7", "78200.0", "78308.0", "77475.0", "288.73", "20045226.7"],
]


def test_ohlc_columns_are_not_read_in_binance_order():
    """The transposition trap. KuCoin sends open, CLOSE, HIGH, LOW."""
    df = _to_frame(SAMPLE)
    first = df.iloc[-1]  # newest bar, after ascending sort
    assert first["open"] == pytest.approx(78305.8)
    assert first["close"] == pytest.approx(77886.3)
    assert first["high"] == pytest.approx(78532.0)
    assert first["low"] == pytest.approx(77703.4)


def test_every_bar_is_internally_consistent():
    """high >= max(open, close) and low <= min(open, close), always.

    This is the assertion that a positional mis-map fails, and it is worth
    stating directly: a high below the low silently poisons ATR, which sets
    the chandelier stop distance on every live position.
    """
    df = _to_frame(SAMPLE)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df["low"]).all()


def test_rows_are_sorted_oldest_first():
    """KuCoin returns newest-first; the rest of the stack assumes ascending."""
    df = _to_frame(SAMPLE)
    assert df.index.is_monotonic_increasing


def test_timestamps_are_seconds_not_milliseconds():
    """A ms/s mix-up puts the bars in 1970 or the year 58000."""
    df = _to_frame(SAMPLE)
    assert df.index[0].year == 2026


def test_empty_payload_returns_an_empty_frame():
    df = _to_frame([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_duplicate_timestamps_collapse():
    df = _to_frame([*SAMPLE, SAMPLE[0]])
    assert len(df) == 3


@pytest.mark.parametrize(
    "given,expected",
    [
        ("BINANCE:BTCUSDT", "BTC-USDT"),
        ("BTCUSDT", "BTC-USDT"),
        ("SOLUSDT", "SOL-USDT"),
        ("CRYPTO:HYPEUSD", "HYPE-USDT"),   # bare USD is promoted, as on Binance
        ("BTC-USDT", "BTC-USDT"),
        ("ETHUSDC", "ETH-USDC"),
    ],
)
def test_symbol_mapping(given, expected):
    assert KuCoinFeed()._to_kucoin_symbol(given) == expected


def test_unsupported_resolution_raises_rather_than_guessing():
    """180 (3h) has no KuCoin equivalent. Silently substituting 4h would make
    every signal subtly wrong on a timeframe nobody asked for."""
    import asyncio

    with pytest.raises(ValueError, match="no KuCoin equivalent"):
        asyncio.run(KuCoinFeed().fetch_ohlcv("BTCUSDT", "180", bars=10))
