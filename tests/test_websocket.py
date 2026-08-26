"""Centrifugo payload handling, using the exact examples from Nobitex's docs."""

import json

from nbtrend.data.nobitex_ws import (
    _normalise,
    book_top_from_payload,
    candle_channel,
    market_stats_channel,
    orderbook_channel,
    private_trades_channel,
    trades_channel,
)
from nbtrend.units import RIAL_PER_TOMAN


def test_channel_names_match_the_documented_patterns():
    assert orderbook_channel("btcirt") == "public:orderbook-BTCIRT"
    assert trades_channel("BTCIRT") == "public:trades-BTCIRT"
    assert candle_channel("BTCIRT", "15") == "public:candle-BTCIRT-15"
    assert market_stats_channel("BTCIRT") == "public:market-stats-BTCIRT"
    assert market_stats_channel() == "public:market-stats-all"
    assert private_trades_channel("abc123") == "private:trades#abc123"


def test_orderbook_payload_from_the_docs():
    payload = json.loads(
        '{"asks": [["35077909990", "0.009433"], ["35078000000", "0.000274"]],'
        ' "bids": [["35020080080", "0.185784"], ["35020070060", "0.086916"]],'
        ' "lastTradePrice": "35077909990", "lastUpdate": 1726581829816}'
    )
    out = _normalise("public:orderbook-BTCIRT", payload)
    assert out["asks"][0] == (35077909990.0, 0.009433)
    assert out["bids"][0] == (35020080080.0, 0.185784)

    top = book_top_from_payload("BTCIRT", out)
    assert top.best_bid == 35020080080.0
    assert top.best_ask == 35077909990.0
    assert top.spread_bps > 0


def test_candle_channel_is_converted_from_toman():
    """The candle channel is the one public channel quoted in toman."""
    payload = {"t": 1731852900, "o": 6240000001.0, "h": 6250000000.0,
               "l": 6238000000.0, "c": 6238031033.0, "v": 1.26}
    out = _normalise("public:candle-BTCIRT-15", payload)
    assert out["close"] == 6238031033.0 * RIAL_PER_TOMAN
    assert out["ts"] == 1731852900


def test_orderbook_channel_is_not_converted():
    """...while the orderbook channel is already rial and must be left alone."""
    payload = {"asks": [["100", "1"]], "bids": [["99", "1"]],
               "lastTradePrice": "99", "lastUpdate": 1}
    out = _normalise("public:orderbook-BTCIRT", payload)
    assert out["asks"][0][0] == 100.0   # not 1000.0


def test_trades_payload():
    payload = {"price": "120000000000", "time": 1762781164192,
               "type": "sell", "volume": "0.000003"}
    out = _normalise("public:trades-BTCIRT", payload)
    assert out["price"] == 120000000000.0
    assert out["side"] == "sell"


def test_market_stats_single_and_all():
    single = {"isClosed": False, "bestSell": "121073861950", "bestBuy": "120000000000",
              "latest": "114879999920", "dayChange": "-5.12"}
    out = _normalise("public:market-stats-BTCIRT", single)
    assert out["bestSell"] == 121073861950.0
    assert out["dayChange"] == -5.12
    assert out["isClosed"] is False

    nested = _normalise("public:market-stats-all", {"btc-irt": single, "usdt-irt": single})
    assert nested["btc-irt"]["latest"] == 114879999920.0


def test_empty_book_returns_none():
    assert book_top_from_payload("BTCIRT", {"bids": [], "asks": []}) is None
