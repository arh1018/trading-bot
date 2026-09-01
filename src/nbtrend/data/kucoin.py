"""KuCoin public REST feed for global prices.

Same interface as `binance.BinanceFeed`. This exists because Binance and
TradingView are both unreachable from some networks -- from the Iranian VM this
project deploys to, Binance refuses the connection outright and TradingView and
most other venues return 403. KuCoin answers, and unlike MEXC (the other
reachable venue) it serves HISTORICAL ranges rather than only the most recent
500 bars, which the 400-bar signal warmup needs headroom above.

Two differences from Binance are easy to get wrong and are handled here:

  * Column order. KuCoin sends [time, open, CLOSE, HIGH, LOW, volume, turnover]
    -- close and high are transposed relative to Binance's
    [time, open, high, low, close]. Mapping positionally from the Binance
    layout yields a series where the "high" is sometimes below the "low", which
    corrupts ATR and every stop that depends on it without ever raising.
  * Ordering. Rows come back newest-first, and timestamps are SECONDS, not
    milliseconds.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pandas as pd

from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

BASE_URL = "https://api.kucoin.com"

# KuCoin caps a single response at 1500 candles.
_MAX_CANDLES = 1500

RESOLUTION_MAP: dict[str, str] = {
    "1": "1min", "5": "5min", "15": "15min", "30": "30min",
    "60": "1hour", "180": "None", "240": "4hour", "360": "6hour", "720": "12hour",
    "D": "1day", "2D": "1week", "3D": "1week",
}

_SECONDS_PER_BAR: dict[str, int] = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
    "1hour": 3600, "4hour": 14400, "6hour": 21600, "12hour": 43200,
    "1day": 86400, "1week": 604800,
}


class KuCoinFeed:
    def __init__(self, base_url: str = BASE_URL, timeout_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _to_kucoin_symbol(self, symbol: str) -> str:
        """`BINANCE:BTCUSDT` / `BTCUSDT` / `CRYPTO:HYPEUSD` -> `BTC-USDT`.

        KuCoin quotes USDT and hyphenates the pair. A trailing bare USD is
        promoted to USDT for the same reason as the Binance adapter.
        """
        pair = symbol.split(":", 1)[-1].upper().replace("-", "")
        for quote in ("USDT", "USDC", "BTC", "ETH"):
            if pair.endswith(quote) and len(pair) > len(quote):
                return f"{pair[: -len(quote)]}-{quote}"
        if pair.endswith("USD"):
            return f"{pair[:-3]}-USDT"
        return pair

    async def fetch_ohlcv(
        self, symbol: str, resolution: str, bars: int = 5000
    ) -> pd.DataFrame:
        interval = RESOLUTION_MAP.get(resolution)
        if interval is None or interval == "None":
            raise ValueError(f"resolution {resolution!r} has no KuCoin equivalent")

        pair = self._to_kucoin_symbol(symbol)
        step = _SECONDS_PER_BAR[interval]
        collected: list[list] = []
        # Seed the window at "now". Without an explicit startAt/endAt KuCoin
        # returns its default 100 candles regardless of what you asked for,
        # which is below the 400-bar signal warmup -- so the very first request
        # has to carry a window too, not just the paging ones.
        end_at: int = int(time.time())

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            while len(collected) < bars:
                want = min(_MAX_CANDLES, bars - len(collected))
                params: dict[str, object] = {
                    "type": interval,
                    "symbol": pair,
                    "endAt": end_at,
                    "startAt": max(0, end_at - want * step),
                }

                resp = await client.get("/api/v1/market/candles", params=params)
                resp.raise_for_status()
                body = resp.json()
                if body.get("code") != "200000":
                    raise RuntimeError(f"kucoin error for {pair}: {body.get('msg', body.get('code'))}")

                chunk = body.get("data") or []
                if not chunk:
                    break

                collected.extend(chunk)
                # Rows are newest-first, so the OLDEST row is the last one.
                oldest = int(chunk[-1][0])
                if oldest >= end_at:
                    break  # no progress; stop rather than spin
                end_at = oldest - 1
                if len(chunk) < want:
                    break
                await asyncio.sleep(0.2)

        return _to_frame(collected)


def _to_frame(candles: list[list]) -> pd.DataFrame:
    if not candles:
        return empty_ohlcv()

    # [time, open, close, high, low, volume, turnover] -- NOT the Binance order.
    df = pd.DataFrame(
        [
            [int(c[0]), float(c[1]), float(c[3]), float(c[4]), float(c[2]), float(c[5])]
            for c in candles
        ],
        columns=["open_s", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset="open_s").sort_values("open_s")
    df.index = pd.to_datetime(df["open_s"], unit="s", utc=True)
    return normalise_index(df)[["open", "high", "low", "close", "volume"]]


def fetch_ohlcv_sync(symbol: str, resolution: str, bars: int = 5000) -> pd.DataFrame:
    return asyncio.run(KuCoinFeed().fetch_ohlcv(symbol, resolution, bars))
