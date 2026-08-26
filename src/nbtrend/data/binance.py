"""Binance public REST fallback for global prices.

Same interface as `tradingview.TradingViewFeed`. Binance publishes a
documented, versioned, keyless klines endpoint, so this is the more reliable
of the two -- TradingView is the default only because it aggregates many
venues and is what the config asks for. Run this one as a cross-check.

Note the venue difference: `BINANCE:BTCUSDT` on TradingView and `BTCUSDT`
here are the same book, so the two feeds should agree to the tick. If they
diverge, the scraper has drifted.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pandas as pd

from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
_MAX_LIMIT = 1000

RESOLUTION_MAP: dict[str, str] = {
    "1": "1m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "180": "3h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "2D": "3d", "3D": "3d",
}


class BinanceFeed:
    def __init__(self, base_url: str = BASE_URL, timeout_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _to_binance_symbol(self, symbol: str) -> str:
        """Normalise a TradingView symbol to a Binance pair.

        Accepts `BINANCE:BTCUSDT`, a bare `BTCUSDT`, or the aggregate-index
        form `CRYPTO:HYPEUSD` that `build_universe.py` emits for tokens Binance
        does not list. Binance quotes in USDT, never USD, so a trailing USD is
        promoted -- otherwise the fallback requests `HYPEUSD` and gets a 400.
        """
        pair = symbol.split(":", 1)[-1].upper()
        if pair.endswith("USD") and not pair.endswith("BUSD"):
            pair = pair + "T"
        return pair

    async def fetch_ohlcv(
        self, symbol: str, resolution: str, bars: int = 5000
    ) -> pd.DataFrame:
        interval = RESOLUTION_MAP.get(resolution)
        if interval is None:
            raise ValueError(f"resolution {resolution!r} has no Binance equivalent")

        pair = self._to_binance_symbol(symbol)
        collected: list[list] = []
        end_time: int | None = None

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as client:
            while len(collected) < bars:
                params = {
                    "symbol": pair,
                    "interval": interval,
                    "limit": min(_MAX_LIMIT, bars - len(collected)),
                }
                if end_time is not None:
                    params["endTime"] = end_time

                resp = await client.get("/api/v3/klines", params=params)
                resp.raise_for_status()
                chunk = resp.json()
                if not chunk:
                    break

                collected = chunk + collected
                end_time = int(chunk[0][0]) - 1
                if len(chunk) < params["limit"]:
                    break
                await asyncio.sleep(0.15)   # stay well inside the weight limit

        return _to_frame(collected)


def _to_frame(klines: list[list]) -> pd.DataFrame:
    if not klines:
        return empty_ohlcv()

    df = pd.DataFrame(
        [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines],
        columns=["open_ms", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset="open_ms").sort_values("open_ms")
    df.index = pd.to_datetime(df["open_ms"], unit="ms", utc=True)
    return normalise_index(df)[["open", "high", "low", "close", "volume"]]


def fetch_ohlcv_sync(symbol: str, resolution: str, bars: int = 5000) -> pd.DataFrame:
    return asyncio.run(BinanceFeed().fetch_ohlcv(symbol, resolution, bars))
