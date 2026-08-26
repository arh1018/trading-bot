"""TradingView datafeed (unofficial, unauthenticated).

TradingView has no public REST history API. What it does have is the
websocket its own charts speak, which serves history to anonymous sessions.
This module implements that protocol.

Be clear-eyed about what this is: a scraped, undocumented endpoint with no
stability contract. It works, it is free, and it can break without notice.
`nbtrend.data.binance` is a drop-in fallback on the same interface -- switch
with `data.global_feed: binance` in config.yaml. For anything with money on
it, run the fallback as a cross-check rather than trusting one scraper.

Protocol sketch
---------------
Messages are length-prefixed frames: ``~m~<byte_len>~m~<payload>``. Payload is
either JSON (``{"m": method, "p": [args]}``) or the heartbeat ``~h~<n>``,
which must be echoed back verbatim or the server drops the socket.

Handshake:
    set_auth_token   -> "unauthorized_user_token"
    chart_create_session
    resolve_symbol   -> {"symbol": "BINANCE:BTCUSDT", "adjustment": "splits"}
    create_series    -> (session, series_id, series_id, symbol_id, res, bars)
History then arrives as ``timescale_update`` frames and ends with
``series_completed``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
from typing import Any

import pandas as pd
import websockets

from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"~m~(\d+)~m~")
_HEARTBEAT_RE = re.compile(r"^~h~\d+$")

# nbtrend resolution -> TradingView resolution. Identical for everything we
# use, but kept explicit so a divergence is a config error, not a silent
# request for the wrong bar size.
RESOLUTION_MAP: dict[str, str] = {
    "1": "1", "5": "5", "15": "15", "30": "30",
    "60": "60", "180": "180", "240": "240", "360": "360", "720": "720",
    "D": "1D", "2D": "2D", "3D": "3D",
}


class TradingViewError(RuntimeError):
    pass


def _random_id(prefix: str, n: int = 12) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _encode(payload: str) -> str:
    return f"~m~{len(payload)}~m~{payload}"


def _encode_message(method: str, params: list[Any]) -> str:
    return _encode(json.dumps({"m": method, "p": params}, separators=(",", ":")))


def _decode(raw: str) -> list[str]:
    """Split a websocket text frame into its length-prefixed payloads."""
    out: list[str] = []
    pos = 0
    while pos < len(raw):
        match = _FRAME_RE.match(raw, pos)
        if not match:
            break
        length = int(match.group(1))
        start = match.end()
        out.append(raw[start : start + length])
        pos = start + length
    return out


class TradingViewFeed:
    """Fetches OHLCV history for a `EXCHANGE:SYMBOL` pair."""

    def __init__(
        self,
        ws_url: str = "wss://data.tradingview.com/socket.io/websocket?from=chart%2F",
        origin: str = "https://www.tradingview.com",
        auth_token: str = "unauthorized_user_token",
        timeout_s: float = 30.0,
    ):
        self.ws_url = ws_url
        self.origin = origin
        self.auth_token = auth_token
        self.timeout_s = timeout_s

    async def fetch_ohlcv(
        self, symbol: str, resolution: str, bars: int = 5000
    ) -> pd.DataFrame:
        """Return up to `bars` of history, ascending, indexed by UTC time.

        Prices are in the quote currency of `symbol` (USD/USDT), NOT rial.
        """
        tv_resolution = RESOLUTION_MAP.get(resolution)
        if tv_resolution is None:
            raise TradingViewError(f"resolution {resolution!r} has no TradingView equivalent")

        try:
            return await asyncio.wait_for(
                self._fetch(symbol, tv_resolution, bars), timeout=self.timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise TradingViewError(
                f"timed out after {self.timeout_s}s fetching {symbol}; "
                "the TradingView datafeed is unofficial -- consider data.global_feed: binance"
            ) from exc

    async def _fetch(self, symbol: str, resolution: str, bars: int) -> pd.DataFrame:
        chart_session = _random_id("cs_")
        series_id = "s1"
        symbol_id = "sym1"

        async with websockets.connect(
            self.ws_url,
            origin=self.origin,
            additional_headers={"User-Agent": "Mozilla/5.0 nbtrend/0.1"},
            max_size=None,
            ping_interval=None,   # TradingView drives its own ~h~ heartbeat
        ) as ws:
            for message in (
                _encode_message("set_auth_token", [self.auth_token]),
                _encode_message("chart_create_session", [chart_session, ""]),
                _encode_message(
                    "resolve_symbol",
                    [
                        chart_session,
                        symbol_id,
                        "=" + json.dumps({"symbol": symbol, "adjustment": "splits"}),
                    ],
                ),
                _encode_message(
                    "create_series",
                    [chart_session, series_id, series_id, symbol_id, resolution, bars, ""],
                ),
            ):
                await ws.send(message)

            rows: dict[int, list[float]] = {}
            async for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")

                for payload in _decode(raw):
                    if _HEARTBEAT_RE.match(payload):
                        await ws.send(_encode(payload))   # echo or be disconnected
                        continue

                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    method = msg.get("m")
                    if method == "timescale_update":
                        rows.update(_extract_bars(msg, series_id))
                    elif method == "series_completed":
                        return _to_frame(rows)
                    elif method in ("symbol_error", "series_error", "critical_error"):
                        raise TradingViewError(f"{method} for {symbol}: {msg.get('p')}")

            return _to_frame(rows)


def _extract_bars(msg: dict, series_id: str) -> dict[int, list[float]]:
    """Pull `{ts: [o,h,l,c,v]}` out of a timescale_update frame."""
    out: dict[int, list[float]] = {}
    params = msg.get("p") or []
    if len(params) < 2 or not isinstance(params[1], dict):
        return out

    series = params[1].get(series_id) or {}
    for bar in series.get("s") or []:
        values = bar.get("v") or []
        if len(values) < 5:
            continue
        ts = int(values[0])
        volume = float(values[5]) if len(values) > 5 and values[5] is not None else 0.0
        out[ts] = [float(values[1]), float(values[2]), float(values[3]), float(values[4]), volume]
    return out


def _to_frame(rows: dict[int, list[float]]) -> pd.DataFrame:
    if not rows:
        return empty_ohlcv()

    ordered = sorted(rows.items())
    df = pd.DataFrame(
        [values for _, values in ordered],
        columns=["open", "high", "low", "close", "volume"],
        index=pd.to_datetime([ts for ts, _ in ordered], unit="s", utc=True),
    )
    return normalise_index(df)


def fetch_ohlcv_sync(
    symbol: str, resolution: str, bars: int = 5000, **kwargs: Any
) -> pd.DataFrame:
    return asyncio.run(TradingViewFeed(**kwargs).fetch_ohlcv(symbol, resolution, bars))
