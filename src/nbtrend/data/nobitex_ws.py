"""Nobitex websocket client (Centrifugo protocol v2, JSON).

Implemented directly rather than via `centrifuge-python` so the dependency
surface stays small and the framing is visible when it misbehaves.

Protocol notes that matter
--------------------------
* Commands are ``{"<method>": {...}, "id": n}``; replies echo the ``id``.
* Pushes arrive as ``{"push": {"channel": ..., "pub": {"data": "<json string>"}}}``.
  Note ``data`` is a JSON **string**, not an object -- it needs a second
  ``json.loads``. Missing that is the most common integration bug here.
* A frame may carry several newline-delimited JSON objects.
* The server sends an empty object ``{}`` as ping. The client must reply with
  an empty object ``{}`` within 25s or be disconnected. There is no other
  keepalive, so `websockets`' own ping is disabled.
* ``delta: "fossil"`` is deliberately NOT requested. It saves ~60% bandwidth
  but requires implementing Fossil delta decompression; the official SDKs hide
  that, a hand-rolled client would have to do it, and getting it subtly wrong
  corrupts the orderbook rather than failing loudly.

Units: orderbook, trades and market-stats channels publish RIAL. The candle
channel publishes TOMAN, like the REST OHLC endpoint -- converted on the way
out so callers only ever see rial.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from ..core.types import BookTop
from ..units import toman_to_rial

log = logging.getLogger(__name__)

Handler = Callable[[str, dict], Awaitable[None] | None]

PING_TIMEOUT_S = 25
MAX_CHANNELS_PER_CONNECTION = 450


# -- channel name builders --------------------------------------------------
def orderbook_channel(symbol: str) -> str:
    return f"public:orderbook-{symbol.upper()}"


def trades_channel(symbol: str) -> str:
    return f"public:trades-{symbol.upper()}"


def candle_channel(symbol: str, resolution: str) -> str:
    return f"public:candle-{symbol.upper()}-{resolution}"


def market_stats_channel(symbol: str | None = None) -> str:
    return f"public:market-stats-{symbol.upper()}" if symbol else "public:market-stats-all"


def private_orders_channel(auth_param: str) -> str:
    return f"private:orders#{auth_param}"


def private_trades_channel(auth_param: str) -> str:
    return f"private:trades#{auth_param}"


class NobitexWS:
    """Subscribe to Nobitex channels and dispatch decoded payloads.

    Usage::

        ws = NobitexWS(url)
        ws.on(orderbook_channel("BTCIRT"), handle_book)
        await ws.run()          # reconnects until cancelled
    """

    def __init__(
        self,
        url: str,
        token: str | None = None,
        reconnect_base_s: float = 1.0,
        reconnect_max_s: float = 60.0,
    ):
        self.url = url
        self.token = token
        self.reconnect_base_s = reconnect_base_s
        self.reconnect_max_s = reconnect_max_s

        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._next_id = 0
        self._ws: Any | None = None
        self._connected = asyncio.Event()
        self._last_message_ts: float = 0.0

    # -- registration ------------------------------------------------------
    def on(self, channel: str, handler: Handler) -> None:
        if len(self._handlers) >= MAX_CHANNELS_PER_CONNECTION and channel not in self._handlers:
            raise RuntimeError(
                f"Nobitex allows {MAX_CHANNELS_PER_CONNECTION} channels per connection; "
                "split the universe across several NobitexWS instances"
            )
        self._handlers[channel].append(handler)

    @property
    def channels(self) -> list[str]:
        return list(self._handlers)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def seconds_since_message(self) -> float:
        return time.time() - self._last_message_ts if self._last_message_ts else float("inf")

    # -- lifecycle ---------------------------------------------------------
    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Connect and stay connected, with exponential backoff on failure."""
        attempt = 0
        while not (stop and stop.is_set()):
            try:
                await self._session(stop)
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                delay = min(self.reconnect_base_s * 2 ** (attempt - 1), self.reconnect_max_s)
                log.warning(
                    "websocket session ended (%s: %s); reconnecting in %.1fs",
                    type(exc).__name__, exc, delay,
                )
                self._connected.clear()
                await asyncio.sleep(delay)

    async def _session(self, stop: asyncio.Event | None) -> None:
        async with websockets.connect(
            self.url,
            ping_interval=None,      # Centrifugo drives its own {} ping
            max_size=None,
            additional_headers={"User-Agent": "nbtrend/0.1"},
        ) as ws:
            self._ws = ws
            self._next_id = 0

            connect_params: dict[str, Any] = {}
            if self.token:
                connect_params["token"] = self.token
            await self._send({"connect": connect_params, "id": self._alloc_id()})

            for channel in self._handlers:
                await self._send({"subscribe": {"channel": channel}, "id": self._alloc_id()})

            log.info("subscribing to %d channel(s)", len(self._handlers))

            async for raw in ws:
                self._last_message_ts = time.time()
                await self._handle_frame(raw)
                if stop and stop.is_set():
                    break

        self._connected.clear()

    async def _send(self, command: dict) -> None:
        if self._ws is None:
            raise RuntimeError("not connected")
        await self._ws.send(json.dumps(command, separators=(",", ":")))

    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -- inbound -----------------------------------------------------------
    async def _handle_frame(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")

        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log.debug("undecodable frame: %.120s", line)
                continue
            await self._handle_message(message)

    async def _handle_message(self, message: dict) -> None:
        # Empty object = server ping. Echo it or get dropped after 25s.
        if not message:
            with contextlib.suppress(Exception):
                await self._send({})
            return

        if "push" in message:
            await self._handle_push(message["push"])
            return

        if "error" in message:
            log.error("centrifugo error: %s", message["error"])
            return

        if "connect" in message:
            self._connected.set()
            info = message["connect"]
            log.info("connected: client=%s ping=%ss", info.get("client"), info.get("ping"))
            return

        if "subscribe" in message:
            log.debug("subscribed (id=%s)", message.get("id"))

    async def _handle_push(self, push: dict) -> None:
        channel = push.get("channel", "")
        pub = push.get("pub") or {}
        payload = pub.get("data")
        if payload is None:
            return

        # `data` is a JSON string, not an object.
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                log.warning("undecodable payload on %s", channel)
                return

        payload = _normalise(channel, payload)

        for handler in self._handlers.get(channel, []):
            try:
                result = handler(channel, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("handler for %s raised", channel)


# -- payload normalisation --------------------------------------------------
def _normalise(channel: str, payload: dict) -> dict:
    """Coerce string numbers to floats and fix units, per channel family."""
    if channel.startswith("public:candle-"):
        # The only TOMAN-denominated channel. Match the REST OHLC quirk.
        return {
            "ts": int(payload["t"]),
            "open": toman_to_rial(float(payload["o"])),
            "high": toman_to_rial(float(payload["h"])),
            "low": toman_to_rial(float(payload["l"])),
            "close": toman_to_rial(float(payload["c"])),
            "volume": float(payload["v"]),
        }

    if channel.startswith("public:orderbook-"):
        return {
            "bids": [(float(p), float(a)) for p, a in payload.get("bids") or []],
            "asks": [(float(p), float(a)) for p, a in payload.get("asks") or []],
            "last_trade": float(payload.get("lastTradePrice") or 0),
            "ts_ms": int(payload.get("lastUpdate") or 0),
        }

    if channel.startswith("public:trades-"):
        return {
            "price": float(payload["price"]),
            "volume": float(payload["volume"]),
            "side": payload.get("type"),
            "ts_ms": int(payload["time"]),
        }

    if channel.startswith("public:market-stats-"):
        return _normalise_stats(payload)

    return payload


_STAT_FLOATS = (
    "bestSell", "bestBuy", "volumeSrc", "volumeDst", "latest", "mark",
    "dayLow", "dayHigh", "dayOpen", "dayClose", "dayChange",
)


def _normalise_stats(payload: dict) -> dict:
    # market-stats-all nests one dict per market under "btc-irt" style keys.
    if payload and all(isinstance(v, dict) for v in payload.values()):
        return {k: _normalise_stats(v) for k, v in payload.items()}

    out: dict[str, Any] = dict(payload)
    for key in _STAT_FLOATS:
        if payload.get(key) is not None:
            out[key] = float(payload[key])
    return out


def book_top_from_payload(symbol: str, payload: dict) -> BookTop | None:
    """Convert a normalised orderbook push into a `BookTop`."""
    bids, asks = payload.get("bids") or [], payload.get("asks") or []
    if not bids or not asks:
        return None
    return BookTop(
        symbol=symbol,
        best_bid=bids[0][0],
        best_ask=asks[0][0],
        last_trade=payload.get("last_trade") or bids[0][0],
        ts_ms=payload.get("ts_ms") or int(time.time() * 1000),
    )
