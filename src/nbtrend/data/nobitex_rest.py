"""Nobitex REST adapter.

Unit handling is the whole reason this class exists rather than raw httpx
calls: `/market/udf/history` speaks toman and everything else speaks rial.
Conversion happens here, once, and every value that leaves this module is
rial. See `nbtrend.units` for the evidence.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.types import BookTop, Order, OrderStatus, OrderType, Side
from ..units import toman_to_rial
from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

# `/market/udf/history` caps a single response; page through longer ranges.
_MAX_BARS_PER_CALL = 500


class NobitexError(RuntimeError):
    def __init__(self, message: str, code: str | None = None, payload: dict | None = None):
        super().__init__(message)
        self.code = code
        self.payload = payload or {}


class RateLimited(NobitexError):
    def __init__(self, message: str, back_off: float):
        super().__init__(message, code="TooManyRequests")
        self.back_off = back_off


class NobitexREST:
    """Thin, typed client. Public endpoints need no token."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 20.0,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        """Supports both Nobitex credential types.

        `token` is a login token (``Authorization: Token <hex>``).
        `api_key` + `api_secret` are an API key pair, which is signed per
        request with Ed25519 -- see `nobitex_auth`. Passing an API key's
        public half as `token` returns 401, because API keys are not bearer
        tokens.
        """
        self.base_url = base_url.rstrip("/")
        self._token = token
        # Nobitex asks bots to identify themselves as TraderBot/<name>.
        headers = {"User-Agent": "TraderBot/nbtrend", "Content-Type": "application/json"}

        auth = None
        if api_key and api_secret:
            from .nobitex_auth import NobitexAPIKeyAuth

            auth = NobitexAPIKeyAuth(api_key, api_secret)
        elif token:
            headers["Authorization"] = f"Token {token}"

        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, auth=auth
        )
        self._authenticated = bool(auth or token)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NobitexREST:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- plumbing ----------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get("status") == "failed":
            code = data.get("code", "")
            if code == "TooManyRequests":
                raise RateLimited(data.get("message", "rate limited"), float(data.get("backOff", 10)))
            raise NobitexError(data.get("message", "request failed"), code, data)
        return data

    def _get(self, path: str, **params: Any) -> dict:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, json=payload)

    # -- market data (public) ----------------------------------------------
    def orderbook(self, symbol: str) -> BookTop:
        """Top of book, RIAL. `bids` are descending, `asks` ascending."""
        data = self._get(f"/v3/orderbook/{symbol}")
        bids, asks = data.get("bids") or [], data.get("asks") or []
        if not bids or not asks:
            raise NobitexError(f"empty orderbook for {symbol}")
        return BookTop(
            symbol=symbol,
            best_bid=float(bids[0][0]),
            best_ask=float(asks[0][0]),
            last_trade=float(data.get("lastTradePrice", bids[0][0])),
            ts_ms=int(data.get("lastUpdate", time.time() * 1000)),
        )

    def full_orderbook(self, symbol: str) -> dict[str, list[tuple[float, float]]]:
        data = self._get(f"/v3/orderbook/{symbol}")
        return {
            side: [(float(p), float(a)) for p, a in data.get(side) or []]
            for side in ("bids", "asks")
        }

    def market_stats(self, src: str, dst: str = "rls") -> dict:
        """24h stats, RIAL."""
        data = self._get("/market/stats", srcCurrency=src, dstCurrency=dst)
        return data.get("stats", {})

    def candles(
        self, symbol: str, resolution: str, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """OHLCV for a Nobitex market, converted TOMAN -> RIAL.

        Returns a DataFrame indexed by UTC timestamp with open/high/low/close/
        volume columns, ascending, de-duplicated across pages.
        """
        frames: list[pd.DataFrame] = []
        cursor = end_ts

        while cursor > start_ts:
            data = self._get(
                "/market/udf/history",
                symbol=symbol,
                resolution=resolution,
                to=cursor,
                countback=_MAX_BARS_PER_CALL,
            )
            if data.get("s") == "no_data" or not data.get("t"):
                break
            if data.get("s") != "ok":
                raise NobitexError(data.get("errmsg", "udf/history failed"), payload=data)

            chunk = pd.DataFrame(
                {
                    "ts": data["t"],
                    # toman -> rial happens HERE and nowhere else
                    "open": [toman_to_rial(v) for v in data["o"]],
                    "high": [toman_to_rial(v) for v in data["h"]],
                    "low": [toman_to_rial(v) for v in data["l"]],
                    "close": [toman_to_rial(v) for v in data["c"]],
                    "volume": data["v"],
                }
            )
            frames.append(chunk)

            oldest = int(chunk["ts"].min())
            if oldest >= cursor:  # no progress, avoid an infinite loop
                break
            cursor = oldest - 1

        if not frames:
            return _empty_ohlcv()

        df = pd.concat(frames, ignore_index=True)
        df = df[df["ts"] >= start_ts]
        return _finalise_ohlcv(df)

    # -- account (needs a token) -------------------------------------------
    def profile(self) -> dict:
        return self._get("/users/profile").get("profile", {})

    def ws_auth_param(self) -> str:
        """The constant per-user suffix for `private:*` channel names."""
        prof = self.profile()
        param = prof.get("websocketAuthParam")
        if not param:
            raise NobitexError("profile has no websocketAuthParam")
        return str(param)

    def ws_token(self) -> str:
        """Short-lived JWT for subscribing to private channels."""
        return str(self._get("/auth/ws/token/")["token"])

    def wallets(self, currencies: list[str] | None = None) -> dict[str, float]:
        """Free balance per currency. Rial balances are in RIAL."""
        if currencies:
            data = self._get("/v2/wallets", currencies=",".join(currencies))
            wallets = data.get("wallets", {})
            return {k.lower(): float(v.get("activeBalance", 0)) for k, v in wallets.items()}
        data = self._get("/users/wallets/list")
        return {
            w["currency"].lower(): float(w.get("activeBalance", 0))
            for w in data.get("wallets", [])
        }

    # -- trading -----------------------------------------------------------
    def add_order(
        self,
        src: str,
        dst: str,
        side: Side,
        amount: float,
        price: float | None = None,
        execution: OrderType = OrderType.LIMIT,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Place an order. `price` and `stop_price` are RIAL for dst=rls."""
        payload: dict[str, Any] = {
            "type": side.value,
            "srcCurrency": src,
            "dstCurrency": dst,
            "amount": f"{amount:.10f}".rstrip("0").rstrip("."),
            "execution": execution.value,
        }
        if execution in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is not None:
            payload["price"] = int(price) if float(price).is_integer() else price
        if execution in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            if stop_price is None:
                raise ValueError(f"{execution.value} requires stop_price")
            payload["stopPrice"] = int(stop_price)
        if client_order_id:
            payload["clientOrderId"] = client_order_id[:32]

        data = self._post("/market/orders/add", payload)
        return _parse_order(data["order"], f"{src.upper()}{'IRT' if dst == 'rls' else dst.upper()}")

    def order_status(
        self, order_id: int | None = None, client_order_id: str | None = None
    ) -> Order:
        if order_id is None and client_order_id is None:
            raise ValueError("pass order_id or client_order_id")
        payload = {k: v for k, v in {"order": order_id, "clientOrderId": client_order_id}.items() if v}
        data = self._post("/market/orders/status", payload)
        return _parse_order(data["order"], "")

    def cancel_order(
        self, order_id: int | None = None, client_order_id: str | None = None
    ) -> bool:
        payload: dict[str, Any] = {"status": "canceled"}
        if order_id is not None:
            payload["order"] = order_id
        if client_order_id is not None:
            payload["clientOrderId"] = client_order_id
        return self._post("/market/orders/update-status", payload).get("status") == "ok"

    def open_orders(self, src: str | None = None, dst: str | None = None) -> list[Order]:
        data = self._get(
            "/market/orders/list", srcCurrency=src, dstCurrency=dst, status="open", details=2
        )
        return [_parse_order(o, "") for o in data.get("orders", [])]


# -- helpers ---------------------------------------------------------------
_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "active": OrderStatus.ACTIVE,
    "open": OrderStatus.ACTIVE,
    "inactive": OrderStatus.NEW,
    "done": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


def _parse_order(o: dict, symbol: str) -> Order:
    matched = float(o.get("matchedAmount") or 0)
    amount = float(o.get("amount") or 0)
    status = _STATUS_MAP.get(str(o.get("status", "")).lower(), OrderStatus.NEW)
    if status is OrderStatus.ACTIVE and 0 < matched < amount:
        status = OrderStatus.PARTIAL

    price_raw = o.get("price")
    price = None if price_raw in (None, "market") else float(price_raw)

    return Order(
        symbol=symbol,
        side=Side(str(o.get("type", "buy")).lower()),
        amount=amount,
        price=price,
        order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
        client_order_id=o.get("clientOrderId"),
        exchange_id=int(o["id"]) if o.get("id") is not None else None,
        status=status,
        filled_amount=matched,
        avg_fill_price=float(o.get("averagePrice") or 0),
        fee=float(o.get("fee") or 0),
    )


def _empty_ohlcv() -> pd.DataFrame:
    return empty_ohlcv()


def _finalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = normalise_index(df)
    return df[["open", "high", "low", "close", "volume"]].astype(float)
