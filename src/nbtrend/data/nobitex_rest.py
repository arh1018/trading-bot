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
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.types import (
    BookTop,
    MarginMarket,
    MarginPosition,
    Order,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
)
from ..units import toman_to_rial
from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

# Status codes worth trying again. Everything else in the 4xx range is a
# statement about the REQUEST, not about luck: a 401 does not become authorised
# on the third attempt, and a 400 does not become well-formed. Retrying those
# is not merely useless -- Nobitex blocks an IP for 30 minutes after roughly
# 100 failed auth attempts, so a blanket retry turns one bad credential into
# four strikes against that budget every call.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(retry_state) -> bool:
    outcome = retry_state.outcome
    if outcome is None or not outcome.failed:
        return False
    exc = outcome.exception()
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return False

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
        retry=_is_retryable,
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

    def trades_list(self) -> list[dict]:
        """Our own executed trades, newest first. Bounded history."""
        return self._get("/market/trades/list").get("trades", []) or []

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
            # The two wallet endpoints do NOT share a schema. /v2/wallets keys
            # by UPPERCASE currency and reports `balance` + `blocked`, with no
            # `activeBalance` field at all -- reading `activeBalance` here
            # silently returns 0 for every currency, which reads as an empty
            # account and makes the bot refuse to trade. /users/wallets/list
            # below is the one that has `activeBalance`.
            return {
                k.lower(): float(v.get("balance", 0)) - float(v.get("blocked", 0))
                for k, v in wallets.items()
            }
        data = self._get("/users/wallets/list")
        return {
            w["currency"].lower(): float(w.get("activeBalance", 0))
            for w in data.get("wallets", [])
        }

    def balances_detailed(self) -> dict[str, tuple[float, float]]:
        """(free, blocked) per currency.

        Neither number alone is right for sizing an ask. `activeBalance`
        excludes units locked in OUR OWN resting quote, so a symbol mid-quote
        reads as empty (ONEIRT: 1796.48944 held, 0.08944 active). But total
        balance includes units locked in orders that have NOT been cancelled,
        so sizing off it asks to sell stock that is not free -- TNSRIRT tried
        to sell 38.7 against 0.63 free, and every order came back the opaque
        "Order Validation Failed".

        The caller knows which blocked units are its own and about to be
        released, so it gets both numbers and decides.
        """
        data = self._get("/users/wallets/list")
        out: dict[str, tuple[float, float]] = {}
        for w in data.get("wallets", []):
            total = float(w.get("balance", 0))
            free = float(w.get("activeBalance", 0))
            out[w["currency"].lower()] = (free, max(0.0, total - free))
        return out

    def total_balances(self) -> dict[str, float]:
        """Balance per currency INCLUDING units blocked in working orders.

        `wallets()` returns `activeBalance`, which excludes anything committed
        to an open order -- the right view for "what can I spend right now".
        It is the wrong view for a market maker sizing an ask, because its own
        resting quote blocks the very inventory it wants to sell: ONEIRT held
        1796.48944 units with an activeBalance of 0.08944, so it quoted
        bid-only and looked like a 0% ask fill rate.
        """
        data = self._get("/users/wallets/list")
        return {
            w["currency"].lower(): float(w.get("balance", 0))
            for w in data.get("wallets", [])
        }

    def margin_wallets(self) -> dict[str, float]:
        """Free balance in the MARGIN wallet, which is separate from spot.

        Margin orders draw collateral from here and nowhere else, so a healthy
        spot balance says nothing about whether a short can be opened. Reading
        this is what lets the runner refuse a short it cannot fund, instead of
        firing orders that come back `InsufficientBalance` and burn the
        300-per-10-minutes order budget.
        """
        data = self._get("/users/wallets/list", type="margin")
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
        order = _parse_order(
            data["order"], f"{src.upper()}{'IRT' if dst == 'rls' else dst.upper()}"
        )
        # Carry the clientOrderId back. The response does not always echo it,
        # and it is the ONLY handle that reliably cancels on this exchange --
        # without it the caller gets "Both id and clientOrderId cannot be
        # null", the order never cancels, its balance stays locked, and every
        # later ask tries to sell stock that is still committed. That produced
        # 5,380 "Order Validation Failed" in a single short run.
        if client_order_id and not order.client_order_id:
            order.client_order_id = client_order_id[:32]
        return order

    def order_status(
        self, order_id: int | None = None, client_order_id: str | None = None
    ) -> Order:
        """Fetch one order.

        The field is `id`, NOT `order`. Nobitex's own parameter table says
        `order`, but the endpoint ignores it -- note 2 of the same section and
        the `NullIdAndClientOrderId` error code both say `id`, and the live API
        agrees. Sending `order` yields "Both id and clientOrderId cannot be
        null" or a bare 404.

        `clientOrderId` is only searched among open/active/inactive orders, so
        it cannot look up an order that has already filled. Prefer the numeric
        id whenever it is known.
        """
        if order_id is None and client_order_id is None:
            raise ValueError("pass order_id or client_order_id")
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["id"] = order_id
        if client_order_id is not None:
            payload["clientOrderId"] = client_order_id
        data = self._post("/market/orders/status", payload)
        return _parse_order(data["order"], "")

    def cancel_order(
        self, order_id: int | None = None, client_order_id: str | None = None
    ) -> bool:
        # `clientOrderId` FIRST, and it is the only one that reliably works.
        #
        # `/market/orders/update-status` ignores the numeric `id` for margin
        # orders: cancelling six live margin sells by id returned "Both id and
        # clientOrderId cannot be null" every time, while the same orders
        # cancelled instantly by clientOrderId. The order LIST compounds this
        # -- it returns `"id": null` and a display name ("Aptos") rather than a
        # currency code -- so the id is often not even knowable. Every order
        # this bot places carries a clientOrderId for exactly this reason.
        # The numeric id field here is `order`, NOT `id`. These two endpoints
        # genuinely disagree: `/market/orders/status` takes `id` (verified
        # live), while `/market/orders/update-status` takes `order`. Sending
        # `id` to update-status returns "Both id and clientOrderId cannot be
        # null" -- the server simply does not read it.
        #
        # This matters because position CLOSE orders carry no clientOrderId:
        # Nobitex does not propagate ours onto them, so the numeric id is the
        # only handle, and with the wrong field name they were uncancellable.
        # Four stale close orders sat unfilled in the book because of it.
        payload: dict[str, Any] = {"status": "canceled"}
        if client_order_id is not None:
            payload["clientOrderId"] = client_order_id
        elif order_id is not None:
            payload["order"] = order_id
        return self._post("/market/orders/update-status", payload).get("status") == "ok"

    def open_orders(self, src: str | None = None, dst: str | None = None) -> list[Order]:
        data = self._get(
            "/market/orders/list", srcCurrency=src, dstCurrency=dst, status="open", details=2
        )
        return [_parse_order(o, "") for o in data.get("orders", [])]

    # -- margin ------------------------------------------------------------
    #
    # Margin is a separate world from spot in three ways that matter:
    #
    #   * Collateral lives in a SEPARATE wallet. A spot balance is not usable
    #     as margin until moved with `transfer`, and an unfunded margin wallet
    #     fails orders with `InsufficientBalance`.
    #   * A filled margin order opens a POSITION -- its own object, with its
    #     own collateral and liquidation price. It is not a wallet balance,
    #     which is why the spot runner's "position == balance" model does not
    #     carry over.
    #   * A `sell` order opens a SHORT. Loss on a short is unbounded above, and
    #     crossing `liquidationPrice` forfeits the collateral outright.

    def margin_markets(self) -> dict[str, MarginMarket]:
        """Which markets support margin, and at what leverage."""
        data = self._get("/margin/markets/list")
        out: dict[str, MarginMarket] = {}
        for symbol, m in (data.get("markets") or {}).items():
            out[symbol] = MarginMarket(
                symbol=symbol,
                src=str(m.get("srcCurrency", "")),
                dst=str(m.get("dstCurrency", "")),
                max_leverage=float(m.get("maxLeverage") or 1),
                sell_enabled=bool(m.get("sellEnabled")),
                buy_enabled=bool(m.get("buyEnabled")),
                position_fee_rate=float(m.get("positionFeeRate") or 0),
            )
        return out

    def transfer(self, currency: str, amount: float, src: str, dst: str) -> bool:
        """Move collateral between the `spot` and `margin` wallets.

        The margin wallet is created by the first transfer into it. The rate
        limit here is 10/minute -- far tighter than the order endpoints.
        """
        if src == dst:
            raise ValueError("transfer src and dst wallets must differ")
        if src not in ("spot", "margin") or dst not in ("spot", "margin"):
            raise ValueError("wallet type must be 'spot' or 'margin'")
        payload = {
            "currency": currency,
            "amount": f"{amount:.10f}".rstrip("0").rstrip("."),
            "src": src,
            "dst": dst,
        }
        return self._post("/wallets/transfer", payload).get("status") == "ok"

    def add_margin_order(
        self,
        src: str,
        dst: str,
        side: Side,
        amount: float,
        leverage: float,
        price: float | None = None,
        execution: OrderType = OrderType.LIMIT,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Open a margin position. `side=SELL` opens a SHORT.

        `leverage` must be between 1 and the market's `maxLeverage` in steps of
        0.5; anything else is rejected with `LeverageTooHigh`. Validated here
        so a typo costs a ValueError rather than a rejected live order.
        """
        if leverage < 1:
            raise ValueError(f"leverage must be >= 1, got {leverage}")
        if round(leverage * 2) != leverage * 2:
            raise ValueError(f"leverage must be a multiple of 0.5, got {leverage}")

        payload: dict[str, Any] = {
            "type": side.value,
            "srcCurrency": src,
            "dstCurrency": dst,
            "amount": f"{amount:.10f}".rstrip("0").rstrip("."),
            "execution": execution.value,
            "leverage": f"{leverage:g}",
        }
        if execution in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is not None:
            payload["price"] = int(price) if float(price).is_integer() else price
        if execution in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            if stop_price is None:
                raise ValueError(f"{execution.value} requires stop_price")
            payload["stopPrice"] = int(stop_price)
        if client_order_id:
            payload["clientOrderId"] = client_order_id[:32]

        data = self._post("/margin/orders/add", payload)
        return _parse_order(data["order"], f"{src.upper()}{'IRT' if dst == 'rls' else dst.upper()}")

    def add_oco_order(
        self,
        src: str,
        dst: str,
        side: Side,
        amount: float,
        price: float,
        stop_price: float,
        stop_limit_price: float,
        leverage: float | None = None,
        last_price: float | None = None,
    ) -> list[Order]:
        """One-Cancels-Other: a take-profit and a stop-loss that cancel each other.

        This is the only way to hold a stop AT THE EXCHANGE. Our chandelier
        stop is evaluated once per decision cycle, so a gap between cycles is
        unprotected -- and on a leveraged position that gap is exactly where
        liquidation happens. An OCO sits in the book continuously.

        `leverage` routes the pair to the margin endpoint; omit it for spot.

        Nobitex requires (note 4 of the OCO docs):
            sell:  price > last market price > stopPrice
            buy:   price < last market price < stopPrice
        Violating it returns `PriceConditionFailed`. When `last_price` is
        supplied that is checked here, because a rejected order still costs a
        round trip and a slot in the 300/10min budget.
        """
        if last_price is not None:
            if side is Side.SELL and not (price > last_price > stop_price):
                raise ValueError(
                    f"OCO sell needs price > last > stopPrice; got "
                    f"{price:,.0f} > {last_price:,.0f} > {stop_price:,.0f}"
                )
            if side is Side.BUY and not (price < last_price < stop_price):
                raise ValueError(
                    f"OCO buy needs price < last < stopPrice; got "
                    f"{price:,.0f} < {last_price:,.0f} < {stop_price:,.0f}"
                )

        payload: dict[str, Any] = {
            "mode": "oco",
            "type": side.value,
            "srcCurrency": src,
            "dstCurrency": dst,
            "amount": f"{amount:.10f}".rstrip("0").rstrip("."),
            "price": int(price) if float(price).is_integer() else price,
            "stopPrice": int(stop_price) if float(stop_price).is_integer() else stop_price,
            "stopLimitPrice": (
                int(stop_limit_price) if float(stop_limit_price).is_integer() else stop_limit_price
            ),
        }

        path = "/market/orders/add"
        if leverage is not None:
            if leverage < 1 or round(leverage * 2) != leverage * 2:
                raise ValueError(f"leverage must be >= 1 in steps of 0.5, got {leverage}")
            payload["leverage"] = f"{leverage:g}"
            path = "/margin/orders/add"

        data = self._post(path, payload)
        symbol = f"{src.upper()}{'IRT' if dst == 'rls' else dst.upper()}"
        # An OCO responds with `orders` (a pair), not the single `order` key.
        return [_parse_order(o, symbol) for o in data.get("orders", [])]

    def positions(
        self, src: str | None = None, dst: str | None = None, status: str | None = "active"
    ) -> list[MarginPosition]:
        data = self._get("/positions/list", srcCurrency=src, dstCurrency=dst, status=status)
        return [_parse_position(p) for p in data.get("positions", [])]

    def position_status(self, position_id: int) -> MarginPosition:
        data = self._get(f"/positions/{int(position_id)}/status")
        return _parse_position(data.get("position", data))

    def close_position(self, position_id: int, amount: float, price: float | None = None) -> Order:
        """Close (or partly close) a position with the opposite order.

        Never retry this on a transport error, for a sharper version of the
        spot reason: a retried close that actually landed the first time closes
        the position twice, and the second one opens a NEW position facing the
        other way.
        """
        payload: dict[str, Any] = {"amount": f"{amount:.10f}".rstrip("0").rstrip(".")}
        if price is not None:
            payload["price"] = int(price) if float(price).is_integer() else price
        data = self._post(f"/positions/{int(position_id)}/close", payload)
        return _parse_order(data["order"], "")


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


def _num(value: Any) -> float | None:
    """Nobitex sends monetary fields as strings, nulls, and -- for negative
    PNL -- with a UNICODE MINUS (U+2212), which float() rejects."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).replace("−", "-").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_position(p: dict) -> MarginPosition:
    src = str(p.get("srcCurrency", "")).upper()
    dst = str(p.get("dstCurrency", "")).lower()
    return MarginPosition(
        id=int(p["id"]),
        symbol=f"{src}{'IRT' if dst == 'rls' else dst.upper()}",
        side=PositionSide(str(p.get("side", "sell")).lower()),
        status=str(p.get("status", "")),
        collateral=_num(p.get("collateral")) or 0.0,
        leverage=_num(p.get("leverage")) or 1.0,
        liquidation_price=_num(p.get("liquidationPrice")),
        entry_price=_num(p.get("entryPrice")),
        liability=_num(p.get("liability")) or 0.0,
        delegated_amount=_num(p.get("delegatedAmount")) or 0.0,
        margin_ratio=_num(p.get("marginRatio")),
        unrealized_pnl=_num(p.get("unrealizedPNL")) or 0.0,
        mark_price=_num(p.get("markPrice")),
        expiration_date=p.get("expirationDate"),
        extension_fee=_num(p.get("extensionFee")) or 0.0,
    )


def _empty_ohlcv() -> pd.DataFrame:
    return empty_ohlcv()


def _finalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = normalise_index(df)
    return df[["open", "high", "low", "close", "volume"]].astype(float)
