"""Live Nobitex broker.

Guard rails, because this one spends money:

* Refuses to construct without a token.
* Every order carries a `clientOrderId`, so a timed-out request can be
  reconciled instead of blindly retried into a double fill.
* Order submission is NOT retried on transport errors. A retried POST that
  actually succeeded the first time places the order twice; reconcile by
  clientOrderId instead.
* Order placement shares a 300-request/10-minute limit across spot and margin.
  `_OrderRateGuard` enforces it locally so the exchange never has to.
"""

from __future__ import annotations

import logging
import time
from collections import deque

from ..core.types import BookTop, Order, OrderStatus, OrderType, Side
from ..data.nobitex_rest import NobitexError, NobitexREST
from .base import new_client_order_id

log = logging.getLogger(__name__)

ORDER_LIMIT = 300
ORDER_WINDOW_S = 600


class _OrderRateGuard:
    def __init__(self, limit: int = ORDER_LIMIT, window_s: int = ORDER_WINDOW_S):
        self.limit = limit
        self.window_s = window_s
        self._stamps: deque[float] = deque()

    def check(self) -> None:
        now = time.time()
        while self._stamps and now - self._stamps[0] > self.window_s:
            self._stamps.popleft()
        if len(self._stamps) >= self.limit:
            wait = self.window_s - (now - self._stamps[0])
            raise NobitexError(
                f"local order rate limit reached ({self.limit}/{self.window_s}s); "
                f"next slot in {wait:.0f}s",
                code="LocalRateLimit",
            )
        self._stamps.append(now)


class NobitexBroker:
    def __init__(self, rest: NobitexREST, symbol_specs: dict[str, object], dry_run: bool = False):
        if not rest._token:
            raise RuntimeError("NobitexBroker needs an API token; set NOBITEX_API_TOKEN")
        self.rest = rest
        self.specs = symbol_specs
        self.dry_run = dry_run
        self._guard = _OrderRateGuard()
        self._by_coid: dict[str, Order] = {}

    # -- read --------------------------------------------------------------
    def balances(self) -> dict[str, float]:
        return self.rest.wallets()

    def book(self, symbol: str) -> BookTop:
        return self.rest.orderbook(symbol)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol is None:
            return self.rest.open_orders()
        spec = self.specs[symbol]
        return self.rest.open_orders(src=spec.src, dst=spec.dst)

    # -- write -------------------------------------------------------------
    def submit(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: float | None = None,
        order_type: OrderType = OrderType.LIMIT,
        client_order_id: str | None = None,
    ) -> Order:
        spec = self.specs[symbol]
        coid = client_order_id or new_client_order_id()

        if self.dry_run:
            log.warning(
                "[dry-run] would %s %.8f %s @ %s", side.value, amount, symbol, price
            )
            return Order(
                symbol=symbol, side=side, amount=amount, price=price,
                order_type=order_type, client_order_id=coid, status=OrderStatus.NEW,
            )

        self._guard.check()
        try:
            order = self.rest.add_order(
                src=spec.src,
                dst=spec.dst,
                side=side,
                amount=amount,
                price=price,
                execution=order_type,
                client_order_id=coid,
            )
        except Exception:
            # Do NOT retry -- reconcile instead. The order may well be live.
            log.exception(
                "order submission failed for %s; reconciling by clientOrderId %s", symbol, coid
            )
            recovered = self._reconcile(coid)
            if recovered is not None:
                return recovered
            raise

        order.symbol = symbol
        order.client_order_id = coid
        self._by_coid[coid] = order
        log.info(
            "submitted %s %s %.8f @ %s (id=%s)", side.value, symbol, amount, price, order.exchange_id
        )
        return order

    def status(self, order: Order) -> Order:
        fresh = self.rest.order_status(
            order_id=order.exchange_id, client_order_id=order.client_order_id
        )
        fresh.symbol = order.symbol
        fresh.client_order_id = order.client_order_id
        return fresh

    def cancel(self, order: Order) -> bool:
        try:
            return self.rest.cancel_order(
                order_id=order.exchange_id, client_order_id=order.client_order_id
            )
        except NobitexError as exc:
            log.warning("cancel failed for %s: %s", order.client_order_id, exc)
            return False

    def _reconcile(self, coid: str) -> Order | None:
        """Did the order land despite the error?"""
        try:
            return self.rest.order_status(client_order_id=coid)
        except Exception:
            return None
