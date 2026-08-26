"""Order router: turns a target weight into fills.

Strategy is a passive limit that chases the touch. Crossing the spread on
every rebalance costs the taker fee (0.30%) plus the spread (~5-20bps in the
IRT majors, far worse in the tail). Posting and re-pricing a few times pays
the maker fee instead, and on a trend follower holding for days the extra
minutes of latency cost far less than the fee difference.

After `max_reposts` attempts the router crosses, because a trend entry that
never fills is a worse outcome than one that fills 20bps wide.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import SymbolSpec
from ..core.types import Order, OrderStatus, OrderType, Side
from ..units import round_to_step
from .base import Broker, limit_price_through_touch, new_client_order_id

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionReport:
    symbol: str
    side: Side
    requested: float
    filled: float
    avg_price: float
    attempts: int
    crossed: bool
    orders: list[Order]

    @property
    def complete(self) -> bool:
        return self.filled >= self.requested * 0.999


class OrderRouter:
    def __init__(self, broker: Broker, cfg: dict, min_order_rial: float):
        self.broker = broker
        self.offset_bps = float(cfg.get("limit_offset_bps", 5))
        self.repost_after_s = float(cfg.get("repost_after_s", 45))
        self.max_reposts = int(cfg.get("max_reposts", 4))
        self.poll_interval_s = float(cfg.get("poll_interval_s", 5))
        self.use_limit = cfg.get("order_type", "limit") == "limit"
        self.min_order_rial = min_order_rial

    def execute(self, spec: SymbolSpec, side: Side, amount: float) -> ExecutionReport:
        symbol = spec.nobitex
        amount = round_to_step(amount, spec.amount_step)
        orders: list[Order] = []

        book = self.broker.book(symbol)
        if amount * book.mid < self.min_order_rial:
            log.info(
                "%s: %.8f is below the %s rial minimum; skipping",
                symbol, amount, f"{self.min_order_rial:,.0f}",
            )
            return ExecutionReport(symbol, side, amount, 0.0, 0.0, 0, False, orders)

        if not self.use_limit:
            order = self.broker.submit(symbol, side, amount, None, OrderType.MARKET)
            orders.append(order)
            return _report(symbol, side, amount, orders, attempts=1, crossed=True)

        remaining = amount
        for attempt in range(1, self.max_reposts + 1):
            book = self.broker.book(symbol)
            price = limit_price_through_touch(book, side, self.offset_bps, spec.price_step)

            order = self.broker.submit(
                symbol, side, remaining, price, OrderType.LIMIT, new_client_order_id()
            )
            orders.append(order)

            filled = self._wait_for_fill(order)
            remaining = round_to_step(max(0.0, remaining - filled), spec.amount_step)

            if remaining * book.mid < self.min_order_rial:
                return _report(symbol, side, amount, orders, attempt, crossed=False)

            self.broker.cancel(order)
            log.info(
                "%s: repost %d/%d, %.8f still unfilled", symbol, attempt, self.max_reposts, remaining
            )

        # Out of patience -- cross the spread for the remainder.
        order = self.broker.submit(symbol, side, remaining, None, OrderType.MARKET)
        orders.append(order)
        log.info("%s: crossed the spread for the residual %.8f", symbol, remaining)
        return _report(symbol, side, amount, orders, self.max_reposts + 1, crossed=True)

    def _wait_for_fill(self, order: Order) -> float:
        deadline = time.time() + self.repost_after_s
        filled = order.filled_amount
        while time.time() < deadline:
            fresh = self.broker.status(order)
            filled = fresh.filled_amount
            if fresh.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                return filled
            time.sleep(self.poll_interval_s)
        return filled


def _report(
    symbol: str, side: Side, requested: float, orders: list[Order], attempts: int, crossed: bool
) -> ExecutionReport:
    filled = sum(o.filled_amount for o in orders)
    notional = sum(o.filled_amount * o.avg_fill_price for o in orders)
    return ExecutionReport(
        symbol=symbol,
        side=side,
        requested=requested,
        filled=filled,
        avg_price=notional / filled if filled else 0.0,
        attempts=attempts,
        crossed=crossed,
        orders=orders,
    )
