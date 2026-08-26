"""Broker interface.

`PaperBroker` and `NobitexBroker` implement the same protocol so the live
runner is identical in both modes. The only difference is whether an order
leaves the process. That symmetry is deliberate: a paper run that exercises a
different code path than the live run proves nothing.
"""

from __future__ import annotations

import time
import uuid
from typing import Protocol, runtime_checkable

from ..core.types import BookTop, Order, OrderType, Side


@runtime_checkable
class Broker(Protocol):
    def balances(self) -> dict[str, float]: ...

    def book(self, symbol: str) -> BookTop: ...

    def submit(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: float | None = None,
        order_type: OrderType = OrderType.LIMIT,
        client_order_id: str | None = None,
    ) -> Order: ...

    def status(self, order: Order) -> Order: ...

    def cancel(self, order: Order) -> bool: ...

    def open_orders(self, symbol: str | None = None) -> list[Order]: ...


SETTLEMENT_TOLERANCE = 0.98
"""Fraction of the expected credit that counts as settled.

Fees are deducted from the credited side, so the observed delta is always a
little smaller than the notional. Requiring the full amount would never
confirm.
"""


def credited_currency(side: Side, base: str, quote: str) -> str:
    """Which wallet receives value from a fill.

    Nobitex blocks the debited side at submission but credits the other side
    only after settlement, so this is the wallet worth watching.
    """
    return base if side is Side.BUY else quote


def new_client_order_id(prefix: str = "nbt") -> str:
    """Nobitex allows 32 chars and requires uniqueness among live orders."""
    return f"{prefix}{uuid.uuid4().hex[:16]}{int(time.time()) % 100000}"[:32]


def limit_price_through_touch(
    book: BookTop, side: Side, offset_bps: float, price_step: float
) -> float:
    """Post a limit slightly *through* the touch.

    Joining the touch exactly is usually a long queue wait in the IRT books;
    a few bps through it takes liquidity from the front of the opposite side
    while still capping the worst-case price, unlike a market order.
    """
    from ..units import round_to_step

    if side is Side.BUY:
        raw = book.best_ask * (1 + offset_bps / 1e4)
    else:
        raw = book.best_bid * (1 - offset_bps / 1e4)

    rounded = round_to_step(raw, price_step)
    # Rounding down on a buy can put the price below the ask and never fill;
    # step it back up.
    if side is Side.BUY and rounded < raw:
        rounded += price_step
    return rounded
