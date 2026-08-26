"""Paper broker: real market data, simulated fills.

Fills are modelled by walking the live orderbook, so a size that would eat
three levels is filled at the volume-weighted price of those three levels and
reports the slippage honestly. A paper broker that fills everything at the
touch will make any strategy look good and teach you nothing about whether
the IRT books are deep enough for your size.
"""

from __future__ import annotations

import logging
import time

from ..core.types import BookTop, Fill, Order, OrderStatus, OrderType, Side
from ..data.nobitex_rest import NobitexREST
from .base import new_client_order_id

log = logging.getLogger(__name__)


class PaperBroker:
    def __init__(
        self,
        rest: NobitexREST,
        starting_balances: dict[str, float],
        maker_fee: float = 0.0025,
        taker_fee: float = 0.0030,
    ):
        self.rest = rest
        self._balances = dict(starting_balances)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.fills: list[Fill] = []
        self._orders: dict[str, Order] = {}

    # -- read --------------------------------------------------------------
    def balances(self) -> dict[str, float]:
        return dict(self._balances)

    def book(self, symbol: str) -> BookTop:
        return self.rest.orderbook(symbol)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.ACTIVE, OrderStatus.PARTIAL)
            and (symbol is None or o.symbol == symbol)
        ]

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
        coid = client_order_id or new_client_order_id("paper")
        levels = self.rest.full_orderbook(symbol)
        side_levels = levels["asks"] if side is Side.BUY else levels["bids"]

        filled, avg_price = _walk_book(side_levels, amount, price, side)
        fee_rate = self.taker_fee if order_type is OrderType.MARKET else self.maker_fee
        fee = filled * avg_price * fee_rate

        order = Order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            client_order_id=coid,
            status=OrderStatus.FILLED if filled >= amount * 0.999 else
                   (OrderStatus.PARTIAL if filled > 0 else OrderStatus.ACTIVE),
            filled_amount=filled,
            avg_fill_price=avg_price,
            fee=fee,
            created_ts=int(time.time()),
        )
        self._orders[coid] = order

        if filled > 0:
            self._settle(symbol, side, filled, avg_price, fee)
            self.fills.append(
                Fill(symbol, side, filled, avg_price, fee, int(time.time()))
            )
            log.info(
                "[paper] %s %s %.8f @ %s rial (fee %s)",
                side.value, symbol, filled, f"{avg_price:,.0f}", f"{fee:,.0f}",
            )
        return order

    def status(self, order: Order) -> Order:
        return self._orders.get(order.client_order_id or "", order)

    def await_settlement(self, currency: str, baseline: float, expected_delta: float) -> bool:
        """No-op: `_settle` mutates the local ledger inline, so a paper fill is
        already visible by the time `submit` returns. Present so the paper and
        live paths stay behaviourally identical."""
        return True

    def cancel(self, order: Order) -> bool:
        stored = self._orders.get(order.client_order_id or "")
        if stored and stored.status in (OrderStatus.ACTIVE, OrderStatus.PARTIAL):
            stored.status = OrderStatus.CANCELED
            return True
        return False

    # -- internals ---------------------------------------------------------
    def _settle(self, symbol: str, side: Side, amount: float, price: float, fee: float) -> None:
        base, quote = _split_symbol(symbol)
        notional = amount * price
        if side is Side.BUY:
            self._balances[quote] = self._balances.get(quote, 0.0) - notional - fee
            self._balances[base] = self._balances.get(base, 0.0) + amount
        else:
            self._balances[base] = self._balances.get(base, 0.0) - amount
            self._balances[quote] = self._balances.get(quote, 0.0) + notional - fee


def _walk_book(
    levels: list[tuple[float, float]], amount: float, limit_price: float | None, side: Side
) -> tuple[float, float]:
    """Volume-weighted fill from consuming the book, respecting a limit."""
    remaining = amount
    notional = 0.0

    for level_price, level_amount in levels:
        if remaining <= 0:
            break
        if limit_price is not None:
            if side is Side.BUY and level_price > limit_price:
                break
            if side is Side.SELL and level_price < limit_price:
                break
        take = min(remaining, level_amount)
        notional += take * level_price
        remaining -= take

    filled = amount - remaining
    return (filled, notional / filled) if filled > 0 else (0.0, 0.0)


def _split_symbol(symbol: str) -> tuple[str, str]:
    """`BTCIRT` -> (`btc`, `rls`). IRT markets settle in rial."""
    for suffix, quote in (("IRT", "rls"), ("USDT", "usdt")):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)].lower(), quote
    raise ValueError(f"cannot split symbol {symbol!r}")
