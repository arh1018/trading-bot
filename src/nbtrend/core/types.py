"""Domain types. All monetary values are RIAL unless the field says otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Candle:
    ts: int          # unix seconds, bar OPEN time
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class BookTop:
    """Top of book, rial."""
    symbol: str
    best_bid: float
    best_ask: float
    last_trade: float
    ts_ms: int

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float:
        return 1e4 * (self.best_ask - self.best_bid) / self.mid if self.mid else float("inf")


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy output for one symbol at one bar."""
    symbol: str
    ts: int
    score: float             # composite trend score, [-1, +1]
    target_weight: float     # fraction of equity, post risk-sizing
    components: dict[str, float] = field(default_factory=dict)
    regime_ok: bool = True
    reason: str = ""


@dataclass(slots=True)
class Position:
    symbol: str
    amount: float = 0.0        # base asset units
    avg_price: float = 0.0     # rial
    entry_ts: int = 0
    peak_price: float = 0.0    # for the chandelier trailing stop

    @property
    def is_open(self) -> bool:
        return abs(self.amount) > 0

    def notional(self, price: float) -> float:
        return self.amount * price

    def unrealised(self, price: float) -> float:
        return (price - self.avg_price) * self.amount


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    amount: float
    price: float | None = None      # None => market
    order_type: OrderType = OrderType.LIMIT
    client_order_id: str | None = None
    exchange_id: int | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    created_ts: int = 0

    @property
    def remaining(self) -> float:
        return max(0.0, self.amount - self.filled_amount)


@dataclass(slots=True)
class Fill:
    symbol: str
    side: Side
    amount: float
    price: float
    fee: float
    ts: int
    order_id: int | None = None


class PositionSide(str, Enum):
    """Direction of a margin position.

    Nobitex opens a position from the order `type`: a `sell` order opens a
    SHORT (you borrow the asset and owe it back), a `buy` order opens a LONG.
    """

    LONG = "buy"
    SHORT = "sell"


@dataclass(slots=True)
class MarginMarket:
    """One entry of `GET /margin/markets/list`."""

    symbol: str
    src: str
    dst: str
    max_leverage: float
    sell_enabled: bool
    buy_enabled: bool
    position_fee_rate: float


@dataclass(slots=True)
class MarginPosition:
    """An open margin position.

    Unlike a spot holding, this is a distinct exchange object with its own
    collateral and a `liquidation_price`. Losses are NOT bounded by the
    position size: if the mark crosses liquidation the collateral is seized.
    That is the whole reason the live runner cannot treat a position as if it
    were a wallet balance.

    All monetary fields are RIAL for dst=rls markets.
    """

    id: int
    symbol: str
    side: PositionSide
    status: str
    collateral: float
    leverage: float
    liquidation_price: float | None
    entry_price: float | None
    liability: float
    delegated_amount: float
    margin_ratio: float | None
    unrealized_pnl: float
    mark_price: float | None = None
    expiration_date: str | None = None
    extension_fee: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status.lower() in ("open", "active")

    def distance_to_liquidation(self, price: float | None = None) -> float | None:
        """Fraction the mark can move against the position before liquidation.

        Returns None when the exchange has not published a liquidation price
        (it does not for an unleveraged position). A SHORT is liquidated by the
        price rising, a LONG by it falling, so the sign is handled per side.
        """
        mark = price if price is not None else self.mark_price
        if not mark or not self.liquidation_price:
            return None
        if self.side is PositionSide.SHORT:
            return (self.liquidation_price - mark) / mark
        return (mark - self.liquidation_price) / mark
