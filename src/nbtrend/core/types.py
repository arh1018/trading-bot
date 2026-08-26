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
