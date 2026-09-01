"""Margin execution: shorts and leverage on Nobitex.

This is deliberately a SEPARATE broker rather than a flag on the spot one,
because the two have incompatible mental models:

  * Spot: a position IS a wallet balance. You hold 0.5 SOL; selling reduces it.
  * Margin: a position is a distinct exchange object with its own collateral,
    its own liquidation price, and a daily financing charge. Selling does not
    reduce a balance -- you close a position by id, and closing "too much"
    opens a NEW position facing the other way.

Three hazards that spot trading simply does not have:

  * LIQUIDATION. Cross `liquidationPrice` and the collateral is seized. This
    is not a drawdown you ride out; the position is gone and does not recover
    when the price does. The portfolio drawdown stop cannot save you, because
    liquidation happens at the exchange between our decision cycles.
  * UNBOUNDED LOSS on shorts. A long can lose 100%. A short loses without
    limit as the price rises.
  * FINANCING. `positionFeeRate` is charged DAILY on the delegated amount --
    0.05%/day on this account, ~18%/year. A position held through a flat
    market bleeds even when the signal is right.

Collateral must be moved into the margin wallet first with
`NobitexREST.transfer`; margin orders draw on that wallet, not spot.
"""

from __future__ import annotations

import logging
import time

from ..core.types import MarginPosition, Order, OrderType, PositionSide, Side
from ..data.nobitex_rest import NobitexError, NobitexREST
from .base import new_client_order_id
from .nobitex import _OrderRateGuard

log = logging.getLogger(__name__)


class MarginBroker:
    """Opens and closes leveraged / short positions.

    Order placement shares Nobitex's 300-per-10-minutes budget with spot, so
    the same rate guard type is used here; pass the spot broker's guard in to
    share one budget across both.
    """

    def __init__(
        self,
        rest: NobitexREST,
        symbol_specs: dict[str, object],
        max_leverage: float = 1.0,
        min_liquidation_distance: float = 0.15,
        dry_run: bool = False,
        guard: _OrderRateGuard | None = None,
    ):
        if not rest._token:
            raise RuntimeError("MarginBroker needs credentials")
        self.rest = rest
        self.specs = symbol_specs
        self.max_leverage = float(max_leverage)
        self.min_liquidation_distance = float(min_liquidation_distance)
        self.dry_run = dry_run
        self._guard = guard or _OrderRateGuard()
        self._markets: dict[str, object] | None = None
        self._collateral = 0.0
        self._collateral_at = 0.0
        # Shares the rate-limited /users/wallets/list budget with spot
        # balance reads, so this is cached for a whole cycle, not seconds.
        self._collateral_ttl = 30.0

    # -- capability --------------------------------------------------------
    def markets(self, refresh: bool = False) -> dict[str, object]:
        if self._markets is None or refresh:
            self._markets = self.rest.margin_markets()
        return self._markets

    def collateral(self, refresh: bool = False) -> float:
        """Free rial in the MARGIN wallet, cached briefly.

        Spot balance is irrelevant here -- margin orders draw on this wallet
        alone. Returns 0.0 rather than raising if the read fails, so the
        caller's "no collateral, no short" rule fails closed.
        """
        now = time.time()
        if not refresh and self._collateral_at and now - self._collateral_at < self._collateral_ttl:
            return self._collateral
        try:
            self._collateral = float(self.rest.margin_wallets().get("rls", 0.0))
        except Exception:
            log.warning("could not read the margin wallet; treating collateral as 0", exc_info=True)
            self._collateral = 0.0
        self._collateral_at = now
        return self._collateral

    def can_short(self, symbol: str, min_collateral: float = 0.0) -> bool:
        """Shortable market AND enough collateral to actually open one.

        The collateral half matters: with an empty margin wallet every short
        is rejected `InsufficientBalance`, and at ~50 signalling symbols a
        cycle that exhausts the 300-per-10-minutes order budget on nothing.
        """
        m = self.markets().get(symbol)
        if not (m and m.sell_enabled):
            return False
        return not (min_collateral > 0 and self.collateral() < min_collateral)

    def max_leverage_for(self, symbol: str) -> float:
        m = self.markets().get(symbol)
        exchange_max = float(m.max_leverage) if m else 1.0
        return min(self.max_leverage, exchange_max)

    def clamp_leverage(self, symbol: str, wanted: float) -> float:
        """Nobitex accepts 1..maxLeverage in steps of 0.5, and rejects anything
        else outright. Round DOWN so a clamp never increases risk."""
        capped = min(float(wanted), self.max_leverage_for(symbol))
        stepped = int(capped * 2) / 2.0
        return max(1.0, stepped)

    # -- read --------------------------------------------------------------
    def positions(self, symbol: str | None = None) -> list[MarginPosition]:
        if symbol is None:
            return self.rest.positions(status="active")
        spec = self.specs[symbol]
        return self.rest.positions(src=spec.src, dst=spec.dst, status="active")

    def position_for(self, symbol: str) -> MarginPosition | None:
        for p in self.positions(symbol):
            if p.symbol == symbol and p.is_open:
                return p
        return None

    def at_risk(self, position: MarginPosition, price: float | None = None) -> bool:
        """True when the position is uncomfortably close to liquidation.

        Checked every cycle, because liquidation is irreversible and happens at
        the exchange without warning between our decisions.
        """
        room = position.distance_to_liquidation(price)
        return room is not None and room <= self.min_liquidation_distance

    # -- write -------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        side: Side,
        amount: float,
        price: float | None = None,
        leverage: float = 1.0,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """`side=SELL` opens a SHORT; `side=BUY` opens a leveraged long."""
        spec = self.specs[symbol]
        if side is Side.SELL and not self.can_short(symbol):
            raise NobitexError(
                f"{symbol} does not support margin selling", code="UnsupportedMarginSrc"
            )

        lev = self.clamp_leverage(symbol, leverage)
        coid = new_client_order_id("mgn")

        if self.dry_run:
            log.warning(
                "[dry-run] would open %s %s %.8f @ %s at %gx",
                side.value, symbol, amount, price, lev,
            )
            return Order(
                symbol=symbol, side=side, amount=amount, price=price,
                order_type=order_type, client_order_id=coid,
            )

        self._guard.check()
        try:
            order = self.rest.add_margin_order(
                src=spec.src, dst=spec.dst, side=side, amount=amount,
                leverage=lev, price=price, execution=order_type,
                client_order_id=coid,
            )
        except Exception:
            # Never retried, for a sharper reason than on spot: a duplicate
            # margin order opens a SECOND leveraged position.
            log.exception("margin order failed for %s (coid=%s); NOT retrying", symbol, coid)
            raise

        order.symbol = symbol
        order.client_order_id = coid
        log.info(
            "opened %s %s %.8f @ %s at %gx (id=%s)",
            side.value, symbol, amount, price, lev, order.exchange_id,
        )
        return order

    def close_position(
        self, position: MarginPosition, amount: float | None = None, price: float | None = None
    ) -> Order:
        """Close all or part of a position.

        `amount` defaults to the full liability. Closing MORE than is owed does
        not simply flatten -- Nobitex opens a new position in the opposite
        direction with the excess -- so the request is clamped.
        """
        owed = abs(position.liability) or abs(position.delegated_amount)
        units = min(abs(amount), owed) if amount is not None else owed
        if units <= 0:
            raise ValueError(f"nothing to close on position {position.id}")

        if self.dry_run:
            log.warning("[dry-run] would close position %s (%.8f)", position.id, units)
            return Order(
                symbol=position.symbol,
                side=Side.BUY if position.side is PositionSide.SHORT else Side.SELL,
                amount=units, price=price,
            )

        self._guard.check()
        order = self.rest.close_position(position.id, amount=units, price=price)
        order.symbol = position.symbol
        log.info(
            "closing position %s (%s %s) amount=%.8f",
            position.id, position.symbol, position.side.name, units,
        )
        return order

    def cancel_working_orders(self) -> int:
        """Cancel every margin order still resting in the book.

        Called at startup. The in-flight-order tracking that stops the runner
        stacking duplicate shorts lives in memory, so a restart forgets it --
        and an unfilled order is not a position, so nothing else would notice
        it either. Clearing the book means the first cycle starts from a state
        it can actually see. Anything still wanted is re-opened at the current
        price moments later.

        Cancels by `clientOrderId`: the order list returns `"id": null` for
        these, and `update-status` ignores the numeric id for margin orders.
        """
        cancelled = 0
        try:
            data = self.rest._get("/market/orders/list", status="open", details=2)
        except Exception:
            log.warning("could not list working orders at startup", exc_info=True)
            return 0

        for raw in data.get("orders", []):
            if str(raw.get("tradeType", "")).lower() != "margin":
                continue
            coid = raw.get("clientOrderId")
            if not coid:
                log.warning("working margin order with no clientOrderId; cannot cancel it")
                continue
            try:
                if self.rest.cancel_order(client_order_id=coid):
                    cancelled += 1
            except Exception:
                log.warning("could not cancel working margin order %s", coid, exc_info=True)
        if cancelled:
            log.warning("cancelled %d working margin order(s) at startup", cancelled)
        return cancelled

    def close_all(self, price_for: dict[str, float] | None = None) -> list[Order]:
        """Flatten every open position. The panic button."""
        out = []
        for p in self.positions():
            px = (price_for or {}).get(p.symbol)
            try:
                out.append(self.close_position(p, price=px))
            except Exception:
                log.exception("could not close position %s (%s)", p.id, p.symbol)
        return out
