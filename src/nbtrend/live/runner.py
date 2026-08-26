"""Live/paper trading loop.

Shape of the loop
-----------------
Two clocks run at once, and conflating them is how live trading diverges from
the backtest:

* The **decision clock** is the strategy bar (4h by default). Signals are
  recomputed only when a bar closes, from the same global-price history the
  backtest used. Recomputing on every tick would produce a different, noisier
  strategy than the one that was validated.
* The **market clock** is the websocket. It maintains the live orderbook and
  the basis check, and it is what execution prices against -- but it never
  moves the target weight.

Safety interlocks, all of which flatten or refuse to trade rather than guess:
  * stale websocket (no message for `max_staleness_s`)
  * basis outside `max_basis_deviation` -- local price dislocated from global
  * `isClosed` on the market-stats channel -- Nobitex halts markets
  * equity drawdown past `max_drawdown_stop`
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..backtest.engine import Backtester
from ..config import Config, SymbolSpec
from ..core.types import BookTop, Side
from ..data.feed import DataFeed
from ..data.fx import compute_basis
from ..data.nobitex_rest import NobitexREST
from ..data.nobitex_ws import (
    NobitexWS,
    book_top_from_payload,
    market_stats_channel,
    orderbook_channel,
)
from ..execution.base import Broker
from ..execution.nobitex import NobitexBroker
from ..execution.paper import PaperBroker
from ..execution.router import OrderRouter
from ..units import round_to_step

log = logging.getLogger(__name__)

MAX_STALENESS_S = 120.0


@dataclass
class SymbolState:
    spec: SymbolSpec
    book: BookTop | None = None
    is_closed: bool = False
    target_weight: float = 0.0
    score: float = 0.0
    last_signal_ts: int | None = None
    basis: float = 0.0

    @property
    def tradeable(self) -> bool:
        return self.book is not None and not self.is_closed


@dataclass
class RunnerState:
    """Persisted across restarts so a crash does not lose the book."""
    equity_peak_rial: float = 0.0
    halted: bool = False
    last_bar_ts: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunnerState:
        if not path.exists():
            return cls()
        try:
            return cls(**json.loads(path.read_text()))
        except Exception:
            log.warning("could not read %s; starting with fresh state", path)
            return cls()


class LiveRunner:
    def __init__(self, cfg: Config, broker: Broker | None = None):
        self.cfg = cfg
        self.rest = NobitexREST(cfg.rest_url, cfg.creds.api_token)
        self.feed = DataFeed(cfg)
        self.backtester = Backtester(cfg)

        self.states: dict[str, SymbolState] = {
            spec.nobitex: SymbolState(spec=spec) for spec in cfg.enabled_symbols
        }
        self.fx_state = SymbolState(spec=cfg.fx)

        self.broker = broker or self._build_broker()
        self.router = OrderRouter(
            self.broker, cfg.execution, float(cfg.costs["min_order_rial"])
        )

        self.state_path = Path("data/state/runner.json")
        self.state = RunnerState.load(self.state_path)
        self.ws = NobitexWS(cfg.ws_url)
        self._stop = asyncio.Event()

    def _build_broker(self) -> Broker:
        if self.cfg.is_live:
            self.cfg.creds.require_token()
            specs = {s.nobitex: s for s in [*self.cfg.enabled_symbols, self.cfg.fx]}
            log.warning("LIVE MODE -- orders will be sent to Nobitex")
            return NobitexBroker(self.rest, specs)

        starting = {"rls": float(self.cfg.backtest["initial_equity_rial"])}
        log.info("paper mode -- starting with %s rial", f"{starting['rls']:,.0f}")
        return PaperBroker(
            self.rest, starting,
            maker_fee=float(self.cfg.costs["maker_fee"]),
            taker_fee=float(self.cfg.costs["taker_fee"]),
        )

    # -- websocket wiring --------------------------------------------------
    def _wire(self) -> None:
        symbols = [*self.states, self.cfg.fx.nobitex]
        for symbol in symbols:
            self.ws.on(orderbook_channel(symbol), self._make_book_handler(symbol))
            self.ws.on(market_stats_channel(symbol), self._make_stats_handler(symbol))

    def _state_for(self, symbol: str) -> SymbolState | None:
        if symbol == self.cfg.fx.nobitex:
            return self.fx_state
        return self.states.get(symbol)

    def _make_book_handler(self, symbol: str):
        def handler(_channel: str, payload: dict) -> None:
            state = self._state_for(symbol)
            if state is None:
                return
            top = book_top_from_payload(symbol, payload)
            if top is not None:
                state.book = top
        return handler

    def _make_stats_handler(self, symbol: str):
        def handler(_channel: str, payload: dict) -> None:
            state = self._state_for(symbol)
            if state is not None and "isClosed" in payload:
                was = state.is_closed
                state.is_closed = bool(payload["isClosed"])
                if state.is_closed and not was:
                    log.warning("%s market is CLOSED", symbol)
        return handler

    # -- main loop ---------------------------------------------------------
    async def run(self, once: bool = False) -> None:
        self._wire()
        ws_task = asyncio.create_task(self.ws.run(self._stop))

        try:
            await self._await_books(timeout_s=45)
            while not self._stop.is_set():
                try:
                    await self.rebalance()
                except Exception:
                    log.exception("rebalance failed; will retry next cycle")

                if once:
                    break
                await self._sleep_until_next_bar()
        finally:
            self._stop.set()
            ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ws_task
            self.state.save(self.state_path)
            self.rest.close()

    async def _await_books(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if all(s.book for s in self.states.values()) and self.fx_state.book:
                log.info("orderbooks warm for %d market(s)", len(self.states) + 1)
                return
            await asyncio.sleep(1)
        # Seed whatever is missing from REST so the first cycle can proceed.
        log.warning("websocket books incomplete after %.0fs; seeding from REST", timeout_s)
        for symbol, state in [*self.states.items(), (self.cfg.fx.nobitex, self.fx_state)]:
            if state.book is None:
                state.book = self.rest.orderbook(symbol)

    async def _sleep_until_next_bar(self) -> None:
        seconds = _bar_seconds(str(self.cfg.data["timeframe"]))
        now = time.time()
        wait = seconds - (now % seconds) + 5   # a few seconds past the close
        log.info("sleeping %.0fs until the next %s bar", wait, self.cfg.data["timeframe"])
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=wait)

    # -- decision ----------------------------------------------------------
    async def rebalance(self) -> None:
        if self.ws.seconds_since_message > MAX_STALENESS_S:
            log.error(
                "websocket stale for %.0fs; skipping this cycle",
                self.ws.seconds_since_message,
            )
            return

        equity = await asyncio.to_thread(self.equity_rial)
        self.state.equity_peak_rial = max(self.state.equity_peak_rial, equity)
        drawdown = equity / self.state.equity_peak_rial - 1.0 if self.state.equity_peak_rial else 0.0

        if drawdown <= -float(self.cfg.risk["max_drawdown_stop"]) and not self.state.halted:
            log.error("drawdown %.1f%% breached the limit -- flattening", drawdown * 100)
            self.state.halted = True

        datasets = await self.feed.build_all(self.cfg.enabled_symbols)
        max_basis = float(self.cfg.execution["max_basis_deviation"])

        # --- pass 1: score every symbol and apply the per-symbol interlocks ---
        tradeable: list[SymbolState] = []
        for symbol, state in self.states.items():
            dataset = datasets.get(symbol)
            if dataset is None or dataset.frame.empty:
                log.warning("%s: no data this cycle", symbol)
                state.target_weight = 0.0
                continue

            signals = self.backtester.build_signals(dataset.frame)
            last = signals.iloc[-1]
            state.score = float(last["score"])
            state.target_weight = 0.0 if self.state.halted else float(last["target_weight"])

            if not state.tradeable:
                log.warning("%s: not tradeable (closed or no book)", symbol)
                state.target_weight = 0.0
                continue

            # Basis interlock: refuse a market dislocated from global pricing.
            if self.fx_state.book:
                basis = compute_basis(
                    symbol,
                    state.book.mid,
                    float(dataset.frame["close"].iloc[-1]),
                    self.fx_state.book.mid,
                )
                state.basis = basis.basis
                if abs(basis.basis) > max_basis:
                    log.error(
                        "%s: basis %.2f%% exceeds the %.2f%% limit -- refusing to trade",
                        symbol, basis.basis * 100, max_basis * 100,
                    )
                    state.target_weight = 0.0
                    continue

            tradeable.append(state)

        # --- pass 2: cap gross exposure across the whole book ---------------
        # Per-symbol caps do not bound the portfolio: four symbols each at the
        # 0.35 limit is 1.4x gross, which on a spot book simply runs out of
        # rial and comes back as InsufficientBalance. Scaling every weight by
        # the same factor preserves relative conviction.
        self._cap_gross(tradeable)

        # --- pass 3: execute ------------------------------------------------
        # The router polls fills with a blocking `time.sleep` and can spend
        # `repost_after_s` (45s) per attempt. Run it off the event loop so the
        # websocket keeps answering Centrifugo's 25s ping.
        for state in tradeable:
            await asyncio.to_thread(self._apply_target, state, equity)

        self.state.save(self.state_path)

    def _cap_gross(self, states: list[SymbolState]) -> None:
        """Scale target weights down so the book respects max_gross_exposure."""
        max_gross = float(self.cfg.risk["max_gross_exposure"])
        gross = sum(abs(s.target_weight) for s in states)
        if gross <= max_gross or gross == 0:
            return

        factor = max_gross / gross
        log.info(
            "gross exposure %.2f exceeds the %.2f limit; scaling every weight by %.3f",
            gross, max_gross, factor,
        )
        for state in states:
            state.target_weight *= factor

    def _apply_target(self, state: SymbolState, equity: float) -> None:
        spec = state.spec
        assert state.book is not None
        price = state.book.mid

        base, _ = _split_symbol(spec.nobitex)
        held = self.broker.balances().get(base, 0.0)
        current_weight = held * price / equity if equity else 0.0

        gap = state.target_weight - current_weight
        min_rebalance = float(self.cfg.execution.get("min_rebalance_weight", 0.0))
        is_state_change = (state.target_weight == 0.0) != (current_weight == 0.0)

        if not is_state_change and abs(gap) < min_rebalance:
            log.debug("%s: gap %.3f inside the no-trade band", spec.nobitex, gap)
            return

        delta_amount = round_to_step(abs(gap) * equity / price, spec.amount_step)
        if delta_amount <= 0:
            return

        side = Side.BUY if gap > 0 else Side.SELL
        if side is Side.SELL:
            delta_amount = min(delta_amount, held)

        log.info(
            "%s: score %+.2f | weight %.3f -> %.3f | %s %.8f",
            spec.nobitex, state.score, current_weight, state.target_weight,
            side.value, delta_amount,
        )
        report = self.router.execute(spec, side, delta_amount)
        if report.filled:
            log.info(
                "%s: filled %.8f @ %s rial (%d attempt(s)%s)",
                spec.nobitex, report.filled, f"{report.avg_price:,.0f}", report.attempts,
                ", crossed" if report.crossed else "",
            )

    def equity_rial(self) -> float:
        """Mark the whole book to the live mid."""
        balances = self.broker.balances()
        total = balances.get("rls", 0.0)
        for symbol, state in self.states.items():
            base, _ = _split_symbol(symbol)
            amount = balances.get(base, 0.0)
            if amount and state.book:
                total += amount * state.book.mid
        usdt = balances.get("usdt", 0.0)
        if usdt and self.fx_state.book:
            total += usdt * self.fx_state.book.mid
        return total

    def stop(self) -> None:
        self._stop.set()


def _split_symbol(symbol: str) -> tuple[str, str]:
    for suffix, quote in (("IRT", "rls"), ("USDT", "usdt")):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)].lower(), quote
    raise ValueError(f"cannot split symbol {symbol!r}")


def _bar_seconds(resolution: str) -> int:
    if resolution.endswith("D"):
        days = int(resolution[:-1]) if resolution[:-1] else 1
        return days * 86400
    return int(resolution) * 60
