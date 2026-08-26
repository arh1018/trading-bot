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
from ..execution.base import Broker, credited_currency
from ..execution.nobitex import NobitexBroker
from ..execution.paper import PaperBroker
from ..execution.router import OrderRouter
from ..units import round_to_step
from .lock import SingleInstanceLock

log = logging.getLogger(__name__)

MAX_STALENESS_S = 120.0
HALT_RESUME_FRACTION = 0.5
"""Resume trading once drawdown recovers to half the stop (hysteresis)."""


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

    STALE_PEAK_FACTOR = 3.0
    """Discard a stored peak more than this multiple above current equity."""

    equity_peak_rial: float = 0.0
    halted: bool = False
    last_bar_ts: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(
        cls,
        path: Path,
        current_equity: float | None = None,
        max_drawdown_stop: float | None = None,
    ) -> RunnerState:
        """Restore persisted state, discarding a peak that cannot belong here.

        `equity_peak_rial` is only meaningful for the account and mode that
        produced it. A paper run ends with a peak of 1e9 (the configured paper
        bankroll); loading that into a live account holding 10M reads as a 99%
        drawdown and trips the kill switch before the first order is placed.
        State is namespaced by mode to prevent that, and this is the backstop
        for a stale file, a withdrawal, or a changed paper bankroll.
        """
        if not path.exists():
            return cls()
        try:
            state = cls(**json.loads(path.read_text()))
        except Exception:
            log.warning("could not read %s; starting with fresh state", path)
            return cls()

        if current_equity and state.equity_peak_rial > current_equity * cls.STALE_PEAK_FACTOR:
            log.warning(
                "stored equity peak %s rial is implausible against current equity %s; "
                "resetting rather than tripping the drawdown stop",
                f"{state.equity_peak_rial:,.0f}", f"{current_equity:,.0f}",
            )
            state.equity_peak_rial = current_equity
            state.halted = False

        # A halt is a circuit breaker, not a latch. Trusting the stored flag
        # means one breach silently disarms the bot forever: every symbol gets
        # target_weight 0.0, so it sells the book every cycle and never buys
        # again, with nothing in the log to say why. Re-derive it from what the
        # account actually looks like now.
        if state.halted:
            drawdown = (
                current_equity / state.equity_peak_rial - 1.0
                if current_equity and state.equity_peak_rial
                else 0.0
            )
            still_breached = (
                max_drawdown_stop is not None and drawdown <= -abs(max_drawdown_stop)
            )
            if still_breached:
                log.error(
                    "resuming halted: drawdown %.1f%% is still past the limit; "
                    "the book stays flat until equity recovers",
                    drawdown * 100,
                )
            else:
                log.warning(
                    "clearing a persisted halt: drawdown %.1f%% is within the limit",
                    drawdown * 100,
                )
                state.halted = False
        return state


class LiveRunner:
    # Class-level default so selection works on a partially built runner
    # (the risk tests exercise _limit_positions without a full __init__).
    _book: set[str] = set()

    def __init__(self, cfg: Config, broker: Broker | None = None):
        self.cfg = cfg
        self.rest = NobitexREST(cfg.rest_url, cfg.creds.api_token,
                    api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret)
        self.feed = DataFeed(cfg)
        self.backtester = Backtester(cfg)

        # Symbols selected last cycle, for incumbency hysteresis.
        self._book: set[str] = set()
        self.states: dict[str, SymbolState] = {
            spec.nobitex: SymbolState(spec=spec) for spec in cfg.enabled_symbols
        }
        self.fx_state = SymbolState(spec=cfg.fx)

        self.broker = broker or self._build_broker()
        self.router = OrderRouter(
            self.broker, cfg.execution, float(cfg.costs["min_order_rial"])
        )

        # Namespaced by mode: a paper peak must never gate a live account.
        self.state_path = Path(f"data/state/runner-{cfg.mode}.json")
        self.state = RunnerState.load(
            self.state_path,
            self._safe_equity(),
            float(cfg.risk["max_drawdown_stop"]),
        )
        self.ws = NobitexWS(cfg.ws_url)
        self._stop = asyncio.Event()
        # One runner per account. Two live runners do not merely duplicate
        # work: their target books differ, so each treats the other's fills as
        # drift and reverses them, paying spread and fees both ways.
        self._lock = SingleInstanceLock(
            Path(f"data/state/{cfg.mode}.lock"), label=cfg.mode
        )

    def _safe_equity(self) -> float | None:
        """Best-effort rial equity for the stale-peak check.

        Runs before the main loop, so it must never raise: a failure here
        should skip the check, not stop the bot starting.

        This must value the POSITIONS, not just the cash. A fully invested
        account holds almost no rial by design, so a cash-only reading made
        every restart look like a catastrophic loss: observed live, a 9,839,505
        peak was discarded against an apparent equity of 29,125. The peak
        recovers on the next cycle, but until it does the 25% drawdown stop is
        measured from a garbage high-water mark -- and over a multi-day run with
        restarts that quietly disarms the main safety net.

        The websocket books are not warm this early, so marks come from REST,
        and only for currencies actually held (a handful of calls).
        """
        try:
            balances = self.broker.balances()
        except Exception:
            log.debug("could not read balances while loading state", exc_info=True)
            return None

        total = balances.get("rls", 0.0)
        for symbol in self.states:
            base, quote = _split_symbol(symbol)
            amount = balances.get(base, 0.0)
            if not amount or quote != "rls":
                continue
            try:
                total += amount * self.rest.orderbook(symbol).mid
            except Exception:
                # An unmarkable holding is better skipped than guessed at; the
                # worst case is the same cash-only reading we had before.
                log.debug("could not mark %s while loading state", symbol, exc_info=True)
        return total or None

    def _build_broker(self) -> Broker:
        if self.cfg.is_live:
            self.cfg.creds.require_token()
            specs = {s.nobitex: s for s in [*self.cfg.enabled_symbols, self.cfg.fx]}
            log.warning("LIVE MODE -- orders will be sent to Nobitex")
            return NobitexBroker(
                self.rest,
                specs,
                settlement_poll_s=float(self.cfg.execution.get("settlement_poll_s", 0.5)),
                settlement_timeout_s=float(self.cfg.execution.get("settlement_timeout_s", 20.0)),
                balance_ttl_s=float(self.cfg.execution.get("balance_ttl_s", 3.0)),
            )

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
    async def run(
        self,
        once: bool = False,
        minutes: float | None = None,
        interval_s: float | None = None,
    ) -> None:
        """Drive the strategy.

        By default the decision clock is the strategy bar, which is what the
        backtest validated. `minutes` bounds the session and `interval_s`
        overrides the cadence -- useful for a supervised live run shorter than
        one bar, where sleeping to the next 4h close would waste the session.
        Recomputing faster than the bar does not change the signal (it is
        derived from closed bars), it just re-checks fills and drift.
        """
        self._wire()
        ws_task = asyncio.create_task(self.ws.run(self._stop))

        deadline = time.time() + minutes * 60 if minutes else None

        # Refuse to start rather than trade against another instance.
        self._lock.acquire()

        try:
            await self._await_books(timeout_s=45)
            while not self._stop.is_set():
                try:
                    await self.rebalance()
                except Exception:
                    log.exception("rebalance failed; will retry next cycle")

                if once:
                    break
                if deadline and time.time() >= deadline:
                    log.info("session limit of %.0f minutes reached", minutes)
                    break

                if interval_s:
                    wait = interval_s
                    if deadline:
                        wait = min(wait, max(0.0, deadline - time.time()))
                    if wait <= 0:
                        break
                    log.info("next cycle in %.0fs", wait)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=wait)
                else:
                    await self._sleep_until_next_bar()
        finally:
            self._stop.set()
            ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ws_task
            self.state.save(self.state_path)
            self.rest.close()
            self._lock.release()

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
        if not self._book:
            # Seed incumbency from what the account actually holds. Without
            # this, the first cycle after any restart has no memory of the book
            # and can churn out positions it opened moments earlier.
            self._book = await asyncio.to_thread(self._held_symbols)
            log.info("seeded incumbency from holdings: %s", ", ".join(sorted(self._book)) or "nothing")
        self.state.equity_peak_rial = max(self.state.equity_peak_rial, equity)
        drawdown = equity / self.state.equity_peak_rial - 1.0 if self.state.equity_peak_rial else 0.0

        stop = float(self.cfg.risk["max_drawdown_stop"])
        if drawdown <= -stop and not self.state.halted:
            log.error("drawdown %.1f%% breached the limit -- flattening", drawdown * 100)
            self.state.halted = True
        elif self.state.halted and drawdown > -stop * HALT_RESUME_FRACTION:
            # Hysteresis, so a halt is not a one-way latch: resume only once
            # equity has recovered well clear of the stop, not the instant it
            # ticks back over the line.
            log.warning("drawdown recovered to %.1f%% -- resuming", drawdown * 100)
            self.state.halted = False

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
                    # Scaled markets (1K_SHIB, 1M_PEPE, ...) quote a bundle of
                    # units against a per-unit global feed. Without this the
                    # basis reads ~99,826% and the market is refused forever.
                    state.spec.multiplier,
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

        # --- pass 2: keep only as many positions as equity can fund ---------
        tradeable = self._limit_positions(tradeable, equity)

        # Gross exposure was already capped inside the selection search above:
        # it evaluates each candidate book *after* capping, which is the only
        # way to know whether a position clears the exchange minimum.

        # --- pass 3b: drop anything the gross cap pushed under the minimum ---
        self._drop_unfundable(tradeable, equity)

        # --- pass 4: execute ------------------------------------------------
        # The router polls fills with a blocking `time.sleep` and can spend
        # `repost_after_s` (45s) per attempt. Run it off the event loop so the
        # websocket keeps answering Centrifugo's 25s ping.
        for state in tradeable:
            await asyncio.to_thread(self._apply_target, state, equity)

        self.state.save(self.state_path)

    def _limit_positions(self, states: list[SymbolState], equity: float) -> list[SymbolState]:
        """Choose the book: the most positions that can all actually be funded.

        Nobitex rejects any IRT order under `min_order_rial` (3,000,000), so a
        wide universe on a small account places nothing -- it fails silently,
        one skip at a time.

        Ranking by conviction alone is not enough. Volatility targeting sizes
        each name differently and `_cap_gross` then scales the whole book, so
        picking the top three and hoping produced one funded position and two
        rejections in a live run. Instead this searches for the largest k whose
        top-k selection *survives* gross capping with every position still
        above the minimum -- the freed weight from a rejected name is exactly
        what lifts the others over the line.

        The universe therefore stays the opportunity set: widening it improves
        selection without needing more capital.
        """
        min_order = float(self.cfg.costs["min_order_rial"])
        max_gross = float(self.cfg.risk["max_gross_exposure"])
        configured = int(self.cfg.risk.get("max_positions", 0))

        wanted = [s for s in states if s.target_weight != 0.0]
        if not wanted or equity <= 0 or min_order <= 0:
            return states

        # Highest conviction first, with a bonus for names already in the book.
        #
        # Pure score ranking makes the last fundable slot flip on noise. A live
        # cycle bought CVX, then the next cycle wanted to evict it for ETH on a
        # score difference of a few hundredths -- paying a full round trip
        # (0.22% fees plus ~0.6% spread) to swap one trending name for another
        # barely-better one. On a small account only ~3 slots are fundable, so
        # that boundary is crossed constantly. An incumbent keeps its slot
        # unless a challenger beats it by a real margin.
        bonus = float(self.cfg.execution.get("incumbent_score_bonus", 0.0))
        wanted.sort(key=lambda s: -(abs(s.score) + (bonus if s.spec.nobitex in self._book else 0.0)))
        log.info(
            "ranking (score, * = incumbent): %s",
            ", ".join(
                f"{s.spec.nobitex} {abs(s.score):+.2f}"
                + ("*" if s.spec.nobitex in self._book else "")
                for s in wanted[:8]
            ),
        )
        ceiling = int((equity * max_gross) // min_order) or 1
        limit = min(configured or ceiling, ceiling, len(wanted))
        original = {id(s): s.target_weight for s in wanted}

        # Greedy by conviction, skipping any name that would break the book.
        #
        # Considering only score-ranked *prefixes* is not enough: the top name
        # may be the volatile one that cannot clear the minimum, which would
        # veto every larger book behind it and fund nothing -- even when a
        # lower-ranked name is perfectly fundable on its own. So a candidate
        # that breaks fundability is skipped, not fatal.
        selected: list[SymbolState] = []
        for candidate in wanted:
            if len(selected) >= limit:
                break

            trial = [*selected, candidate]
            for state in trial:
                state.target_weight = original[id(state)]
            self._cap_gross(trial)

            # The 3,000,000 minimum is a limit on ORDERS, not on holdings.
            # Keeping a position already in the book places no order at all, so
            # an incumbent whose vol-targeted size lands just under the minimum
            # is still perfectly holdable. Testing it as if it had to be opened
            # from flat produced a stalemate live: CVX sized to 2,970,000 was
            # dropped from selection, the slot went to a name there was no cash
            # to buy, and the CVX sell was then itself skipped for being under
            # the minimum -- so every cycle churned intent and traded nothing.
            if all(
                abs(s.target_weight) * equity >= min_order
                or s.spec.nobitex in self._book
                for s in trial
            ):
                selected = trial
            else:
                candidate.target_weight = 0.0
                # Restore the survivors to their last good sizing.
                for state in selected:
                    state.target_weight = original[id(state)]
                self._cap_gross(selected)

        chosen = {id(s) for s in selected}
        for state in wanted:
            if id(state) not in chosen:
                state.target_weight = 0.0

        self._book = {s.spec.nobitex for s in selected}
        held = [s.spec.nobitex for s in selected]
        dropped = [s.spec.nobitex for s in wanted if id(s) not in chosen]
        log.info(
            "equity %s rial funds %d of %d signal(s) at the %s minimum; holding %s%s",
            f"{equity:,.0f}", len(selected), len(wanted), f"{min_order:,.0f}",
            ", ".join(held) or "nothing",
            f"; skipping {len(dropped)} ({', '.join(dropped[:6])}"
            + ("..." if len(dropped) > 6 else "") + ")" if dropped else "",
        )
        return states

    def _drop_unfundable(self, states: list[SymbolState], equity: float) -> None:
        """Remove positions the gross cap pushed below the exchange minimum.

        `_limit_positions` picks how many names an equal split could fund, but
        the weights are not equal: volatility targeting sizes each name
        differently and `_cap_gross` then scales the whole book. The result is
        that a kept name can still land under 3,000,000 rial and be rejected
        at submission -- observed live, where two of three chosen positions
        were skipped one at a time.

        Dropping the smallest offender and re-capping frees its weight for the
        survivors, so the book converges on fewer, fundable positions rather
        than several unfillable ones.
        """
        min_order = float(self.cfg.costs["min_order_rial"])
        if equity <= 0 or min_order <= 0:
            return

        for _ in range(len(states)):
            live = [s for s in states if s.target_weight != 0.0]
            if not live:
                return

            # Same open-vs-hold distinction as in `_limit_positions`: a name we
            # already own needs no order to stay in the book, so a sub-minimum
            # target is not a reason to drop it. Dropping it instead sets a sell
            # that is itself under the minimum and gets skipped -- the book then
            # re-decides the same thing every cycle and never actually trades.
            underfunded = [
                s
                for s in live
                if abs(s.target_weight) * equity < min_order
                and s.spec.nobitex not in self._book
            ]
            if not underfunded:
                return

            victim = min(underfunded, key=lambda s: abs(s.target_weight))
            log.info(
                "%s: %s rial is under the %s minimum after capping; dropping it and "
                "re-spreading across the rest",
                victim.spec.nobitex,
                f"{abs(victim.target_weight) * equity:,.0f}",
                f"{min_order:,.0f}",
            )
            victim.target_weight = 0.0
            self._cap_gross([s for s in states if s.target_weight != 0.0])

    def _cap_gross(self, states: list[SymbolState]) -> None:
        """Scale target weights down so the book respects max_gross_exposure.

        The cap is reduced by a cost allowance first. Fees are charged on top
        of notional, so investing exactly 100% of equity needs 100% + fees in
        cash and overdraws the rial balance -- observed in a shadow test as a
        -0.2% `rls` position, which live would be an InsufficientBalance
        rejection on the last order of the cycle.
        """
        cost_rate = float(self.cfg.costs["taker_fee"]) + float(self.cfg.costs["slippage"])
        max_gross = float(self.cfg.risk["max_gross_exposure"]) / (1.0 + cost_rate)

        gross = sum(abs(s.target_weight) for s in states)
        if gross <= max_gross or gross == 0:
            return

        factor = max_gross / gross
        log.info(
            "gross exposure %.3f exceeds the %.3f cost-adjusted limit; "
            "scaling every weight by %.4f",
            gross, max_gross, factor,
        )
        for state in states:
            state.target_weight *= factor

    def _apply_target(self, state: SymbolState, equity: float) -> None:
        spec = state.spec
        assert state.book is not None
        price = state.book.mid

        base, _ = _split_symbol(spec.nobitex)
        # One snapshot for the whole decision: holding, spendable cash and the
        # settlement baseline all come from it. Re-reading per use is what
        # earned a 429 from /users/wallets/list.
        balances = self.broker.balances()
        held = balances.get(base, 0.0)
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
        else:
            # Never submit a buy the rial balance cannot fund. Belt and braces
            # alongside the cost-adjusted gross cap: rounding, slippage and a
            # moving book can still push the last order of a cycle over.
            cost_rate = float(self.cfg.costs["taker_fee"]) + float(self.cfg.costs["slippage"])
            cash = balances.get("rls", 0.0)
            affordable = max(0.0, cash / (price * (1.0 + cost_rate)))
            if affordable < delta_amount:
                log.info(
                    "%s: trimming buy %.8f -> %.8f, only %s rial of cash",
                    spec.nobitex, delta_amount, affordable, f"{cash:,.0f}",
                )
                delta_amount = round_to_step(affordable, spec.amount_step)
            if delta_amount <= 0:
                return

        log.info(
            "%s: score %+.2f | weight %.3f -> %.3f | %s %.8f",
            spec.nobitex, state.score, current_weight, state.target_weight,
            side.value, delta_amount,
        )
        credited = credited_currency(side, base, "rls")
        baseline = balances.get(credited, 0.0)

        report = self.router.execute(spec, side, delta_amount)

        if report.filled:
            # Nobitex credits the wallet ~2s after acknowledging the fill.
            # Confirm it landed before the loop moves to the next symbol,
            # whose sizing reads this same rial balance.
            expected = (
                report.filled if side is Side.BUY else report.filled * report.avg_price
            )
            self.broker.await_settlement(credited, baseline, expected)

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

    def _held_symbols(self) -> set[str]:
        """Symbols we actually own, for incumbency.

        The floor is a dust threshold, not the order minimum: a position sitting
        just under 3,000,000 rial is still a position we paid to open, and
        treating it as vacant is what let a freshly bought name be evicted on
        the next cycle.
        """
        min_order = float(self.cfg.costs["min_order_rial"]) * 0.25
        balances = self.broker.balances()
        held = set()
        for symbol, state in self.states.items():
            base, _ = _split_symbol(symbol)
            amount = balances.get(base, 0.0)
            if amount and state.book and amount * state.book.mid >= min_order:
                held.add(symbol)
        return held

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
