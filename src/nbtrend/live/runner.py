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
    margin = None
    _short_warned = False
    _short_notional_used = 0.0
    _pending_shorts: dict[str, str] = {}
    _last_equity: float = 0.0
    _suspect_equity: float = 0.0
    MAX_EQUITY_JUMP = 5.0
    """Refuse to size positions from an equity reading this many times the
    previous cycle's -- no real account moves that far in one interval."""

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
        # Margin is opt-in and stays None unless explicitly enabled, so a
        # short signal has nothing to execute against and gets flattened.
        self._short_warned = False
        self._short_notional_used = 0.0
        # symbol -> clientOrderId of a short order still working.
        self._pending_shorts: dict[str, str] = {}
        self._last_equity = 0.0
        self._suspect_equity = 0.0
        self.margin = self._build_margin_broker()
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
            # In-flight short tracking is in memory, so a restart forgets what
            # is resting in the book and would open duplicates on top of it.
            # Clear it once, up front; anything still wanted is re-opened at
            # the current price on the first cycle.
            if self.margin is not None:
                await asyncio.to_thread(self.margin.cancel_working_orders)

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

        # Equity sanity gate.
        #
        # One live cycle read 3,725,744,251 rial against a real account of
        # ~88,000,000 -- 42x -- and funded 56 of 56 signals on the strength of
        # it. REST could not reproduce it, so the cause is a bad websocket book
        # mid on some symbol. Equity feeds position SIZE, so a reading like
        # that is not a cosmetic log error: it decides how much money to spend.
        # An account cannot plausibly change by this much in one cycle, so
        # treat it as bad data and skip rather than trade on it.
        if self._last_equity and equity > self._last_equity * self.MAX_EQUITY_JUMP:
            # A jump this large is either bad data or a real deposit, and the
            # two are told apart by whether it PERSISTS. A glitched book mid
            # will not reproduce next cycle; a deposit will. So the first
            # sighting is refused and remembered, and a second, corroborating
            # reading is accepted.
            #
            # Without that second chance this would latch: rejecting without
            # updating the baseline means a genuine deposit is refused forever.
            # A 9,840,000 -> 29,700,000 deposit really happened on this account.
            corroborated = (
                self._suspect_equity
                and abs(equity - self._suspect_equity) <= self._suspect_equity * 0.10
            )
            if not corroborated:
                self._suspect_equity = equity
                log.error(
                    "equity %s rial is %.1fx last cycle's %s -- skipping this cycle "
                    "rather than sizing positions from it; will accept it if the "
                    "next cycle agrees",
                    f"{equity:,.0f}", equity / self._last_equity, f"{self._last_equity:,.0f}",
                )
                return
            log.warning(
                "equity %s rial confirmed across two cycles; treating the jump as real",
                f"{equity:,.0f}",
            )

        self._suspect_equity = 0.0
        self._last_equity = equity
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

            # A negative weight means SHORT, which spot cannot express: the
            # runner sizes positions as wallet balances, so a negative target
            # would try to sell more than is held. Shorts require margin, so
            # unless margin is enabled AND this market allows selling, a short
            # signal is flattened to cash rather than acted on.
            if state.target_weight < 0 and not self._can_short(symbol):
                if not self._short_warned:
                    log.warning(
                        "shorts are signalled but margin is off (or the market "
                        "forbids selling); treating every short as flat"
                    )
                    self._short_warned = True
                state.target_weight = 0.0

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

        # Cash held back from the strategy. Sizing runs on INVESTABLE equity,
        # not total equity: if the reserve stayed in the number the book would
        # target gross 1.0 of everything and spend the reserve on the next dip.
        # `equity` itself is left intact so the drawdown stop and the equity
        # peak still measure the real account.
        investable = max(0.0, equity - self._cash_reserve())
        # Short budget is per-cycle; reset before any target is applied.
        self._short_notional_used = 0.0
        self._reconcile_pending_shorts()

        # --- pass 2: keep only as many positions as equity can fund ---------
        tradeable = self._limit_positions(tradeable, investable)

        # Gross exposure was already capped inside the selection search above:
        # it evaluates each candidate book *after* capping, which is the only
        # way to know whether a position clears the exchange minimum.

        # --- pass 3b: drop anything the gross cap pushed under the minimum ---
        self._drop_unfundable(tradeable, investable)

        # --- pass 4: execute ------------------------------------------------
        # The router polls fills with a blocking `time.sleep` and can spend
        # `repost_after_s` (45s) per attempt. Run it off the event loop so the
        # websocket keeps answering Centrifugo's 25s ping.
        # --- pass 3c: park idle cash in USDT, not rial ---------------------
        #
        # Rial is not a neutral resting place, it is a losing position. Over the
        # backtest window holding USDT returned +249.9% in rial terms while the
        # strategy returned +2.9% to +76.3% -- the rial devalued ~71% against
        # the dollar, and every hour the book sits in rial it pays that.
        #
        # `fx_floor_weight` was in the config describing exactly this ("the rial
        # leg is a one-way carry") but was parsed and never used, so the
        # safeguard did not exist. The unallocated part of the book is now a
        # USDT position instead of idle rial.
        fx_target = self._fx_target_weight(tradeable)
        self.fx_state.target_weight = fx_target
        if fx_target > 0 and self.fx_state.tradeable:
            tradeable = [*tradeable, self.fx_state]

        # SELLS FIRST, then buys.
        #
        # The book is normally fully invested, so a rotation is "sell A to
        # afford B". Executing in arbitrary order means B's buy is attempted
        # while the cash is still tied up in A -- it gets trimmed to whatever
        # rial happens to be spare, lands under the 3,000,000 minimum, and is
        # skipped. The rotation then needs a second cycle, and only completes
        # if the signal still holds. Doing every reduction first makes the
        # proceeds available to the buys in the SAME cycle.
        for state in self._sells_before_buys(tradeable, investable):
            await asyncio.to_thread(self._apply_target, state, investable)

        # Incumbency must reflect what we actually HOLD, not what we chose.
        # Selection runs in pass 2 and execution in pass 4, and most selected
        # names never get bought -- their order lands under the exchange
        # minimum and is skipped. Recording intent meant a name bought on the
        # last of the cash was not an incumbent next cycle, so it was re-judged
        # as a fresh candidate and sold straight back: live, BCH was bought at
        # 544,754,000 and sold at 543,255,000 one cycle later, paying a round
        # trip for nothing. Re-reading holdings costs one balances call, which
        # the broker caches anyway.
        try:
            self._book = await asyncio.to_thread(self._held_symbols)
        except Exception:
            # Keep the previous book rather than dropping to empty: an empty
            # book means nothing is an incumbent, which is exactly the churn
            # this guards against.
            log.debug("could not refresh incumbency from holdings", exc_info=True)

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
            # Quiet inside the search: each trial book is a hypothesis, not a
            # decision, and announcing every one of them is what made a cycle
            # unreadable. The book that survives is summarised once below.
            self._cap_gross(trial, quiet=True)

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
                # Restore the survivors to their last good sizing. No re-cap is
                # needed: `selected` was already capped when it was accepted,
                # and `_cap_gross` is idempotent on an in-limit book -- calling
                # it again only re-emitted the same scaling line. With 25
                # incumbents that turned one cycle into ~350 log lines
                # oscillating between two fixed points.
                for state in selected:
                    state.target_weight = original[id(state)]
                self._restore_cap(selected)

        chosen = {id(s) for s in selected}
        for state in wanted:
            if id(state) not in chosen:
                state.target_weight = 0.0

        final_gross = sum(abs(s.target_weight) for s in selected)
        if final_gross:
            log.info(
                "gross exposure settled at %.3f across %d name(s)",
                final_gross, len(selected),
            )
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

    def _restore_cap(self, states: list[SymbolState]) -> None:
        """Re-apply the gross cap without logging.

        Used on the rejection path of the selection search, where the book
        being restored was already capped when it was accepted. The scaling is
        real but says nothing new, and at 25+ incumbents re-announcing it once
        per rejected candidate buried the cycle in identical lines.
        """
        self._cap_gross(states, quiet=True)

    def _cap_gross(self, states: list[SymbolState], quiet: bool = False) -> None:
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
        log.log(
            logging.DEBUG if quiet else logging.INFO,
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

        # A short is a MARGIN position, not a negative wallet balance, so it
        # cannot go through the spot path below -- that path clamps every sell
        # to what is held (`min(delta_amount, held)`) and would silently turn
        # a short signal into "go flat".
        if self.margin is not None:
            existing_short = self._open_short(spec.nobitex)
            if state.target_weight < 0:
                self._apply_short_target(state, equity, existing_short)
                return
            if existing_short is not None:
                # Signal has turned non-negative: close the short before the
                # spot path considers buying.
                self._close_short(existing_short, price)

        base, _ = _split_symbol(spec.nobitex)
        # One snapshot for the whole decision: holding, spendable cash and the
        # settlement baseline all come from it. Re-reading per use is what
        # earned a 429 from /users/wallets/list.
        balances = self.broker.balances()
        held = balances.get(base, 0.0)
        current_weight = held * price / equity if equity else 0.0

        gap = state.target_weight - current_weight
        min_rebalance = self._rebalance_band(equity)
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
            # Spendable cash excludes the reserve. Sizing already works on
            # investable equity, but rounding and a moving book can still push
            # the last order of a cycle over -- and the whole point of the
            # reserve is that it is there when it is wanted.
            cash = max(0.0, balances.get("rls", 0.0) - self._cash_reserve())
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

        # The MARGIN wallet is a separate balance that `balances()` does not
        # see. Moving 6,558,554 rial into it read as an instant -8% loss:
        # sizing shrank and the drawdown stop started measuring against a peak
        # that included money the account still had. Open positions count too,
        # as posted collateral plus unrealised PNL -- that collateral is locked
        # out of the wallet's free balance, so it would otherwise vanish twice.
        total += self._margin_equity()
        return total

    def _margin_equity(self) -> float:
        """Collateral in the margin wallet plus the value of open positions.

        Never raises: an unreadable margin wallet degrades to "spot only",
        which understates equity but keeps the loop alive.
        """
        if self.margin is None:
            return 0.0
        total = 0.0
        try:
            # Go through the broker's CACHED collateral read, not a fresh
            # `/users/wallets/list?type=margin`. That endpoint is shared with
            # the spot balance reads and is rate limited per IP: an uncached
            # call here every cycle earned a 429 that aborted the whole
            # rebalance, and shortening the cycle interval multiplies it.
            total += float(self.margin.collateral())
        except Exception:
            log.debug("could not read the margin wallet for equity", exc_info=True)
        try:
            for p in self.margin.positions():
                total += float(p.collateral) + float(p.unrealized_pnl)
        except Exception:
            log.debug("could not read positions for equity", exc_info=True)
        return total

    COLLATERAL_SAFETY = 0.80
    """Use at most this share of margin collateral, so the last short is not
    opened at exactly 100% of margin where any adverse tick is a margin call."""

    def _short_budget(self, equity: float, leverage: float) -> float:
        """Maximum total short NOTIONAL allowed right now."""
        margin_cfg = self.cfg.raw.get("margin", {})
        strategy_cap = float(margin_cfg.get("max_short_gross", 0.0) or 0.0) * equity
        try:
            collateral = self.margin.collateral() if self.margin else 0.0
        except Exception:
            collateral = 0.0
        exchange_cap = collateral * max(1.0, leverage) * self.COLLATERAL_SAFETY
        if strategy_cap <= 0:
            return exchange_cap
        return min(strategy_cap, exchange_cap)

    def _fx_target_weight(self, tradeable: list[SymbolState]) -> float:
        """How much of the book should sit in USDT rather than rial.

        Whatever the crypto sleeve does not claim, subject to a floor. Held to
        a sane range: never negative, never more than the whole book, and the
        cash reserve is deliberately NOT included -- that rial is being kept
        aside deliberately (to fund the margin wallet by hand) and converting
        it would defeat the point.
        """
        floor = float(self.cfg.risk.get("fx_floor_weight", 0.0) or 0.0)
        crypto_gross = sum(abs(s.target_weight) for s in tradeable)
        leftover = max(0.0, 1.0 - crypto_gross)
        return max(0.0, min(1.0, max(floor, leftover)))

    def _sells_before_buys(
        self, states: list[SymbolState], equity: float
    ) -> list[SymbolState]:
        """Order execution so every reduction happens before any purchase.

        Uses ONE balance snapshot: the broker caches for `balance_ttl_s`, but
        the point is a consistent view, not just fewer calls -- classifying
        some symbols against pre-trade balances and others against post-trade
        ones would put buys back ahead of sells.
        """
        try:
            balances = self.broker.balances()
        except Exception:
            log.warning("could not read balances for ordering; using list order", exc_info=True)
            return states

        def reduces(state: SymbolState) -> bool:
            if state.book is None or equity <= 0:
                return False
            base, _ = _split_symbol(state.spec.nobitex)
            current_w = balances.get(base, 0.0) * state.book.mid / equity
            return state.target_weight < current_w

        # Stable sort: reductions keep their relative order, and so do buys,
        # which preserves the conviction ranking within each group.
        return sorted(states, key=lambda s: 0 if reduces(s) else 1)

    BAND_HEADROOM = 1.10
    """Keep the no-trade band this far above the exchange minimum, so rounding
    and a moving price cannot drop a band-passing trade back under it."""

    def _rebalance_band(self, equity: float) -> float:
        """No-trade band as a fraction of equity, never below the exchange floor.

        The band is configured as a FRACTION but `min_order_rial` is a fixed
        RIAL amount, so the two drift apart as equity moves. Configured at 0.035
        when equity was 90,000,000 (a 3,150,000 trade, comfortably over the
        3,000,000 minimum), it became 2,901,853 once equity fell to 82,910,076
        -- under the minimum. Every trade then passed the band and was rejected
        by the exchange, and the bot cycled for 17 hours without placing an
        order.

        Deriving the floor from current equity makes that self-correcting
        instead of a number that silently expires.
        """
        configured = float(self.cfg.execution.get("min_rebalance_weight", 0.0))
        if equity <= 0:
            return configured
        floor = float(self.cfg.costs["min_order_rial"]) / equity * self.BAND_HEADROOM
        return max(configured, floor)

    def _cash_reserve(self) -> float:
        """Rial deliberately kept out of the strategy's hands.

        Used to park cash the operator needs for something else -- funding the
        margin wallet, for instance, which cannot be done from the API while
        the key lacks the transfer scope. Without this the bot redeploys idle
        rial into spot positions on the next cycle and the money is never there
        when it is wanted.
        """
        return float(self.cfg.execution.get("cash_reserve_rial", 0.0) or 0.0)

    def _reconcile_pending_shorts(self) -> None:
        """Drop tracked short orders that are no longer working.

        An order leaves this set when it fills, is cancelled, or cannot be
        found. Anything still Active blocks a second order on that symbol,
        which is what stops the book stacking duplicates while a limit rests.
        """
        if not self._pending_shorts:
            return
        for symbol, coid in list(self._pending_shorts.items()):
            try:
                order = self.rest.order_status(client_order_id=coid)
            except Exception:
                # Unknown state: forget it rather than blocking the symbol
                # forever. A duplicate is caught next cycle by position_for.
                log.debug("could not read short order %s for %s", coid, symbol, exc_info=True)
                self._pending_shorts.pop(symbol, None)
                continue
            if str(order.status.value).lower() not in ("new", "active", "partial"):
                self._pending_shorts.pop(symbol, None)

    def _open_short(self, symbol: str):
        """The open short position for this symbol, if any. Never raises."""
        if self.margin is None:
            return None
        try:
            p = self.margin.position_for(symbol)
        except Exception:
            log.warning("could not read positions for %s", symbol, exc_info=True)
            return None
        from ..core.types import PositionSide

        return p if p is not None and p.side is PositionSide.SHORT else None

    def _close_short(self, position, price: float | None) -> None:
        try:
            self.margin.close_position(position, price=price)
        except Exception:
            log.exception("could not close short %s on %s", position.id, position.symbol)

    def _apply_short_target(self, state: SymbolState, equity: float, existing) -> None:
        """Open, resize or close a short so it matches `target_weight`.

        Sizing note: `target_weight` is a fraction of EQUITY, and that is the
        exposure we want -- the leverage only decides how much collateral is
        posted against it. Multiplying the exposure by leverage as well would
        take a 5x setting to 5x the intended risk.
        """
        spec = state.spec
        symbol = spec.nobitex
        price = state.book.mid
        min_order = float(self.cfg.costs["min_order_rial"])
        margin_cfg = self.cfg.raw.get("margin", {})
        leverage = float(margin_cfg.get("max_leverage", 1.0))

        # Size multiplier for shorts. Vol targeting spreads the book thinly
        # across every signal, which left each short at ~3.7M notional -- barely
        # over the 3,000,000 exchange minimum, so fees and spread ate a large
        # share of any move. Scaling here concentrates the SAME short budget
        # into fewer, larger positions rather than adding exposure: the budget
        # ceilings below still bind, so this cannot raise total short risk
        # above `max_short_gross` or what the collateral supports.
        size_multiplier = float(margin_cfg.get("short_size_multiplier", 1.0) or 1.0)
        wanted_notional = abs(state.target_weight) * equity * max(1.0, size_multiplier)

        # Two independent ceilings on the short book, whichever binds first.
        #
        #   * `max_short_gross` -- a strategy limit, as a fraction of equity.
        #   * collateral x leverage -- a hard exchange limit. Position size is
        #     computed from total equity (~88M) while collateral is a small
        #     separate wallet (~6.5M), so without this the book cheerfully
        #     sizes shorts it cannot post margin for and every one comes back
        #     InsufficientBalance, burning the shared 300-per-10-minutes order
        #     budget and blocking SPOT trading with it.
        #
        # A safety factor keeps the last position from sitting at exactly 100%
        # of collateral, where any adverse tick is an immediate margin call.
        budget = self._short_budget(equity, leverage)
        remaining = max(0.0, budget - self._short_notional_used)
        if wanted_notional > remaining:
            if remaining < min_order:
                log.info(
                    "%s: short budget exhausted (%s of %s rial used); skipping",
                    symbol, f"{self._short_notional_used:,.0f}", f"{budget:,.0f}",
                )
                return
            log.info(
                "%s: trimming short %s -> %s rial to fit the budget",
                symbol, f"{wanted_notional:,.0f}", f"{remaining:,.0f}",
            )
            wanted_notional = remaining

        if wanted_notional < min_order:
            if existing is not None:
                log.info("%s: short target below the minimum; closing", symbol)
                self._close_short(existing, price)
            return

        if existing is not None:
            # Liquidation proximity is checked every cycle: it happens at the
            # exchange between our decisions, so the drawdown stop never sees it.
            if self.margin.at_risk(existing, price):
                log.error(
                    "%s: short %s is within %.0f%% of liquidation -- closing",
                    symbol, existing.id, self.margin.min_liquidation_distance * 100,
                )
                self._close_short(existing, price)
                return

            held_notional = abs(existing.liability) * price
            gap = (wanted_notional - held_notional) / equity if equity else 0.0
            if abs(gap) < float(self.cfg.execution.get("min_rebalance_weight", 0.0)):
                return
            if gap < 0:
                units = round_to_step(abs(gap) * equity / price, spec.amount_step)
                if units * price >= min_order:
                    log.info("%s: trimming short by %.8f", symbol, units)
                    try:
                        self.margin.close_position(existing, amount=units, price=price)
                    except Exception:
                        log.exception("could not trim short on %s", symbol)
            return

        units = round_to_step(wanted_notional / price, spec.amount_step)
        if units <= 0 or units * price < min_order:
            return

        # A margin order that has not filled yet is NOT a position, so
        # `position_for` cannot see it. Opening again on the next cycle stacked
        # three APTIRT and three FILIRT sells in six minutes -- 3x the intended
        # short exposure the moment they filled. Track what is in flight by the
        # clientOrderId we generate, since the order list returns `"id": null`
        # and a display name rather than a symbol.
        if symbol in self._pending_shorts:
            log.info("%s: a short order is already working; not stacking another", symbol)
            return

        log.info(
            "%s: score %+.2f | weight %.3f | OPEN SHORT %.8f at %gx",
            symbol, state.score, state.target_weight, units, leverage,
        )
        try:
            order = self.margin.open_position(
                symbol, Side.SELL, units, price=price, leverage=leverage
            )
            self._short_notional_used += units * price
            if order.client_order_id:
                self._pending_shorts[symbol] = order.client_order_id
        except Exception:
            log.exception("could not open short on %s", symbol)

    def _build_margin_broker(self):
        """Only in live mode, only when explicitly enabled, never in paper."""
        margin_cfg = self.cfg.raw.get("margin", {})
        if not margin_cfg.get("enabled") or not self.cfg.is_live:
            return None

        from ..execution.margin import MarginBroker

        specs = {s.nobitex: s for s in [*self.cfg.enabled_symbols, self.cfg.fx]}
        log.warning(
            "MARGIN ENABLED -- leverage up to %gx. Positions can be LIQUIDATED "
            "between cycles; that loss is permanent.",
            float(margin_cfg.get("max_leverage", 1.0)),
        )
        return MarginBroker(
            self.rest,
            specs,
            max_leverage=float(margin_cfg.get("max_leverage", 1.0)),
            min_liquidation_distance=float(margin_cfg.get("min_liquidation_distance", 0.15)),
        )

    def _can_short(self, symbol: str) -> bool:
        """Shorting needs margin enabled AND a market that permits selling.

        Deliberately fails closed: any error reading margin capability means
        no short, because the failure mode of guessing wrong is a spot sell
        order for units the account does not own.
        """
        if self.margin is None:
            return False
        try:
            # Require collateral for a whole minimum order, or the short is
            # rejected `InsufficientBalance` and wastes a slot in the
            # 300-per-10-minutes budget -- ~50 times a cycle.
            return self.margin.can_short(
                symbol, min_collateral=float(self.cfg.costs["min_order_rial"])
            )
        except Exception:
            log.debug("could not read margin capability for %s", symbol, exc_info=True)
            return False

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
