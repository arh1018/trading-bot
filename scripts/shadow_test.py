"""Shadow test: run the full live stack against real markets, with no real money.

This exercises every component the live path uses -- websocket, orderbook
maintenance, signal derivation, risk sizing, basis interlocks, and the
limit-chase router -- but the broker is `PaperBroker`, so no order ever leaves
the process. Fills are simulated by walking the real orderbook, so reported
slippage is what that size would actually have cost.

    python scripts/shadow_test.py --minutes 15

Two clocks, matching the live runner:
  * decision  -- full signal recompute + rebalance (default every 5 min here,
                 versus one bar in production, so a short run sees several)
  * monitor   -- mark-to-market and interlock checks off the live websocket
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from nbtrend.config import load_config
from nbtrend.data.fx import compute_basis
from nbtrend.live.runner import LiveRunner, _split_symbol

console = Console()


@dataclass
class Observation:
    ts: float
    equity: float
    basis: dict[str, float] = field(default_factory=dict)
    spreads: dict[str, float] = field(default_factory=dict)


async def shadow_test(minutes: float, decision_every_s: float, monitor_every_s: float) -> None:
    cfg = load_config()
    if cfg.is_live:
        raise SystemExit("refusing to shadow test with NBTREND_MODE=live")

    runner = LiveRunner(cfg)
    console.rule(f"[bold]shadow test[/bold] -- {minutes:.0f} min, paper broker, real market data")
    console.print(
        f"universe   : {', '.join(s.nobitex for s in cfg.enabled_symbols)}\n"
        f"timeframe  : {cfg.data['timeframe']}\n"
        f"fees       : maker {cfg.costs['maker_fee']:.4%} / taker {cfg.costs['taker_fee']:.4%}\n"
        f"start cash : {cfg.backtest['initial_equity_rial']:,.0f} rial\n"
    )

    runner._wire()
    ws_task = asyncio.create_task(runner.ws.run(runner._stop))
    await runner._await_books(timeout_s=45)

    start = time.time()
    deadline = start + minutes * 60
    start_equity = runner.equity_rial()

    observations: list[Observation] = []
    decisions = 0
    last_decision = 0.0

    try:
        while time.time() < deadline:
            now = time.time()

            if now - last_decision >= decision_every_s:
                decisions += 1
                console.print(
                    f"[dim]{time.strftime('%H:%M:%S')}[/dim] "
                    f"[bold]decision cycle {decisions}[/bold]"
                )
                try:
                    await runner.rebalance()
                except Exception:
                    logging.getLogger(__name__).exception("rebalance failed")
                last_decision = now

            # Monitor: mark the book and record the interlock inputs.
            equity = runner.equity_rial()
            obs = Observation(ts=now, equity=equity)
            for symbol, state in runner.states.items():
                if state.book:
                    obs.spreads[symbol] = state.book.spread_bps
                if state.book and runner.fx_state.book:
                    dataset_close = getattr(state, "_last_global", None)
                    if dataset_close:
                        obs.basis[symbol] = compute_basis(
                            symbol, state.book.mid, dataset_close, runner.fx_state.book.mid
                        ).basis
                    else:
                        obs.basis[symbol] = state.basis
            observations.append(obs)

            await asyncio.sleep(monitor_every_s)
    finally:
        runner._stop.set()
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task

    _report(runner, observations, start_equity, start, decisions)
    runner.rest.close()


def _report(runner, observations, start_equity, start, decisions) -> None:
    elapsed = (time.time() - start) / 60
    end_equity = observations[-1].equity if observations else start_equity
    broker = runner.broker

    console.rule("[bold]shadow test report[/bold]")
    console.print(
        f"duration        : {elapsed:.1f} min\n"
        f"decision cycles : {decisions}\n"
        f"monitor samples : {len(observations)}\n"
        f"ws connected    : {runner.ws.is_connected}\n"
        f"ws channels     : {len(runner.ws.channels)}\n"
        f"last ws message : {runner.ws.seconds_since_message:.1f}s ago\n"
    )

    fills = getattr(broker, "fills", [])
    if fills:
        table = Table(title="simulated fills (walked against the real book)")
        for col in ("time", "symbol", "side", "amount", "price (rial)", "fee (rial)"):
            table.add_column(col, justify="right")
        for f in fills:
            table.add_row(
                time.strftime("%H:%M:%S", time.localtime(f.ts)),
                f.symbol, f.side.value, f"{f.amount:.8f}",
                f"{f.price:,.0f}", f"{f.fee:,.0f}",
            )
        console.print(table)
        console.print(f"total fees paid : {sum(f.fee for f in fills):,.0f} rial")
    else:
        console.print("[yellow]no fills -- signals produced no tradeable delta[/yellow]")

    balances = broker.balances()
    table = Table(title="final book")
    for col in ("asset", "amount", "mark (rial)", "weight"):
        table.add_column(col, justify="right")
    for asset, amount in sorted(balances.items()):
        if abs(amount) < 1e-12:
            continue
        if asset == "rls":
            mark = amount
        else:
            symbol = next(
                (s for s in runner.states if _split_symbol(s)[0] == asset), None
            )
            state = runner.states.get(symbol) if symbol else None
            mark = amount * state.book.mid if state and state.book else 0.0
        table.add_row(asset, f"{amount:,.8f}".rstrip("0").rstrip("."),
                      f"{mark:,.0f}", f"{mark / end_equity:.1%}" if end_equity else "-")
    console.print(table)

    pnl = end_equity - start_equity
    console.print(
        f"\nstart equity : {start_equity:,.0f} rial\n"
        f"end equity   : {end_equity:,.0f} rial\n"
        f"P&L          : {pnl:+,.0f} rial ({pnl / start_equity:+.3%})"
    )

    if observations:
        equities = [o.equity for o in observations]
        console.print(
            f"equity range : {min(equities):,.0f} .. {max(equities):,.0f} rial"
        )
        table = Table(title="market conditions observed")
        for col in ("symbol", "mean basis", "min basis", "max basis", "mean spread"):
            table.add_column(col, justify="right")
        for symbol in runner.states:
            bases = [o.basis[symbol] for o in observations if symbol in o.basis]
            spreads = [o.spreads[symbol] for o in observations if symbol in o.spreads]
            if not bases:
                continue
            table.add_row(
                symbol,
                f"{sum(bases) / len(bases):+.3%}",
                f"{min(bases):+.3%}", f"{max(bases):+.3%}",
                f"{sum(spreads) / len(spreads):.1f} bps" if spreads else "-",
            )
        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--decision-every", type=float, default=300.0,
                        help="seconds between full signal recomputes")
    parser.add_argument("--monitor-every", type=float, default=15.0,
                        help="seconds between mark-to-market samples")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    for noisy in ("httpx", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    asyncio.run(shadow_test(args.minutes, args.decision_every, args.monitor_every))


if __name__ == "__main__":
    main()
