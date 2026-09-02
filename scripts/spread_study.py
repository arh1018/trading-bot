"""Is the bid-ask spread wide enough to trade against the fee?

The arithmetic that decides it, before any strategy:

    buy at the bid, sell at the ask -> you pay the fee TWICE
    breakeven spread = maker_fee + maker_fee

Both legs of a market-making round trip REST, so the cost is maker+maker -- not
maker+taker, which is what a strategy that crosses the spread pays. This
account's tiers: IRT maker 0.08% / taker 0.10%, USDT maker 0.06% / taker 0.09%.
Breakeven is therefore 16 bps on IRT markets and 12 bps on USDT ones. A market
quoting tighter than that cannot be made profitably no matter how good the
execution is -- the fee eats the whole capture and then some.

Beyond breakeven there are three costs this samples but cannot fully price:

  * ADVERSE SELECTION. A resting bid fills when the market is falling and a
    resting ask fills when it is rising, so the fills you get are systematically
    the ones you did not want. The quoted spread is the gross prize; adverse
    selection is what is taken back.
  * INVENTORY RISK. Only one side filling leaves a position, and holding it is
    a directional bet nobody chose.
  * THE ORDER RATE LIMIT. Nobitex allows 300 order placements per 10 minutes,
    shared across spot and margin. Quoting two sides on N symbols and requoting
    every R seconds costs 2*N*(600/R) placements per window -- this is usually
    the binding constraint, and it is reported below.

Samples the book repeatedly so the number is a distribution, not one lucky tick.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

from rich.console import Console
from rich.table import Table

from nbtrend.config import load_config
from nbtrend.data.nobitex_rest import NobitexREST

console = Console()


async def _sample(symbols: list[str], rounds: int, pause: float) -> None:
    cfg = load_config()
    # Fees differ by QUOTE currency and by side, and for market making both
    # legs rest passively -- so the cost is maker+maker, not maker+taker.
    # Account tiers: IRT maker 0.08%/taker 0.10%; USDT maker 0.06%/taker 0.09%.
    irt_maker = float(cfg.costs.get("maker_fee_irt", 0.0008))
    usdt_maker = float(cfg.costs.get("maker_fee_usdt", 0.0006))

    def breakeven_for(symbol: str) -> float:
        mk = usdt_maker if symbol.upper().endswith("USDT") else irt_maker
        return 2 * mk * 10_000

    breakeven_bps = 2 * irt_maker * 10_000
    min_order = float(cfg.costs["min_order_rial"])

    console.print(
        f"IRT  maker {irt_maker:.4%} x2 -> [bold]breakeven {2 * irt_maker * 10_000:.1f} bps[/bold]\n"
        f"USDT maker {usdt_maker:.4%} x2 -> [bold]breakeven {2 * usdt_maker * 10_000:.1f} bps[/bold]\n"
    )

    samples: dict[str, list[float]] = {s: [] for s in symbols}
    depth: dict[str, list[float]] = {s: [] for s in symbols}

    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                     api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        for i in range(rounds):
            for sym in symbols:
                try:
                    book = api.full_orderbook(sym)
                    bids, asks = book["bids"], book["asks"]
                    if not bids or not asks:
                        continue
                    bid, ask = bids[0][0], asks[0][0]
                    mid = (bid + ask) / 2
                    if mid <= 0 or ask <= bid:
                        continue
                    samples[sym].append((ask - bid) / mid * 10_000)
                    # Rial available at the touch -- a spread you cannot fill
                    # any size into is not a tradeable spread.
                    depth[sym].append(min(bids[0][0] * bids[0][1], asks[0][0] * asks[0][1]))
                except Exception:
                    continue
            if i < rounds - 1:
                await asyncio.sleep(pause)

    table = Table(title=f"live spread vs {breakeven_bps:.0f} bps breakeven ({rounds} samples)")
    for col in ("symbol", "median bps", "breakeven", "p25 bps", "net edge bps",
                "touch depth (rial)", "verdict"):
        table.add_column(col)

    viable = []
    for sym in symbols:
        xs = samples[sym]
        if len(xs) < max(2, rounds // 2):
            continue
        med = statistics.median(xs)
        p25 = statistics.quantiles(xs, n=4)[0] if len(xs) >= 4 else min(xs)
        d = statistics.median(depth[sym]) if depth[sym] else 0.0
        be = breakeven_for(sym)
        net = med - be
        ok = net > 0 and d >= min_order
        if ok:
            viable.append((net, sym, med, d))
        table.add_row(
            sym, f"{med:.1f}", f"{be:.0f}", f"{p25:.1f}",
            f"[{'green' if net > 0 else 'red'}]{net:+.1f}[/]",
            f"{d:,.0f}",
            "[green]viable[/]" if ok else ("[yellow]thin[/]" if net > 0 else "[red]fee-eaten[/]"),
        )
    console.print(table)

    console.print(
        f"\n[bold]{len(viable)}/{len(symbols)}[/bold] markets quote wider than the fee "
        f"AND show at least {min_order:,.0f} rial at the touch"
    )

    # The constraint that usually decides feasibility.
    console.print("\n[bold]order budget[/bold] (300 placements / 10 min, shared with spot):")
    for requote_s in (10, 30, 60, 120):
        per_window = 600 / requote_s
        max_symbols = 300 / (2 * per_window)
        console.print(
            f"  requote every {requote_s:>3}s -> {2 * per_window:>5.0f} orders/symbol/10min "
            f"-> at most [bold]{max_symbols:.1f}[/bold] symbols quoted two-sided"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    rounds = int(args[0]) if args and args[0].isdigit() else 5
    syms = [a for a in args if not a.isdigit()]
    if not syms:
        syms = [s.nobitex for s in load_config().enabled_symbols[:20]]
    start = time.time()
    asyncio.run(_sample(syms, rounds=rounds, pause=3.0))
    console.print(f"\nsampled in {time.time() - start:.0f}s")
