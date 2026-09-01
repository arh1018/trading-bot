"""Fewer, bigger positions -- or more, smaller ones?

The live book holds ~29 names at ~3.4% each, which is not a choice: it is what
`min_order_rial` allows at the current equity. This sweeps the position cap on
the same data and the same signals to find out whether that is also the right
number, or merely the reachable one.

Spot only: unlevered, long-only. No borrowing, no liquidation.
"""

from __future__ import annotations

import asyncio
import copy
import sys

from rich.console import Console
from rich.table import Table

from nbtrend.backtest.portfolio import PortfolioBacktester
from nbtrend.config import load_config
from nbtrend.data.feed import DataFeed

console = Console()

CAPS = (3, 5, 8, 12, 20, 29, 0)  # 0 = uncapped


async def _study(equity: float, limit: int | None) -> None:
    cfg = load_config()
    specs = cfg.enabled_symbols[:limit] if limit else cfg.enabled_symbols
    console.print(f"building {len(specs)} datasets ...")

    datasets = await DataFeed(cfg).build_all(specs)
    frames = {s: d.frame for s, d in datasets.items() if d.frame is not None and not d.frame.empty}
    console.print(f"usable: {len(frames)} symbols\n")

    table = Table(title=f"position-count sweep @ {equity:,.0f} rial (long-only, unlevered)")
    for col in ("max positions", "return", "CAGR", "Sharpe", "max DD",
                "avg held", "trades", "costs"):
        table.add_column(col)

    rows = []
    for cap in CAPS:
        c = copy.deepcopy(cfg)
        c.raw["strategy"]["allow_short"] = False
        c.raw["risk"]["max_leverage"] = 1.0
        try:
            r = PortfolioBacktester(c, max_positions=cap or None).run(frames, initial_equity=equity)
        except Exception as exc:
            console.print(f"[yellow]cap={cap}: {exc}[/yellow]")
            continue
        rows.append((cap, r))
        table.add_row(
            "uncapped" if cap == 0 else str(cap),
            f"{r['total_return']:+.1%}", f"{r['cagr']:+.1%}", f"{r['sharpe']:.2f}",
            f"{r['max_drawdown']:.1%}", f"{r['avg_positions']:.1f}",
            str(r["trades"]), f"{r['costs_pct']:.1%}",
        )

    console.print(table)
    if rows:
        best = max(rows, key=lambda kv: kv[1]["sharpe"])
        cur = next((r for c, r in rows if c == 29), None)
        label = "uncapped" if best[0] == 0 else str(best[0])
        console.print(f"\nbest Sharpe at cap [bold]{label}[/bold]: {best[1]['sharpe']:.2f}")
        if cur:
            d = best[1]["sharpe"] - cur["sharpe"]
            console.print(
                f"vs the current ~29-position book ({cur['sharpe']:.2f}): [bold]{d:+.2f}[/bold] Sharpe, "
                f"{best[1]['total_return'] - cur['total_return']:+.1%} return"
            )


if __name__ == "__main__":
    eq = float(sys.argv[1]) if len(sys.argv) > 1 else 90_000_000.0
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(_study(eq, lim))
