"""What does leverage actually buy, and what does it cost?

Leverage scales both return and volatility, so Sharpe is roughly invariant to
it -- the interesting columns are DRAWDOWN and liquidations, not return.

Two effects the arithmetic alone does not show, and which this measures:

  * Drawdown does not scale linearly once liquidation exists. A 2x book that
    gets liquidated at the bottom does not recover when the price does; the
    position is gone.
  * Costs scale with gross exposure, so a 2x book pays 2x the fees on the same
    signal. At ~147bps a round trip that is a real drag, not a rounding error.

Long-only. The short study already showed shorts lose money here, so mixing the
two would confound the question.
"""

from __future__ import annotations

import asyncio
import copy
import sys

from rich.console import Console
from rich.table import Table

from nbtrend.backtest.engine import Backtester
from nbtrend.backtest.metrics import summarise
from nbtrend.config import load_config
from nbtrend.data.feed import DataFeed
from nbtrend.risk.sizing import bars_per_year

console = Console()

LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)


async def _study(symbols: list[str]) -> None:
    base = load_config()
    feed = DataFeed(base)
    ppy = bars_per_year(str(base.data["timeframe"]))
    base_weight = float(base.risk["max_weight_per_symbol"])

    frames = {}
    for symbol in symbols:
        spec = base.symbol(symbol)
        try:
            frame = (await feed.build_dataset(spec, days=None)).frame
        except Exception as exc:
            console.print(f"[yellow]skip {symbol}: {exc}[/yellow]")
            continue
        if not frame.empty and len(frame) >= 400:
            frames[symbol] = (spec, frame)

    agg: dict[float, list] = {lev: [] for lev in LEVERAGES}

    for symbol, (spec, frame) in frames.items():
        table = Table(title=f"{symbol} -- leverage sweep (long-only)")
        for col in ("lev", "return", "Sharpe", "max DD", "Calmar", "costs", "liquidations"):
            table.add_column(col)

        for lev in LEVERAGES:
            cfg = copy.deepcopy(base)
            cfg.raw["strategy"]["allow_short"] = False
            cfg.raw["risk"]["max_leverage"] = lev
            # Scale the per-symbol cap with leverage: same signal, bigger book.
            cfg.raw["risk"]["max_weight_per_symbol"] = base_weight * lev

            res = Backtester(cfg).run(frame, symbol, amount_step=spec.amount_step)
            m = summarise(res, ppy, fx=frame.get("fx"))
            liq = sum(1 for t in res.trades if t.exit_reason == "liquidation")
            agg[lev].append((m.sharpe, m.max_drawdown, m.total_return, liq))

            table.add_row(
                f"{lev:g}x", f"{m.total_return:+.1%}", f"{m.sharpe:.2f}",
                f"{m.max_drawdown:.1%}", f"{m.calmar:.2f}",
                f"{m.costs_pct_of_equity:.1%}",
                f"[red]{liq}[/red]" if liq else "0",
            )
        console.print(table)

    summary = Table(title="across all symbols")
    for col in ("lev", "mean Sharpe", "mean return", "worst DD", "total liquidations"):
        summary.add_column(col)
    for lev in LEVERAGES:
        rows = agg[lev]
        if not rows:
            continue
        summary.add_row(
            f"{lev:g}x",
            f"{sum(r[0] for r in rows) / len(rows):.2f}",
            f"{sum(r[2] for r in rows) / len(rows):+.1%}",
            f"{min(r[1] for r in rows):.1%}",
            f"{sum(r[3] for r in rows)}",
        )
    console.print(summary)


if __name__ == "__main__":
    asyncio.run(_study(sys.argv[1:]))
