"""Does the short side actually earn anything?

The walk-forward that justified this strategy was long-only. Enabling shorts is
a different bet: it needs the trend signal to be predictive when it points DOWN,
which is not implied by it working when it points up. Crypto in rial terms has a
structural upward drift (rial devaluation), so a short book fights the carry.

This runs each symbol twice -- long-only and long/short -- on identical data and
reports the difference. Leverage is deliberately NOT modelled: leverage scales
both return and volatility, so it cannot turn a negative edge positive. If the
short side does not pay at 1x, it does not pay at 5x either -- it just loses
faster.
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


async def _study(symbols: list[str]) -> None:
    base = load_config()
    feed = DataFeed(base)

    long_cfg = copy.deepcopy(base)
    long_cfg.raw["strategy"]["allow_short"] = False
    short_cfg = copy.deepcopy(base)
    short_cfg.raw["strategy"]["allow_short"] = True

    ppy = bars_per_year(str(base.data["timeframe"]))
    table = Table(title="long-only vs long/short (identical data, 1x)")
    for col in ("symbol", "LO return", "LO Sharpe", "LS return", "LS Sharpe",
                "delta Sharpe", "trades LO/LS"):
        table.add_column(col)

    deltas: list[float] = []
    for symbol in symbols:
        spec = base.symbol(symbol)
        try:
            frame = (await feed.build_dataset(spec, days=None)).frame
        except Exception as exc:
            console.print(f"[yellow]skip {symbol}: {exc}[/yellow]")
            continue
        if frame.empty or len(frame) < 400:
            console.print(f"[yellow]skip {symbol}: only {len(frame)} bars[/yellow]")
            continue

        rows = {}
        for name, cfg in (("LO", long_cfg), ("LS", short_cfg)):
            res = Backtester(cfg).run(frame, symbol, amount_step=spec.amount_step)
            rows[name] = summarise(res, ppy, fx=frame.get("fx"))

        d = rows["LS"].sharpe - rows["LO"].sharpe
        deltas.append(d)
        table.add_row(
            symbol,
            f"{rows['LO'].total_return:+.1%}", f"{rows['LO'].sharpe:.2f}",
            f"{rows['LS'].total_return:+.1%}", f"{rows['LS'].sharpe:.2f}",
            f"[{'green' if d > 0 else 'red'}]{d:+.2f}[/]",
            f"{rows['LO'].num_trades}/{rows['LS'].num_trades}",
        )

    console.print(table)
    if deltas:
        better = sum(1 for d in deltas if d > 0)
        mean = sum(deltas) / len(deltas)
        console.print(
            f"\nshorts improved Sharpe on [bold]{better}/{len(deltas)}[/bold] symbols; "
            f"mean delta [bold]{mean:+.2f}[/bold]"
        )
        verdict = (
            "shorts add measurable edge" if mean > 0.15 and better > len(deltas) * 0.6
            else "shorts do NOT pay for themselves on this evidence"
        )
        console.print(f"verdict: [bold]{verdict}[/bold]")


if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(_study(args))
