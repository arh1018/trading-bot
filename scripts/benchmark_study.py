"""Is the strategy beating doing nothing?

Trading is only worth it if it beats the passive alternatives available on the
same account. In a rial-denominated market there are three, and the first
walk-forward hinted the answer: the strategy returned +18.9% in RIAL but
-47.6% in USD terms, meaning the gain was rial devaluation rather than alpha.

Benchmarks, all in rial:

  * strategy      -- trend following, paying real costs
  * buy & hold    -- own the coin for the whole window, one round trip
  * hold USDT     -- capture rial devaluation with no market risk at all
  * hold rial     -- 0% by definition, the thing devaluation erodes

If "hold USDT" wins, the honest conclusion is that the account is being paid
for currency debasement and the trading is subtracting from it.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.table import Table

from nbtrend.backtest.engine import Backtester
from nbtrend.backtest.metrics import max_drawdown, summarise
from nbtrend.config import load_config
from nbtrend.data.feed import DataFeed
from nbtrend.risk.sizing import bars_per_year

console = Console()


async def _run(symbols: list[str]) -> None:
    cfg = load_config()
    cfg.raw["strategy"]["allow_short"] = False
    cfg.raw["risk"]["max_leverage"] = 1.0
    feed = DataFeed(cfg)
    ppy = bars_per_year(str(cfg.data["timeframe"]))

    table = Table(title="strategy vs doing nothing (rial terms, real costs)")
    for col in ("symbol", "strategy", "buy & hold", "hold USDT",
                "strat Sharpe", "strat maxDD", "hold maxDD", "trades", "costs"):
        table.add_column(col)

    wins = 0
    total = 0
    for symbol in symbols:
        spec = cfg.symbol(symbol)
        try:
            frame = (await feed.build_dataset(spec, days=None)).frame
        except Exception as exc:
            console.print(f"[yellow]skip {symbol}: {exc}[/yellow]")
            continue
        if frame.empty or len(frame) < 400:
            continue

        res = Backtester(cfg).run(frame, symbol, amount_step=spec.amount_step)
        m = summarise(res, ppy, fx=frame.get("fx"))

        local = frame["local_close"].ffill().dropna()
        hold = float(local.iloc[-1] / local.iloc[0] - 1.0) if len(local) > 1 else 0.0
        hold_dd = float(max_drawdown(local))

        fx = frame.get("fx")
        usdt = (
            float(fx.ffill().dropna().iloc[-1] / fx.ffill().dropna().iloc[0] - 1.0)
            if fx is not None and fx.notna().sum() > 1
            else float("nan")
        )

        total += 1
        if m.total_return > max(hold, usdt if usdt == usdt else -9):
            wins += 1

        table.add_row(
            symbol,
            f"{m.total_return:+.1%}", f"{hold:+.1%}",
            "n/a" if usdt != usdt else f"{usdt:+.1%}",
            f"{m.sharpe:.2f}", f"{m.max_drawdown:.1%}", f"{hold_dd:.1%}",
            str(m.num_trades), f"{m.costs_pct_of_equity:.1%}",
        )

    console.print(table)
    if total:
        console.print(
            f"\nstrategy beat BOTH passive benchmarks on "
            f"[bold]{wins}/{total}[/bold] symbols"
        )


if __name__ == "__main__":
    asyncio.run(_run(sys.argv[1:]))
