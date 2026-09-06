"""Does a slow long book beat holding USDT, and is the parameter robust?

The market maker loses by construction: measured over 16 completed round trips
its median GROSS edge was -5.8 bps, before the 16 bps of fees. A resting bid
only fills when someone sells into it, so a passive book buys into weakness and
its ask sits above a market walking away. On spot, unable to short, that loss is
one-sided.

This measures the opposite posture -- hold a basket, pay the spread once rather
than on every cycle -- against the only benchmark that matters on a rial
account: simply holding USDT, which captures devaluation with no market risk.

Everything is measured in USDT terms so rial devaluation is stripped out of
both sides. What is left is the question actually being asked: does owning
crypto beat owning dollars, after the cost of switching?
"""

from __future__ import annotations

import sys
import time

import pandas as pd
from rich.console import Console
from rich.table import Table

from nbtrend.config import load_config
from nbtrend.data.nobitex_rest import NobitexREST

console = Console()

# One taker fee plus half the median spread, charged on every switch.
SWITCH_COST = 0.0041

DEFAULT_SYMBOLS = ["BTCIRT", "ETHIRT", "XRPIRT", "DOGEIRT",
                   "ADAIRT", "SOLIRT", "LTCIRT", "TRXIRT"]


def load_prices(days: int, symbols: list[str]) -> pd.DataFrame:
    cfg = load_config()
    end = int(time.time())
    start = end - days * 86400
    frames: dict[str, pd.Series] = {}
    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                     api_key=cfg.creds.api_key,
                     api_secret=cfg.creds.api_secret) as api:
        for sym in [*symbols, "USDTIRT"]:
            try:
                df = api.candles(sym, "D", start, end)
            except Exception:
                continue
            if df is not None and len(df):
                frames[sym] = df["close"]
    return pd.DataFrame(frames).dropna()


def equity(basket: pd.Series, usdt_ret: pd.Series, hold: pd.Series) -> pd.Series:
    """Rial equity curve for a signal, charging a cost on every switch."""
    ret = basket.pct_change().fillna(0.0)
    held = hold.shift(1).fillna(False)
    switches = held.ne(held.shift(1)).fillna(False)
    total = ret.where(held, 0.0) + usdt_ret - switches * SWITCH_COST
    return (1.0 + total).cumprod()


def summarise(curve: pd.Series) -> tuple[float, float]:
    total = (curve.iloc[-1] - 1.0) * 100.0
    drawdown = (curve / curve.cummax() - 1.0).min() * 100.0
    return total, drawdown


def main(days: int = 900) -> None:
    px = load_prices(days, DEFAULT_SYMBOLS)
    if px.empty or "USDTIRT" not in px:
        console.print("[red]no price data[/red]")
        return

    usdt = px["USDTIRT"]
    coins = px.drop(columns=["USDTIRT"])
    # In USDT terms: this is crypto's own performance, with rial removed.
    in_usdt = coins.div(usdt, axis=0)
    basket = in_usdt.div(in_usdt.iloc[0]).mean(axis=1)
    usdt_ret = usdt.pct_change().fillna(0.0)

    console.print(f"\n[bold]{len(px)} days[/bold]: "
                  f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    table = Table(title="Strategy versus simply holding dollars")
    for col in ("strategy", "total", "max drawdown", "switches"):
        table.add_column(col, justify="right")

    hold_usdt = (1.0 + usdt_ret).cumprod()
    bench_total, bench_dd = summarise(hold_usdt)
    table.add_row("hold USDT", f"{bench_total:+.1f}%", f"{bench_dd:.1f}%", "0")

    always = pd.Series(True, index=px.index)
    total, dd = summarise(equity(basket, usdt_ret, always))
    table.add_row("hold basket, no filter", f"{total:+.1f}%", f"{dd:.1f}%", "1")

    best = None
    for window in (20, 50, 100):
        signal = (basket > basket.rolling(window).mean()).fillna(False)
        curve = equity(basket, usdt_ret, signal)
        total, dd = summarise(curve)
        switches = int(signal.ne(signal.shift(1)).sum())
        table.add_row(f"basket over its {window}d mean",
                      f"{total:+.1f}%", f"{dd:.1f}%", str(switches))
        if best is None or total > best[1]:
            best = (window, total)
    console.print(table)

    # A parameter that works at exactly one value is fitted to the sample.
    console.print("\n[bold]Sensitivity[/bold] "
                  "[dim]-- a plateau is a signal, a spike is a fit[/dim]")
    sens = Table()
    for col in ("window", "total", "max drawdown"):
        sens.add_column(col, justify="right")
    for window in (60, 80, 100, 120, 150, 200):
        signal = (basket > basket.rolling(window).mean()).fillna(False)
        total, dd = summarise(equity(basket, usdt_ret, signal))
        sens.add_row(f"{window}d", f"{total:+.1f}%", f"{dd:.1f}%")
    console.print(sens)

    # Split sample: an edge present in only one half is not an edge.
    console.print("\n[bold]Split sample[/bold] "
                  "[dim]-- both halves must beat the benchmark[/dim]")
    window = best[0] if best else 100
    signal = (basket > basket.rolling(window).mean()).fillna(False)
    half = len(px) // 2
    split = Table()
    for col in ("half", f"{window}d filter", "hold USDT", "edge"):
        split.add_column(col, justify="right")
    for label, chunk in (("first", slice(0, half)), ("second", slice(half, None))):
        ret = basket.pct_change().fillna(0.0)
        held = signal.shift(1).fillna(False)
        switches = held.ne(held.shift(1)).fillna(False)
        total_ret = ret.where(held, 0.0) + usdt_ret - switches * SWITCH_COST
        curve = (1.0 + total_ret.iloc[chunk]).cumprod()
        bench = (1.0 + usdt_ret.iloc[chunk]).cumprod()
        edge = (curve.iloc[-1] / bench.iloc[-1] - 1.0) * 100.0
        split.add_row(label,
                      f"{(curve.iloc[-1] - 1) * 100:+.1f}%",
                      f"{(bench.iloc[-1] - 1) * 100:+.1f}%",
                      f"[{'green' if edge > 0 else 'red'}]{edge:+.1f}%[/]")
    console.print(split)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900)
