"""P&L, decomposed. Not cash flow, not equity change -- actual profit.

Three numbers get confused with each other in this project, and each one has
misled a report already:

  * CASH FLOW is what `_fills.py` shows. Selling inventory looks like profit
    because rial arrives, but nothing was earned -- an asset changed form.
    That is how a +16,500,073 "profit" turned out to be a liquidation.
  * EQUITY CHANGE mixes trading with deposits, withdrawals and marking. A
    6,558,554 transfer to the margin wallet once read as an instant -8% loss.
  * P&L is realized round trips plus the mark on what is still held.

This computes the third, per market:

    realized  = sum(sell proceeds) - sum(buy cost) - fees      [closed volume]
    unrealized= units still held x (mark - average cost)       [open position]
    total     = realized + unrealized

Average cost is tracked FIFO-style through the fill sequence, so a partially
sold position attributes the right cost to the part that was sold.

Caveat this cannot escape: /market/trades/list returns a bounded history, so a
position opened before the window has no recorded cost basis. Those are listed
separately rather than silently valued at zero, which would invent profit.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.table import Table

from nbtrend.config import load_config
from nbtrend.data.nobitex_rest import NobitexREST

console = Console()


def _ts(raw) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, int | float):
        return float(raw) / 1000.0 if float(raw) > 1e11 else float(raw)
    try:
        return datetime.fromisoformat(str(raw)).timestamp()
    except ValueError:
        return 0.0


def main(hours: float) -> None:
    cfg = load_config()
    import time

    cutoff = time.time() - hours * 3600

    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                     api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        trades = api._get("/market/trades/list").get("trades", [])
        trades = [t for t in trades if _ts(t.get("timestamp")) >= cutoff]
        trades.sort(key=lambda t: _ts(t.get("timestamp")))

        # Walk the fills in order, tracking average cost per market.
        state = defaultdict(lambda: {"units": 0.0, "cost": 0.0, "realized": 0.0,
                                     "fees": 0.0, "buys": 0, "sells": 0,
                                     "unmatched_sold": 0.0})
        for t in trades:
            sym = t.get("market") or ""
            side = str(t.get("type", "")).lower()
            amt = float(t.get("amount") or 0)
            total = float(t.get("total") or 0)
            fee = float(t.get("fee") or 0)
            if not amt or side not in ("buy", "sell"):
                continue
            d = state[sym]
            d["fees"] += fee
            if side == "buy":
                d["units"] += amt
                d["cost"] += total
                d["buys"] += 1
            else:
                d["sells"] += 1
                if d["units"] <= 0:
                    # Sold something bought before this window: no cost basis.
                    d["unmatched_sold"] += total
                    continue
                sold = min(amt, d["units"])
                avg = d["cost"] / d["units"]
                d["realized"] += (total / amt) * sold - avg * sold
                d["units"] -= sold
                d["cost"] -= avg * sold

        # How far did the RIAL move while we held inventory? Anything bought
        # at one USDT/IRT rate and marked at another carries a currency gain
        # or loss that has nothing to do with spread capture. On a book that
        # holds overnight this can dwarf the edge.
        fx_open = fx_now = None
        try:
            fx_now = api.orderbook("USDTIRT").mid
            hist = api.candles("USDTIRT", "60", int(cutoff) - 3600, int(time.time()))
            if not hist.empty:
                fx_open = float(hist["close"].iloc[0])
        except Exception:
            pass

        # Mark whatever is still open.
        table = Table(title=f"P&L by market, last {hours:g}h (mark-to-market)")
        for col in ("market", "buys", "sells", "realized", "unrealized",
                    "fees", "net P&L"):
            table.add_column(col, justify="right")

        tr = tu = tf = 0.0
        unmatched = 0.0
        for sym, d in sorted(state.items()):
            mark_val = 0.0
            if d["units"] > 1e-12:
                base = sym.split("-")[0].lower()
                try:
                    mid = api.orderbook(base.upper() + "IRT").mid
                    mark_val = d["units"] * mid - d["cost"]
                except Exception:
                    mark_val = 0.0
            net = d["realized"] + mark_val - d["fees"]
            tr += d["realized"]
            tu += mark_val
            tf += d["fees"]
            unmatched += d["unmatched_sold"]
            table.add_row(
                sym, str(d["buys"]), str(d["sells"]),
                f"{d['realized']:+,.0f}", f"{mark_val:+,.0f}",
                f"-{d['fees']:,.0f}",
                f"[{'green' if net >= 0 else 'red'}]{net:+,.0f}[/]",
            )
        total_net = tr + tu - tf
        table.add_row(
            "TOTAL", "", "", f"{tr:+,.0f}", f"{tu:+,.0f}", f"-{tf:,.0f}",
            f"[bold {'green' if total_net >= 0 else 'red'}]{total_net:+,.0f}[/]",
        )
        console.print(table)

        if fx_open and fx_now:
            drift = (fx_now / fx_open - 1) * 100
            console.print(
                f"\n[bold]USDT/IRT over the window[/bold]: {fx_open:,.0f} -> {fx_now:,.0f} "
                f"({drift:+.2f}%)"
            )
            if abs(drift) > 0.1:
                console.print(
                    f"  Inventory held across that move carries a currency component.\n"
                    f"  Of the {tu:+,.0f} unrealized, roughly "
                    f"[bold]{tu * (drift / 100) / (1 + drift / 100):+,.0f}[/bold] is rial "
                    f"drift rather than spread capture."
                )

        console.print(
            f"\n  realized (closed round trips) : {tr:>+15,.0f}"
            f"\n  unrealized (open inventory)   : {tu:>+15,.0f}"
            f"\n  fees                          : {-tf:>+15,.0f}"
            f"\n  [bold]NET P&L                       : {total_net:>+15,.0f}[/bold]"
        )
        if unmatched:
            console.print(
                f"\n[yellow]{unmatched:,.0f} rial of sales had no cost basis in this "
                f"window[/yellow] -- positions opened earlier (the liquidations). "
                "Excluded rather than counted as profit, which would invent it."
            )


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 24.0)
