"""Does a higher edge floor actually keep more money?

The maker quotes any market whose spread clears `breakeven + min_edge_bps`.
That floor was 8 bps against a 16 bps breakeven, and a day at that setting
turned +35,704 of gross capture into -108,894 net: fees took 29,826 and one
COTI round trip gave back 114,772 when the price fell 2.7% between our buy and
our sell.

Two stories explain that, and they call for opposite fixes:

  * FEES. Too many marginal fills, each barely clearing costs. Raising the
    floor trades volume for margin and should help.
  * ADVERSE SELECTION. The spread was never the problem -- COTI was quotable
    at 46 bps and still lost. Raising the floor changes nothing, because the
    losing fills were in wide markets too.

This replays completed round trips against candidate floors to see which. It
uses realised pairs only: a buy matched to the sell that closed it, so the
number is money that actually moved, not a mark.
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
    try:
        return datetime.fromisoformat(str(raw)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def main(hours: float = 24.0) -> None:
    cfg = load_config()
    import time

    cutoff = time.time() - hours * 3600

    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                     api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        trades = [t for t in api._get("/market/trades/list").get("trades", [])
                  if _ts(t.get("timestamp")) >= cutoff]

        # Current spread per market, as the stand-in for "what we saw when we
        # quoted". The book at fill time is not in the history, so this is an
        # approximation and the report says so rather than implying precision.
        spreads: dict[str, float] = {}
        for market in {t["market"] for t in trades}:
            sym = market.replace("-RLS", "IRT").replace("-", "")
            try:
                ob = api.orderbook(sym)
                spreads[market] = (ob.best_ask - ob.best_bid) / ob.mid * 1e4
            except Exception:
                pass

    if not trades:
        console.print("[yellow]no trades in the window[/yellow]")
        return

    # Match buys to sells per market, FIFO, and realise the closed volume.
    legs: dict[str, list[tuple[float, float]]] = defaultdict(list)   # market -> [(px, amt)]
    realised: dict[str, float] = defaultdict(float)
    fees: dict[str, float] = defaultdict(float)
    pairs: dict[str, int] = defaultdict(int)

    for t in sorted(trades, key=lambda x: _ts(x.get("timestamp"))):
        m, px, amt = t["market"], float(t["price"]), float(t["amount"])
        # FEE UNITS DIFFER BY SIDE. A buy is charged in the BASE currency
        # (0.025192 COOKIE) and a sell in rial (2,466.696). Multiplying the
        # wrong one by the price inflated a 62,000,000 rial account's costs to
        # billions and made every floor look catastrophic.
        raw_fee = float(t.get("fee") or 0)
        fees[m] += raw_fee * px if t["type"] == "buy" else raw_fee
        if t["type"] == "buy":
            legs[m].append((px, amt))
            continue
        left = amt
        while left > 1e-12 and legs[m]:
            bpx, bamt = legs[m][0]
            take = min(left, bamt)
            realised[m] += (px - bpx) * take
            left -= take
            if take >= bamt - 1e-12:
                legs[m].pop(0)
                pairs[m] += 1
            else:
                legs[m][0] = (bpx, bamt - take)

    table = Table(title=f"Round trips by market, last {hours:g}h")
    for col in ("market", "spread now", "pairs", "realized", "fees", "net"):
        table.add_column(col, justify="right")

    markets = sorted(set(realised) | set(fees), key=lambda m: realised[m] - fees[m])
    for m in markets:
        net = realised[m] - fees[m]
        sp = spreads.get(m)
        table.add_row(
            m,
            f"{sp:.1f} bps" if sp is not None else "?",
            str(pairs[m]),
            f"{realised[m]:+,.0f}",
            f"{-fees[m]:+,.0f}",
            f"[{'green' if net >= 0 else 'red'}]{net:+,.0f}[/]",
        )
    console.print(table)

    # The actual question: filter by the spread we would have required.
    console.print("\n[bold]What each edge floor would have kept[/bold]")
    console.print("[dim]breakeven is 16.0 bps; a floor of N refuses any market "
                  "under 16+N bps[/dim]\n")
    base = Table()
    for col in ("min_edge", "required", "markets kept", "realized", "fees", "net"):
        base.add_column(col, justify="right")

    for floor in (0.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0):
        need = 16.0 + floor
        kept = [m for m in markets if (spreads.get(m) or 0.0) >= need]
        r = sum(realised[m] for m in kept)
        f = sum(fees[m] for m in kept)
        base.add_row(
            f"{floor:.0f} bps", f"{need:.0f} bps", f"{len(kept)}/{len(markets)}",
            f"{r:+,.0f}", f"{-f:+,.0f}",
            f"[{'green' if r - f >= 0 else 'red'}]{r - f:+,.0f}[/]",
        )
    console.print(base)
    console.print(
        "\n[dim]Spreads are read NOW, not at fill time, so a market that has "
        "since widened or tightened is misfiled. Treat this as a direction, "
        "not a forecast.[/dim]"
    )


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 24.0)
