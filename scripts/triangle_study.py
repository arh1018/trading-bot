"""Is there a profitable triangle? IRT -> X -> USDT -> IRT.

The cycle: buy X with rial, sell X for tether, sell tether back to rial. If the
three books disagree about what X is worth, the round trip returns more rial
than it started with.

The arithmetic that decides it, before any strategy:

    every leg CROSSES the spread, so every leg pays TAKER
    IRT taker 0.10% + USDT taker 0.09% + IRT taker 0.10% = 29 bps

A mispricing under 29 bps is not an opportunity, it is a 29 bps fee paid for
the privilege of moving money in a circle.

Two things this measures that a mid-price calculation would miss:

  * EXECUTABLE prices. You buy at the ask and sell at the bid, never at the
    mid. Using mids overstates every opportunity by roughly the sum of the
    three half-spreads -- which on these books is larger than the edge itself.
  * DEPTH. The touch quantity bounds the trade. An arbitrage you can only do
    for 50,000 rial is not worth the leg risk, and Nobitex will reject it
    anyway under the 3,000,000 minimum.

Also reports the REVERSE cycle (IRT -> USDT -> X -> IRT), since a mispricing
has a sign and only one direction pays.
"""

from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.table import Table

from nbtrend.config import load_config
from nbtrend.data.nobitex_rest import NobitexREST

console = Console()


def _top(book: dict) -> tuple[float, float, float, float]:
    """(bid, bid_qty, ask, ask_qty) from a full orderbook."""
    bids, asks = book.get("bids") or [], book.get("asks") or []
    if not bids or not asks:
        raise ValueError("empty book")
    return bids[0][0], bids[0][1], asks[0][0], asks[0][1]


def _scan(api, coins: list[str], fees: tuple[float, float]) -> list[dict]:
    irt_taker, usdt_taker = fees
    drag = irt_taker + usdt_taker + irt_taker
    out = []

    fx = api.full_orderbook("USDTIRT")
    fx_bid, fx_bid_q, fx_ask, fx_ask_q = _top(fx)

    for coin in coins:
        try:
            irt = api.full_orderbook(f"{coin}IRT")
            usdt = api.full_orderbook(f"{coin}USDT")
            i_bid, i_bid_q, i_ask, i_ask_q = _top(irt)
            u_bid, u_bid_q, u_ask, u_ask_q = _top(usdt)
        except Exception:
            continue

        # FORWARD: rial -> coin (pay ask) -> usdt (hit bid) -> rial (hit fx bid)
        start = 1_000_000.0
        units = start / i_ask * (1 - irt_taker)
        tether = units * u_bid * (1 - usdt_taker)
        back = tether * fx_bid * (1 - irt_taker)
        fwd_bps = (back / start - 1) * 10_000

        # REVERSE: rial -> usdt (pay fx ask) -> coin (pay usdt ask) -> rial
        tether2 = start / fx_ask * (1 - irt_taker)
        units2 = tether2 / u_ask * (1 - usdt_taker)
        back2 = units2 * i_bid * (1 - irt_taker)
        rev_bps = (back2 / start - 1) * 10_000

        # Executable size is the tightest leg, in rial.
        fwd_size = min(i_ask * i_ask_q, u_bid * u_bid_q * fx_bid, fx_bid_q * fx_bid)
        rev_size = min(fx_ask_q * fx_ask, u_ask_q * u_ask * fx_ask, i_bid * i_bid_q)

        out.append({
            "coin": coin, "fwd": fwd_bps, "rev": rev_bps,
            "fwd_size": fwd_size, "rev_size": rev_size, "drag": drag * 10_000,
        })
    return out


def main(coins: list[str], rounds: int) -> None:
    cfg = load_config()
    irt_taker = float(cfg.costs.get("taker_fee_irt", 0.0010))
    usdt_taker = float(cfg.costs.get("taker_fee_usdt", 0.0009))
    min_order = float(cfg.costs["min_order_rial"])
    drag = (irt_taker + usdt_taker + irt_taker) * 10_000

    console.print(
        f"taker {irt_taker:.2%} + {usdt_taker:.2%} + {irt_taker:.2%} "
        f"-> [bold]{drag:.0f} bps to beat[/bold] (net of fees, already applied below)\n"
    )

    best: dict[str, dict] = {}
    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                     api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        for i in range(rounds):
            for row in _scan(api, coins, (irt_taker, usdt_taker)):
                cur = best.get(row["coin"])
                edge = max(row["fwd"], row["rev"])
                if cur is None or edge > max(cur["fwd"], cur["rev"]):
                    best[row["coin"]] = row
            if i < rounds - 1:
                time.sleep(2)

    table = Table(title=f"triangular arbitrage, net of {drag:.0f} bps fees "
                        f"({rounds} samples, best per coin)")
    for col in ("coin", "fwd bps", "rev bps", "best size (rial)", "verdict"):
        table.add_column(col)

    live = 0
    for coin, r in sorted(best.items(), key=lambda kv: -max(kv[1]["fwd"], kv[1]["rev"])):
        edge = max(r["fwd"], r["rev"])
        size = r["fwd_size"] if r["fwd"] >= r["rev"] else r["rev_size"]
        ok = edge > 0 and size >= min_order
        if ok:
            live += 1
        verdict = ("[green]PROFITABLE[/]" if ok
                   else "[yellow]too small[/]" if edge > 0 else "[red]fee-eaten[/]")
        table.add_row(
            coin,
            f"[{'green' if r['fwd'] > 0 else 'red'}]{r['fwd']:+.1f}[/]",
            f"[{'green' if r['rev'] > 0 else 'red'}]{r['rev']:+.1f}[/]",
            f"{size:,.0f}", verdict,
        )
    console.print(table)
    console.print(
        f"\n[bold]{live}/{len(best)}[/bold] coins show a cycle that clears fees "
        f"AND {min_order:,.0f} rial of depth"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 4
    syms = [a.upper() for a in args if not a.isdigit()]
    main(syms or ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "TRX", "LTC"], n)
