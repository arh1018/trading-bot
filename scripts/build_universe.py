"""Generate config/universe.yaml from the live Nobitex market list.

Hand-maintaining 100+ symbols goes stale the moment Nobitex lists or delists
anything, so the universe is generated and re-generatable:

    python scripts/build_universe.py --top 120

What it resolves that a hand-written list gets wrong:

* **Scaled units.** Nobitex quotes `1K_SHIBIRT` per 1,000 SHIB, `1M_PEPEIRT`
  per 1,000,000 PEPE, `100K_FLOKIIRT` per 100,000 FLOKI. The global feed is
  per 1 unit. Without the multiplier the basis check sees a 1000x
  "dislocation" and refuses to trade every one of them.
* **Which global feed actually exists.** Binance does not list every token
  Nobitex does. Pairs are checked against Binance's `exchangeInfo`, and
  anything missing falls back to TradingView's aggregated `CRYPTO:<BASE>USD`
  index rather than pointing at a symbol that will never resolve.
* **Tokenized equities.** Nobitex lists AAPL, AMZN and friends against rial.
  Those are stocks, not crypto: they need an equity feed, they stop trading
  outside market hours, and a 24/7 trend model will happily hold them across
  a weekend gap. They are emitted `enabled: false` with a comment.
* **Exchange precision.** `amount_step` comes from Nobitex's own
  `/v2/options` precision table, not a guess.

Ranking is by 24h rial volume (`volumeDst`), because a trend signal you
cannot fill is worthless -- the tail of the IRT book is extremely thin.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

NOBITEX = "https://apiv2.nobitex.ir"
BINANCE = "https://api.binance.com"

# Nobitex prefixes a scaled market with its multiplier.
SCALE_RE = re.compile(r"^(100K|1K|1M|1B|10K)_(.+)$")
SCALES = {"1K": 1_000, "10K": 10_000, "100K": 100_000, "1M": 1_000_000, "1B": 1_000_000_000}

# Tokenized equities listed against rial. Mapped so they are usable, but left
# disabled: they are not 24/7 instruments and this is a 24/7 trend model.
EQUITY_FEEDS = {
    "AAPL": "NASDAQ:AAPL", "AMZN": "NASDAQ:AMZN", "GOOGL": "NASDAQ:GOOGL",
    "META": "NASDAQ:META", "MSFT": "NASDAQ:MSFT", "NVDA": "NASDAQ:NVDA",
    "TSLA": "NASDAQ:TSLA", "NFLX": "NASDAQ:NFLX", "AMD": "NASDAQ:AMD",
    "INTC": "NASDAQ:INTC", "COIN": "NASDAQ:COIN", "MSTR": "NASDAQ:MSTR",
    "PLTR": "NASDAQ:PLTR", "AVGO": "NASDAQ:AVGO", "COST": "NASDAQ:COST",
    "PEP": "NASDAQ:PEP", "ADBE": "NASDAQ:ADBE", "QQQ": "NASDAQ:QQQ",
    "SPY": "AMEX:SPY", "GLD": "AMEX:GLD", "SLV": "AMEX:SLV",
    "KO": "NYSE:KO", "MCD": "NYSE:MCD", "NKE": "NYSE:NKE", "DIS": "NYSE:DIS",
    "JPM": "NYSE:JPM", "V": "NYSE:V", "MA": "NYSE:MA", "BABA": "NYSE:BABA",
    "PFE": "NYSE:PFE", "XOM": "NYSE:XOM", "WMT": "NYSE:WMT", "UNH": "NYSE:UNH",
}

# Stablecoins and fiat proxies: no trend to follow, and USDTIRT is the FX leg
# handled separately in the `fx:` block.
EXCLUDE_BASES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "PAXG", "XAUT"}


def fetch_irt_markets(client: httpx.Client) -> list[str]:
    data = client.get(f"{NOBITEX}/v3/orderbook/all", timeout=60).json()
    return sorted(
        symbol for symbol, book in data.items()
        if symbol.endswith("IRT") and isinstance(book, dict) and book.get("bids") and book.get("asks")
    )


def split_symbol(symbol: str) -> tuple[str, int]:
    """`1M_PEPEIRT` -> (`PEPE`, 1_000_000); `BTCIRT` -> (`BTC`, 1)."""
    base = symbol[: -len("IRT")]
    match = SCALE_RE.match(base)
    if match:
        return match.group(2), SCALES[match.group(1)]
    return base, 1


def nobitex_currency(symbol: str) -> str:
    """srcCurrency for the trading API keeps the scale prefix, lowercased."""
    return symbol[: -len("IRT")].lower()


def fetch_stats(client: httpx.Client, currencies: list[str]) -> dict[str, dict]:
    """Batch /market/stats. One bad currency fails the whole call, so on error
    the batch is bisected until the offender is isolated."""
    if not currencies:
        return {}
    try:
        resp = client.get(
            f"{NOBITEX}/market/stats",
            params={"srcCurrency": ",".join(currencies), "dstCurrency": "rls"},
            timeout=60,
        ).json()
        stats = resp.get("stats")
        if stats is None:
            raise ValueError(resp.get("message", "no stats"))
        return stats
    except Exception:
        if len(currencies) == 1:
            return {}
        mid = len(currencies) // 2
        return {
            **fetch_stats(client, currencies[:mid]),
            **fetch_stats(client, currencies[mid:]),
        }


def fetch_binance_pairs(client: httpx.Client) -> set[str]:
    try:
        info = client.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=60).json()
        return {
            s["symbol"] for s in info.get("symbols", [])
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
        }
    except Exception as exc:
        print(f"warning: could not reach Binance ({exc}); assuming all pairs exist", file=sys.stderr)
        return set()


def fetch_precisions(client: httpx.Client) -> dict[str, float]:
    try:
        options = client.get(f"{NOBITEX}/v2/options", timeout=60).json()
        out: dict[str, float] = {}
        for coin in options.get("coins", []):
            try:
                out[coin["coin"].lower()] = float(coin["displayPrecision"])
            except (KeyError, TypeError, ValueError):
                continue
        return out
    except Exception as exc:
        print(f"warning: could not read /v2/options ({exc})", file=sys.stderr)
        return {}


def global_feed(base: str, binance_pairs: set[str]) -> tuple[str, str]:
    """Return (tradingview_symbol, kind)."""
    if base in EQUITY_FEEDS:
        return EQUITY_FEEDS[base], "equity"
    pair = f"{base}USDT"
    if not binance_pairs or pair in binance_pairs:
        return f"BINANCE:{pair}", "crypto"
    return f"CRYPTO:{base}USD", "aggregate"


def price_step(latest_rial: float) -> int:
    """Coarse tick, scaled to price magnitude. Nobitex rejects sub-tick prices
    and the exact tick is not published per market."""
    for threshold, step in ((1e10, 10_000), (1e8, 1_000), (1e6, 100), (1e4, 10)):
        if latest_rial >= threshold:
            return step
    return 1


def amount_step(base: str, precisions: dict[str, float], multiplier: int) -> float:
    key = base.lower()
    step = precisions.get(key)
    if step and step > 0:
        # A scaled market trades in units of `multiplier`, so the step is coarser.
        return step * multiplier if multiplier > 1 else step
    return 0.01 if multiplier > 1 else 1e-6


def build(top: int, min_volume_rial: float) -> dict:
    with httpx.Client(headers={"User-Agent": "nbtrend/0.1"}) as client:
        print("fetching Nobitex IRT markets ...", file=sys.stderr)
        markets = fetch_irt_markets(client)
        print(f"  {len(markets)} markets with live books", file=sys.stderr)

        currencies = [nobitex_currency(m) for m in markets]
        print("fetching 24h stats ...", file=sys.stderr)
        stats = fetch_stats(client, currencies)
        print(f"  stats for {len(stats)} markets", file=sys.stderr)

        print("fetching Binance pairs and Nobitex precisions ...", file=sys.stderr)
        binance_pairs = fetch_binance_pairs(client)
        precisions = fetch_precisions(client)

    rows = []
    for symbol in markets:
        base, multiplier = split_symbol(symbol)
        if base in EXCLUDE_BASES:
            continue

        stat = stats.get(f"{nobitex_currency(symbol)}-rls", {})
        try:
            volume = float(stat.get("volumeDst", 0))
            latest = float(stat.get("latest", 0))
        except (TypeError, ValueError):
            continue
        if volume < min_volume_rial or latest <= 0:
            continue

        feed, kind = global_feed(base, binance_pairs)
        rows.append(
            {
                "symbol": symbol, "base": base, "multiplier": multiplier,
                "volume": volume, "latest": latest, "feed": feed, "kind": kind,
            }
        )

    rows.sort(key=lambda r: -r["volume"])
    selected = rows[:top]

    entries = []
    for row in selected:
        entry = {
            "nobitex": row["symbol"],
            "src": nobitex_currency(row["symbol"]),
            "dst": "rls",
            "tradingview": row["feed"],
            "amount_step": amount_step(row["base"], precisions, row["multiplier"]),
            "price_step": price_step(row["latest"]),
            "enabled": row["kind"] != "equity",
        }
        if row["multiplier"] > 1:
            entry["multiplier"] = row["multiplier"]
        entries.append(entry)

    return {"entries": entries, "rows": selected, "total": len(rows)}


def render(entries: list[dict], rows: list[dict], total: int) -> str:
    header = f'''# ---------------------------------------------------------------------------
# Trading universe -- GENERATED by scripts/build_universe.py
#
# Regenerate after any Nobitex listing change:
#     python scripts/build_universe.py --top {len(entries)}
#
# Ranked by 24h rial volume: a trend signal you cannot fill is worthless, and
# the tail of the IRT book is very thin. {total} markets cleared the volume
# floor; the top {len(entries)} are listed.
#
# Fields
#   nobitex      Nobitex market symbol, UPPERCASE. Used verbatim for websocket
#                channels and /v3/orderbook.
#   src / dst    srcCurrency + dstCurrency for POST /market/orders/add.
#   tradingview  Global reference feed. BINANCE:* where Binance lists the
#                pair, CRYPTO:*USD (TradingView's aggregated index) where it
#                does not, NASDAQ:/NYSE:/AMEX: for tokenized equities.
#   multiplier   Present only on scaled markets. Nobitex quotes 1K_SHIBIRT per
#                1,000 SHIB while the global feed is per 1 SHIB, so the fair
#                price is global * fx * multiplier. Omitting this makes the
#                basis check see a 1000x dislocation and refuse to trade.
#   amount_step  Order size rounding, from Nobitex /v2/options precision.
#   price_step   Price rounding, in RIAL, scaled to price magnitude.
#   enabled      Tokenized equities are emitted disabled -- they are not 24/7
#                instruments, and a 24/7 trend model will hold them straight
#                through a weekend gap.
# ---------------------------------------------------------------------------

fx:
  # The rial leg. Not a trading target -- this IS the exchange rate that
  # converts every global USD price into rial.
  nobitex: USDTIRT
  src: usdt
  dst: rls
  tradingview: null
  amount_step: 0.01
  price_step: 10

symbols:
'''
    lines = [header]
    by_symbol = {r["symbol"]: r for r in rows}
    for entry in entries:
        row = by_symbol[entry["nobitex"]]
        note = ""
        if row["kind"] == "equity":
            note = "   # tokenized equity -- not 24/7"
        elif row["kind"] == "aggregate":
            note = "   # not on Binance; TradingView aggregate index"
        lines.append(
            f"  # {row['volume'] / 1e12:>8.2f}T rial 24h volume{note}\n"
            + "".join(
                f"  {'- ' if i == 0 else '  '}{k}: {_scalar(v)}\n"
                for i, (k, v) in enumerate(entry.items())
            )
        )
    return "".join(lines)


def _scalar(value: object) -> str:
    """Render one YAML scalar.

    `yaml.safe_dump` cannot be used per-value here: on a bare scalar it emits a
    full document, trailing `...` end-marker included, which corrupts the file.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        # Keep small steps readable rather than in scientific notation.
        text = f"{value:.10f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return f'"{text}"' if any(c in text for c in ":#{}[],&*?|>=!%@`") else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=120, help="how many markets to emit")
    parser.add_argument("--min-volume", type=float, default=5e9,
                        help="minimum 24h rial volume to consider (default 5e9 = 500M toman)")
    parser.add_argument("--out", type=Path, default=Path("config/universe.yaml"))
    args = parser.parse_args()

    result = build(args.top, args.min_volume)
    text = render(result["entries"], result["rows"], result["total"])
    args.out.write_text(text)

    kinds: dict[str, int] = {}
    for row in result["rows"]:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    scaled = sum(1 for r in result["rows"] if r["multiplier"] > 1)

    print(
        f"\nwrote {len(result['entries'])} symbols to {args.out}\n"
        f"  by feed : {kinds}\n"
        f"  scaled  : {scaled} markets with a unit multiplier",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
