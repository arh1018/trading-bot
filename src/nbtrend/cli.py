"""Command line interface: `nbtrend <command>`."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .backtest.engine import Backtester
from .backtest.metrics import buy_and_hold, format_metrics, max_drawdown, summarise
from .backtest.walkforward import walk_forward
from .config import load_config
from .data.feed import DataFeed
from .data.fx import compute_basis
from .data.nobitex_rest import NobitexREST
from .risk.sizing import bars_per_year

app = typer.Typer(add_completion=False, help="Trend following on Nobitex rial markets.")
console = Console()


def _short_param(dotted: str) -> str:
    """`strategy.signal.macd.weight` -> `macd.weight`.

    The last segment alone is ambiguous -- three different estimators each
    have a `weight`, and a table of `weight=0.5, weight=0.3` tells you
    nothing about which is which.
    """
    parts = dotted.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else dotted


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


@app.command()
def fetch(
    symbol: str | None = typer.Option(None, help="Only this Nobitex symbol, e.g. BTCIRT."),
    days: int | None = typer.Option(None, help="Override data.history_days."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Download global + local history into the parquet cache."""
    _setup_logging(log_level)
    cfg = load_config()
    feed = DataFeed(cfg)
    specs = [cfg.symbol(symbol)] if symbol else cfg.enabled_symbols

    if days:
        datasets = {
            spec.nobitex: asyncio.run(feed.build_dataset(spec, days=days)) for spec in specs
        }
    else:
        datasets = asyncio.run(feed.build_all(specs))

    table = Table(title="cached history")
    for column in ("symbol", "bars", "from", "to", "mean basis"):
        table.add_column(column)
    for name, dataset in datasets.items():
        frame = dataset.frame
        table.add_row(
            name, str(len(frame)),
            str(frame.index[0].date()), str(frame.index[-1].date()),
            f"{frame['basis'].mean():+.2%}",
        )
    console.print(table)


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Nobitex symbol, e.g. BTCIRT."),
    days: int | None = typer.Option(None, help="History window."),
    benchmark: bool = typer.Option(True, help="Compare against buy & hold."),
    save: Path | None = typer.Option(None, help="Write the equity curve to CSV."),
    log_level: str = typer.Option("WARNING"),
) -> None:
    """Backtest one symbol on cached/fresh history."""
    _setup_logging(log_level)
    cfg = load_config()
    spec = cfg.symbol(symbol)

    feed = DataFeed(cfg)
    dataset = asyncio.run(feed.build_dataset(spec, days=days))
    frame = dataset.frame

    result = Backtester(cfg).run(frame, symbol, amount_step=spec.amount_step)
    ppy = bars_per_year(str(cfg.data["timeframe"]))
    metrics = summarise(result, ppy, fx=frame.get("fx"))

    console.print(format_metrics(metrics, f"{symbol} -- {cfg.data['timeframe']} trend following"))

    if benchmark:
        hold = buy_and_hold(frame["local_close"], result.initial_equity)
        if not hold.empty:
            console.print(
                f"\n[bold]Buy & hold[/bold]  {hold.iloc[-1] / hold.iloc[0] - 1:+.1%} return, "
                f"{max_drawdown(hold):.1%} max drawdown"
            )
    if save:
        result.equity.to_csv(save)
        console.print(f"equity curve -> {save}")


@app.command()
def optimize(
    symbol: str = typer.Argument(..., help="Nobitex symbol, e.g. BTCIRT."),
    days: int | None = typer.Option(None),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Walk-forward parameter search. Reports OUT-OF-SAMPLE results only."""
    _setup_logging(log_level)
    cfg = load_config()
    spec = cfg.symbol(symbol)

    feed = DataFeed(cfg)
    frame = asyncio.run(feed.build_dataset(spec, days=days)).frame

    console.print(f"[bold]walk-forward[/bold] on {len(frame)} bars of {symbol}")
    result = walk_forward(cfg, frame, symbol, spec.amount_step)

    table = Table(title="folds")
    for column in ("#", "test window", "OOS Sharpe", "OOS return", "trades", "chosen params"):
        table.add_column(column, overflow="fold")
    for i, fold in enumerate(result.folds, 1):
        m = fold.test_metrics
        table.add_row(
            str(i),
            f"{fold.test_start.date()} -> {fold.test_end.date()}",
            f"{m.sharpe:.2f}" if m else "-",
            f"{m.total_return:+.1%}" if m else "-",
            str(m.num_trades) if m else "-",
            ", ".join(f"{_short_param(k)}={v}" for k, v in fold.best_params.items()),
        )
    console.print(table)

    if result.oos_metrics:
        console.print(format_metrics(result.oos_metrics, f"{symbol} -- stitched out-of-sample"))
        console.print(
            f"\nmean in-sample score {result.in_sample_score:.2f} vs "
            f"out-of-sample Sharpe {result.oos_metrics.sharpe:.2f} "
            f"([bold]overfit gap {result.overfit_gap:+.2f}[/bold])"
        )

    if result.param_stability:
        console.print("\n[bold]parameter stability[/bold] (coefficient of variation across folds; "
                      "high means the value is fitting noise and should be pinned)")
        for key, cv in sorted(result.param_stability.items(), key=lambda kv: -kv[1]):
            console.print(f"  {_short_param(key):<28} {cv:.2f}")


@app.command()
def signals(log_level: str = typer.Option("WARNING")) -> None:
    """Current trend score and target weight for every enabled symbol."""
    _setup_logging(log_level)
    cfg = load_config()
    feed = DataFeed(cfg)
    backtester = Backtester(cfg)
    datasets = asyncio.run(feed.build_all())

    table = Table(title=f"signals ({cfg.data['timeframe']} bars)")
    for column in ("symbol", "score", "regime", "direction", "target w", "basis", "global USD"):
        table.add_column(column, justify="right")

    for symbol, dataset in datasets.items():
        frame = dataset.frame
        row = backtester.build_signals(frame).iloc[-1]
        colour = "green" if row["direction"] > 0 else ("red" if row["direction"] < 0 else "dim")
        table.add_row(
            symbol,
            f"[{colour}]{row['score']:+.2f}[/{colour}]",
            "yes" if row["regime"] else "no",
            {1.0: "long", -1.0: "short"}.get(float(row["direction"]), "flat"),
            f"{row['target_weight']:.1%}",
            f"{frame['basis'].iloc[-1]:+.2%}",
            f"${frame['close'].iloc[-1]:,.0f}",
        )
    console.print(table)


@app.command()
def basis(log_level: str = typer.Option("WARNING")) -> None:
    """Live local-versus-global pricing for every enabled market."""
    _setup_logging(log_level)
    cfg = load_config()

    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                    api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        fx_book = api.orderbook(cfg.fx.nobitex)
        console.print(
            f"[bold]USDT/IRT[/bold]  {fx_book.mid:,.0f} rial "
            f"({fx_book.mid / 10:,.0f} toman), spread {fx_book.spread_bps:.1f} bps\n"
        )

        feed = DataFeed(cfg)
        table = Table(title="basis vs global")
        for column in ("symbol", "local (rial)", "global (USD)", "fair (rial)", "basis", "spread"):
            table.add_column(column, justify="right")

        for spec in cfg.enabled_symbols:
            book = api.orderbook(spec.nobitex)
            global_df = asyncio.run(feed.fetch_global(spec, bars=5))
            if global_df.empty:
                continue
            global_usd = float(global_df["close"].iloc[-1])
            b = compute_basis(spec.nobitex, book.mid, global_usd, fx_book.mid)
            colour = "yellow" if abs(b.basis) > 0.02 else "white"
            table.add_row(
                spec.nobitex, f"{b.local_rial:,.0f}", f"${global_usd:,.2f}",
                f"{b.fair_rial:,.0f}",
                f"[{colour}]{b.basis:+.2%}[/{colour}]",
                f"{book.spread_bps:.0f} bps",
            )
        console.print(table)


@app.command()
def run(
    once: bool = typer.Option(False, help="Run a single rebalance and exit."),
    minutes: float | None = typer.Option(None, help="Stop after this many minutes."),
    interval: float | None = typer.Option(
        None, help="Seconds between cycles (default: wait for the next bar close)."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the live-mode confirmation prompt."
    ),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Start the trading loop. Honours NBTREND_MODE (paper | live)."""
    _setup_logging(log_level)
    cfg = load_config()

    if cfg.is_live:
        console.print("[bold red]LIVE MODE[/bold red] -- real orders will be placed.")
        # A non-interactive run (background, cron) has no stdin to prompt on,
        # so --yes is required rather than silently proceeding.
        if not yes and not typer.confirm("Continue?"):
            raise typer.Abort()
    else:
        console.print("[bold green]PAPER MODE[/bold green] -- no orders leave this process.")

    from .live.lock import AlreadyRunning
    from .live.runner import LiveRunner

    runner = LiveRunner(cfg)
    try:
        asyncio.run(runner.run(once=once, minutes=minutes, interval_s=interval))
    except AlreadyRunning as exc:
        console.print(f"[bold red]refusing to start[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\nstopped")


async def _socket_loop(cfg, mm, api, symbols, deadline, fallback_s: float,
                       protected: set[str] | None = None) -> None:
    """Quote from websocket book events rather than a polling clock.

    The blocking work -- placing and cancelling orders -- runs in a thread, not
    on the event loop. A REST call inside the websocket handler is what starved
    Centrifugo's 25s ping and got the trend runner dropped for "no pong".
    """
    import contextlib

    from .data.nobitex_ws import NobitexWS, book_top_from_payload, orderbook_channel
    from .live.maker import SocketMakerRunner

    ws = NobitexWS(cfg.ws_url)
    # `--requote` is the per-symbol cooldown, NOT just the REST fallback sleep.
    # It was only wired to the fallback, so the socket path used its 2s default
    # and asked for 12 x 2 x (600/2) = 7,200 placements per 10 minutes against a
    # 240 budget -- it burned the whole allowance in seconds, then refused every
    # quote including the asks that sell held inventory.
    runner = SocketMakerRunner(cfg, mm, api, ws, min_requote_gap_s=fallback_s)
    runner.protected = set(protected or ())

    def _handler(symbol: str):
        # Handlers are called as (channel, payload) -- see nobitex_ws.Handler.
        def _on(_channel: str, payload: dict) -> None:
            top = book_top_from_payload(symbol, payload)
            if top is not None:
                runner.on_book(symbol, top)
        return _on

    for sym in symbols:
        ws.on(orderbook_channel(sym), _handler(sym))

    stop = asyncio.Event()
    ws_task = asyncio.create_task(ws.run(stop))
    console.print(f"websocket: subscribed to {len(symbols)} orderbook channel(s)")

    try:
        # Wait for the first books so we never quote off an empty view.
        for _ in range(60):
            if all(runner.latest_book(s) for s in symbols):
                break
            await asyncio.sleep(1)

        await asyncio.to_thread(mm.refresh_balances)
        while time.time() < deadline:
            due = runner.take_due()
            if not due:
                await asyncio.sleep(1.0)
                continue
            if ws.seconds_since_message > 120:
                console.print("[yellow]websocket stale; pulling quotes[/yellow]")
                for sym in symbols:
                    await asyncio.to_thread(runner.cancel_working, sym)
                await asyncio.sleep(fallback_s)
                continue
            await asyncio.to_thread(runner.sweep, due)
    finally:
        stop.set()
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task
        for sym in symbols:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(runner.cancel_working, sym)


@app.command()
def doctor(log_level: str = typer.Option("WARNING")) -> None:
    """Check connectivity, credentials, and the rial/toman unit convention."""
    _setup_logging(log_level)
    cfg = load_config()
    ok = True

    console.print(f"mode           : {cfg.mode}")
    console.print(f"REST           : {cfg.rest_url}")
    console.print(f"websocket      : {cfg.ws_url}")
    if cfg.creds.uses_api_key:
        scheme = "API key pair (Ed25519-signed)"
    elif cfg.creds.api_token:
        scheme = "login token (bearer)"
    else:
        scheme = "[yellow]none -- public data only[/yellow]"
    console.print(f"credentials    : {scheme}")
    console.print(f"timeframe      : {cfg.data['timeframe']}")
    console.print(f"universe       : {', '.join(s.nobitex for s in cfg.enabled_symbols)}\n")

    try:
        with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                    api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
            book = api.orderbook("BTCIRT")
            console.print(f"[green]OK[/green] orderbook BTCIRT: {book.mid:,.0f} rial")

            import time as _time
            now = int(_time.time())
            candles = api.candles("BTCIRT", "60", now - 86400, now)
            close = float(candles["close"].iloc[-1])
            console.print(f"[green]OK[/green] candles: {len(candles)} bars, last {close:,.0f} rial")

            from .units import assert_plausible_rial
            assert_plausible_rial("candle vs orderbook", close, book.last_trade)
            console.print("[green]OK[/green] toman->rial conversion agrees with the orderbook")
    except Exception as exc:
        ok = False
        console.print(f"[red]FAIL[/red] Nobitex: {exc}")

    try:
        feed = DataFeed(cfg)
        spec = cfg.enabled_symbols[0]
        global_df = asyncio.run(feed.fetch_global(spec, bars=10))
        console.print(
            f"[green]OK[/green] global feed ({cfg.data['global_feed']}): "
            f"{spec.tradingview} = ${global_df['close'].iloc[-1]:,.2f}"
        )
    except Exception as exc:
        ok = False
        console.print(f"[red]FAIL[/red] global feed: {exc}")

    if cfg.creds.can_trade and not cfg.creds.uses_api_key and cfg.creds.api_token:
        looks_like_api_key = (
            len(cfg.creds.api_token.strip()) == 44 and cfg.creds.api_token.strip().endswith("=")
        )
        if looks_like_api_key:
            console.print(
                "\n[yellow]WARN[/yellow] NOBITEX_API_TOKEN looks like an API key "
                "(44 chars, base64), not a login token.\n"
                "     Nobitex API keys are NOT bearer tokens -- each request must be signed\n"
                "     with Ed25519, which needs BOTH halves of the pair. Set NOBITEX_API_KEY\n"
                "     (public) and NOBITEX_API_SECRET (private) instead. The private key is\n"
                "     shown only once, when the key is created."
            )

    if cfg.creds.can_trade:
        try:
            with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                    api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
                wallets = api.wallets(["rls", "usdt", "btc"])
                console.print(f"[green]OK[/green] wallets: {wallets}")
                console.print(f"[green]OK[/green] ws auth param: {api.ws_auth_param()[:8]}...")
        except Exception as exc:
            ok = False
            console.print(f"[red]FAIL[/red] authenticated endpoints: {exc}")

    console.print(f"\n{'[green]all checks passed[/green]' if ok else '[red]some checks failed[/red]'}")


@app.command()
def margin(log_level: str = typer.Option("WARNING")) -> None:
    """Report margin capability, wallet and open positions. Places NO orders."""
    _setup_logging(log_level)
    cfg = load_config()

    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        markets = api.margin_markets()
        universe = {s.nobitex for s in cfg.enabled_symbols}
        tradeable = {k: v for k, v in markets.items() if k in universe}
        shortable = [k for k, v in tradeable.items() if v.sell_enabled]

        console.print(f"margin markets       : {len(markets)} total")
        console.print(f"  in our universe    : {len(tradeable)}")
        console.print(f"  shortable          : {len(shortable)}")
        if tradeable:
            lev = max(v.max_leverage for v in tradeable.values())
            fee = max(v.position_fee_rate for v in tradeable.values())
            console.print(f"  max leverage       : {lev:g}x")
            console.print(f"  position fee rate  : {fee:.4%}")

        wallets = api.wallets()
        console.print(f"\nspot rial            : {wallets.get('rls', 0.0):,.0f}")

        positions = api.positions(status="active")
        if not positions:
            console.print("open positions       : none")
            return

        table = Table(title="open margin positions")
        for col in ("id", "symbol", "side", "lev", "collateral", "liq price",
                    "to liq", "unreal PNL"):
            table.add_column(col)
        for p in positions:
            room = p.distance_to_liquidation()
            table.add_row(
                str(p.id), p.symbol, p.side.name, f"{p.leverage:g}x",
                f"{p.collateral:,.0f}",
                f"{p.liquidation_price:,.0f}" if p.liquidation_price else "-",
                f"{room:+.1%}" if room is not None else "-",
                f"{p.unrealized_pnl:,.0f}",
            )
        console.print(table)


@app.command("make")
def make(
    symbols: str = typer.Option(
        "AVAXIRT,XRPIRT,LINKIRT,DOTIRT",
        help="Markets to quote. Defaults to the four measured above breakeven.",
    ),
    minutes: float = typer.Option(60.0, help="Stop after this many minutes."),
    requote: float = typer.Option(30.0, help="Seconds between requotes."),
    notional: float = typer.Option(3_000_000.0, help="Rial per quote."),
    max_inventory: float = typer.Option(15_000_000.0, help="Rial inventory cap per symbol."),
    min_edge_bps: float = typer.Option(4.0, help="Required bps above breakeven."),
    min_cash: float = typer.Option(
        0.0, help="Rial to keep unspent across the whole book (portfolio floor)."
    ),
    min_quote: float = typer.Option(
        0.0,
        help="Smallest quote the exchange accepts, in rial. 0 reads "
             "costs.min_order_rial from the config, which is the measured value.",
    ),
    live: bool = typer.Option(False, "--live", help="Place real orders (default: dry run)."),
    socket: bool = typer.Option(
        True, help="Drive quoting from the websocket book instead of polling."
    ),
    protect: str = typer.Option(
        "",
        help="Markets the bot must never quote, even if held (comma separated). "
             "Empty by default; the gold pairs were listed here while the "
             "account owner held a position in them by hand.",
    ),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Market-make the spread. Dry run unless --live is passed."""
    _setup_logging(log_level)
    cfg = load_config()

    from .live.maker import MakerRunner, MarketMaker

    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    specs = {s.nobitex: s for s in [*cfg.enabled_symbols, cfg.fx]}
    missing = [s for s in wanted if s not in specs]
    if missing:
        console.print(f"[red]not in the universe:[/red] {', '.join(missing)}")
        raise typer.Exit(code=1)

    maker_fee = float(cfg.costs.get("maker_fee_irt", cfg.costs["maker_fee"]))
    with NobitexREST(cfg.rest_url, cfg.creds.api_token,
                api_key=cfg.creds.api_key, api_secret=cfg.creds.api_secret) as api:
        # THE EXCHANGE MINIMUM HAS ONE SOURCE. It was measured against the live
        # book and written to costs.min_order_rial; the maker used to carry its
        # own 3,000,000 default that nothing ever overwrote, so the measured
        # 550,000 sat in the config being ignored. Every holding worth between
        # the two was refused an ask -- sellable on Nobitex, blocked by us --
        # which is why buys outnumbered sells better than two to one.
        min_quote_rial = min_quote or float(cfg.costs.get("min_order_rial", 550_000))

        mm = MarketMaker(
            api, specs, wanted,
            maker_fee=maker_fee,
            min_edge_bps=min_edge_bps,
            quote_notional_rial=notional,
            max_inventory_rial=max_inventory,
            min_cash_rial=min_cash,
            min_quote_rial=min_quote_rial,
            requote_s=requote,
            dry_run=not live,
        )
        runner = MakerRunner(cfg, mm, api)
        protected = {p.strip().upper() for p in protect.split(',') if p.strip()}
        runner.protected = protected

        mode = "[bold red]LIVE[/bold red]" if live else "[bold green]DRY RUN[/bold green]"
        console.print(
            f"{mode} market making {len(wanted)} symbol(s): {', '.join(wanted)}\n"
            f"breakeven {mm.breakeven_bps():.1f} bps (maker {maker_fee:.4%} x2), "
            f"floor {min_edge_bps:.1f} bps above it\n"
            f"budget supports {mm.max_quotable_symbols():.1f} symbols at {requote:.0f}s requote\n"
            f"cash floor {min_cash:,.0f} rial; inventory cap {max_inventory:,.0f}/symbol\n"
            f"min quote {min_quote_rial:,.0f} rial "
            f"({'--min-quote' if min_quote else 'costs.min_order_rial'})\n"
        )
        if len(wanted) > mm.max_quotable_symbols():
            console.print(
                "[yellow]warning:[/yellow] more symbols than the order budget supports; "
                "quotes will be skipped once it runs out\n"
            )

        deadline = time.time() + minutes * 60
        try:
            if socket:
                asyncio.run(_socket_loop(cfg, mm, api, wanted, deadline, requote, protected))
            else:
                while time.time() < deadline:
                    runner.sweep(wanted)
                    time.sleep(requote)
        except KeyboardInterrupt:
            console.print("\nstopping; pulling quotes")
        finally:
            for sym in wanted:
                runner.cancel_working(sym)

        table = Table(title="market making session")
        for col in ("symbol", "fills", "inventory", "realized rial"):
            table.add_column(col)
        for sym in wanted:
            b = mm.books[sym]
            table.add_row(sym, str(b.fills), f"{b.inventory:.8f}", f"{b.realized_rial:,.0f}")
        console.print(table)


if __name__ == "__main__":
    app()
