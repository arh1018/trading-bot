"""Facade over the three data sources the strategy needs.

Assembles, per symbol:
  * `global_df` -- OHLCV in USD from TradingView (or Binance), the signal input
  * `local_df`  -- OHLCV in RIAL from Nobitex, the execution reference
  * `fx`        -- USDT/IRT close series, the bridge between them

The three feeds do not share a bar clock. Nobitex 4h bars close at :30 past
the hour (Iran is UTC+3:30) while Binance/TradingView close on the hour, so
naive joins drop or duplicate rows. `build_dataset` reindexes the local and FX
series onto the global clock with a backward `merge_asof`, which uses the last
*already-closed* local bar for each global bar and therefore never reads a
price from the future.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import pandas as pd

from ..config import Config, SymbolSpec
from .binance import BinanceFeed
from .kucoin import KuCoinFeed
from .nobitex_rest import NobitexREST
from .store import CandleStore
from .tradingview import TradingViewFeed

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SymbolDataset:
    """Everything the strategy needs for one symbol, on one shared clock."""
    spec: SymbolSpec
    frame: pd.DataFrame     # global o/h/l/c/v + local_close + fx + fair + basis

    @property
    def symbol(self) -> str:
        return self.spec.nobitex

    def global_ohlcv(self) -> pd.DataFrame:
        return self.frame[["open", "high", "low", "close", "volume"]]

    def __len__(self) -> int:
        return len(self.frame)


class DataFeed:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = CandleStore(cfg.data["cache_dir"])
        self.resolution: str = str(cfg.data["timeframe"])
        self._source: str = cfg.data.get("global_feed", "tradingview")
        # USDTIRT is the same series for every symbol. Fetching it per symbol
        # is 120 identical REST calls per cycle -- slow, and a pointless share
        # of the rate limit.
        self._fx_cache: pd.DataFrame | None = None
        self._fx_fetched_at = 0.0
        self._fx_lock = asyncio.Lock()
        self._fetch_semaphore = asyncio.Semaphore(
            int(cfg.data.get("max_concurrent_fetches", 8))
        )
        # TradingView circuit breaker. Its datafeed rate-limits concurrent
        # anonymous sockets (HTTP 429), so on a large universe every symbol
        # fails over to Binance anyway -- after paying ~2s for the rejected
        # connection. Retrying it 112 times a cycle is slow for us and abusive
        # to a free service, so a run of failures trips the breaker and the
        # rest of the session goes straight to Binance.
        self._tv_failures = 0
        self._tv_disabled = False
        self._tv_failure_limit = int(cfg.data.get("tradingview_failure_limit", 3))
        # Datasets are cached between cycles. The strategy decides on closed
        # bars, so refetching 4h candles every 2 minutes cannot change a
        # signal -- it only burns rate limit, and that is what earns a 429.
        # Short cycles stay useful because fills, drift and stops are checked
        # against the live websocket book, which is not rate limited.
        self._dataset_cache: dict[str, tuple[float, SymbolDataset]] = {}
        self._min_refetch_s = float(cfg.data.get("min_refetch_s", 240))
        # A global feed older than this is treated as dead rather than trusted.
        self._max_feed_age_s = float(cfg.data.get("max_feed_age_days", 3)) * 86400

    async def fx_history(self, days: int) -> pd.DataFrame:
        """USDTIRT candles, fetched once per DataFeed lifetime."""
        async with self._fx_lock:
            if self._fx_cache is None:
                self._fx_cache = await asyncio.to_thread(self.fetch_local, self.cfg.fx, days)
                self._fx_fetched_at = time.time()
            return self._fx_cache

    def invalidate_fx(self) -> None:
        """Drop the cached FX series so the next cycle refetches it."""
        self._fx_cache = None

    # -- global (USD) ------------------------------------------------------
    async def fetch_global(self, spec: SymbolSpec, bars: int) -> pd.DataFrame:
        if not spec.tradingview:
            raise ValueError(f"{spec.nobitex} has no `tradingview` symbol in universe.yaml")

        if self._source == "kucoin":
            df = await KuCoinFeed().fetch_ohlcv(spec.tradingview, self.resolution, bars)
        elif self._source == "binance" or self._tv_disabled:
            df = await BinanceFeed().fetch_ohlcv(spec.tradingview, self.resolution, bars)
        else:
            tv = self.cfg.data["tradingview"]
            feed = TradingViewFeed(
                ws_url=tv["ws_url"],
                origin=tv["origin"],
                auth_token=tv["auth_token"],
                timeout_s=float(tv["timeout_s"]),
            )
            try:
                df = await feed.fetch_ohlcv(spec.tradingview, self.resolution, bars)
                self._tv_failures = 0
            except Exception as exc:
                self._tv_failures += 1
                if self._tv_failures >= self._tv_failure_limit and not self._tv_disabled:
                    self._tv_disabled = True
                    log.warning(
                        "TradingView failed %d times in a row (%s); using Binance for the "
                        "rest of this session. Set data.global_feed: binance to skip this.",
                        self._tv_failures, exc,
                    )
                else:
                    log.debug("TradingView failed for %s (%s); using Binance", spec.tradingview, exc)
                df = await BinanceFeed().fetch_ohlcv(spec.tradingview, self.resolution, bars)

        return self.store.append(df, self._source, spec.tradingview, self.resolution)

    # -- local (RIAL) ------------------------------------------------------
    def fetch_local(self, spec: SymbolSpec, days: int) -> pd.DataFrame:
        end = int(time.time())
        start = end - days * 86400
        with NobitexREST(
            self.cfg.rest_url, self.cfg.creds.api_token,
            api_key=self.cfg.creds.api_key, api_secret=self.cfg.creds.api_secret,
        ) as api:
            df = api.candles(spec.nobitex, self.resolution, start, end)
        return self.store.append(df, "nobitex", spec.nobitex, self.resolution)

    # -- assembly ----------------------------------------------------------
    async def build_dataset(
        self, spec: SymbolSpec, bars: int | None = None, days: int | None = None
    ) -> SymbolDataset:
        bars = bars or int(self.cfg.data.get("history_days", 900)) * _bars_per_day(self.resolution)
        days = days or int(self.cfg.data.get("history_days", 900))

        global_df = await self.fetch_global(spec, bars)
        # `fetch_local` is synchronous httpx. Awaiting it directly would block
        # the event loop for seconds, which starves the Centrifugo pong
        # handler and gets the websocket dropped for "no pong".
        local_df = await asyncio.to_thread(self.fetch_local, spec, days)
        fx_df = await self.fx_history(days)

        frame = _merge_on_global_clock(global_df, local_df, fx_df, spec.multiplier)
        return SymbolDataset(spec=spec, frame=frame)

    async def build_all(self, specs: list[SymbolSpec] | None = None) -> dict[str, SymbolDataset]:
        """Build every symbol's dataset concurrently.

        Sequential fetching does not scale: at ~10s per symbol a 120-market
        universe takes 20 minutes per decision cycle, longer than some of the
        bars it is deciding on. Concurrency is bounded by a semaphore so we
        neither trip Nobitex's per-IP rate limit nor open 120 simultaneous
        TradingView sockets.
        """
        specs = specs or self.cfg.enabled_symbols

        # Warm the shared FX series before fanning out, or the first N workers
        # all race for the same lock. Expire it on the same clock as datasets.
        if time.time() - self._fx_fetched_at > self._min_refetch_s:
            self.invalidate_fx()
        await self.fx_history(int(self.cfg.data.get("history_days", 900)))

        now = time.time()

        async def one(spec: SymbolSpec) -> tuple[str, SymbolDataset | None]:
            cached = self._dataset_cache.get(spec.nobitex)
            if cached and now - cached[0] < self._min_refetch_s:
                return spec.nobitex, cached[1]

            async with self._fetch_semaphore:
                try:
                    dataset = await self.build_dataset(spec)
                    self._dataset_cache[spec.nobitex] = (time.time(), dataset)
                    return spec.nobitex, dataset
                except Exception as exc:
                    log.warning("failed to build dataset for %s: %s", spec.nobitex, exc)
                    # Serve stale data rather than dropping the symbol from the
                    # book on one transient failure.
                    return spec.nobitex, cached[1] if cached else None

        results = await asyncio.gather(*(one(spec) for spec in specs))

        out: dict[str, SymbolDataset] = {}
        stale: list[str] = []
        for name, ds in results:
            if ds is None or ds.frame.empty:
                continue
            age_s = (pd.Timestamp.now(tz="UTC") - ds.frame.index[-1]).total_seconds()
            if age_s > self._max_feed_age_s:
                stale.append(f"{name} ({age_s / 86400:.0f}d)")
                continue
            out[name] = ds

        if stale:
            # A delisted pair does not error -- Binance keeps serving the last
            # bars it ever had. XMRUSDT still returns February 2024 data. The
            # basis interlock catches the resulting nonsense, but only after
            # the symbol has been priced against a frozen quote all cycle.
            log.warning(
                "%d symbol(s) have a stale global feed and were dropped "
                "(likely delisted upstream): %s%s",
                len(stale), ", ".join(stale[:6]), "..." if len(stale) > 6 else "",
            )
        failed = [name for name, ds in results if ds is None]
        if failed:
            log.info(
                "built %d/%d datasets (%d failed: %s%s)",
                len(out), len(specs), len(failed), ", ".join(failed[:5]),
                "..." if len(failed) > 5 else "",
            )
        else:
            log.info("built %d/%d datasets", len(out), len(specs))
        return out


def _bars_per_day(resolution: str) -> int:
    per_day = {
        "1": 1440, "5": 288, "15": 96, "30": 48,
        "60": 24, "180": 8, "240": 6, "360": 4, "720": 2,
        "D": 1, "2D": 1, "3D": 1,
    }
    return per_day.get(resolution, 6)


def _merge_on_global_clock(
    global_df: pd.DataFrame,
    local_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    multiplier: int = 1,
) -> pd.DataFrame:
    """Align local rial and FX onto the global bar clock, backward-only."""
    if global_df.empty:
        return global_df

    out = global_df.copy()

    for name, source in (("local_close", local_df), ("fx", fx_df)):
        if source.empty:
            out[name] = pd.NA
            continue
        merged = pd.merge_asof(
            out[[]].reset_index().rename(columns={out.index.name or "index": "dt"}),
            source[["close"]].reset_index().rename(
                columns={source.index.name or "index": "dt", "close": name}
            ),
            on="dt",
            direction="backward",      # never take a bar that has not closed
        )
        out[name] = merged[name].to_numpy()

    # `multiplier` handles Nobitex's scaled markets (1K_SHIB, 1M_PEPE, ...),
    # which quote a bundle of units against a per-unit global feed.
    out["fair_rial"] = out["close"] * out["fx"] * multiplier
    out["basis"] = (out["local_close"] / out["fair_rial"]) - 1.0
    return out


def build_all_sync(cfg: Config, specs: list[SymbolSpec] | None = None) -> dict[str, SymbolDataset]:
    return asyncio.run(DataFeed(cfg).build_all(specs))
