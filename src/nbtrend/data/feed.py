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

    # -- global (USD) ------------------------------------------------------
    async def fetch_global(self, spec: SymbolSpec, bars: int) -> pd.DataFrame:
        if not spec.tradingview:
            raise ValueError(f"{spec.nobitex} has no `tradingview` symbol in universe.yaml")

        if self._source == "binance":
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
            except Exception as exc:
                log.warning(
                    "TradingView failed for %s (%s); falling back to Binance",
                    spec.tradingview, exc,
                )
                df = await BinanceFeed().fetch_ohlcv(spec.tradingview, self.resolution, bars)

        return self.store.append(df, self._source, spec.tradingview, self.resolution)

    # -- local (RIAL) ------------------------------------------------------
    def fetch_local(self, spec: SymbolSpec, days: int) -> pd.DataFrame:
        end = int(time.time())
        start = end - days * 86400
        with NobitexREST(self.cfg.rest_url, self.cfg.creds.api_token) as api:
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
        fx_df = await asyncio.to_thread(self.fetch_local, self.cfg.fx, days)

        frame = _merge_on_global_clock(global_df, local_df, fx_df)
        return SymbolDataset(spec=spec, frame=frame)

    async def build_all(self, specs: list[SymbolSpec] | None = None) -> dict[str, SymbolDataset]:
        specs = specs or self.cfg.enabled_symbols
        out: dict[str, SymbolDataset] = {}
        for spec in specs:
            try:
                out[spec.nobitex] = await self.build_dataset(spec)
                log.info("%s: %d aligned bars", spec.nobitex, len(out[spec.nobitex]))
            except Exception:
                log.exception("failed to build dataset for %s", spec.nobitex)
        return out


def _bars_per_day(resolution: str) -> int:
    per_day = {
        "1": 1440, "5": 288, "15": 96, "30": 48,
        "60": 24, "180": 8, "240": 6, "360": 4, "720": 2,
        "D": 1, "2D": 1, "3D": 1,
    }
    return per_day.get(resolution, 6)


def _merge_on_global_clock(
    global_df: pd.DataFrame, local_df: pd.DataFrame, fx_df: pd.DataFrame
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

    out["fair_rial"] = out["close"] * out["fx"]
    out["basis"] = (out["local_close"] / out["fair_rial"]) - 1.0
    return out


def build_all_sync(cfg: Config, specs: list[SymbolSpec] | None = None) -> dict[str, SymbolDataset]:
    return asyncio.run(DataFeed(cfg).build_all(specs))
