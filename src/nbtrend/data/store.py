"""Parquet-backed OHLCV cache.

Trend signals need hundreds of warmup bars, and refetching them on every run
wastes both time and the exchange's rate limit. The store keeps one file per
(source, symbol, resolution) and merges new bars into it.

The last bar of any fetch is usually still forming. `append` drops it before
writing so a partial bar never lands in the cache and gets mistaken for a
closed one on the next run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ._util import empty_ohlcv, normalise_index

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class CandleStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, source: str, symbol: str, resolution: str) -> Path:
        safe = symbol.replace(":", "_").replace("/", "_")
        return self.root / f"{source}__{safe}__{resolution}.parquet"

    def load(self, source: str, symbol: str, resolution: str) -> pd.DataFrame:
        path = self.path(source, symbol, resolution)
        if not path.exists():
            return _empty()
        df = pd.read_parquet(path)
        return normalise_index(df).sort_index()

    def save(self, df: pd.DataFrame, source: str, symbol: str, resolution: str) -> Path:
        path = self.path(source, symbol, resolution)
        df[OHLCV_COLUMNS].sort_index().to_parquet(path)
        return path

    def append(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        resolution: str,
        drop_last: bool = True,
    ) -> pd.DataFrame:
        """Merge `df` into the cache, newest values winning on overlap."""
        if df.empty:
            return self.load(source, symbol, resolution)

        incoming = df[OHLCV_COLUMNS].copy()
        if drop_last and len(incoming) > 1:
            incoming = incoming.iloc[:-1]

        existing = self.load(source, symbol, resolution)
        merged = pd.concat([existing, incoming])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        self.save(merged, source, symbol, resolution)
        log.debug("cached %d bars for %s/%s/%s", len(merged), source, symbol, resolution)
        return merged

    def coverage(self, source: str, symbol: str, resolution: str) -> tuple | None:
        df = self.load(source, symbol, resolution)
        if df.empty:
            return None
        return (df.index[0], df.index[-1], len(df))


def _empty() -> pd.DataFrame:
    return empty_ohlcv()


def align(frames: dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
    """Join one column from many symbols onto a shared index.

    Forward-fill only -- an asset that has not printed yet must stay NaN, or
    the backtest trades a price that did not exist.
    """
    series = {name: df[column] for name, df in frames.items() if not df.empty}
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().ffill()
