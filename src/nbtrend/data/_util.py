"""Shared frame hygiene for the data adapters.

pandas >= 2 preserves the resolution of a DatetimeIndex, so `unit="s"`
(TradingView, Nobitex) and `unit="ms"` (Binance) produce `datetime64[s, UTC]`
and `datetime64[ms, UTC]` indexes respectively. pandas 3 refuses to
`merge_asof` across two different resolutions, which means the feed that
happens to be cached works and the one fetched fresh raises. Every adapter
normalises to nanoseconds on the way out so the resolution can never depend
on which source a frame came from.
"""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalise_index(df: pd.DataFrame, name: str = "dt") -> pd.DataFrame:
    """Coerce the index to a UTC, nanosecond-resolution DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index = df.index.as_unit("ns")
    df.index.name = name
    return df


def empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame(columns=OHLCV_COLUMNS, dtype=float)
    df.index = pd.DatetimeIndex([], tz="UTC", name="dt").as_unit("ns")
    return df
