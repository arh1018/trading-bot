"""Data-layer contracts: unit conversion at the edges, clock alignment, cache."""

from __future__ import annotations

import pandas as pd
import pytest

from nbtrend.data._util import empty_ohlcv, normalise_index
from nbtrend.data.feed import _bars_per_day, _merge_on_global_clock
from nbtrend.data.fx import (
    basis_series,
    compute_basis,
    fair_rial_price,
    implied_fx,
    synthesise_irt_history,
)
from nbtrend.data.store import CandleStore


def _frame(index, close):
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": [1.0] * len(close)},
        index=index,
    )


def test_index_resolutions_are_unified():
    """TradingView yields datetime64[s], Binance datetime64[ms]; pandas 3
    refuses to merge_asof across resolutions."""
    seconds = pd.to_datetime([1_700_000_000, 1_700_014_400], unit="s", utc=True)
    millis = pd.to_datetime([1_700_000_000_000, 1_700_014_400_000], unit="ms", utc=True)

    a = normalise_index(_frame(seconds, [1.0, 2.0]))
    b = normalise_index(_frame(millis, [1.0, 2.0]))
    assert a.index.dtype == b.index.dtype
    pd.testing.assert_index_equal(a.index, b.index)


def test_naive_index_is_localised_to_utc():
    naive = pd.date_range("2024-01-01", periods=3, freq="4h")
    out = normalise_index(_frame(naive, [1.0, 2.0, 3.0]))
    assert str(out.index.tz) == "UTC"


def test_merge_never_takes_an_unclosed_local_bar():
    """Nobitex 4h bars close at :30 (UTC+3:30); the global clock is on the
    hour. Alignment must reach backward, never forward."""
    global_idx = pd.date_range("2024-01-01 04:00", periods=3, freq="4h", tz="UTC")
    local_idx = pd.date_range("2024-01-01 00:30", periods=3, freq="4h", tz="UTC")

    global_df = _frame(global_idx, [100.0, 110.0, 120.0])
    local_df = _frame(local_idx, [1000.0, 1100.0, 1200.0])
    fx_df = _frame(local_idx, [2.0, 2.0, 2.0])

    out = _merge_on_global_clock(global_df, local_df, fx_df)
    # Global 04:00 must use the local bar that closed at 00:30, not 04:30.
    assert out["local_close"].iloc[0] == 1000.0
    assert out["fair_rial"].iloc[0] == 200.0


def test_basis_is_computed_from_the_identity():
    b = compute_basis("BTCIRT", local_rial=210.0, global_usd=100.0, fx_rial_per_usdt=2.0)
    assert b.fair_rial == fair_rial_price(100.0, 2.0) == 200.0
    assert b.basis == pytest.approx(0.05)
    assert b.basis_bps == pytest.approx(500.0)
    assert b.is_premium


def test_implied_fx_recovers_the_rate():
    local = pd.Series([200.0, 400.0])
    glob = pd.Series([100.0, 200.0])
    assert implied_fx(local, glob).tolist() == [2.0, 2.0]


def test_basis_series_is_zero_when_pricing_is_fair():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    glob = pd.Series([100.0, 110.0, 120.0], index=idx)
    fx = pd.Series([2.0, 2.0, 2.0], index=idx)
    out = basis_series(glob * fx, glob, fx)
    assert out.abs().max() == pytest.approx(0.0)


def test_synthesised_irt_history_applies_fx_and_basis():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    glob = _frame(idx, [100.0, 200.0, 300.0])
    fx = pd.Series([2.0, 2.0, 2.0], index=idx)

    plain = synthesise_irt_history(glob, fx)
    assert plain["close"].tolist() == [200.0, 400.0, 600.0]

    premium = synthesise_irt_history(glob, fx, basis=0.05)
    assert premium["close"].iloc[0] == pytest.approx(210.0)


def test_bars_per_day():
    assert _bars_per_day("240") == 6
    assert _bars_per_day("60") == 24
    assert _bars_per_day("D") == 1


def test_store_round_trip_and_merge(tmp_path):
    store = CandleStore(tmp_path)
    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    store.append(_frame(idx, [1.0, 2.0, 3.0, 4.0, 5.0]), "src", "BTCIRT", "240")

    # drop_last=True: the newest bar is still forming and must not be cached.
    cached = store.load("src", "BTCIRT", "240")
    assert len(cached) == 4
    assert cached["close"].tolist() == [1.0, 2.0, 3.0, 4.0]

    # Overlapping refetch: newer values win, no duplicates.
    store.append(_frame(idx[2:], [99.0, 98.0, 97.0]), "src", "BTCIRT", "240")
    merged = store.load("src", "BTCIRT", "240")
    assert not merged.index.has_duplicates
    assert merged["close"].loc[idx[2]] == 99.0


def test_empty_store_returns_a_typed_frame(tmp_path):
    out = CandleStore(tmp_path).load("src", "NOPE", "240")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.dtype == empty_ohlcv().index.dtype
