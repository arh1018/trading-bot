"""The rial leg: USDT/IRT, and the basis between local and global prices.

Central identity of this project
--------------------------------
Every Nobitex IRT price decomposes into three factors:

    P_irt = P_usd * FX_usdt_irt * (1 + basis)

and therefore, in log returns:

    r_irt = r_usd + r_fx + r_basis

Those three terms have nothing to do with each other:

* `r_usd`   -- global crypto trend. Deep, liquid, 24/7, and what the trend
  model is actually trying to capture.
* `r_fx`    -- rial devaluation. Driven by Iranian monetary policy, sanctions
  and capital controls. Strongly one-directional over multi-year horizons,
  punctuated by jumps. Not a crypto signal.
* `r_basis` -- Nobitex's local premium/discount versus global. Mean-reverting,
  usually small, but it blows out when capital controls tighten or the
  exchange has an outage.

Running a trend model directly on `P_irt` silently mixes all three. The model
then reports "momentum" that is really the rial sliding, sizes it with a
volatility estimate contaminated by FX jumps, and cannot tell you which bet
lost money. So: **signal on `P_usd`, execute on `P_irt`, monitor `basis`.**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Basis:
    """A point-in-time comparison of local versus global pricing."""
    symbol: str
    local_rial: float
    global_usd: float
    fx_rial_per_usdt: float
    fair_rial: float
    basis: float          # (local / fair) - 1

    @property
    def basis_bps(self) -> float:
        return self.basis * 1e4

    @property
    def is_premium(self) -> bool:
        return self.basis > 0


def fair_rial_price(
    global_usd: float, fx_rial_per_usdt: float, multiplier: int = 1
) -> float:
    """What the IRT market *should* print, given the global price and FX.

    `multiplier` covers Nobitex's scaled markets: 1K_SHIBIRT is quoted per
    1,000 SHIB, so its fair price is 1,000x the per-unit global price.
    """
    return global_usd * fx_rial_per_usdt * multiplier


def compute_basis(
    symbol: str,
    local_rial: float,
    global_usd: float,
    fx_rial_per_usdt: float,
    multiplier: int = 1,
) -> Basis:
    fair = fair_rial_price(global_usd, fx_rial_per_usdt, multiplier)
    return Basis(
        symbol=symbol,
        local_rial=local_rial,
        global_usd=global_usd,
        fx_rial_per_usdt=fx_rial_per_usdt,
        fair_rial=fair,
        basis=(local_rial / fair - 1.0) if fair > 0 else 0.0,
    )


def implied_fx(local_rial: pd.Series, global_usd: pd.Series) -> pd.Series:
    """FX rate implied by a crypto pair quoted in both rial and USD.

    Useful as a sanity check on the USDTIRT book, and as a stand-in when the
    USDTIRT market is halted (`isClosed`) but crypto markets are still trading
    -- which does happen on Nobitex.
    """
    aligned_local, aligned_global = local_rial.align(global_usd, join="inner")
    return aligned_local / aligned_global.replace(0, np.nan)


def basis_series(
    local_rial: pd.Series, global_usd: pd.Series, fx_rial_per_usdt: pd.Series
) -> pd.Series:
    """Historical basis, for deciding whether `max_basis_deviation` is sane."""
    df = pd.DataFrame(
        {"local": local_rial, "glob": global_usd, "fx": fx_rial_per_usdt}
    ).dropna()
    fair = df["glob"] * df["fx"]
    return (df["local"] / fair.replace(0, np.nan)) - 1.0


def synthesise_irt_history(
    global_usd: pd.DataFrame, fx_rial: pd.Series, basis: float | pd.Series = 0.0
) -> pd.DataFrame:
    """Build an IRT OHLCV series from global prices and the FX rate.

    Nobitex minute/hour history only reaches back to 1401 (2022) for some
    markets, and thinly-traded pairs have gaps. For a long backtest it is
    often better to reconstruct the rial series than to trust a sparse local
    one -- as long as you remember the reconstruction assumes a constant (or
    supplied) basis and therefore understates local dislocation risk.
    """
    aligned_fx = fx_rial.reindex(global_usd.index).ffill()
    multiplier = aligned_fx * (1.0 + basis)

    out = pd.DataFrame(index=global_usd.index)
    for col in ("open", "high", "low", "close"):
        out[col] = global_usd[col] * multiplier
    out["volume"] = global_usd["volume"]
    return out.dropna()
