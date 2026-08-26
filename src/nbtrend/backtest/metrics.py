"""Performance statistics.

Two things here are specific to trading rial markets and are easy to get
wrong elsewhere:

1. **Everything is nominal rial.** A 60% rial return in a year where the rial
   lost 45% against the dollar is roughly break-even in purchasing power.
   `summarise` therefore also reports the return measured in USD terms when an
   FX series is supplied. Judge the strategy on both; the rial number is what
   the exchange shows you, the USD number is whether the strategy worked.
2. **Risk-free rate is not zero.** Rial bank deposits and money-market funds
   pay a substantial nominal rate. A Sharpe computed against rf=0 flatters
   every rial strategy. `rf_annual` defaults to 0 but should be set to the
   real alternative rate before believing any of these numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Metrics:
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_bars_held: float
    exposure: float
    turnover: float
    costs_pct_of_equity: float
    usd_total_return: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe_ratio(returns: pd.Series, periods_per_year: float, rf_annual: float = 0.0) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    excess = returns - (rf_annual / periods_per_year)
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, rf_annual: float = 0.0) -> float:
    excess = returns - (rf_annual / periods_per_year)
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def summarise(
    result,
    periods_per_year: float,
    rf_annual: float = 0.0,
    fx: pd.Series | None = None,
) -> Metrics:
    """Collapse a `BacktestResult` into comparable statistics."""
    equity = result.equity
    returns = result.returns

    if equity.empty:
        return Metrics(*([0.0] * 13), num_trades=0)

    years = len(equity) / periods_per_year
    total_return = float(equity.iloc[-1] / result.initial_equity - 1.0)
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else 0.0

    trades = result.trades
    wins = [t for t in trades if t.pnl_rial > 0]
    losses = [t for t in trades if t.pnl_rial <= 0]
    gross_win = sum(t.pnl_rial for t in wins)
    gross_loss = abs(sum(t.pnl_rial for t in losses))

    weights = result.positions["weight"]
    dd = max_drawdown(equity)

    usd_return = None
    if fx is not None and not fx.empty:
        fx_aligned = fx.reindex(equity.index).ffill().bfill()
        usd_equity = equity / fx_aligned
        if usd_equity.iloc[0] > 0:
            usd_return = float(usd_equity.iloc[-1] / usd_equity.iloc[0] - 1.0)

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        annual_vol=float(returns.std() * np.sqrt(periods_per_year)),
        sharpe=sharpe_ratio(returns, periods_per_year, rf_annual),
        sortino=sortino_ratio(returns, periods_per_year, rf_annual),
        max_drawdown=dd,
        calmar=float(cagr / abs(dd)) if dd < 0 else 0.0,
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        num_trades=len(trades),
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        exposure=float((weights.abs() > 1e-9).mean()),
        turnover=float(weights.diff().abs().sum()),
        costs_pct_of_equity=float(result.costs_paid / result.initial_equity),
        usd_total_return=usd_return,
    )


def buy_and_hold(prices: pd.Series, initial_equity: float) -> pd.Series:
    """Benchmark. For a rial book this is the bar to beat, and it is a high
    bar: holding BTC through a devaluing currency has both legs working."""
    clean = prices.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return initial_equity * clean / clean.iloc[0]


def format_metrics(m: Metrics, name: str = "") -> str:
    lines = [
        f"{'=' * 52}",
        f"{name}".center(52) if name else "",
        f"{'=' * 52}",
        f"{'Total return (rial)':<28} {m.total_return:>+12.1%}",
    ]
    if m.usd_total_return is not None:
        lines.append(f"{'Total return (USD terms)':<28} {m.usd_total_return:>+12.1%}")
    lines += [
        f"{'CAGR':<28} {m.cagr:>+12.1%}",
        f"{'Annualised vol':<28} {m.annual_vol:>12.1%}",
        f"{'Sharpe':<28} {m.sharpe:>12.2f}",
        f"{'Sortino':<28} {m.sortino:>12.2f}",
        f"{'Max drawdown':<28} {m.max_drawdown:>12.1%}",
        f"{'Calmar':<28} {m.calmar:>12.2f}",
        f"{'-' * 52}",
        f"{'Trades':<28} {m.num_trades:>12d}",
        f"{'Win rate':<28} {m.win_rate:>12.1%}",
        f"{'Profit factor':<28} {m.profit_factor:>12.2f}",
        f"{'Avg bars held':<28} {m.avg_bars_held:>12.1f}",
        f"{'Time in market':<28} {m.exposure:>12.1%}",
        f"{'Turnover (sum |dw|)':<28} {m.turnover:>12.1f}",
        f"{'Costs paid':<28} {m.costs_pct_of_equity:>12.1%}",
        f"{'=' * 52}",
    ]
    return "\n".join(line for line in lines if line)
