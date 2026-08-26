"""Parameter search with walk-forward validation.

The point of this module is to answer "which trend metrics work here?"
*without* fooling yourself. A plain grid search over the whole history will
always find a parameter set with a great Sharpe -- on ~4000 bars and a few
hundred combinations, the best in-sample result is mostly luck.

Walk-forward instead:

    |---- train 365d ----|-- test 90d --|
              |---- train 365d ----|-- test 90d --|
                        |---- train 365d ----|-- test 90d --|

Parameters are fitted on each training window and then traded, untouched, on
the window that follows. The out-of-sample segments are stitched into one
equity curve. That curve is the only honest estimate of what the strategy
would have done, and it is normally much worse than the in-sample fit -- the
gap between them is the size of your overfit.
"""

from __future__ import annotations

import copy
import itertools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..risk.sizing import bars_per_year
from .engine import Backtester
from .metrics import Metrics, summarise

log = logging.getLogger(__name__)


# Parameters worth searching, and sane ranges. Deliberately small: every extra
# axis multiplies the number of fits and the amount of overfitting.
DEFAULT_GRID: dict[str, list[Any]] = {
    "strategy.signal.macd.weight": [0.3, 0.5, 0.7],
    "strategy.signal.tsmom.weight": [0.1, 0.3, 0.5],
    "strategy.signal.donchian.weight": [0.0, 0.2, 0.4],
    "strategy.thresholds.entry": [0.2, 0.3, 0.4],
    "strategy.regime.min_efficiency_ratio": [0.0, 0.25, 0.35],
    "risk.atr_stop_mult": [2.0, 3.0, 4.0],
}


@dataclass(slots=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_score: float
    test_metrics: Metrics | None = None
    # Kept so the stitched curve can report real trade-level statistics
    # rather than zeros.
    test_trades: list = field(default_factory=list)
    test_positions: pd.DataFrame | None = None
    test_costs: float = 0.0


@dataclass(slots=True)
class WalkForwardResult:
    folds: list[Fold]
    oos_equity: pd.Series
    oos_metrics: Metrics | None = None
    in_sample_score: float = 0.0
    param_stability: dict[str, float] = field(default_factory=dict)

    @property
    def overfit_gap(self) -> float:
        """In-sample score minus out-of-sample score. Large is bad."""
        oos = self.oos_metrics.sharpe if self.oos_metrics else 0.0
        return self.in_sample_score - oos


def set_nested(cfg_dict: dict, dotted_key: str, value: Any) -> None:
    node = cfg_dict
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def iter_grid(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def _apply(cfg: Config, params: dict[str, Any]) -> Config:
    clone = copy.deepcopy(cfg)
    for key, value in params.items():
        set_nested(clone.raw, key, value)
    return clone


def default_objective(m: Metrics) -> float:
    """What "best" means.

    Sharpe alone picks parameter sets that trade constantly and happen to have
    smooth in-sample returns. This penalises deep drawdowns and rewards having
    actually taken enough trades to be a distribution rather than an anecdote.
    """
    if m.num_trades < 5:
        return -np.inf
    penalty = 1.0 + max(0.0, abs(m.max_drawdown) - 0.20) * 5.0
    return float(m.sharpe / penalty)


def evaluate(
    cfg: Config,
    frame: pd.DataFrame,
    symbol: str,
    amount_step: float,
    objective: Callable[[Metrics], float] = default_objective,
) -> tuple[float, Metrics]:
    result = Backtester(cfg).run(frame, symbol, amount_step=amount_step)
    ppy = bars_per_year(str(cfg.data["timeframe"]))
    metrics = summarise(result, ppy, fx=frame.get("fx"))
    return objective(metrics), metrics


def search(
    cfg: Config,
    frame: pd.DataFrame,
    symbol: str,
    amount_step: float,
    grid: dict[str, list[Any]] | None = None,
    objective: Callable[[Metrics], float] = default_objective,
) -> tuple[dict[str, Any], float]:
    """Exhaustive grid search on one window. Returns (best_params, score)."""
    grid = grid or DEFAULT_GRID
    best_params: dict[str, Any] = {}
    best_score = -np.inf

    for params in iter_grid(grid):
        try:
            score, _ = evaluate(_apply(cfg, params), frame, symbol, amount_step, objective)
        except Exception:
            log.debug("params failed: %s", params, exc_info=True)
            continue
        if score > best_score:
            best_score, best_params = score, params

    return best_params, float(best_score)


def walk_forward(
    cfg: Config,
    frame: pd.DataFrame,
    symbol: str,
    amount_step: float,
    grid: dict[str, list[Any]] | None = None,
    objective: Callable[[Metrics], float] = default_objective,
    warmup_bars: int | None = None,
) -> WalkForwardResult:
    """Roll a train/test window through the sample and stitch the OOS results."""
    grid = grid or DEFAULT_GRID
    resolution = str(cfg.data["timeframe"])
    ppy = bars_per_year(resolution)
    per_day = ppy / 365.0

    train_bars = int(float(cfg.backtest["train_days"]) * per_day)
    test_bars = int(float(cfg.backtest["test_days"]) * per_day)
    warmup = int(warmup_bars if warmup_bars is not None else cfg.data.get("warmup_bars", 400))

    if len(frame) < warmup + train_bars + test_bars:
        raise ValueError(
            f"need at least {warmup + train_bars + test_bars} bars for walk-forward "
            f"({warmup} warmup + {train_bars} train + {test_bars} test), have {len(frame)}"
        )

    folds: list[Fold] = []
    oos_segments: list[pd.Series] = []
    equity_level = float(cfg.backtest["initial_equity_rial"])

    start = 0
    while start + warmup + train_bars + test_bars <= len(frame):
        train_slice = frame.iloc[start : start + warmup + train_bars]
        # The test window carries the warmup with it, so indicators are warm
        # at the first traded bar -- but only the post-warmup part is scored.
        test_slice = frame.iloc[
            start + train_bars : start + warmup + train_bars + test_bars
        ]

        best_params, train_score = search(
            cfg, train_slice, symbol, amount_step, grid, objective
        )
        if not best_params:
            start += test_bars
            continue

        test_cfg = _apply(cfg, best_params)
        test_result = Backtester(test_cfg).run(
            test_slice, symbol, amount_step=amount_step, initial_equity=equity_level
        )

        traded = test_result.equity.iloc[warmup:] if len(test_result.equity) > warmup else test_result.equity
        if not traded.empty:
            oos_segments.append(traded)
            equity_level = float(traded.iloc[-1])

        folds.append(
            Fold(
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=traded.index[0] if not traded.empty else test_slice.index[0],
                test_end=test_slice.index[-1],
                best_params=best_params,
                train_score=train_score,
                test_metrics=summarise(test_result, ppy, fx=test_slice.get("fx")),
                test_trades=[
                    t for t in test_result.trades
                    if traded.empty or t.entry_ts >= traded.index[0]
                ],
                test_positions=test_result.positions.iloc[warmup:]
                if len(test_result.positions) > warmup
                else test_result.positions,
                test_costs=test_result.costs_paid,
            )
        )
        log.info(
            "fold %d: train->%s score=%.2f  test->%s",
            len(folds), train_slice.index[-1].date(), train_score, test_slice.index[-1].date(),
        )
        start += test_bars

    oos_equity = (
        pd.concat(oos_segments)[lambda s: ~s.index.duplicated(keep="first")].sort_index()
        if oos_segments
        else pd.Series(dtype=float)
    )

    result = WalkForwardResult(
        folds=folds,
        oos_equity=oos_equity,
        in_sample_score=float(np.mean([f.train_score for f in folds])) if folds else 0.0,
        param_stability=_stability(folds),
    )

    if not oos_equity.empty:
        stitched = _StitchedResult(oos_equity, float(cfg.backtest["initial_equity_rial"]), folds)
        result.oos_metrics = summarise(stitched, ppy, fx=frame.get("fx"))

    return result


def _stability(folds: list[Fold]) -> dict[str, float]:
    """How much each parameter moved between folds, normalised.

    A parameter that jumps to a different value every fold is not being
    "optimised" -- it is fitting noise, and should be pinned to a constant.
    """
    if not folds:
        return {}
    out: dict[str, float] = {}
    for key in folds[0].best_params:
        values = [float(f.best_params.get(key, np.nan)) for f in folds]
        values = [v for v in values if not np.isnan(v)]
        if len(values) < 2:
            continue
        mean = np.mean(values)
        out[key] = float(np.std(values) / abs(mean)) if mean else float(np.std(values))
    return out


class _StitchedResult:
    """Adapts a stitched OOS equity curve to the `summarise` interface.

    Trades, positions and costs are concatenated from the folds rather than
    recomputed, so win rate, turnover and costs describe the out-of-sample
    record instead of silently reporting zero.
    """

    def __init__(self, equity: pd.Series, initial_equity: float, folds: list[Fold]):
        self.equity = equity
        self.initial_equity = initial_equity
        self.trades = [t for fold in folds for t in fold.test_trades]
        self.costs_paid = sum(fold.test_costs for fold in folds)

        frames = [f.test_positions for f in folds if f.test_positions is not None]
        if frames:
            positions = pd.concat(frames)
            self.positions = positions[~positions.index.duplicated(keep="first")].sort_index()
        else:
            self.positions = pd.DataFrame({"weight": np.zeros(len(equity))}, index=equity.index)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)
