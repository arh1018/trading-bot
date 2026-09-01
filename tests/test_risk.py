"""Sizing and portfolio-level exposure limits."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from nbtrend.config import load_config
from nbtrend.risk.sizing import (
    RiskLimits,
    bars_per_year,
    chandelier_exit_hit,
    drawdown_breached,
    normalise_gross,
    order_size_from_weight,
    vol_target_weight,
)


@pytest.fixture
def limits():
    return RiskLimits.from_config(load_config().risk)


def test_bars_per_year_matches_the_calendar():
    assert bars_per_year("D") == 365
    assert bars_per_year("240") == 365 * 6
    assert bars_per_year("60") == 365 * 24
    with pytest.raises(KeyError):
        bars_per_year("7m")


def test_vol_targeting_sizes_down_a_volatile_asset(limits):
    rng = np.random.default_rng(1)
    quiet = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))))
    wild = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.05, 300))))
    direction = pd.Series(1.0, index=quiet.index)

    w_quiet = vol_target_weight(direction, quiet, limits, 2190).iloc[-1]
    w_wild = vol_target_weight(direction, wild, limits, 2190).iloc[-1]
    assert w_quiet > w_wild


def test_vol_targeting_respects_the_per_symbol_cap(limits):
    flat = pd.Series(np.full(300, 100.0) + np.arange(300) * 1e-9)
    direction = pd.Series(1.0, index=flat.index)
    weights = vol_target_weight(direction, flat, limits, 2190)
    assert weights.max() <= limits.max_weight_per_symbol + 1e-12


def test_vol_estimate_does_not_use_the_current_bar(limits):
    """Sizing at t may only use volatility known at t-1."""
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    direction = pd.Series(1.0, index=close.index)

    full = vol_target_weight(direction, close, limits, 2190)
    truncated = vol_target_weight(direction.iloc[:200], close.iloc[:200], limits, 2190)
    pd.testing.assert_series_equal(full.iloc[:200], truncated, check_names=False)


def test_normalise_gross_scales_the_whole_book():
    weights = pd.DataFrame({"a": [0.5, 0.1], "b": [0.7, 0.2]})
    out = normalise_gross(weights, max_gross=1.0)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    # A row already inside the cap must be left alone.
    assert out.iloc[1].tolist() == pytest.approx([0.1, 0.2])


def test_live_runner_caps_gross_exposure():
    """Four symbols at the 0.35 per-symbol cap is 1.4x gross -- impossible on
    a spot book, and it fails as InsufficientBalance rather than loudly."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.35, 0.35, 0.35, 0.277)]
    runner._cap_gross(states)

    gross = sum(s.target_weight for s in states)
    # At or just under the cap -- `_cap_gross` also reserves a cost allowance,
    # see test_gross_cap_reserves_cash_for_fees.
    assert gross <= float(cfg.risk["max_gross_exposure"])
    assert gross == pytest.approx(float(cfg.risk["max_gross_exposure"]), rel=0.01)
    # Relative conviction preserved.
    assert states[0].target_weight > states[3].target_weight


def test_gross_cap_leaves_a_small_book_alone():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.2, 0.3)]
    runner._cap_gross(states)
    assert [s.target_weight for s in states] == pytest.approx([0.2, 0.3])


def test_order_size_skips_below_the_exchange_minimum():
    assert order_size_from_weight(0.0001, 1e9, 1.55e11, 0.0, 1e-6, 3e6) == 0.0
    assert order_size_from_weight(0.3, 1e9, 1.55e11, 0.0, 1e-6, 3e6) > 0


def test_chandelier_stop_triggers_only_past_the_threshold():
    assert not chandelier_exit_hit(True, price=95, peak_price=100, atr_value=2, multiple=3)
    assert chandelier_exit_hit(True, price=93, peak_price=100, atr_value=2, multiple=3)


def test_drawdown_kill_switch():
    equity = pd.Series([100.0, 120.0, 110.0, 85.0])
    breached = drawdown_breached(equity, 0.25)
    assert not breached.iloc[2]
    assert breached.iloc[3]


def test_gross_cap_reserves_cash_for_fees():
    """Investing exactly 100% of equity overdraws, because fees are charged on
    top of notional. Observed in a shadow test as a negative `rls` balance."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    states = [SymbolState(spec=cfg.symbol("BTCIRT"), target_weight=w) for w in (0.35, 0.35, 0.35, 0.277)]
    runner._cap_gross(states)

    gross = sum(s.target_weight for s in states)
    cost_rate = float(cfg.costs["taker_fee"]) + float(cfg.costs["slippage"])
    assert gross < float(cfg.risk["max_gross_exposure"])
    # Enough cash left to pay the fees on the whole book.
    assert 1.0 - gross >= gross * cost_rate * 0.99


def test_position_limit_is_derived_from_equity():
    """A wide universe on a small account otherwise places no orders at all:
    every position falls under the 3,000,000 rial minimum and is skipped."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:10]
    states = [SymbolState(spec=s, target_weight=0.3, score=0.9 - i * 0.05)
              for i, s in enumerate(specs)]

    # 10M rial / 3M minimum -> 3 fundable positions.
    runner._limit_positions(states, equity=10_000_000)
    funded = [s for s in states if s.target_weight != 0.0]
    assert len(funded) == 3


def test_position_limit_keeps_the_highest_conviction_names():
    """When the cap binds, the surviving book is the highest-scoring names."""
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 2
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:5]
    scores = [0.1, 0.9, 0.3, 0.8, 0.2]
    states = [SymbolState(spec=s, target_weight=0.3, score=sc)
              for s, sc in zip(specs, scores, strict=True)]

    runner._limit_positions(states, equity=100_000_000)
    funded = {s.spec.nobitex for s in states if s.target_weight != 0.0}
    assert funded == {specs[1].nobitex, specs[3].nobitex}


def test_selection_declines_rather_than_oversizing():
    """If even one position cannot clear the minimum at its vol-targeted
    weight, the book is empty. Scaling a position UP to clear the minimum
    would take more risk than the model sized for -- silently."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:5]
    states = [SymbolState(spec=s, target_weight=0.3, score=0.9) for s in specs]

    # 0.3 x 7,000,000 = 2.1M, under the 3M minimum.
    runner._limit_positions(states, equity=7_000_000)
    assert all(s.target_weight == 0.0 for s in states)


def test_selection_prefers_a_smaller_fundable_book_to_a_larger_broken_one():
    """Observed live: top-3 by score produced one fill and two rejections.
    Freed weight from a dropped name is what lifts the rest over the minimum.
    """
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    equity = 9_950_000.0
    specs = cfg.universe[:3]
    # The live weights: only the largest clears 3M on its own.
    states = [
        SymbolState(spec=specs[0], target_weight=0.350, score=0.95),
        SymbolState(spec=specs[1], target_weight=0.231, score=0.96),
        SymbolState(spec=specs[2], target_weight=0.197, score=0.93),
    ]
    runner._limit_positions(states, equity)

    funded = [s for s in states if s.target_weight != 0.0]
    assert funded, "should fund at least the one viable position"
    for s in funded:
        assert s.target_weight * equity >= float(cfg.costs["min_order_rial"])


def test_a_large_account_keeps_every_signal():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    specs = cfg.universe[:5]
    states = [SymbolState(spec=s, target_weight=0.3, score=0.5) for s in specs]

    runner._limit_positions(states, equity=10_000_000_000)
    assert all(s.target_weight != 0.0 for s in states)


def test_paper_equity_peak_cannot_halt_a_live_account():
    """A paper run ends with a 1e9 peak. Loaded into a live account holding
    10M that reads as -99% and trips the kill switch before the first order --
    observed live."""
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state.json"
    path.write_text(json.dumps({"equity_peak_rial": 1_000_000_000.0, "halted": True,
                                "last_bar_ts": {}}))
    try:
        state = RunnerState.load(path, current_equity=9_950_000)
        assert state.equity_peak_rial == 9_950_000
        assert not state.halted
    finally:
        path.unlink()


def test_a_plausible_peak_survives_reload():
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state2.json"
    path.write_text(json.dumps({"equity_peak_rial": 11_000_000.0, "halted": False,
                                "last_bar_ts": {}}))
    try:
        state = RunnerState.load(path, current_equity=9_950_000)
        assert state.equity_peak_rial == 11_000_000.0
    finally:
        path.unlink()


def test_state_path_is_namespaced_by_mode():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "nbtrend" / "live" / "runner.py"
    ).read_text()
    assert 'f"data/state/runner-{cfg.mode}.json"' in source


def test_gross_cap_can_push_a_kept_position_under_the_minimum():
    """Observed live: _limit_positions kept 3 names, then vol targeting and
    _cap_gross scaled two of them under 3,000,000 rial and both were rejected
    one at a time. Dropping the smallest frees weight for the survivors."""
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    equity = 9_950_000.0
    specs = cfg.universe[:3]
    weights = [0.350, 0.231, 0.197]      # the live numbers
    states = [SymbolState(spec=s, target_weight=w, score=0.9)
              for s, w in zip(specs, weights, strict=True)]

    runner._drop_unfundable(states, equity)

    live = [s for s in states if s.target_weight != 0.0]
    assert live, "should keep at least one fundable position"
    for s in live:
        assert s.target_weight * equity >= float(cfg.costs["min_order_rial"])


def test_drop_unfundable_leaves_a_well_funded_book_alone():
    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    specs = cfg.universe[:2]
    states = [SymbolState(spec=s, target_weight=0.35, score=0.9) for s in specs]

    runner._drop_unfundable(states, equity=1_000_000_000)
    assert all(s.target_weight == 0.35 for s in states)


def test_a_persisted_halt_does_not_survive_a_healthy_restart():
    """The bug that made the live bot sell its book every cycle, forever.

    `halted` was saved true after the stale-peak incident, but the peak on disk
    was plausible against the account, so the stale-peak reset -- the only path
    that cleared the flag -- never fired. Every symbol then scored
    `target_weight = 0.0` and the runner liquidated on every pass, silently.
    """
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state3.json"
    path.write_text(json.dumps({"equity_peak_rial": 9_952_362.0, "halted": True,
                                "last_bar_ts": {}}))
    try:
        state = RunnerState.load(path, current_equity=9_861_642, max_drawdown_stop=0.25)
        assert not state.halted, "a -0.9% drawdown must not keep the kill switch on"
        assert state.equity_peak_rial == 9_952_362.0
    finally:
        path.unlink()


def test_a_halt_survives_a_restart_that_is_still_in_breach():
    import json

    from nbtrend.live.runner import RunnerState

    path = pathlib.Path(__file__).parent / "_tmp_state4.json"
    path.write_text(json.dumps({"equity_peak_rial": 10_000_000.0, "halted": True,
                                "last_bar_ts": {}}))
    try:
        # 6M against a 10M peak is -40%, past the 25% stop.
        state = RunnerState.load(path, current_equity=6_000_000, max_drawdown_stop=0.25)
        assert state.halted, "restarting must not launder a real drawdown breach"
    finally:
        path.unlink()


def test_an_incumbent_is_not_evicted_on_a_marginal_score_edge():
    """Anti-churn. Observed live: CVX was bought one cycle and slated for
    eviction the next on a few hundredths of score, paying a full round trip
    (0.22% fees plus a ~0.6% spread) to swap one trending name for another."""
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 1
    cfg.raw["execution"]["incumbent_score_bonus"] = 0.10
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    incumbent, challenger = cfg.universe[0], cfg.universe[1]
    runner._book = {incumbent.nobitex}
    states = [
        SymbolState(spec=incumbent, target_weight=0.3, score=0.87),
        SymbolState(spec=challenger, target_weight=0.3, score=0.90),
    ]

    runner._limit_positions(states, equity=100_000_000)
    funded = {s.spec.nobitex for s in states if s.target_weight != 0.0}
    assert funded == {incumbent.nobitex}, "a 0.03 edge must not pay a round trip"


def test_a_decisively_better_challenger_still_takes_the_slot():
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 1
    cfg.raw["execution"]["incumbent_score_bonus"] = 0.10
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    incumbent, challenger = cfg.universe[0], cfg.universe[1]
    runner._book = {incumbent.nobitex}
    states = [
        SymbolState(spec=incumbent, target_weight=0.3, score=0.50),
        SymbolState(spec=challenger, target_weight=0.3, score=0.95),
    ]

    runner._limit_positions(states, equity=100_000_000)
    funded = {s.spec.nobitex for s in states if s.target_weight != 0.0}
    assert funded == {challenger.nobitex}


def test_an_incumbent_just_under_the_minimum_keeps_its_slot():
    """The 3,000,000 rial floor is a minimum ORDER size, not a minimum holding.

    Live stalemate: CVX was held, its vol-targeted size came to 2,970,000, so
    selection dropped it and handed the slot to a name there was no cash to buy
    -- while the CVX sell was itself skipped for being under the minimum. The
    book could not move in either direction.
    """
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 1
    cfg.raw["risk"]["max_weight_per_symbol"] = 1.0
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    incumbent = cfg.universe[0]
    runner._book = {incumbent.nobitex}
    equity = 10_000_000
    # 0.297 * 10,000,000 = 2,970,000 -- just under the 3,000,000 minimum.
    states = [SymbolState(spec=incumbent, target_weight=0.297, score=0.87)]

    runner._limit_positions(states, equity=equity)
    assert states[0].target_weight != 0.0, "holding a position places no order"


def test_drop_unfundable_leaves_incumbents_alone():
    """The other half of the CVX stalemate.

    `_limit_positions` was taught that the 3,000,000 rial floor limits orders,
    not holdings -- but `_drop_unfundable` then dropped the same incumbent a
    moment later, setting a sell that was itself under the minimum and skipped.
    The book re-decided the identical trade every cycle and never moved.
    """
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    incumbent, newcomer = cfg.universe[0], cfg.universe[1]
    runner._book = {incumbent.nobitex}
    equity = 10_000_000
    states = [
        SymbolState(spec=incumbent, target_weight=0.254, score=0.87),  # 2,540,000
        SymbolState(spec=newcomer, target_weight=0.10, score=0.80),    # 1,000,000
    ]

    runner._drop_unfundable(states, equity=equity)
    assert states[0].target_weight != 0.0, "an owned position needs no order to hold"
    assert states[1].target_weight == 0.0, "an unowned sub-minimum name is still dropped"


def test_startup_equity_values_positions_not_just_cash():
    """A fully invested account holds almost no rial.

    Marking only the cash made every restart look catastrophic -- observed
    live, a 9,839,505 peak discarded against an apparent equity of 29,125 --
    which resets the high-water mark the drawdown stop is measured from.
    """
    import copy
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    spec = next(s for s in cfg.universe if s.nobitex == "BTCIRT")
    runner.states = {"BTCIRT": SymbolState(spec=spec)}
    runner.broker = SimpleNamespace(balances=lambda: {"rls": 29_125.0, "btc": 0.06})
    runner.rest = SimpleNamespace(orderbook=lambda _s: SimpleNamespace(mid=150_000_000.0))

    equity = runner._safe_equity()
    assert equity == 29_125.0 + 0.06 * 150_000_000.0


def test_startup_equity_survives_an_unmarkable_holding():
    import copy
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    spec = next(s for s in cfg.universe if s.nobitex == "BTCIRT")
    runner.states = {"BTCIRT": SymbolState(spec=spec)}
    runner.broker = SimpleNamespace(balances=lambda: {"rls": 29_125.0, "btc": 0.06})

    def _boom(_symbol):
        raise RuntimeError("orderbook unavailable")

    runner.rest = SimpleNamespace(orderbook=_boom)
    assert runner._safe_equity() == 29_125.0, "must degrade, not raise"


def test_selection_does_not_re_announce_a_book_it_already_capped(caplog):
    """The log flood at 25 incumbents.

    Every rejected candidate restored the survivors and re-capped them, but
    that book was already in-limit from when it was accepted, so `_cap_gross`
    only re-emitted an identical line. Live, one cycle produced ~350 lines
    oscillating between gross 7.491 and 7.841 -- two fixed points, alternating
    accept/reject. The scaling itself was correct; the noise made a real cycle
    impossible to read.
    """
    import copy
    import logging

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 0
    cfg.raw["risk"]["max_weight_per_symbol"] = 0.35
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    runner._book = set()

    # Enough names, each oversized, that the cap binds on nearly every trial.
    states = [
        SymbolState(spec=spec, target_weight=0.30, score=0.90 - i * 0.01)
        for i, spec in enumerate(cfg.universe[:30])
    ]

    with caplog.at_level(logging.INFO, logger="nbtrend.live.runner"):
        runner._limit_positions(states, equity=100_000_000)

    scaling_lines = [r for r in caplog.records if "scaling every weight" in r.message]
    assert not scaling_lines, (
        f"the search must not narrate itself; got {len(scaling_lines)} scaling lines"
    )
    settled = [r for r in caplog.records if "gross exposure settled" in r.message]
    assert len(settled) == 1, "the surviving book is summarised exactly once"


def test_the_gross_cap_still_actually_binds():
    """Quieting the search must not stop it capping. Same setup as above, but
    asserting on the weights rather than the log."""
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 0
    cfg.raw["risk"]["max_weight_per_symbol"] = 0.35
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    runner._book = set()

    states = [
        SymbolState(spec=spec, target_weight=0.30, score=0.90 - i * 0.01)
        for i, spec in enumerate(cfg.universe[:30])
    ]
    runner._limit_positions(states, equity=100_000_000)

    gross = sum(abs(s.target_weight) for s in states)
    cost_rate = float(cfg.costs["taker_fee"]) + float(cfg.costs["slippage"])
    limit = float(cfg.risk["max_gross_exposure"]) / (1.0 + cost_rate)
    assert gross <= limit + 1e-9, f"gross {gross:.4f} breached the {limit:.4f} cap"


def test_incumbency_tracks_holdings_not_intent(monkeypatch):
    """The BCH round trip.

    Selection (pass 2) recorded `_book` from the names it CHOSE, but execution
    is pass 4 and most chosen names are never bought -- their order lands under
    the exchange minimum and is skipped. A name bought on the last of the cash
    was therefore not an incumbent on the next cycle, got re-judged as a fresh
    candidate, and was sold straight back. Live: BCH bought at 544,754,000 and
    sold at 543,255,000 one cycle later, paying a full round trip for nothing.

    `_limit_positions` must leave `_book` alone; only observed holdings set it.
    """
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["max_positions"] = 2
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    before = {"ALREADY_HELD"}
    runner._book = before

    states = [
        SymbolState(spec=cfg.universe[0], target_weight=0.3, score=0.90),
        SymbolState(spec=cfg.universe[1], target_weight=0.3, score=0.88),
    ]
    runner._limit_positions(states, equity=100_000_000)

    assert runner._book is before, (
        "selection must not overwrite incumbency with names it merely intends "
        "to buy; holdings are the only source of truth"
    )


def test_cash_reserve_is_withheld_from_sizing():
    """Reserved rial must not be counted as investable.

    Sizing on TOTAL equity would target gross 1.0 of everything including the
    reserve, so the bot spends it on the next dip and the money is not there
    when the operator needs it -- here, to fund the margin wallet by hand,
    which the API key has no scope to do.
    """
    import copy

    from nbtrend.live.runner import LiveRunner

    cfg = copy.deepcopy(load_config())
    cfg.raw["execution"]["cash_reserve_rial"] = 1_000_000
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    equity = 90_000_000.0
    assert runner._cash_reserve() == 1_000_000
    assert max(0.0, equity - runner._cash_reserve()) == 89_000_000.0


def test_cash_reserve_defaults_to_zero_when_absent():
    import copy

    from nbtrend.live.runner import LiveRunner

    cfg = copy.deepcopy(load_config())
    cfg.raw["execution"].pop("cash_reserve_rial", None)
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    assert runner._cash_reserve() == 0.0


def test_a_reserve_larger_than_equity_cannot_make_investable_negative():
    """A negative investable equity would flip every weight's sign."""
    import copy

    from nbtrend.live.runner import LiveRunner

    cfg = copy.deepcopy(load_config())
    cfg.raw["execution"]["cash_reserve_rial"] = 500_000_000
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    assert max(0.0, 90_000_000.0 - runner._cash_reserve()) == 0.0


def test_sells_are_executed_before_buys():
    """A fully invested book rotates by "sell A to afford B".

    In arbitrary order B's buy is attempted while the cash is still in A, gets
    trimmed to whatever rial is spare, lands under the 3,000,000 minimum and is
    skipped -- so the rotation needs a second cycle and only completes if the
    signal still holds. 403 live cycles produced 6 buys.
    """
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = {s.nobitex: s for s in cfg.universe}
    sol = specs["SOLIRT"]
    btc = specs["BTCIRT"]
    equity = 100_000_000.0

    # SOL is held and being cut to zero; BTC is a fresh buy.
    seller = SymbolState(spec=sol, target_weight=0.0, score=0.1)
    seller.book = SimpleNamespace(mid=200_000_000.0)
    buyer = SymbolState(spec=btc, target_weight=0.30, score=0.9)
    buyer.book = SimpleNamespace(mid=150_000_000_000.0)

    base_sol, _ = "sol", "rls"
    runner.broker = SimpleNamespace(balances=lambda: {base_sol: 0.2, "rls": 0.0})

    # Buyer deliberately placed first, the order that loses the cash.
    ordered = runner._sells_before_buys([buyer, seller], equity)
    assert ordered[0] is seller, "the reduction must run first to fund the buy"
    assert ordered[1] is buyer


def test_ordering_falls_back_to_list_order_if_balances_fail():
    """An unreadable balance must not stop the cycle executing."""
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = load_config()
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    def boom():
        raise RuntimeError("api down")

    runner.broker = SimpleNamespace(balances=boom)
    states = [SymbolState(spec=s) for s in cfg.universe[:3]]
    assert runner._sells_before_buys(states, 1_000_000.0) == states


def test_rebalance_band_stays_above_the_exchange_minimum():
    """A band below min_order_rial/equity lets through adjustments the exchange
    rejects -- turnover on paper, skipped orders in practice."""
    cfg = load_config()
    band = float(cfg.execution["min_rebalance_weight"])
    min_order = float(cfg.costs["min_order_rial"])
    equity = 90_000_000.0
    assert band * equity >= min_order, (
        f"band {band} allows {band * equity:,.0f} rial trades, "
        f"under the {min_order:,.0f} minimum"
    )


def test_idle_book_is_parked_in_usdt_not_rial():
    """Rial is not a neutral resting place, it is a losing position.

    Over the backtest window holding USDT returned +249.9% in rial terms while
    the strategy returned +2.9% to +76.3%: the rial devalued ~71% against the
    dollar. `fx_floor_weight` described this in the config ("the rial leg is a
    one-way carry") but was parsed and never used, so nothing implemented it.
    """
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:3]
    # Crypto sleeve claims 40% of the book; the other 60% must go to USDT.
    states = [SymbolState(spec=s, target_weight=w) for s, w in zip(specs, [0.2, 0.1, 0.1], strict=True)]
    assert runner._fx_target_weight(states) == pytest.approx(0.60)


def test_a_fully_invested_book_leaves_nothing_for_usdt():
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:2]
    states = [SymbolState(spec=s, target_weight=w) for s, w in zip(specs, [0.6, 0.4], strict=True)]
    assert runner._fx_target_weight(states) == pytest.approx(0.0)


def test_fx_weight_never_goes_negative_when_gross_exceeds_one():
    """An over-allocated book must not produce a negative USDT target, which
    would read as a short on the currency leg."""
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:2]
    states = [SymbolState(spec=s, target_weight=w) for s, w in zip(specs, [0.9, 0.5], strict=True)]
    assert runner._fx_target_weight(states) == 0.0


def test_fx_floor_raises_the_usdt_leg_even_when_fully_invested():
    import copy

    from nbtrend.live.runner import LiveRunner, SymbolState

    cfg = copy.deepcopy(load_config())
    cfg.raw["risk"]["fx_floor_weight"] = 0.25
    runner = object.__new__(LiveRunner)
    runner.cfg = cfg

    specs = cfg.universe[:2]
    states = [SymbolState(spec=s, target_weight=w) for s, w in zip(specs, [0.6, 0.4], strict=True)]
    assert runner._fx_target_weight(states) == pytest.approx(0.25)


def test_an_implausible_equity_jump_is_rejected():
    """One live cycle read 3,725,744,251 rial against a real ~88,000,000
    account -- 42x -- and funded 56 of 56 signals on it. Equity feeds position
    SIZE, so a bad reading decides how much money gets spent."""
    from nbtrend.live.runner import LiveRunner

    jump = LiveRunner.MAX_EQUITY_JUMP
    # The 42x glitch is caught.
    assert 88_000_000 * jump < 3_725_744_251
    # Ordinary movement passes untouched.
    assert not 88_000_000 * jump < 90_000_000
    # A real deposit that happened on this account (9.84M -> 29.7M) passes too;
    # the threshold must not be so tight that funding the account trips it.
    assert not 9_840_000 * jump < 29_700_000


def test_a_rejected_equity_jump_is_accepted_once_it_is_corroborated():
    """The guard must not latch. Rejecting without updating the baseline would
    refuse a genuine deposit forever -- a glitch does not reproduce, a deposit
    does, which is exactly what separates them."""
    from nbtrend.live.runner import LiveRunner

    runner = object.__new__(LiveRunner)
    runner._last_equity = 88_000_000.0
    runner._suspect_equity = 0.0

    glitch = 3_725_744_251.0
    assert glitch > runner._last_equity * LiveRunner.MAX_EQUITY_JUMP
    # First sighting: remembered, not acted on.
    runner._suspect_equity = glitch

    # A second, agreeing reading corroborates it.
    again = glitch * 1.02
    assert abs(again - runner._suspect_equity) <= runner._suspect_equity * 0.10

    # A different wild value does not corroborate the first.
    unrelated = glitch * 3
    assert not abs(unrelated - runner._suspect_equity) <= runner._suspect_equity * 0.10
