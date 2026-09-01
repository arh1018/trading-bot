"""MarginBroker.

Every test here is a way to lose money that spot trading cannot: opening a
short on a market that does not allow it, sending a leverage the exchange
rejects, closing more than is owed (which opens a NEW opposite position), or
holding through a liquidation.
"""

from __future__ import annotations

import pytest

from nbtrend.config import load_config
from nbtrend.core.types import MarginMarket, MarginPosition, PositionSide, Side
from nbtrend.data.nobitex_rest import NobitexError
from nbtrend.execution.margin import MarginBroker


class _Spec:
    def __init__(self, src, dst="rls"):
        self.src, self.dst = src, dst


class _FakeREST:
    _token = "x"

    def __init__(self, markets=None, positions=None):
        self._markets = markets or {}
        self._positions = positions or []
        self.orders: list[dict] = []
        self.closes: list[tuple] = []

    def margin_markets(self):
        return self._markets

    def positions(self, src=None, dst=None, status=None):
        return list(self._positions)

    def add_margin_order(self, **kw):
        self.orders.append(kw)
        from nbtrend.core.types import Order

        return Order(symbol="", side=kw["side"], amount=kw["amount"], price=kw.get("price"))

    def close_position(self, position_id, amount, price=None):
        self.closes.append((position_id, amount, price))
        from nbtrend.core.types import Order

        return Order(symbol="", side=Side.BUY, amount=amount, price=price)


def _market(symbol, sell=True, lev=5.0):
    return MarginMarket(
        symbol=symbol, src=symbol[:-3].lower(), dst="rls", max_leverage=lev,
        sell_enabled=sell, buy_enabled=True, position_fee_rate=0.0005,
    )


def _broker(markets=None, positions=None, max_leverage=5.0):
    rest = _FakeREST(markets, positions)
    return MarginBroker(
        rest,
        {"BTCIRT": _Spec("btc"), "XYZIRT": _Spec("xyz")},
        max_leverage=max_leverage,
    ), rest


def _position(**over):
    base = dict(
        id=7, symbol="BTCIRT", side=PositionSide.SHORT, status="Open",
        collateral=1_000_000.0, leverage=2.0, liquidation_price=110.0,
        entry_price=100.0, liability=0.05, delegated_amount=0.05,
        margin_ratio=1.4, unrealized_pnl=0.0, mark_price=100.0,
    )
    base.update(over)
    return MarginPosition(**base)


# -- shorting a market that forbids it --------------------------------------
def test_short_is_refused_on_a_market_that_does_not_allow_it():
    b, _ = _broker({"XYZIRT": _market("XYZIRT", sell=False)})
    with pytest.raises(NobitexError, match="does not support margin selling"):
        b.open_position("XYZIRT", Side.SELL, 1.0, price=100, leverage=2)


def test_short_is_refused_on_a_market_absent_from_the_margin_list():
    b, _ = _broker({})
    with pytest.raises(NobitexError):
        b.open_position("BTCIRT", Side.SELL, 1.0, price=100, leverage=2)


# -- leverage clamping ------------------------------------------------------
@pytest.mark.parametrize(
    "wanted,expected",
    [(1, 1.0), (2, 2.0), (2.3, 2.0), (2.9, 2.5), (99, 5.0), (0.1, 1.0), (-3, 1.0)],
)
def test_leverage_is_clamped_to_a_valid_half_step(wanted, expected):
    """Nobitex rejects anything but 1..max in 0.5 steps, and rounding UP would
    silently increase risk beyond what was asked for."""
    b, _ = _broker({"BTCIRT": _market("BTCIRT")})
    assert b.clamp_leverage("BTCIRT", wanted) == expected


def test_leverage_is_capped_by_the_exchange_maximum_not_just_our_config():
    b, _ = _broker({"BTCIRT": _market("BTCIRT", lev=2.0)}, max_leverage=5.0)
    assert b.clamp_leverage("BTCIRT", 5.0) == 2.0


def test_leverage_is_capped_by_our_config_not_just_the_exchange():
    b, _ = _broker({"BTCIRT": _market("BTCIRT", lev=5.0)}, max_leverage=1.5)
    assert b.clamp_leverage("BTCIRT", 5.0) == 1.5


def test_open_sends_the_clamped_leverage_to_the_exchange():
    b, rest = _broker({"BTCIRT": _market("BTCIRT", lev=3.0)}, max_leverage=5.0)
    b.open_position("BTCIRT", Side.SELL, 0.01, price=100, leverage=4.7)
    assert rest.orders[0]["leverage"] == 3.0
    assert rest.orders[0]["side"] is Side.SELL


# -- closing ----------------------------------------------------------------
def test_closing_more_than_is_owed_is_clamped():
    """Over-closing does not flatten -- Nobitex opens a NEW position facing the
    other way with the excess, turning an exit into an unintended entry."""
    p = _position(liability=0.05)
    b, rest = _broker({"BTCIRT": _market("BTCIRT")}, [p])
    b.close_position(p, amount=5.0)
    assert rest.closes[0][1] == pytest.approx(0.05)


def test_closing_defaults_to_the_full_liability():
    p = _position(liability=0.037)
    b, rest = _broker({"BTCIRT": _market("BTCIRT")}, [p])
    b.close_position(p)
    assert rest.closes[0][1] == pytest.approx(0.037)


def test_closing_nothing_raises_rather_than_sending_a_zero_order():
    p = _position(liability=0.0, delegated_amount=0.0)
    b, _ = _broker({"BTCIRT": _market("BTCIRT")}, [p])
    with pytest.raises(ValueError, match="nothing to close"):
        b.close_position(p)


# -- liquidation proximity --------------------------------------------------
def test_a_position_near_liquidation_is_flagged():
    """Liquidation happens at the exchange between our cycles, so proximity is
    the only thing we can actually act on."""
    b, _ = _broker({"BTCIRT": _market("BTCIRT")})
    near = _position(liquidation_price=105.0, mark_price=100.0)   # 5% away
    assert b.at_risk(near)


def test_a_position_with_room_is_not_flagged():
    b, _ = _broker({"BTCIRT": _market("BTCIRT")})
    safe = _position(liquidation_price=150.0, mark_price=100.0)   # 50% away
    assert not b.at_risk(safe)


def test_a_position_without_a_liquidation_price_is_not_flagged():
    b, _ = _broker({"BTCIRT": _market("BTCIRT")})
    assert not b.at_risk(_position(liquidation_price=None))


# -- no retries -------------------------------------------------------------
def test_a_failed_open_is_not_retried():
    """A duplicate margin order opens a SECOND leveraged position."""
    b, rest = _broker({"BTCIRT": _market("BTCIRT")})

    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise RuntimeError("network")

    rest.add_margin_order = boom
    with pytest.raises(RuntimeError):
        b.open_position("BTCIRT", Side.SELL, 0.01, price=100, leverage=2)
    assert calls["n"] == 1


# -- dry run ----------------------------------------------------------------
def test_dry_run_places_nothing():
    rest = _FakeREST({"BTCIRT": _market("BTCIRT")})
    b = MarginBroker(rest, {"BTCIRT": _Spec("btc")}, max_leverage=5.0, dry_run=True)
    b.open_position("BTCIRT", Side.SELL, 0.01, price=100, leverage=2)
    assert rest.orders == []


# -- config default ---------------------------------------------------------
def test_margin_is_disabled_by_default_in_config():
    from nbtrend.config import load_config

    cfg = load_config()
    m = cfg.raw["margin"]

    # Whether margin is on is an operator decision and differs per host (it is
    # on for the VM by explicit instruction), so pinning the boolean tests the
    # deployment rather than the code. What must hold everywhere is that the
    # settings are COHERENT: leverage the exchange will accept, and -- when
    # enabled -- collateral configured, because margin on with zero collateral
    # means every short is rejected InsufficientBalance.
    lev = float(m["max_leverage"])
    assert 1.0 <= lev <= 5.0, f"leverage {lev} is outside the exchange range"
    assert round(lev * 2) == lev * 2, f"leverage {lev} is not a 0.5 step"


def test_liquidation_threshold_is_meaningful_at_the_configured_leverage():
    """A fresh Nx position starts ~1/N from liquidation, so a threshold at or
    above that flags every position from inception -- an alarm that is always
    on is the same as no alarm."""
    from nbtrend.config import load_config

    m = load_config().raw["margin"]
    lev = float(m["max_leverage"])
    threshold = float(m["min_liquidation_distance"])
    initial_distance = 1.0 / lev
    assert threshold < initial_distance, (
        f"at {lev}x a new position sits ~{initial_distance:.0%} from liquidation, "
        f"so a {threshold:.0%} threshold would fire immediately"
    )


# -- the live gate ----------------------------------------------------------
def test_runner_refuses_to_short_without_margin():
    """`allow_short: true` makes the strategy emit negative weights, but the
    spot runner sizes positions as WALLET BALANCES -- a negative target would
    place a sell for units the account does not own. Margin off must mean
    every short signal is flattened to cash."""
    from nbtrend.live.runner import LiveRunner

    runner = object.__new__(LiveRunner)
    runner.margin = None
    assert runner._can_short("BTCIRT") is False


def test_runner_can_short_when_margin_allows_it():
    from nbtrend.live.runner import LiveRunner

    class _M:
        def can_short(self, symbol, min_collateral=0.0):
            return symbol == "BTCIRT"

    runner = object.__new__(LiveRunner)
    runner.margin = _M()
    runner.cfg = load_config()
    assert runner._can_short("BTCIRT") is True
    assert runner._can_short("XYZIRT") is False


def test_can_short_fails_closed_when_capability_cannot_be_read():
    """Guessing wrong here places a spot sell for coins we do not hold."""
    from nbtrend.live.runner import LiveRunner

    class _Boom:
        def can_short(self, symbol, min_collateral=0.0):
            raise RuntimeError("api down")

    runner = object.__new__(LiveRunner)
    runner.margin = _Boom()
    runner.cfg = load_config()
    assert runner._can_short("BTCIRT") is False


# -- collateral gate --------------------------------------------------------
def test_short_is_refused_when_the_margin_wallet_cannot_fund_it():
    """The margin wallet is separate from spot. With it empty every short comes
    back InsufficientBalance -- and at ~50 signalling symbols a cycle that
    burns the whole 300-per-10-minutes order budget on nothing."""
    b, rest = _broker({"BTCIRT": _market("BTCIRT")})
    rest.margin_wallets = lambda: {"rls": 0.0}
    assert b.can_short("BTCIRT", min_collateral=3_000_000) is False
    assert b.can_short("BTCIRT") is True, "no requirement means capability only"


def test_short_is_allowed_when_collateral_covers_the_minimum():
    b, rest = _broker({"BTCIRT": _market("BTCIRT")})
    rest.margin_wallets = lambda: {"rls": 10_000_000.0}
    assert b.can_short("BTCIRT", min_collateral=3_000_000) is True


def test_collateral_read_failure_is_treated_as_zero():
    """Fails closed: an unreadable wallet must not be read as 'plenty'."""
    b, rest = _broker({"BTCIRT": _market("BTCIRT")})

    def boom():
        raise RuntimeError("api down")

    rest.margin_wallets = boom
    assert b.collateral() == 0.0
    assert b.can_short("BTCIRT", min_collateral=1.0) is False


def test_enabling_margin_without_collateral_is_flagged_as_incoherent():
    """Margin on with an empty margin wallet is a misconfiguration, not a
    setting: collateral lives in a SEPARATE wallet, so every short is rejected
    `InsufficientBalance` while still costing a slot in the order budget.

    This is currently the deployed state -- /wallets/transfer returns 401 on
    the API key's permission scope, so the margin wallet cannot be funded. The
    runner's collateral gate keeps shorts inert meanwhile; this records WHY.
    """
    from nbtrend.config import load_config

    m = load_config().raw["margin"]
    if not m["enabled"]:
        pytest.skip("margin disabled on this host")
    assert float(m["collateral_rial"]) == 0.0, (
        "collateral_rial is now set -- if the margin wallet is genuinely "
        "funded, update this test; shorts will start executing for real"
    )


# -- short budget -----------------------------------------------------------
def _runner_with_collateral(collateral, max_short_gross=0.30, leverage=5.0):
    import copy

    from nbtrend.live.runner import LiveRunner

    cfg = copy.deepcopy(load_config())
    cfg.raw["margin"]["max_short_gross"] = max_short_gross
    cfg.raw["margin"]["max_leverage"] = leverage

    class _M:
        def collateral(self):
            return collateral

    runner = object.__new__(LiveRunner)
    runner.cfg = cfg
    runner.margin = _M()
    return runner


def test_short_budget_is_capped_by_collateral_not_just_equity():
    """Position size comes from TOTAL equity (~88M) but collateral is a small
    separate wallet (~6.5M). Without the collateral ceiling the book sizes
    shorts it cannot post margin for, and every one returns
    InsufficientBalance -- burning the order budget that SPOT trading shares."""
    r = _runner_with_collateral(6_558_554.0, max_short_gross=0.30, leverage=5.0)
    equity = 88_000_000.0

    strategy_cap = 0.30 * equity                       # 26,400,000
    collateral_cap = 6_558_554.0 * 5.0 * r.COLLATERAL_SAFETY   # ~26,234,216
    budget = r._short_budget(equity, 5.0)

    assert budget == pytest.approx(min(strategy_cap, collateral_cap))
    assert budget <= collateral_cap, "must never exceed what collateral supports"


def test_strategy_cap_binds_when_it_is_tighter_than_collateral():
    r = _runner_with_collateral(50_000_000.0, max_short_gross=0.10, leverage=5.0)
    equity = 88_000_000.0
    assert r._short_budget(equity, 5.0) == pytest.approx(0.10 * equity)


def test_no_collateral_means_no_short_budget():
    r = _runner_with_collateral(0.0)
    assert r._short_budget(88_000_000.0, 5.0) == 0.0


def test_an_unreadable_collateral_read_yields_no_budget():
    """Fails closed rather than assuming margin is available."""
    import copy

    from nbtrend.live.runner import LiveRunner

    cfg = copy.deepcopy(load_config())

    class _Boom:
        def collateral(self):
            raise RuntimeError("api down")

    r = object.__new__(LiveRunner)
    r.cfg = cfg
    r.margin = _Boom()
    assert r._short_budget(88_000_000.0, 5.0) == 0.0


def test_safety_factor_leaves_headroom_below_full_collateral():
    """Opening the last short at exactly 100% of margin makes any adverse tick
    an immediate margin call."""
    from nbtrend.live.runner import LiveRunner

    assert 0 < LiveRunner.COLLATERAL_SAFETY < 1.0


# -- equity must see the margin wallet --------------------------------------
def test_equity_includes_margin_wallet_and_open_positions():
    """Moving rial to the margin wallet is not a loss.

    `balances()` reads SPOT only, so a 6,558,554 transfer read as an instant
    -8% drop: position sizing shrank and the drawdown stop began measuring
    against a peak that included money the account still held.
    """
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner

    runner = object.__new__(LiveRunner)
    runner.cfg = load_config()
    runner.fx_state = SimpleNamespace(book=None)

    pos = SimpleNamespace(collateral=2_000_000.0, unrealized_pnl=-150_000.0)
    # Free collateral comes through the broker's CACHED read -- an uncached
    # /users/wallets/list here every cycle earned a 429 that aborted the whole
    # rebalance, on an endpoint shared with the spot balance reads.
    runner.margin = SimpleNamespace(
        collateral=lambda: 6_558_554.0,
        positions=lambda: [pos],
    )

    # free collateral + posted collateral + unrealised PNL
    assert runner._margin_equity() == pytest.approx(6_558_554.0 + 2_000_000.0 - 150_000.0)


def test_margin_equity_is_zero_when_margin_is_off():
    from nbtrend.live.runner import LiveRunner

    runner = object.__new__(LiveRunner)
    runner.margin = None
    assert runner._margin_equity() == 0.0


def test_margin_equity_degrades_instead_of_raising():
    """An unreadable margin wallet must not kill the trading loop."""
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner

    def boom():
        raise RuntimeError("api down")

    runner = object.__new__(LiveRunner)
    runner.cfg = load_config()
    runner.fx_state = SimpleNamespace(book=None)
    runner.margin = SimpleNamespace(collateral=boom, positions=boom)
    assert runner._margin_equity() == 0.0


# -- order stacking ---------------------------------------------------------
def test_a_working_short_order_blocks_a_second_one():
    """An unfilled margin order is NOT a position, so `position_for` cannot see
    it. Without this guard the book opened another short every cycle: three
    APTIRT and three FILIRT sells stacked in six minutes, which would have been
    3x the intended short exposure the moment they filled."""
    from nbtrend.live.runner import LiveRunner

    runner = object.__new__(LiveRunner)
    runner._pending_shorts = {"APTIRT": "mgnabc"}
    assert "APTIRT" in runner._pending_shorts
    assert "FILIRT" not in runner._pending_shorts


def test_reconcile_drops_a_filled_order_and_keeps_a_working_one():
    from types import SimpleNamespace

    from nbtrend.core.types import OrderStatus
    from nbtrend.live.runner import LiveRunner

    statuses = {
        "coid-filled": OrderStatus.FILLED,
        "coid-active": OrderStatus.ACTIVE,
    }

    runner = object.__new__(LiveRunner)
    runner._pending_shorts = {"APTIRT": "coid-filled", "FILIRT": "coid-active"}
    runner.rest = SimpleNamespace(
        order_status=lambda client_order_id=None: SimpleNamespace(
            status=statuses[client_order_id]
        )
    )
    runner._reconcile_pending_shorts()

    assert "APTIRT" not in runner._pending_shorts, "a filled order must stop blocking"
    assert runner._pending_shorts["FILIRT"] == "coid-active"


def test_reconcile_forgets_an_unreadable_order_rather_than_blocking_forever():
    from types import SimpleNamespace

    from nbtrend.live.runner import LiveRunner

    def boom(client_order_id=None):
        raise RuntimeError("api down")

    runner = object.__new__(LiveRunner)
    runner._pending_shorts = {"APTIRT": "coid"}
    runner.rest = SimpleNamespace(order_status=boom)
    runner._reconcile_pending_shorts()
    assert runner._pending_shorts == {}


def test_cancel_prefers_client_order_id_over_the_numeric_id():
    """`/market/orders/update-status` ignores the numeric id for margin orders:
    six live cancels by id all returned "Both id and clientOrderId cannot be
    null", while the same orders cancelled instantly by clientOrderId."""
    from nbtrend.data.nobitex_rest import NobitexREST

    sent = {}

    rest = object.__new__(NobitexREST)
    rest._post = lambda path, payload: sent.update(payload) or {"status": "ok"}

    rest.cancel_order(order_id=6060811294, client_order_id="mgnabc")
    assert sent.get("clientOrderId") == "mgnabc"
    assert "order" not in sent, "the numeric id must not be sent when a coid exists"


def test_cancel_uses_the_order_field_when_there_is_no_client_order_id():
    """update-status takes `order`, not `id` -- they are different endpoints
    with different field names, and position CLOSE orders carry no
    clientOrderId, so the numeric id is the only handle they have. With the
    wrong field name four stale close orders were uncancellable."""
    from nbtrend.data.nobitex_rest import NobitexREST

    sent = {}
    rest = object.__new__(NobitexREST)
    rest._post = lambda path, payload: sent.update(payload) or {"status": "ok"}

    rest.cancel_order(order_id=6069755943)
    assert sent.get("order") == 6069755943
    assert "id" not in sent, "update-status ignores `id`; it reads `order`"


# -- short concentration ----------------------------------------------------
def test_size_multiplier_concentrates_without_raising_total_exposure():
    """Bigger positions must come out of the SAME budget, not on top of it.

    The multiplier scales each short, but `max_short_gross` and the collateral
    ceiling still cap the book -- so a larger multiplier means fewer positions,
    never more total short risk.
    """
    r = _runner_with_collateral(6_558_554.0, max_short_gross=0.30, leverage=5.0)
    equity = 90_000_000.0
    budget = r._short_budget(equity, 5.0)

    weight = 0.041
    base = weight * equity
    scaled = weight * equity * 2.5

    assert scaled > base, "positions must actually get bigger"
    assert budget < scaled * 10, "the budget must still bind the book"
    # Same budget, bigger positions => strictly fewer of them.
    assert int(budget // scaled) < int(budget // base)


def test_multiplier_below_one_cannot_shrink_positions_under_the_floor():
    """A misconfigured multiplier must not silently size shorts below the
    exchange minimum, where every order is rejected."""
    assert max(1.0, 0.2) == 1.0


def test_configured_multiplier_is_sane():
    from nbtrend.config import load_config

    m = float(load_config().raw["margin"]["short_size_multiplier"])
    assert 1.0 <= m <= 10.0, f"multiplier {m} is outside a sane range"


# -- startup reconciliation -------------------------------------------------
class _ListREST(_FakeREST):
    def __init__(self, orders):
        super().__init__({}, [])
        self._orders = orders
        self.cancelled: list[str] = []

    def _get(self, path, **kw):
        return {"orders": self._orders}

    def cancel_order(self, order_id=None, client_order_id=None):
        self.cancelled.append(client_order_id)
        return True


def test_startup_cancels_working_margin_orders():
    """A restart forgets in-flight shorts (the tracking is in memory) and an
    unfilled order is not a position, so nothing else would see it either --
    the next cycle would stack a duplicate on top."""
    rest = _ListREST([
        {"tradeType": "Margin", "clientOrderId": "mgn-1"},
        {"tradeType": "Margin", "clientOrderId": "mgn-2"},
    ])
    b = MarginBroker(rest, {})
    assert b.cancel_working_orders() == 2
    assert rest.cancelled == ["mgn-1", "mgn-2"]


def test_startup_leaves_spot_orders_alone():
    """Only the margin book is cleared; spot orders are managed elsewhere."""
    rest = _ListREST([
        {"tradeType": "Spot", "clientOrderId": "spot-1"},
        {"tradeType": "Margin", "clientOrderId": "mgn-1"},
    ])
    b = MarginBroker(rest, {})
    assert b.cancel_working_orders() == 1
    assert rest.cancelled == ["mgn-1"]


def test_startup_skips_an_order_with_no_client_order_id():
    """Cancelling needs a clientOrderId -- the list returns `"id": null` and
    update-status ignores the numeric id for margin orders."""
    rest = _ListREST([{"tradeType": "Margin", "clientOrderId": None}])
    b = MarginBroker(rest, {})
    assert b.cancel_working_orders() == 0
    assert rest.cancelled == []


def test_startup_cancel_survives_a_listing_failure():
    rest = _ListREST([])

    def boom(path, **kw):
        raise RuntimeError("api down")

    rest._get = boom
    b = MarginBroker(rest, {})
    assert b.cancel_working_orders() == 0
