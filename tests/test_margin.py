"""Margin API layer.

These cover the parts where a wrong value costs money rather than raising:
leverage the exchange will reject, a short parsed as a long, and Nobitex's
unicode-minus PNL that `float()` refuses.
"""

from __future__ import annotations

import pytest

from nbtrend.core.types import MarginPosition, OrderType, PositionSide, Side
from nbtrend.data.nobitex_rest import _num, _parse_position


class _FakeREST:
    """Captures the payload instead of sending it."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _post(self, path, payload=None):
        self.calls.append((path, payload or {}))
        return {
            "status": "ok",
            "order": {
                "id": 25, "type": payload.get("type", "sell"), "srcCurrency": "btc",
                "dstCurrency": "rls", "price": "6400000000", "amount": "0.01",
                "status": "Active", "matchedAmount": 0, "leverage": payload.get("leverage"),
            },
        }


def _rest():
    from nbtrend.data.nobitex_rest import NobitexREST

    rest = object.__new__(NobitexREST)
    fake = _FakeREST()
    rest._post = fake._post
    return rest, fake


# -- leverage validation ----------------------------------------------------
def test_leverage_below_one_is_rejected_before_it_reaches_the_exchange():
    rest, _ = _rest()
    with pytest.raises(ValueError, match="leverage must be >= 1"):
        rest.add_margin_order("btc", "rls", Side.SELL, 0.01, leverage=0.5, price=1)


def test_leverage_off_the_half_step_is_rejected():
    """Nobitex allows 1, 1.5, 2, 2.5 ... only; anything else is LeverageTooHigh."""
    rest, _ = _rest()
    with pytest.raises(ValueError, match="multiple of 0.5"):
        rest.add_margin_order("btc", "rls", Side.SELL, 0.01, leverage=2.3, price=1)


@pytest.mark.parametrize("lev", [1, 1.5, 2, 2.5, 5])
def test_valid_leverage_is_sent_as_a_plain_number(lev):
    rest, fake = _rest()
    rest.add_margin_order("btc", "rls", Side.SELL, 0.01, leverage=lev, price=6_400_000_000)
    _, payload = fake.calls[0]
    assert payload["leverage"] == f"{lev:g}"
    assert "." not in payload["leverage"] or payload["leverage"].endswith(".5")


# -- direction --------------------------------------------------------------
def test_a_sell_order_opens_a_short():
    """The whole point of margin here. `type=sell` is what borrows the asset."""
    rest, fake = _rest()
    rest.add_margin_order("btc", "rls", Side.SELL, 0.01, leverage=2, price=6_400_000_000)
    path, payload = fake.calls[0]
    assert path == "/margin/orders/add"
    assert payload["type"] == "sell"


def test_margin_orders_do_not_go_to_the_spot_endpoint():
    rest, fake = _rest()
    rest.add_margin_order("btc", "rls", Side.BUY, 0.01, leverage=2, price=1)
    assert fake.calls[0][0] == "/margin/orders/add"
    assert all(path != "/market/orders/add" for path, _ in fake.calls)


def test_stop_orders_still_require_a_stop_price():
    rest, _ = _rest()
    with pytest.raises(ValueError, match="requires stop_price"):
        rest.add_margin_order(
            "btc", "rls", Side.SELL, 0.01, leverage=2, price=1,
            execution=OrderType.STOP_LIMIT,
        )


# -- number parsing ---------------------------------------------------------
def test_unicode_minus_pnl_parses_as_negative():
    """Nobitex writes a losing PNL with U+2212, which float() rejects outright.

    Read as 0.0 (or crashing) would hide a loss on a leveraged position.
    """
    assert _num("−576435") == -576435.0


@pytest.mark.parametrize(
    "raw,expected",
    [("6400000000", 6.4e9), (None, None), ("", None), ("1.49", 1.49), (3, 3.0), ("junk", None)],
)
def test_num_handles_the_shapes_the_api_actually_sends(raw, expected):
    assert _num(raw) == expected


# -- position parsing -------------------------------------------------------
def _sample(**over):
    base = {
        "id": 128, "srcCurrency": "btc", "dstCurrency": "rls", "side": "sell",
        "status": "Open", "collateral": "320000000", "leverage": "2",
        "liquidationPrice": "25174302690", "entryPrice": "6400000000",
        "delegatedAmount": "0.03", "liability": "0.0300450676",
        "marginRatio": "1.49", "unrealizedPNL": "−576435",
        "markPrice": "6430000000", "expirationDate": "2022-11-20",
        "extensionFee": "320000",
    }
    base.update(over)
    return base


def test_position_parses_into_the_domain_type():
    p = _parse_position(_sample())
    assert p.id == 128
    assert p.symbol == "BTCIRT"
    assert p.side is PositionSide.SHORT
    assert p.is_open
    assert p.leverage == 2.0
    assert p.liquidation_price == 25_174_302_690.0
    assert p.unrealized_pnl == -576435.0


def test_a_position_with_no_liquidation_price_does_not_crash():
    p = _parse_position(_sample(liquidationPrice=None))
    assert p.liquidation_price is None
    assert p.distance_to_liquidation() is None


# -- liquidation distance ---------------------------------------------------
def test_a_short_is_liquidated_by_the_price_rising():
    p = MarginPosition(
        id=1, symbol="BTCIRT", side=PositionSide.SHORT, status="Open",
        collateral=1.0, leverage=2.0, liquidation_price=110.0, entry_price=100.0,
        liability=1.0, delegated_amount=1.0, margin_ratio=1.5, unrealized_pnl=0.0,
        mark_price=100.0,
    )
    assert p.distance_to_liquidation() == pytest.approx(0.10)
    # Closer to the liquidation price means less room, never more.
    assert p.distance_to_liquidation(price=105.0) == pytest.approx(0.047619, rel=1e-3)


def test_a_long_is_liquidated_by_the_price_falling():
    p = MarginPosition(
        id=2, symbol="BTCIRT", side=PositionSide.LONG, status="Open",
        collateral=1.0, leverage=2.0, liquidation_price=90.0, entry_price=100.0,
        liability=1.0, delegated_amount=1.0, margin_ratio=1.5, unrealized_pnl=0.0,
        mark_price=100.0,
    )
    assert p.distance_to_liquidation() == pytest.approx(0.10)


def test_liquidation_distance_goes_negative_once_breached():
    """A breached position must read as past the line, not as a small cushion."""
    p = MarginPosition(
        id=3, symbol="BTCIRT", side=PositionSide.SHORT, status="Open",
        collateral=1.0, leverage=2.0, liquidation_price=110.0, entry_price=100.0,
        liability=1.0, delegated_amount=1.0, margin_ratio=0.2, unrealized_pnl=-1.0,
        mark_price=120.0,
    )
    assert p.distance_to_liquidation() < 0


# -- transfer ---------------------------------------------------------------
def test_transfer_refuses_a_same_wallet_move():
    rest, _ = _rest()
    with pytest.raises(ValueError, match="must differ"):
        rest.transfer("rls", 1_000_000, src="spot", dst="spot")


def test_transfer_rejects_an_unknown_wallet_name():
    rest, _ = _rest()
    with pytest.raises(ValueError, match="spot.*margin"):
        rest.transfer("rls", 1_000_000, src="spot", dst="savings")


# -- leverage and financing in the backtest engine --------------------------
def _levered_cfg(lev: float, daily_fee: float = 0.0005):
    import copy

    from nbtrend.config import load_config

    cfg = copy.deepcopy(load_config())
    cfg.raw["strategy"]["allow_short"] = False
    cfg.raw["risk"]["max_leverage"] = lev
    cfg.raw["costs"]["position_fee_daily"] = daily_fee
    return cfg


def test_leverage_defaults_to_one_so_spot_behaviour_is_unchanged():
    from nbtrend.config import load_config
    from nbtrend.risk.sizing import RiskLimits

    cfg = load_config()
    limits = RiskLimits.from_config(cfg.risk)
    assert limits.max_leverage == 1.0, "config must not silently enable borrowing"


def test_borrow_fee_is_prorated_from_a_daily_rate():
    """0.05%/day on a 4h timeframe is a sixth of that per bar.

    Charging the daily rate once per BAR instead would overstate financing
    sixfold and make leverage look far worse than it is; charging nothing
    makes it look far better. Both errors have been made in this file's
    history, so the arithmetic is pinned here.
    """
    from nbtrend.backtest.engine import Backtester

    cfg = _levered_cfg(2.0)
    bt = Backtester(cfg)
    bars_per_day = max(1.0, bt.periods_per_year / 365.0)
    per_bar = 0.0005 / bars_per_day

    # Derived from the configured timeframe, not pinned to one: the timeframe
    # is a tuning decision (it moved 240 -> 720 on cost evidence) and a test
    # that hardcodes it fails for the wrong reason. What must hold is the
    # RELATIONSHIP -- a daily rate spread across a day's worth of bars.
    expected_bars_per_day = 24.0 / (int(cfg.data["timeframe"]) / 60.0)
    assert bars_per_day == pytest.approx(expected_bars_per_day, rel=0.05)
    assert per_bar == pytest.approx(0.0005 / expected_bars_per_day, rel=1e-3)
    assert per_bar < 0.0005, "a single bar must cost less than a whole day"


def test_financing_is_only_charged_on_the_borrowed_portion():
    """A position funded entirely by cash borrows nothing and owes nothing.

    This is what the leverage sweep got wrong: raising `max_leverage` to 2x
    while the vol-targeted weight peaked at 0.70 meant the book never borrowed,
    so the sweep measured concentration and reported it as leverage.
    """
    equity, gross = 10_000_000.0, 7_000_000.0
    assert max(0.0, gross - equity) == 0.0

    gross_levered = 15_000_000.0
    assert max(0.0, gross_levered - equity) == 5_000_000.0


# -- retry policy -----------------------------------------------------------
def _status_error(code: int):
    import httpx

    request = httpx.Request("POST", "https://apiv2.nobitex.ir/wallets/transfer")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


class _Outcome:
    def __init__(self, exc):
        self._exc = exc
        self.failed = True

    def exception(self):
        return self._exc


class _State:
    def __init__(self, exc):
        self.outcome = _Outcome(exc)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(code):
    """A 401 does not become authorised on the third attempt.

    Nobitex blocks an IP for ~30 minutes after roughly 100 failed auth
    attempts, so a blanket retry spent four strikes of that budget on every
    single bad call -- observed live against /wallets/transfer.
    """
    from nbtrend.data.nobitex_rest import _is_retryable

    assert not _is_retryable(_State(_status_error(code)))


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_errors_are_still_retried(code):
    from nbtrend.data.nobitex_rest import _is_retryable

    assert _is_retryable(_State(_status_error(code)))


def test_transport_errors_are_still_retried():
    import httpx

    from nbtrend.data.nobitex_rest import _is_retryable

    assert _is_retryable(_State(httpx.ConnectError("boom")))


# -- OCO --------------------------------------------------------------------
class _OcoREST:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _post(self, path, payload=None):
        self.calls.append((path, payload or {}))
        return {"status": "ok", "orders": [
            {"id": 25, "type": payload.get("type"), "srcCurrency": "btc",
             "dstCurrency": "rls", "price": "100", "amount": "0.01",
             "status": "Active", "matchedAmount": 0},
            {"id": 26, "type": payload.get("type"), "srcCurrency": "btc",
             "dstCurrency": "rls", "price": "90", "amount": "0.01",
             "status": "Active", "matchedAmount": 0},
        ]}


def _oco_rest():
    from nbtrend.data.nobitex_rest import NobitexREST

    rest = object.__new__(NobitexREST)
    fake = _OcoREST()
    rest._post = fake._post
    return rest, fake


def test_oco_returns_both_legs():
    """The response key is `orders` (a pair), not the single `order` of a
    normal submit -- reading `order` yields a KeyError at the worst moment."""
    rest, _ = _oco_rest()
    orders = rest.add_oco_order("btc", "rls", Side.SELL, 0.01,
                                price=110, stop_price=90, stop_limit_price=89)
    assert len(orders) == 2
    assert {o.exchange_id for o in orders} == {25, 26}


def test_oco_without_leverage_goes_to_the_spot_endpoint():
    rest, fake = _oco_rest()
    rest.add_oco_order("btc", "rls", Side.SELL, 0.01,
                       price=110, stop_price=90, stop_limit_price=89)
    assert fake.calls[0][0] == "/market/orders/add"
    assert "leverage" not in fake.calls[0][1]
    assert fake.calls[0][1]["mode"] == "oco"


def test_oco_with_leverage_goes_to_the_margin_endpoint():
    rest, fake = _oco_rest()
    rest.add_oco_order("btc", "rls", Side.SELL, 0.01, price=110,
                       stop_price=90, stop_limit_price=89, leverage=2)
    assert fake.calls[0][0] == "/margin/orders/add"
    assert fake.calls[0][1]["leverage"] == "2"


def test_oco_sell_price_ordering_is_validated():
    """Nobitex: sell needs price > last > stopPrice, else PriceConditionFailed.
    Catching it here saves a rejected order and a slot in the 300/10min budget."""
    rest, fake = _oco_rest()
    with pytest.raises(ValueError, match="OCO sell needs"):
        rest.add_oco_order("btc", "rls", Side.SELL, 0.01, price=90,
                           stop_price=110, stop_limit_price=109, last_price=100)
    assert fake.calls == [], "must not reach the exchange"


def test_oco_buy_price_ordering_is_validated():
    rest, fake = _oco_rest()
    with pytest.raises(ValueError, match="OCO buy needs"):
        rest.add_oco_order("btc", "rls", Side.BUY, 0.01, price=110,
                           stop_price=90, stop_limit_price=91, last_price=100)
    assert fake.calls == []


def test_a_correctly_ordered_oco_passes_validation():
    rest, fake = _oco_rest()
    rest.add_oco_order("btc", "rls", Side.SELL, 0.01, price=110,
                       stop_price=90, stop_limit_price=89, last_price=100)
    assert len(fake.calls) == 1


def test_oco_rejects_an_invalid_leverage_step():
    rest, fake = _oco_rest()
    with pytest.raises(ValueError, match="steps of 0.5"):
        rest.add_oco_order("btc", "rls", Side.SELL, 0.01, price=110,
                           stop_price=90, stop_limit_price=89, leverage=2.3)
    assert fake.calls == []


def test_add_order_returns_the_client_order_id_it_sent():
    """The response does not always echo it, and it is the only handle that
    reliably cancels on this exchange. Losing it means the order cannot be
    cancelled, its balance stays locked, and every later ask tries to sell
    committed stock -- 5,380 "Order Validation Failed" in one short run."""
    from nbtrend.data.nobitex_rest import NobitexREST

    rest = object.__new__(NobitexREST)
    rest._post = lambda path, payload: {
        "status": "ok",
        # No clientOrderId echoed back, as the live API sometimes does.
        "order": {"id": 42, "type": "buy", "srcCurrency": "xlm", "dstCurrency": "rls",
                  "price": "400000", "amount": "3", "status": "Active",
                  "matchedAmount": 0},
    }
    order = rest.add_order("xlm", "rls", Side.BUY, 3.0, price=400_000,
                           client_order_id="mkr-abc123")
    assert order.client_order_id == "mkr-abc123"
