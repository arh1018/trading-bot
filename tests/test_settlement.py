"""Nobitex credits a wallet ~2s AFTER acknowledging a fill.

Reading balances inside that window returns pre-trade numbers. In a
sequential rebalance loop that makes the next symbol over-estimate available
cash and overdraw the rial wallet -- the same failure the cost-adjusted gross
cap was added to prevent, reintroduced through a stale read.
"""

from __future__ import annotations

import time

import pytest

from nbtrend.core.types import Side
from nbtrend.execution.base import SETTLEMENT_TOLERANCE, credited_currency
from nbtrend.execution.nobitex import NobitexBroker


class _FakeREST:
    """Wallet that only reflects a credit after `delay_s`."""

    def __init__(self, currency: str, before: float, after: float, delay_s: float):
        self._token = "test-token"
        self.currency = currency
        self.before = before
        self.after = after
        self.delay_s = delay_s
        self.started = time.time()
        self.reads = 0

    def wallets(self, currencies=None):
        self.reads += 1
        settled = (time.time() - self.started) >= self.delay_s
        return {self.currency: self.after if settled else self.before}


def _broker(rest, timeout=5.0):
    return NobitexBroker(
        rest, {}, settlement_poll_s=0.05, settlement_timeout_s=timeout
    )


def test_buy_credits_the_base_and_sell_credits_the_quote():
    assert credited_currency(Side.BUY, "btc", "rls") == "btc"
    assert credited_currency(Side.SELL, "btc", "rls") == "rls"


def test_await_settlement_waits_for_the_credit_to_land():
    rest = _FakeREST("btc", before=0.0, after=0.5, delay_s=0.3)
    broker = _broker(rest)

    start = time.time()
    assert broker.await_settlement("btc", baseline=0.0, expected_delta=0.5) is True
    elapsed = time.time() - start

    assert elapsed >= 0.3, "returned before the wallet reflected the credit"
    assert rest.reads > 1, "should poll rather than sleep a fixed interval"


def test_await_settlement_tolerates_fees_taken_from_the_credit():
    """The credited side arrives slightly short because the fee comes out of
    it; demanding the full amount would never confirm."""
    rest = _FakeREST("btc", before=0.0, after=0.499, delay_s=0.0)
    assert _broker(rest).await_settlement("btc", 0.0, 0.5) is True

    short = _FakeREST("btc", before=0.0, after=0.5 * SETTLEMENT_TOLERANCE * 0.5, delay_s=0.0)
    assert _broker(short, timeout=0.3).await_settlement("btc", 0.0, 0.5) is False


def test_await_settlement_times_out_instead_of_hanging():
    rest = _FakeREST("btc", before=0.0, after=0.5, delay_s=999)
    start = time.time()
    assert _broker(rest, timeout=0.3).await_settlement("btc", 0.0, 0.5) is False
    assert time.time() - start < 2.0


def test_await_settlement_survives_a_failing_balance_read():
    class _Flaky(_FakeREST):
        def wallets(self, currencies=None):
            self.reads += 1
            if self.reads < 3:
                raise RuntimeError("transient")
            return {self.currency: self.after}

    rest = _Flaky("btc", before=0.0, after=0.5, delay_s=0.0)
    assert _broker(rest).await_settlement("btc", 0.0, 0.5) is True


def test_zero_delta_is_a_noop():
    rest = _FakeREST("btc", 0.0, 0.0, 0.0)
    assert _broker(rest).await_settlement("btc", 0.0, 0.0) is True
    assert rest.reads == 0


def test_paper_broker_settles_synchronously():
    """The paper path must expose the same method so both modes behave alike."""
    from nbtrend.execution.paper import PaperBroker

    broker = PaperBroker(rest=None, starting_balances={"rls": 1e9})
    assert broker.await_settlement("btc", 0.0, 1.0) is True


def test_runner_confirms_settlement_before_the_next_symbol():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "nbtrend" / "live" / "runner.py"
    ).read_text()
    assert "self.broker.await_settlement(credited, baseline, expected)" in source
    assert "baseline = self.broker.balances().get(credited, 0.0)" in source


def test_v2_wallets_uses_balance_not_activebalance():
    """Nobitex's two wallet endpoints do NOT share a schema.

    /v2/wallets  -> {"USDT": {"balance": "5.007", "blocked": "0"}}   UPPERCASE keys
    /users/wallets/list -> [{"currency": "usdt", "activeBalance": "5.007", ...}]

    Reading `activeBalance` from /v2/wallets returns 0 for every currency,
    which reads as an empty account: the cash clamp then trims every buy to
    zero and settlement confirmation never observes a credit.
    """
    from nbtrend.data.nobitex_rest import NobitexREST

    api = object.__new__(NobitexREST)
    api._get = lambda path, **kw: {
        "status": "ok",
        "wallets": {
            "RLS": {"id": 1, "balance": "609.012626075", "blocked": "0"},
            "USDT": {"id": 2, "balance": "5.007048", "blocked": "1.0"},
        },
    }

    out = NobitexREST.wallets(api, ["rls", "usdt"])
    assert out["rls"] == pytest.approx(609.012626075)
    # Blocked funds are committed to open orders and are not spendable.
    assert out["usdt"] == pytest.approx(4.007048)


def test_users_wallets_list_uses_activebalance():
    from nbtrend.data.nobitex_rest import NobitexREST

    api = object.__new__(NobitexREST)
    api._get = lambda path, **kw: {
        "wallets": [
            {"currency": "rls", "activeBalance": "609.01", "balance": "609.01"},
            {"currency": "usdt", "activeBalance": "5.007", "balance": "6.007"},
        ]
    }
    out = NobitexREST.wallets(api, None)
    assert out["usdt"] == pytest.approx(5.007)
