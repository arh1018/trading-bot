"""The rial/toman convention is the highest-consequence detail in the project."""

import pytest

from nbtrend.units import (
    RIAL_PER_TOMAN,
    UnitSanityError,
    assert_plausible_rial,
    rial_to_toman,
    round_to_step,
    toman_to_rial,
)

# Captured from the live API at the same instant, 2026-08-26.
ORDERBOOK_RIAL = 154_993_852_310      # GET /v3/orderbook/BTCIRT
UDF_TOMAN = 15_499_385_231            # GET /market/udf/history


def test_udf_and_orderbook_differ_by_exactly_ten():
    assert toman_to_rial(UDF_TOMAN) == ORDERBOOK_RIAL


def test_roundtrip():
    assert rial_to_toman(toman_to_rial(1234.5)) == pytest.approx(1234.5)


def test_plausible_rial_accepts_a_small_basis():
    assert_plausible_rial("ok", ORDERBOOK_RIAL * 1.02, ORDERBOOK_RIAL)


def test_plausible_rial_catches_a_missing_conversion():
    with pytest.raises(UnitSanityError, match="never converted"):
        assert_plausible_rial("bad", UDF_TOMAN, ORDERBOOK_RIAL)


def test_plausible_rial_catches_a_double_conversion():
    with pytest.raises(UnitSanityError, match="converted to rial twice"):
        assert_plausible_rial("bad", ORDERBOOK_RIAL * RIAL_PER_TOMAN, ORDERBOOK_RIAL)


@pytest.mark.parametrize(
    "value,step,expected",
    [
        (0.0012345, 0.000001, 0.001234),   # binary float would floor to 0.001233
        (1.23456789, 0.00001, 1.23456),
        (64.7, 1, 64.0),
        (0.5, 0.01, 0.5),
        (99.99, 0, 99.99),                 # step 0 disables rounding
    ],
)
def test_round_to_step_floors_exactly(value, step, expected):
    assert round_to_step(value, step) == pytest.approx(expected)


def test_round_to_step_never_rounds_up():
    """Rounding up produces orders larger than the balance held."""
    for value in (0.123456789, 1.999999, 0.000000999):
        assert round_to_step(value, 0.000001) <= value
