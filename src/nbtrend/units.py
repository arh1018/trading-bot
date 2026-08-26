"""Money units.

Nobitex is **not** internally consistent about rial vs toman, and getting this
wrong is a silent factor-of-ten error in every price, position size and P&L
number in the system. Verified against the live API on 2026-08-26, BTCIRT at
the same instant:

    GET /v3/orderbook/BTCIRT      lastTradePrice = 154_993_852_310   <- RIAL
    GET /market/udf/history       close          =  15_499_385_231   <- TOMAN

So:

    +--------------------------------------+---------+
    | endpoint / channel                   | unit    |
    +--------------------------------------+---------+
    | GET /v3/orderbook/{SYMBOL}           | rial    |
    | GET /market/stats                    | rial    |
    | GET /v2/trades/{SYMBOL}              | rial    |
    | ws  public:orderbook-{SYMBOL}        | rial    |
    | ws  public:trades-{SYMBOL}           | rial    |
    | ws  public:market-stats-{SYMBOL}     | rial    |
    | POST /market/orders/add  (price)     | rial    |
    | GET /market/udf/history              | TOMAN   |
    | ws  public:candle-{SYMBOL}-{res}     | TOMAN   |
    +--------------------------------------+---------+

`dstCurrency: "rls"` in the trading API means the *order* price is in rial;
the OHLC endpoints are the odd ones out because they back the TradingView
charting widget, which Nobitex renders in toman.

**Everything inside nbtrend is rial.** Conversion happens exactly once, at the
edge, in the adapter that owns the endpoint. `assert_plausible_rial` is wired
into the live runner so a future change to Nobitex's convention fails loudly
instead of quietly halving the book.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

RIAL_PER_TOMAN = 10


def toman_to_rial(x: float) -> float:
    return x * RIAL_PER_TOMAN


def rial_to_toman(x: float) -> float:
    return x / RIAL_PER_TOMAN


class UnitSanityError(RuntimeError):
    """Raised when two feeds that should agree differ by a factor of ~10."""


def assert_plausible_rial(
    label: str,
    candidate_rial: float,
    reference_rial: float,
    tolerance: float = 0.35,
) -> None:
    """Cross-check a converted price against a known-rial reference.

    `candidate_rial` typically comes from the OHLC feed (converted from toman)
    and `reference_rial` from the orderbook. A genuine basis between a candle
    close and the live touch is a few percent; a unit bug is 10x or 0.1x.
    """
    if reference_rial <= 0 or candidate_rial <= 0:
        raise UnitSanityError(f"{label}: non-positive price ({candidate_rial}, {reference_rial})")

    ratio = candidate_rial / reference_rial
    if abs(ratio - 1.0) <= tolerance:
        return

    hint = ""
    if 0.5 * RIAL_PER_TOMAN < ratio < 2 * RIAL_PER_TOMAN:
        hint = " -- looks like a toman value was converted to rial twice"
    elif 0.5 / RIAL_PER_TOMAN < ratio < 2 / RIAL_PER_TOMAN:
        hint = " -- looks like a toman value was never converted to rial"

    raise UnitSanityError(
        f"{label}: price {candidate_rial:,.0f} is {ratio:.3f}x the rial "
        f"reference {reference_rial:,.0f}{hint}"
    )


def round_to_step(value: float, step: float) -> float:
    """Round down to an exchange-legal increment.

    Rounding *down* on both size and (buy) price keeps the order inside the
    balance you actually hold; rounding up gets it rejected as
    InsufficientBalance. Decimal, not float: ``0.0012345 / 0.000001`` is
    ``1234.4999...`` in binary floating point, so a plain ``int()`` silently
    loses a step.
    """
    if step <= 0:
        return value
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    quantised = (d_value / d_step).to_integral_value(rounding=ROUND_FLOOR) * d_step
    return float(quantised)
