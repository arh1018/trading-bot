# nbtrend

Trend-following algorithmic trading on **Nobitex rial (IRT) markets**, with
signals computed from **global USD prices** and orders executed locally.

```
global price (TradingView)  ──┐
                              ├──► trend signal ──► risk sizing ──► order router ──► Nobitex
USDT/IRT rate (Nobitex)     ──┤                                                         │
local price (Nobitex WS)    ──┘◄──────────── basis / interlock checks ───────────────────┘
```

---

## The core idea

Every Nobitex IRT price decomposes into three independent factors:

```
P_irt  =  P_usd  ×  FX_usdt_irt  ×  (1 + basis)

r_irt  =  r_usd  +  r_fx  +  r_basis
```

| term | what it is | behaviour |
| --- | --- | --- |
| `r_usd` | global crypto trend | deep, liquid, 24/7 — **this is what the model trades** |
| `r_fx` | rial devaluation | Iranian macro; strongly one-directional, jumpy |
| `r_basis` | Nobitex local premium | mean-reverting, small, blows out during stress |

Running a trend model directly on the rial price silently mixes all three. It
reports "momentum" that is really the currency sliding, sizes it with a
volatility estimate contaminated by FX jumps, and can never tell you which bet
made or lost the money.

So this project **signals on the global price, executes on the rial price, and
monitors the basis as a safety interlock.**

A consequence worth internalising before you risk anything: on the ~2 years of
BTCIRT history tested here, the strategy returned **+52.4% in rial and −53.9%
in USD terms** over the same window. Both numbers are correct. The rial figure
is what the exchange shows you; the USD figure is whether the strategy actually
worked. `nbtrend backtest` always prints both.

---

## Quick start

```bash
make install                    # venv + dependencies (uses uv if available)
make doctor                     # verify connectivity and the rial/toman convention
make fetch                      # download history into the parquet cache
make signals                    # current trend score per symbol
make backtest SYMBOL=BTCIRT     # backtest with a buy & hold benchmark
make optimize SYMBOL=BTCIRT     # walk-forward search, out-of-sample only
make run                        # paper trading loop
make shadow MINUTES=15          # shadow test the full live stack, no real money
```

No credentials are needed for anything except live trading — all market data
used here is public.

### Adding your tokens

Edit [config/universe.yaml](config/universe.yaml). Each entry maps a Nobitex
market to its global reference feed:

```yaml
- nobitex: SOLIRT             # Nobitex market symbol, UPPERCASE
  src: sol                    # srcCurrency for POST /market/orders/add
  dst: rls                    # dstCurrency — rial
  tradingview: "BINANCE:SOLUSDT"   # global reference, the signal input
  amount_step: 0.001          # order size rounding
  price_step: 10              # price rounding, in RIAL
  enabled: true
```

Pick the *most liquid* global venue for `tradingview`, not the local one — the
whole point is to read the price that leads.

### Going live

```bash
cp .env.example .env      # then fill in NOBITEX_API_TOKEN
NBTREND_MODE=live make run
```

`run` prompts for confirmation in live mode. Start with `NOBITEX_TESTNET=1`.

---

## ⚠️ The rial / toman trap

**Nobitex is not internally consistent about units.** Verified against the live
API, BTCIRT at the same instant:

```
GET /v3/orderbook/BTCIRT   lastTradePrice = 154,993,852,310   ← RIAL
GET /market/udf/history    close          =  15,499,385,231   ← TOMAN
```

| endpoint / channel | unit |
| --- | --- |
| `/v3/orderbook/{SYMBOL}`, `/market/stats`, `/v2/trades` | **rial** |
| `public:orderbook-*`, `public:trades-*`, `public:market-stats-*` | **rial** |
| `POST /market/orders/add` (price) | **rial** |
| `/market/udf/history` | **toman** |
| `public:candle-{SYMBOL}-{res}` | **toman** |

The OHLC endpoints are the odd ones out because they back the TradingView
charting widget, which Nobitex renders in toman.

**Everything inside `nbtrend` is rial.** Conversion happens exactly once, at
the edge, in the adapter that owns each endpoint. `nbtrend doctor` cross-checks
a converted candle against the live orderbook, so if Nobitex ever changes this
the project fails loudly instead of quietly trading at 10× or 0.1× size.

---

## Strategy

Three trend estimators, blended. Each sees the same trend through a different
lens and fails differently, so the blend has lower turnover and better
risk-adjusted return than any one alone.

| estimator | source | why it earns its place |
| --- | --- | --- |
| **Normalised MACD** | Baz et al. (2015) | Volatility-normalised MACD over three timescales, squashed by a response function that *decays* past ~1.4σ — conviction falls when a trend is overextended instead of saturating into the blow-off top. The nearest thing to a standard CTA trend metric. |
| **TSMOM** | Moskowitz, Ooi & Pedersen (2012) | Return over a lookback ÷ volatility over that lookback, so a 10% move in a quiet asset outranks 10% in a wild one. Robust, nearly parameter-free. |
| **Donchian breakout** | Turtle | The only one anchored to actual price levels rather than a moving average, so it catches regime changes the smoothers lag. Slow entry, fast exit. |

On top of that:

- **Regime gate** — Kaufman efficiency ratio + ADX. Below the threshold the
  market is chopping and trend estimators are unbiased but high-variance; the
  variance is what pays the fees. Signal is forced flat.
- **Hysteresis** — enter above 0.30, exit below 0.10. Without the dead band a
  score oscillating around one threshold trades every bar, and at ~0.55%
  round-trip that is the entire edge.
- **Volatility targeting** — weight ∝ target_vol / realised_vol, so BTC and
  DOGE contribute comparable risk.
- **No-trade band** — vol targeting makes the target weight drift each bar;
  without a band the book rebalances continuously for nothing.
- **Chandelier ATR trailing stop** and a portfolio drawdown kill switch.

Spot IRT markets cannot be shorted, so the default book is **long/flat**. Set
`strategy.allow_short: true` only if you have enabled Nobitex margin.

### Finding the best metrics honestly

```bash
make optimize SYMBOL=BTCIRT
```

A plain grid search over full history will always find a great Sharpe — on a
few thousand bars and a few hundred combinations, the best in-sample result is
mostly luck. `optimize` runs **walk-forward** instead: fit on 365 days, trade
the next 90 untouched, roll forward, stitch the out-of-sample segments.

It reports the **overfit gap** (in-sample minus out-of-sample) and
**parameter stability** — the coefficient of variation of each parameter across
folds. A parameter that jumps every fold is not being optimised, it is fitting
noise, and should be pinned to a constant.

---

## Costs matter, and your fee tier decides how much

Nobitex spot fees depend on your 30-day volume tier. `config/config.yaml` is
set to **0.11% maker/taker**, plus 0.10% modelled slippage — about **0.42% per
round trip**.

That tier is not a detail; it decides which strategies are viable at all. Same
BTCIRT data, same parameters, only the fee changed:

| fee tier | rial return | Sharpe | CAGR | costs paid |
| --- | --- | --- | --- | --- |
| 0.25/0.30% (entry level) | +25.8% | 1.33 | +13.4% | 13.1% of equity |
| **0.11% (configured)** | **+52.4%** | **1.78** | **+18.6%** | 10.0% of equity |

Even at 0.11%, costs consumed 10% of equity against an 18.6% CAGR — because at
that point *slippage*, not fees, is the larger term. Practical consequences,
all reflected in the defaults:

- Sub-hourly trend following is still not viable. Default timeframe is **4h**.
- The router **posts limit orders and re-prices** rather than crossing, and
  only crosses after `max_reposts` attempts.
- The no-trade band and hysteresis exist specifically to suppress turnover.

If your tier changes, update `costs.maker_fee` / `costs.taker_fee` and re-run
`make optimize` — the best parameters move with the cost structure.

---

## Safety interlocks

The live runner refuses to trade — rather than guessing — when:

| condition | check |
| --- | --- |
| websocket stale | no message for `MAX_STALENESS_S` (120s) |
| local price dislocated | \|basis\| > `execution.max_basis_deviation` (5%) |
| market halted | `isClosed` on the market-stats channel |
| equity drawdown | past `risk.max_drawdown_stop` → flatten |
| order too small | below the 3,000,000 rial IRT minimum → skip |
| order rate | local guard for the 300 orders / 10 min shared limit |

The runner is `asyncio`-based and the Nobitex websocket must answer
Centrifugo's ping within **25 seconds** or the connection is dropped. The
router polls fills with a blocking `time.sleep` (up to `repost_after_s` = 45s
per attempt) and the REST client is synchronous, so both run via
`asyncio.to_thread` — running either on the event loop starves the pong
handler and kills the feed mid-rebalance. `tests/test_async_safety.py` guards
this, including a control case that reproduces the starvation.

Order submission is **never retried** on transport error — a retried POST that
actually succeeded places the order twice. Every order carries a
`clientOrderId` so a timed-out request is reconciled instead.

---

## Project layout

```
config/
  config.yaml            strategy, risk, costs, execution parameters
  universe.yaml          the tokens you trade  ← edit this
src/nbtrend/
  units.py               THE rial/toman boundary — read this first
  config.py              typed config + .env loading
  core/types.py          Candle, BookTop, Order, Position, Signal, Fill
  data/
    tradingview.py       unofficial TradingView websocket datafeed
    binance.py           documented keyless fallback, same interface
    nobitex_rest.py      REST: orderbook, candles, wallets, orders
    nobitex_ws.py        Centrifugo client: orderbook/trades/candle/private
    fx.py                USDT/IRT leg and basis decomposition
    store.py             parquet OHLCV cache
    feed.py              assembles global + local + fx on one clock
  features/
    indicators.py        EMA, ATR, ADX, efficiency ratio, Donchian
    signals.py           the three estimators + composite + hysteresis
  risk/sizing.py         vol targeting, exposure caps, stops
  execution/
    base.py              Broker protocol
    paper.py             simulated fills by walking the real book
    nobitex.py           live broker with rate guard and reconciliation
    router.py            limit-chase execution
  backtest/
    engine.py            event-driven, lookahead-audited
    metrics.py           rial *and* USD-terms performance
    walkforward.py       parameter search with out-of-sample validation
  live/runner.py         the trading loop
  cli.py                 nbtrend <command>
scripts/shadow_test.py   full-stack dry run against real markets
tests/                   64 tests
```

### Data sources

| source | what for | reliability |
| --- | --- | --- |
| **TradingView** | global USD prices | *unofficial* — a scraped websocket with no stability contract. Free, works, can break without notice. |
| **Binance** | fallback global prices | documented, versioned, keyless. Set `data.global_feed: binance`. |
| **Nobitex REST** | local candles, orderbook, orders | official |
| **Nobitex WS** | live orderbook, trades, market status | official (Centrifugo) |

The two global feeds agree to the tick, so running Binance as a cross-check on
TradingView is cheap insurance.

---

## Testing

```bash
make test    # 64 tests
make lint
```

Before risking anything, run the full stack against live markets with no money
at stake:

```bash
make shadow MINUTES=15
```

`scripts/shadow_test.py` drives the real websocket, real orderbooks, real
signal derivation and the real router — only the broker is swapped for
`PaperBroker`, which simulates fills by *walking the actual book*, so reported
slippage is what your size would genuinely have cost. It reports fills, fees,
the final book, mark-to-market P&L, and the basis/spread conditions observed.

The most important test is `tests/test_backtest.py::test_no_lookahead`: it
randomly rewrites every bar after time *t* and asserts the equity curve up to
*t* is byte-identical. Every other number in this project is downstream of that
being true.

---

## Known limitations

- **Nobitex minute-candle history starts at 1401 (2022)** and thin markets have
  gaps. `fx.synthesise_irt_history` can reconstruct a rial series from global ×
  FX, but that assumes a constant basis and therefore *understates* local
  dislocation risk.
- **The backtest fills at bar prices**, not by walking the book. The paper
  broker does walk the real book — use it to check your size is realistic
  before trusting backtest fills.
- **Sharpe is computed against rf = 0.** Rial deposits pay a substantial
  nominal rate; set a realistic `rf_annual` before believing any Sharpe here.
- **No margin/short support** in the default path.
- **`optimize` searches 729 combinations per fold** and takes minutes. Narrow
  `DEFAULT_GRID` in `walkforward.py` before adding axes — every extra axis
  multiplies both runtime and overfitting.

---

## Disclaimer

This is trading software. It can lose money. Backtested performance is not
indicative of future results, the TradingView feed is unofficial and may break,
and the rial/toman distinction means a units bug is a 10× position error. Run it
in paper mode until you understand every interlock above, and never deploy
capital you cannot lose.
