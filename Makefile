.PHONY: help install test lint doctor fetch signals basis backtest optimize run shadow clean

PY := .venv/bin/python
SYMBOL ?= BTCIRT
MINUTES ?= 15

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install the package
	uv venv --python 3.12 .venv || python3 -m venv .venv
	uv pip install --python $(PY) -e '.[dev,research]'

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## Ruff check
	$(PY) -m ruff check src tests

doctor:  ## Verify connectivity, credentials and the rial/toman convention
	$(PY) -m nbtrend.cli doctor

fetch:  ## Download history into the parquet cache
	$(PY) -m nbtrend.cli fetch

signals:  ## Show the current trend score for every enabled symbol
	$(PY) -m nbtrend.cli signals

basis:  ## Show live local-vs-global pricing
	$(PY) -m nbtrend.cli basis

backtest:  ## Backtest one symbol (make backtest SYMBOL=ETHIRT)
	$(PY) -m nbtrend.cli backtest $(SYMBOL)

optimize:  ## Walk-forward parameter search (make optimize SYMBOL=ETHIRT)
	$(PY) -m nbtrend.cli optimize $(SYMBOL)

run:  ## Start the trading loop (paper unless NBTREND_MODE=live)
	$(PY) -m nbtrend.cli run

shadow:  ## Shadow test the full live stack, no real money (make shadow MINUTES=15)
	$(PY) scripts/shadow_test.py --minutes $(MINUTES)

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
	rm -f data/candles/*.parquet
