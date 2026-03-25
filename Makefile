PYTHON ?= python
MODULE ?= src

.PHONY: install download build train backtest predict dashboard test lint docker-build docker-run

install:
	$(PYTHON) -m pip install -e .[dev]

download:
	$(PYTHON) -m $(MODULE) download

build:
	$(PYTHON) -m $(MODULE) build

train:
	$(PYTHON) -m $(MODULE) train

backtest:
	$(PYTHON) -m $(MODULE) backtest

predict:
	$(PYTHON) -m $(MODULE) predict

dashboard:
	$(PYTHON) -m $(MODULE) dashboard

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

docker-build:
	docker build -t btc-tft-forecasting .

docker-run:
	docker compose run --rm btc-tft --help
