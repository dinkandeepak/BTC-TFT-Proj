PYTHON ?= python
MODULE ?= src

.PHONY: install download build train backtest predict test lint

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

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
