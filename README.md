# BTC TFT Forecasting

Standalone BTC forecasting pipeline using Temporal Fusion Transformer (TFT) across `15m`, `1h`, and `4h`.

## What This Project Does
- Downloads Binance spot OHLCV for BTCUSDT using CCXT.
- Downloads Binance USDⓈ-M funding history and open interest (historical when available).
- Builds no-leak processed feature datasets for each timeframe.
- Trains TFT quantile models (`q10/q50/q90`) on returns.
- Runs monthly walk-forward backtests and writes metrics/plots/reports.
- Produces forecasts with automatic fallback from `full` to `price_only` when derivatives coverage is low.

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Copy `.env.example` to `.env` and adjust values.

## CLI Usage
```bash
python -m src download --start-date 2021-01-01 --end-date 2026-01-01
python -m src build --timeframes 15m 1h 4h
python -m src train --timeframe 15m --variant all
python -m src backtest --timeframe 1h --variant full
python -m src predict --timeframe 4h --variant auto
```

## No-Leak Alignment (Funding/OI)
- Bars are indexed by close timestamp in UTC.
- Funding and OI events are merged with `merge_asof(..., direction="backward")`.
- This ensures each bar only sees derivative values published at or before that bar close.
- Validation checks enforce that no aligned derivative timestamp is greater than the bar timestamp.

## Fallback Behavior
- `full` models use price + derivatives.
- `price_only` models train without derivatives and are always available as fallback.
- Prediction command can auto-fallback to `price_only` if recent derivatives coverage is below threshold.

## Output Paths
- Raw data: `data/raw/*.parquet`
- Processed data: `data/processed/btc_15m.parquet`, `btc_1h.parquet`, `btc_4h.parquet`
- Training artifacts: `artifacts/{timeframe}/{variant}/`
- Backtest report: `reports/summary.md`
- Backtest plots: `reports/plots/`

## Known Limitations
- Binance open interest history endpoint can have availability/range limits by symbol/time.
- If historical OI is unavailable, the pipeline stores sampled OI snapshots and continues with price+funding.
- You can later plug in alternate OI vendors without changing model interfaces.
