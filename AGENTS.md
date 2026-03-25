# BTC TFT Forecasting Agent Rules

## Project Scope
- Build a standalone BTC forecasting pipeline for `15m`, `1h`, and `4h`.
- Use Binance spot OHLCV plus USDⓈ-M derivatives (funding and open interest).
- Train Temporal Fusion Transformer (TFT) models with quantile outputs.

## Non-Negotiables
- All timestamps are UTC.
- No data leakage: derivative features must be as-of each bar close using backward carry-forward only.
- Save deterministic artifacts and configs so training and prediction can be reproduced.
- Keep PRs small and reviewable.

## Data Standards
- Symbol is always normalized to `BTCUSDT`.
- Raw data is written to `data/raw/*.parquet`.
- Processed data is written to `data/processed/btc_{timeframe}.parquet`.
- Include validation checks for gaps, duplicates, and leakage constraints.

## Modeling Standards
- Target is return forecasting with quantiles `0.1`, `0.5`, `0.9`.
- Train two variants per timeframe:
  - `full`: price + derivatives.
  - `price_only`: fallback model without derivatives.
- Store checkpoints and dataset metadata in `artifacts/{timeframe}/{variant}/`.

## Backtesting Standards
- Monthly rolling folds with train/val/test segmentation.
- Report MAE, RMSE, directional hit rate, pinball loss, and quantile calibration.
- Save plots under `reports/plots/` and markdown summary under `reports/summary.md`.

## CLI Contract
- `python -m src download`
- `python -m src build`
- `python -m src train`
- `python -m src backtest`
- `python -m src predict`
