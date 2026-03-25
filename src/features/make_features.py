from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.settings import (
    DEFAULT_TIMEFRAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TIMEFRAME_SPECS,
    ensure_directories,
)

LOGGER = logging.getLogger(__name__)

VOL_WINDOWS = {
    "15m": 96,   # 1 day
    "1h": 24,    # 1 day
    "4h": 42,    # 1 week
}

EMA_WINDOWS = {
    "15m": (48, 192),
    "1h": (24, 96),
    "4h": (12, 42),
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def validate_bar_continuity(frame: pd.DataFrame, freq: str, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label}: empty dataframe")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{label}: null timestamps")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{label}: duplicate timestamps found")

    actual = pd.DatetimeIndex(frame["timestamp"].sort_values().unique())
    expected = pd.date_range(start=actual.min(), end=actual.max(), freq=freq, tz="UTC")
    missing = expected.difference(actual)
    if len(missing) > 0:
        sample = ", ".join(str(ts) for ts in missing[:5])
        raise ValueError(f"{label}: missing {len(missing)} timestamps (sample: {sample})")


def validate_no_leakage(
    frame: pd.DataFrame,
    bar_timestamp_col: str,
    source_timestamp_col: str,
    label: str,
) -> None:
    if source_timestamp_col not in frame.columns:
        return
    valid = frame[source_timestamp_col].notna()
    leak_mask = valid & (frame[source_timestamp_col] > frame[bar_timestamp_col])
    if leak_mask.any():
        offending = frame.loc[leak_mask, [bar_timestamp_col, source_timestamp_col]].head(5)
        raise ValueError(f"{label}: leakage detected\n{offending}")


def align_derivatives_asof(
    bars: pd.DataFrame,
    derivatives: pd.DataFrame,
    value_columns: list[str],
    source_time_col: str,
    prefix: str,
) -> pd.DataFrame:
    aligned = bars.sort_values("timestamp").copy()
    published_col = f"{prefix}_published_at"

    if derivatives.empty:
        aligned[published_col] = pd.NaT
        for column in value_columns:
            aligned[column] = np.nan
        return aligned

    right = derivatives[[source_time_col, *value_columns]].copy().sort_values(source_time_col)
    right = right.rename(columns={source_time_col: published_col})

    aligned = pd.merge_asof(
        aligned,
        right,
        left_on="timestamp",
        right_on=published_col,
        direction="backward",
        allow_exact_matches=True,
    )
    validate_no_leakage(aligned, "timestamp", published_col, prefix)
    return aligned


def _compute_price_features(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    result = frame.copy().sort_values("timestamp")
    fast_span, slow_span = EMA_WINDOWS[timeframe]
    vol_window = VOL_WINDOWS[timeframe]

    result["log_return"] = np.log(result["close"]).diff()
    result["target_return"] = np.log(result["close"].shift(-1) / result["close"])
    result["rolling_vol"] = result["log_return"].rolling(vol_window).std()
    result["range_pct"] = (result["high"] - result["low"]) / result["close"].replace(0, np.nan)
    result["volume_log1p"] = np.log1p(result["volume"])

    vol_mean = result["volume_log1p"].rolling(vol_window).mean()
    vol_std = result["volume_log1p"].rolling(vol_window).std().replace(0, np.nan)
    result["volume_zscore"] = (result["volume_log1p"] - vol_mean) / vol_std

    ema_fast = result["close"].ewm(span=fast_span, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow_span, adjust=False).mean()
    result["ema_slope"] = (ema_fast - ema_slow) / result["close"].replace(0, np.nan)

    result["hour"] = result["timestamp"].dt.hour.astype(int)
    result["day_of_week"] = result["timestamp"].dt.dayofweek.astype(int)
    result["month"] = result["timestamp"].dt.month.astype(int)
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_asia_session"] = result["hour"].between(0, 7, inclusive="both").astype(int)
    result["is_europe_session"] = result["hour"].between(7, 15, inclusive="both").astype(int)
    result["is_us_session"] = result["hour"].between(13, 21, inclusive="both").astype(int)
    return result


def _load_ohlcv(timeframe: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"ohlcv_{timeframe}.parquet"
    frame = _load_parquet(path)
    if frame.empty:
        raise FileNotFoundError(f"Missing raw OHLCV file: {path}")
    required = {"timestamp", "open", "high", "low", "close", "volume", "symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV file missing columns: {missing}")
    return frame


def _load_funding() -> pd.DataFrame:
    frame = _load_parquet(RAW_DATA_DIR / "funding_rate.parquet")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return (
        frame[["timestamp", "funding_rate"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )


def _load_open_interest() -> pd.DataFrame:
    hist = _load_parquet(RAW_DATA_DIR / "open_interest_hist.parquet")
    if not hist.empty and {"timestamp", "open_interest"}.issubset(hist.columns):
        return (
            hist[["timestamp", "open_interest"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )

    snapshots = _load_parquet(RAW_DATA_DIR / "open_interest_snapshots.parquet")
    if not snapshots.empty and {"timestamp", "open_interest"}.issubset(snapshots.columns):
        return (
            snapshots[["timestamp", "open_interest"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )

    return pd.DataFrame(columns=["timestamp", "open_interest"])


def _build_single_timeframe(timeframe: str) -> Path:
    spec = TIMEFRAME_SPECS[timeframe]
    bars = _load_ohlcv(timeframe).sort_values("timestamp").reset_index(drop=True)
    validate_bar_continuity(bars, spec.pandas_freq, f"ohlcv_{timeframe}")

    bars = _compute_price_features(bars, timeframe)

    funding = _load_funding()
    bars = align_derivatives_asof(
        bars=bars,
        derivatives=funding,
        value_columns=["funding_rate"],
        source_time_col="timestamp",
        prefix="funding",
    )
    bars["funding_rate_available"] = bars["funding_published_at"].notna().astype(int)
    bars["funding_rate"] = bars["funding_rate"].fillna(0.0)
    bars["funding_rate_change"] = bars["funding_rate"].diff().fillna(0.0)

    open_interest = _load_open_interest()
    bars = align_derivatives_asof(
        bars=bars,
        derivatives=open_interest,
        value_columns=["open_interest"],
        source_time_col="timestamp",
        prefix="oi",
    )
    bars["open_interest_available"] = bars["oi_published_at"].notna().astype(int)
    bars["open_interest"] = bars["open_interest"].ffill().fillna(0.0)
    bars["open_interest_log"] = np.log1p(bars["open_interest"].clip(lower=0))
    bars["open_interest_change"] = bars["open_interest_log"].diff().fillna(0.0)

    validate_no_leakage(bars, "timestamp", "funding_published_at", "funding")
    validate_no_leakage(bars, "timestamp", "oi_published_at", "open_interest")

    bars["symbol"] = bars["symbol"].astype(str)
    bars["time_idx"] = np.arange(len(bars), dtype=int)

    required_cols = [
        "log_return",
        "rolling_vol",
        "range_pct",
        "volume_log1p",
        "volume_zscore",
        "ema_slope",
        "target_return",
    ]
    bars = bars.dropna(subset=required_cols).reset_index(drop=True)
    bars["time_idx"] = np.arange(len(bars), dtype=int)

    output_columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "time_idx",
        "target_return",
        "log_return",
        "rolling_vol",
        "range_pct",
        "volume_log1p",
        "volume_zscore",
        "ema_slope",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_asia_session",
        "is_europe_session",
        "is_us_session",
        "funding_rate",
        "funding_rate_change",
        "funding_rate_available",
        "open_interest_log",
        "open_interest_change",
        "open_interest_available",
        "funding_published_at",
        "oi_published_at",
    ]

    processed = bars[output_columns].copy()
    out_path = PROCESSED_DATA_DIR / f"btc_{timeframe}.parquet"
    processed.to_parquet(out_path, index=False)
    LOGGER.info("Built %s rows -> %s", len(processed), out_path)
    return out_path


def run_build(args: Any) -> None:
    _setup_logging()
    ensure_directories()
    timeframes = tuple(getattr(args, "timeframes", None) or DEFAULT_TIMEFRAMES)

    for timeframe in timeframes:
        _build_single_timeframe(timeframe)
