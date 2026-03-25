from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

from src.settings import DEFAULT_SYMBOL, DEFAULT_TIMEFRAMES, RAW_DATA_DIR, ensure_directories, normalize_symbol

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    symbol: str
    start_ms: int
    end_ms: int
    base_url: str
    max_retries: int
    timeframes: tuple[str, ...]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _to_utc_ms(value: str) -> int:
    timestamp = pd.Timestamp(value, tz="UTC")
    return int(timestamp.value // 1_000_000)


def _runtime_config(args: Any) -> DownloadConfig:
    load_dotenv()
    symbol = normalize_symbol(getattr(args, "symbol", None) or os.getenv("SYMBOL", DEFAULT_SYMBOL))
    start_date = getattr(args, "start_date", None) or os.getenv("START_DATE", "2020-01-01")
    end_date = (
        getattr(args, "end_date", None)
        or os.getenv("END_DATE")
        or datetime.now(UTC).strftime("%Y-%m-%d")
    )
    base_url = str(getattr(args, "base_url", None) or os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com"))
    max_retries = int(getattr(args, "max_retries", None) or os.getenv("MAX_RETRIES", "5"))
    timeframes = tuple(getattr(args, "timeframes", None) or DEFAULT_TIMEFRAMES)

    start_ms = _to_utc_ms(start_date)
    end_ms = _to_utc_ms(end_date)
    if end_ms <= start_ms:
        raise ValueError("END_DATE must be after START_DATE")

    return DownloadConfig(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        base_url=base_url,
        max_retries=max_retries,
        timeframes=timeframes,
    )


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    max_retries: int,
    timeout: int = 30,
) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, (list, dict)):
                return data
            raise ValueError(f"Unexpected response shape from {url}: {type(data)}")
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(f"Request failed after retries: {url}") from exc
            sleep_s = min(2**attempt, 10)
            LOGGER.warning("Request retry %s/%s for %s (%s)", attempt, max_retries, url, exc)
            time.sleep(sleep_s)
    return []


def _timeframe_to_ms(timeframe: str) -> int:
    delta = pd.to_timedelta(timeframe)
    return int(delta.total_seconds() * 1000)


def _format_ccxt_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    raise ValueError(f"Cannot infer CCXT spot symbol from {symbol}")


def fetch_ohlcv(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    ccxt_symbol = _format_ccxt_symbol(symbol)
    timeframe_ms = _timeframe_to_ms(timeframe)
    cursor = start_ms
    rows: list[list[float]] = []

    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts < cursor:
            break
        cursor = last_ts + timeframe_ms
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"]
        )

    frame = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
    frame = frame[(frame["timestamp_ms"] >= start_ms) & (frame["timestamp_ms"] <= end_ms)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["symbol"] = normalize_symbol(symbol)
    frame["timeframe"] = timeframe
    return frame[["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"]]


def fetch_funding_history(
    session: requests.Session,
    base_url: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    max_retries: int,
    limit: int = 1000,
) -> pd.DataFrame:
    cursor = start_ms
    records: list[dict[str, Any]] = []
    url = f"{base_url.rstrip('/')}/fapi/v1/fundingRate"

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        batch = _request_json(session, url, params=params, max_retries=max_retries)
        if isinstance(batch, dict):
            batch = [batch]
        if not batch:
            break
        records.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        if last_time < cursor:
            break
        cursor = last_time + 1
        time.sleep(0.2)

    if not records:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price", "symbol"])

    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"].astype("int64"), unit="ms", utc=True)
    frame["funding_rate"] = frame["fundingRate"].astype(float)
    frame["mark_price"] = frame.get("markPrice", 0.0).astype(float)
    frame["symbol"] = normalize_symbol(symbol)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return frame[["timestamp", "funding_rate", "mark_price", "symbol"]]


def fetch_open_interest_hist(
    session: requests.Session,
    base_url: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    max_retries: int,
    period: str = "15m",
    limit: int = 500,
) -> pd.DataFrame:
    cursor = start_ms
    records: list[dict[str, Any]] = []
    url = f"{base_url.rstrip('/')}/futures/data/openInterestHist"

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        batch = _request_json(session, url, params=params, max_retries=max_retries)
        if isinstance(batch, dict):
            batch = [batch]
        if not batch:
            break
        records.extend(batch)
        last_time = int(batch[-1]["timestamp"])
        if last_time < cursor:
            break
        cursor = last_time + 1
        time.sleep(0.2)

    if not records:
        return pd.DataFrame(
            columns=["timestamp", "open_interest", "open_interest_value", "symbol", "source"]
        )

    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    frame["open_interest"] = frame["sumOpenInterest"].astype(float)
    frame["open_interest_value"] = frame["sumOpenInterestValue"].astype(float)
    frame["symbol"] = normalize_symbol(symbol)
    frame["source"] = "openInterestHist"
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return frame[["timestamp", "open_interest", "open_interest_value", "symbol", "source"]]


def fetch_open_interest_snapshot(
    session: requests.Session,
    base_url: str,
    symbol: str,
    max_retries: int,
) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    result = _request_json(session, url, params=params, max_retries=max_retries)
    if not result:
        return pd.DataFrame(columns=["timestamp", "open_interest", "symbol", "source"])

    payload = result[0] if isinstance(result, list) else result
    timestamp_ms = int(payload.get("time", int(time.time() * 1000)))
    return pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                "open_interest": float(payload["openInterest"]),
                "symbol": normalize_symbol(symbol),
                "source": "snapshot",
            }
        ]
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(frame), path)


def _write_oi_limitation_note(path: Path) -> None:
    message = (
        "Open interest historical endpoint returned no data for the requested range. "
        "Fallback snapshots were stored instead. "
        "Price + funding pipeline remains operational; historical OI can be upgraded later."
    )
    path.write_text(message, encoding="utf-8")
    LOGGER.warning(message)


def run_download(args: Any) -> None:
    _setup_logging()
    ensure_directories()
    config = _runtime_config(args)

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    session = requests.Session()

    for timeframe in config.timeframes:
        LOGGER.info("Downloading spot OHLCV for %s (%s)", config.symbol, timeframe)
        ohlcv = fetch_ohlcv(
            exchange=exchange,
            symbol=config.symbol,
            timeframe=timeframe,
            start_ms=config.start_ms,
            end_ms=config.end_ms,
        )
        _write_parquet(ohlcv, RAW_DATA_DIR / f"ohlcv_{timeframe}.parquet")

    LOGGER.info("Downloading funding history for %s", config.symbol)
    funding = fetch_funding_history(
        session=session,
        base_url=config.base_url,
        symbol=config.symbol,
        start_ms=config.start_ms,
        end_ms=config.end_ms,
        max_retries=config.max_retries,
    )
    _write_parquet(funding, RAW_DATA_DIR / "funding_rate.parquet")

    LOGGER.info("Downloading open interest history for %s", config.symbol)
    open_interest_hist = fetch_open_interest_hist(
        session=session,
        base_url=config.base_url,
        symbol=config.symbol,
        start_ms=config.start_ms,
        end_ms=config.end_ms,
        max_retries=config.max_retries,
    )

    if open_interest_hist.empty:
        LOGGER.warning("Historical OI unavailable; saving snapshot fallback")
        snapshot = fetch_open_interest_snapshot(
            session=session,
            base_url=config.base_url,
            symbol=config.symbol,
            max_retries=config.max_retries,
        )
        _write_parquet(snapshot, RAW_DATA_DIR / "open_interest_snapshots.parquet")
        _write_oi_limitation_note(RAW_DATA_DIR / "open_interest_LIMITATION.md")
    else:
        _write_parquet(open_interest_hist, RAW_DATA_DIR / "open_interest_hist.parquet")

    session.close()
