from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
PLOTS_DIR: Final[Path] = REPORTS_DIR / "plots"

DEFAULT_SYMBOL: Final[str] = "BTCUSDT"
DEFAULT_TIMEFRAMES: Final[tuple[str, str, str]] = ("15m", "1h", "4h")
DEFAULT_QUANTILES: Final[tuple[float, float, float]] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    pandas_freq: str
    minutes: int
    min_encoder_length: int
    max_encoder_length: int
    max_prediction_length: int


TIMEFRAME_SPECS: Final[dict[str, TimeframeSpec]] = {
    "15m": TimeframeSpec(
        name="15m",
        pandas_freq="15min",
        minutes=15,
        min_encoder_length=7 * 24 * 4,   # 7d
        max_encoder_length=14 * 24 * 4,  # 14d
        max_prediction_length=96,
    ),
    "1h": TimeframeSpec(
        name="1h",
        pandas_freq="1h",
        minutes=60,
        min_encoder_length=4 * 7 * 24,    # 4w
        max_encoder_length=12 * 7 * 24,   # 12w
        max_prediction_length=168,
    ),
    "4h": TimeframeSpec(
        name="4h",
        pandas_freq="4h",
        minutes=240,
        min_encoder_length=3 * 30 * 6,    # 3m (approx)
        max_encoder_length=12 * 30 * 6,   # 12m (approx)
        max_prediction_length=84,
    ),
}


def ensure_directories() -> None:
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()
