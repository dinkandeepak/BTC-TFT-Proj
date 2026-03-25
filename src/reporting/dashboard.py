from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import (
    ARTIFACTS_DIR,
    DEFAULT_TIMEFRAMES,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_directories,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_VARIANTS = ("full", "price_only")


@dataclass
class DashboardEntry:
    timeframe: str
    variant: str
    metrics: dict[str, float] | None = None
    fold_count: int = 0
    forecast: pd.DataFrame | None = None
    forecast_plot: Path | None = None
    notes: list[str] = field(default_factory=list)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse JSON at %s: %s", path, exc)
        return None


def _safe_read_forecast(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read forecast file %s: %s", path, exc)
        return None


def _render_forecast_plot(entry: DashboardEntry) -> Path | None:
    import matplotlib.pyplot as plt

    if entry.forecast is None or entry.forecast.empty:
        return None

    frame = entry.forecast.copy()
    required = {"timestamp", "q10_price", "q50_price", "q90_price"}
    if not required.issubset(frame.columns):
        entry.notes.append("Latest forecast exists but is missing expected quantile price columns.")
        return None

    frame = frame.dropna(subset=["timestamp", "q10_price", "q50_price", "q90_price"]).copy()
    if frame.empty:
        entry.notes.append("Latest forecast exists but has no usable rows for plotting.")
        return None

    path = PLOTS_DIR / f"dashboard_{entry.timeframe}_{entry.variant}_forecast.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame["timestamp"], frame["q50_price"], label="Median price", linewidth=1.2)
    ax.fill_between(
        frame["timestamp"],
        frame["q10_price"],
        frame["q90_price"],
        alpha=0.2,
        label="10-90% band",
    )
    ax.set_title(f"Forecast bands: {entry.timeframe} / {entry.variant}")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _collect_entry(timeframe: str, variant: str) -> DashboardEntry:
    entry = DashboardEntry(timeframe=timeframe, variant=variant)
    variant_dir = ARTIFACTS_DIR / timeframe / variant

    if not variant_dir.exists():
        entry.notes.append("Artifact directory not found. Train/backtest this variant first.")
        return entry

    metrics_payload = _safe_read_json(variant_dir / "backtest_metrics.json")
    if metrics_payload and isinstance(metrics_payload, dict):
        aggregate = metrics_payload.get("aggregate", {})
        folds = metrics_payload.get("folds", [])
        if isinstance(aggregate, dict):
            entry.metrics = {
                key: float(value)
                for key, value in aggregate.items()
                if isinstance(value, int | float)
            }
        if isinstance(folds, list):
            entry.fold_count = len(folds)
    else:
        entry.notes.append("Backtest metrics not found.")

    forecast = _safe_read_forecast(variant_dir / "latest_forecast.parquet")
    if forecast is None:
        forecast = _safe_read_forecast(variant_dir / "latest_forecast.csv")
    if forecast is None or forecast.empty:
        entry.notes.append("Latest forecast not found.")
    else:
        entry.forecast = forecast
        entry.forecast_plot = _render_forecast_plot(entry)

    return entry


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def render_dashboard_markdown(entries: list[DashboardEntry]) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# BTC TFT Dashboard",
        "",
        f"Generated: {generated}",
        "",
    ]
    if not entries:
        lines.append("No dashboard entries found.")
        return "\n".join(lines)

    for entry in entries:
        lines.extend(
            [
                f"## {entry.timeframe} / {entry.variant}",
                "",
            ]
        )

        if entry.metrics:
            lines.extend(
                [
                    "| Metric | Value |",
                    "|---|---:|",
                ]
            )
            for key, value in sorted(entry.metrics.items()):
                lines.append(f"| {key} | {_format_metric(value)} |")
            lines.append("")
            lines.append(f"Completed folds: {entry.fold_count}")
            lines.append("")

        if entry.forecast is not None and not entry.forecast.empty:
            forecast = entry.forecast
            start_ts = str(forecast["timestamp"].min())
            end_ts = str(forecast["timestamp"].max())
            latest = forecast.iloc[-1]
            lines.append(f"Forecast horizon rows: {len(forecast)}")
            lines.append(f"Forecast range: {start_ts} -> {end_ts}")
            if "q50_price" in latest:
                lines.append(f"Latest median forecast price: {float(latest['q50_price']):.2f}")
            if "q50_return" in latest:
                lines.append(f"Latest median forecast return: {float(latest['q50_return']):.6f}")
            lines.append("")

        if entry.forecast_plot is not None:
            relative = entry.forecast_plot.relative_to(REPORTS_DIR)
            lines.append(f"![{entry.timeframe}-{entry.variant}]({relative.as_posix()})")
            lines.append("")

        for note in entry.notes:
            lines.append(f"- {note}")
        if entry.notes:
            lines.append("")

    return "\n".join(lines)


def collect_entries(
    timeframes: tuple[str, ...],
    variants: tuple[str, ...],
) -> list[DashboardEntry]:
    entries: list[DashboardEntry] = []
    for timeframe in timeframes:
        for variant in variants:
            entries.append(_collect_entry(timeframe=timeframe, variant=variant))
    return entries


def run_dashboard(args: Any) -> None:
    _setup_logging()
    ensure_directories()

    timeframes = tuple(getattr(args, "timeframes", None) or DEFAULT_TIMEFRAMES)
    variants = tuple(getattr(args, "variants", None) or DEFAULT_VARIANTS)
    output_path = Path(getattr(args, "output", REPORTS_DIR / "dashboard.md"))
    if not output_path.is_absolute():
        output_path = REPORTS_DIR / output_path

    entries = collect_entries(timeframes=timeframes, variants=variants)
    markdown = render_dashboard_markdown(entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    LOGGER.info("Dashboard written to %s", output_path)
