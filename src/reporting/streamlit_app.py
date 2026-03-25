from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.reporting.dashboard import DEFAULT_VARIANTS, DashboardEntry, collect_entries
from src.settings import DEFAULT_TIMEFRAMES


def _metric_frame(entry: DashboardEntry) -> pd.DataFrame:
    if not entry.metrics:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [{"metric": key, "value": float(value)} for key, value in sorted(entry.metrics.items())]
    return pd.DataFrame(rows)


def _render_entry(entry: DashboardEntry) -> None:
    st.subheader(f"{entry.timeframe} / {entry.variant}")

    if entry.metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{entry.metrics.get('mae_mean', float('nan')):.6f}")
        col2.metric("RMSE", f"{entry.metrics.get('rmse_mean', float('nan')):.6f}")
        col3.metric(
            "Hit Rate",
            f"{entry.metrics.get('directional_hit_rate_mean', float('nan')):.4f}",
        )
        col4.metric("Pinball", f"{entry.metrics.get('pinball_loss_mean', float('nan')):.6f}")
        col5.metric("Calib Err", f"{entry.metrics.get('calibration_error_mean', float('nan')):.6f}")
        st.caption(f"Completed folds: {entry.fold_count}")
        st.dataframe(_metric_frame(entry), hide_index=True, use_container_width=True)
    else:
        st.info("No backtest metrics available for this entry.")

    if entry.forecast is not None and not entry.forecast.empty:
        forecast = entry.forecast.copy().sort_values("timestamp")
        required_cols = ["timestamp", "q10_price", "q50_price", "q90_price"]
        available_cols = [col for col in required_cols if col in forecast.columns]

        if len(available_cols) == 4:
            chart_df = forecast[required_cols].set_index("timestamp")
            st.line_chart(chart_df, use_container_width=True)

            latest = forecast.iloc[-1]
            latest_cols = st.columns(3)
            latest_cols[0].metric("Latest Q10 Price", f"{float(latest['q10_price']):.2f}")
            latest_cols[1].metric("Latest Q50 Price", f"{float(latest['q50_price']):.2f}")
            latest_cols[2].metric("Latest Q90 Price", f"{float(latest['q90_price']):.2f}")
        else:
            st.info("Forecast found, but expected quantile price columns are missing.")

        preview_allow = (
            "timestamp",
            "q10_return",
            "q50_return",
            "q90_return",
            "q10_price",
            "q50_price",
            "q90_price",
        )
        preview_columns = [col for col in forecast.columns if col in preview_allow]
        if preview_columns:
            st.dataframe(
                forecast[preview_columns].tail(20),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("No latest forecast available for this entry.")

    for note in entry.notes:
        st.warning(note)


def main() -> None:
    st.set_page_config(page_title="BTC TFT Dashboard", layout="wide")
    st.title("BTC TFT Live Dashboard")
    st.caption(f"Updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    with st.sidebar:
        st.header("Filters")
        timeframes = st.multiselect(
            "Timeframes",
            options=list(DEFAULT_TIMEFRAMES),
            default=list(DEFAULT_TIMEFRAMES),
        )
        variants = st.multiselect(
            "Variants",
            options=list(DEFAULT_VARIANTS),
            default=list(DEFAULT_VARIANTS),
        )

    if not timeframes or not variants:
        st.warning("Select at least one timeframe and one variant.")
        return

    entries = collect_entries(timeframes=tuple(timeframes), variants=tuple(variants))
    if not entries:
        st.info("No dashboard entries found.")
        return

    for entry in entries:
        _render_entry(entry)
        st.divider()


if __name__ == "__main__":
    main()
