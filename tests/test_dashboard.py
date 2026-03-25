from __future__ import annotations

import pandas as pd

from src.reporting.dashboard import DashboardEntry, render_dashboard_markdown


def test_render_dashboard_markdown_includes_metrics_and_forecast() -> None:
    forecast = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-03-01T00:00:00Z", "2026-03-01T01:00:00Z"],
                utc=True,
            ),
            "q10_price": [90000.0, 90100.0],
            "q50_price": [90500.0, 90600.0],
            "q90_price": [91000.0, 91100.0],
            "q50_return": [0.001, 0.002],
        }
    )
    entry = DashboardEntry(
        timeframe="1h",
        variant="full",
        metrics={"mae_mean": 0.01, "rmse_mean": 0.02},
        fold_count=3,
        forecast=forecast,
    )

    markdown = render_dashboard_markdown([entry])
    assert "## 1h / full" in markdown
    assert "| mae_mean | 0.010000 |" in markdown
    assert "Completed folds: 3" in markdown
    assert "Forecast horizon rows: 2" in markdown
    assert "Latest median forecast price: 90600.00" in markdown


def test_render_dashboard_markdown_handles_empty_entries() -> None:
    markdown = render_dashboard_markdown([])
    assert "No dashboard entries found." in markdown
