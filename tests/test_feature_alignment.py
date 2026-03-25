from __future__ import annotations

import pandas as pd
import pytest

from src.features.make_features import (
    align_derivatives_asof,
    validate_bar_continuity,
    validate_no_leakage,
)


def test_asof_alignment_uses_last_known_value() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:15:00Z", "2026-01-01T00:30:00Z", "2026-01-01T00:45:00Z"],
                utc=True,
            ),
            "close": [100.0, 101.0, 102.0],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:10:00Z", "2026-01-01T00:40:00Z"], utc=True),
            "funding_rate": [0.001, 0.002],
        }
    )

    aligned = align_derivatives_asof(
        bars=bars,
        derivatives=funding,
        value_columns=["funding_rate"],
        source_time_col="timestamp",
        prefix="funding",
    )

    assert aligned.loc[0, "funding_rate"] == pytest.approx(0.001)
    assert aligned.loc[1, "funding_rate"] == pytest.approx(0.001)
    assert aligned.loc[2, "funding_rate"] == pytest.approx(0.002)
    assert (aligned["funding_published_at"] <= aligned["timestamp"]).all()


def test_validate_no_leakage_raises_on_future_feature_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T01:00:00Z"], utc=True),
            "funding_published_at": pd.to_datetime(["2026-01-01T01:15:00Z"], utc=True),
        }
    )
    with pytest.raises(ValueError, match="leakage detected"):
        validate_no_leakage(frame, "timestamp", "funding_published_at", "funding")


def test_validate_bar_continuity_detects_missing_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
                utc=True,
            )
        }
    )
    with pytest.raises(ValueError, match="missing"):
        validate_bar_continuity(frame, freq="15min", label="bars")
