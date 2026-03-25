from __future__ import annotations

import pytest

from src.ensemble.core import ModelSnapshot, compute_ensemble


def test_compute_ensemble_filters_by_horizon_and_outputs_long_signal() -> None:
    models = [
        ModelSnapshot(
            name="TFT_1H",
            horizon="1h",
            acc=0.62,
            conf=0.80,
            n=120,
            direction=1,
            price_pred=101.0,
        ),
        ModelSnapshot(
            name="LSTM_1H",
            horizon="1h",
            acc=0.59,
            conf=0.70,
            n=100,
            direction=1,
            price_pred=100.8,
        ),
        ModelSnapshot(
            name="TFT_24H",
            horizon="24h",
            acc=0.63,
            conf=0.75,
            n=140,
            direction=-1,
            price_pred=90.0,
        ),
    ]
    output = compute_ensemble(models=models, current_price=100.0, horizon="1h")

    assert output["horizon"] == "1h"
    assert output["direction"] == 1
    assert output["ensemble_price"] > 100.0
    assert set(output["weights"].keys()) == {"TFT_1H", "LSTM_1H"}


def test_compute_ensemble_handles_cold_start_zero_quality_scores() -> None:
    models = [
        ModelSnapshot("A", "1h", acc=0.5, conf=0.0, n=0, direction=1, price_pred=101.0),
        ModelSnapshot("B", "1h", acc=0.5, conf=0.0, n=0, direction=-1, price_pred=99.0),
    ]
    output = compute_ensemble(models=models, current_price=100.0, horizon="1h")
    assert 0.0 <= output["confidence"] <= 1.0
    assert pytest.approx(output["ensemble_price"], abs=1e-6) == 100.0


def test_compute_ensemble_rejects_non_positive_price() -> None:
    models = [ModelSnapshot("A", "1h", acc=0.6, conf=0.7, n=10, direction=1, price_pred=100.0)]
    with pytest.raises(ValueError, match="current_price"):
        compute_ensemble(models=models, current_price=0.0, horizon="1h")
