from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.ensemble.evaluation import PredictionRecord, RollingEvaluator


def test_rolling_evaluator_computes_directional_accuracy_and_drawdown() -> None:
    evaluator = RollingEvaluator(window_size=10)
    base = datetime(2026, 3, 1, tzinfo=UTC)

    records = [
        PredictionRecord(
            timestamp_open=base,
            timestamp_resolve=base + timedelta(hours=1),
            horizon="1h",
            model_name="ensemble",
            direction_pred=1,
            price_open=100.0,
            price_resolve=101.0,
        ),
        PredictionRecord(
            timestamp_open=base + timedelta(hours=1),
            timestamp_resolve=base + timedelta(hours=2),
            horizon="1h",
            model_name="ensemble",
            direction_pred=-1,
            price_open=101.0,
            price_resolve=100.0,
        ),
        PredictionRecord(
            timestamp_open=base + timedelta(hours=2),
            timestamp_resolve=base + timedelta(hours=3),
            horizon="1h",
            model_name="ensemble",
            direction_pred=1,
            price_open=100.0,
            price_resolve=99.0,
        ),
    ]
    evaluator.add_records(records)
    metrics = evaluator.get_metrics(model_name="ensemble", horizon="1h")

    assert metrics is not None
    assert metrics["n"] == 3
    assert metrics["directional_accuracy"] == 2 / 3
    assert metrics["max_drawdown"] >= 0.0
