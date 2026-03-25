from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.ensemble.core import ModelSnapshot
from src.ensemble.evaluation import PredictionRecord, RollingEvaluator
from src.ensemble.worker import DataSource, auto_update_once


class FakeDataSource(DataSource):
    def __init__(self):
        self.logged: list[dict] = []

    def get_current_price(self, symbol: str) -> float:
        return 100.0

    def get_model_predictions(self, symbol: str, horizon: str) -> list[ModelSnapshot]:
        if horizon == "1h":
            return [
                ModelSnapshot("TFT", "1h", acc=0.55, conf=0.8, n=10, direction=1, price_pred=101.0),
                ModelSnapshot("LSTM", "1h", acc=0.54, conf=0.7, n=8, direction=1, price_pred=100.5),
            ]
        if horizon == "24h":
            return [
                ModelSnapshot(
                    "TFT",
                    "24h",
                    acc=0.56,
                    conf=0.75,
                    n=12,
                    direction=-1,
                    price_pred=98.0,
                ),
                ModelSnapshot(
                    "LSTM",
                    "24h",
                    acc=0.55,
                    conf=0.65,
                    n=9,
                    direction=-1,
                    price_pred=97.5,
                ),
            ]
        return []

    def resolve_pending_predictions(self, now: datetime) -> list[PredictionRecord]:
        base = now - timedelta(hours=2)
        return [
            PredictionRecord(
                timestamp_open=base,
                timestamp_resolve=base + timedelta(hours=1),
                horizon="1h",
                model_name="TFT",
                direction_pred=1,
                price_open=100.0,
                price_resolve=101.0,
            )
        ]

    def log_new_prediction(
        self,
        symbol: str,
        horizon: str,
        ensemble_output: dict,
        model_snapshots: list[ModelSnapshot],
        now: datetime,
    ) -> None:
        self.logged.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "output": ensemble_output,
                "count": len(model_snapshots),
                "now": now,
            }
        )


def test_auto_update_once_runs_for_both_horizons_and_logs_predictions() -> None:
    evaluator = RollingEvaluator(window_size=50)
    source = FakeDataSource()
    outputs = auto_update_once(
        symbol="BTCUSDT",
        data_source=source,
        evaluator=evaluator,
        horizons=("1h", "24h"),
        now=datetime(2026, 3, 1, tzinfo=UTC),
    )

    assert "1h" in outputs
    assert "24h" in outputs
    assert len(source.logged) == 2
    assert all(item["count"] >= 1 for item in source.logged)
