from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Optional

from src.ensemble.core import EnsembleParams, Horizon, ModelSnapshot, compute_ensemble
from src.ensemble.evaluation import PredictionRecord, RollingEvaluator


class DataSource(ABC):
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_model_predictions(self, symbol: str, horizon: Horizon) -> list[ModelSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def resolve_pending_predictions(self, now: datetime) -> list[PredictionRecord]:
        raise NotImplementedError

    @abstractmethod
    def log_new_prediction(
        self,
        symbol: str,
        horizon: Horizon,
        ensemble_output: dict,
        model_snapshots: list[ModelSnapshot],
        now: datetime,
    ) -> None:
        raise NotImplementedError


def _hydrate_snapshots_from_metrics(
    snapshots: list[ModelSnapshot],
    evaluator: RollingEvaluator,
    horizon: Horizon,
) -> list[ModelSnapshot]:
    hydrated: list[ModelSnapshot] = []
    for snapshot in snapshots:
        metrics = evaluator.get_metrics(snapshot.name, horizon)
        if metrics is None:
            hydrated.append(snapshot)
            continue
        hydrated.append(
            ModelSnapshot(
                name=snapshot.name,
                horizon=snapshot.horizon,
                acc=float(metrics["directional_accuracy"]),
                conf=snapshot.conf,
                n=int(metrics["n"]),
                direction=snapshot.direction,
                price_pred=snapshot.price_pred,
            )
        )
    return hydrated


def auto_update_once(
    symbol: str,
    data_source: DataSource,
    evaluator: RollingEvaluator,
    params: EnsembleParams = EnsembleParams(),
    horizons: tuple[Horizon, ...] = ("1h", "24h"),
    now: Optional[datetime] = None,
) -> dict[Horizon, dict]:
    now = now or datetime.now(UTC)

    resolved = data_source.resolve_pending_predictions(now)
    evaluator.add_records(resolved)

    price_now = data_source.get_current_price(symbol)
    outputs: dict[Horizon, dict] = {}
    for horizon in horizons:
        raw_snapshots = data_source.get_model_predictions(symbol, horizon)
        if not raw_snapshots:
            continue
        snapshots = _hydrate_snapshots_from_metrics(raw_snapshots, evaluator, horizon)
        ensemble_output = compute_ensemble(
            models=snapshots,
            current_price=price_now,
            horizon=horizon,
            params=params,
        )
        data_source.log_new_prediction(
            symbol=symbol,
            horizon=horizon,
            ensemble_output=ensemble_output,
            model_snapshots=snapshots,
            now=now,
        )
        outputs[horizon] = ensemble_output
    return outputs


def auto_update_worker(
    symbol: str,
    data_source: DataSource,
    evaluator: RollingEvaluator,
    params: EnsembleParams = EnsembleParams(),
    interval_seconds: int = 60,
    horizons: tuple[Horizon, ...] = ("1h", "24h"),
) -> None:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be >= 1.")

    while True:
        auto_update_once(
            symbol=symbol,
            data_source=data_source,
            evaluator=evaluator,
            params=params,
            horizons=horizons,
        )
        time.sleep(interval_seconds)
