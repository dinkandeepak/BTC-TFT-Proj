from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.ensemble.core import Direction, Horizon


@dataclass
class PredictionRecord:
    timestamp_open: datetime
    timestamp_resolve: datetime
    horizon: Horizon
    model_name: str      # "ensemble" or base model name
    direction_pred: Direction
    price_open: float
    price_resolve: float

    @property
    def ret_actual(self) -> float:
        return (self.price_resolve - self.price_open) / self.price_open

    @property
    def ret_signed(self) -> float:
        return self.direction_pred * self.ret_actual


class RollingEvaluator:
    def __init__(self, window_size: int = 200):
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")
        self.window_size = window_size
        self.records: dict[str, deque[PredictionRecord]] = {}

    def add_record(self, rec: PredictionRecord) -> None:
        key = f"{rec.model_name}:{rec.horizon}"
        if key not in self.records:
            self.records[key] = deque(maxlen=self.window_size)
        self.records[key].append(rec)

    def add_records(self, recs: list[PredictionRecord]) -> None:
        for rec in recs:
            self.add_record(rec)

    def get_metrics(self, model_name: str, horizon: Horizon) -> Optional[dict]:
        key = f"{model_name}:{horizon}"
        if key not in self.records or not self.records[key]:
            return None
        recs = list(self.records[key])

        n = len(recs)
        correct = 0
        eff_rets: list[float] = []
        for rec in recs:
            realized_dir: Direction = 0
            if rec.ret_actual > 0:
                realized_dir = 1
            elif rec.ret_actual < 0:
                realized_dir = -1

            if realized_dir == rec.direction_pred:
                correct += 1
            eff_rets.append(rec.ret_signed)

        acc = correct / n
        mean_signed_ret = sum(eff_rets) / n

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for ret_signed in eff_rets:
            cum += ret_signed
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        return {
            "model_name": model_name,
            "horizon": horizon,
            "n": n,
            "directional_accuracy": acc,
            "mean_signed_return": mean_signed_ret,
            "max_drawdown": max_dd,
        }
