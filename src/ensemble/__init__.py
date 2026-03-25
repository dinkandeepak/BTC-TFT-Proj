"""Ensemble backend logic (multi-horizon weighting, evaluation, worker loop)."""

from src.ensemble.core import Direction, EnsembleParams, Horizon, ModelSnapshot, compute_ensemble
from src.ensemble.evaluation import PredictionRecord, RollingEvaluator
from src.ensemble.worker import DataSource, auto_update_once, auto_update_worker

__all__ = [
    "DataSource",
    "Direction",
    "EnsembleParams",
    "Horizon",
    "ModelSnapshot",
    "PredictionRecord",
    "RollingEvaluator",
    "auto_update_once",
    "auto_update_worker",
    "compute_ensemble",
]
