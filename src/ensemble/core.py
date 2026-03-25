from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Direction = Literal[-1, 0, 1]  # SHORT, FLAT, LONG
Horizon = Literal["1h", "24h"]


@dataclass
class ModelSnapshot:
    name: str
    horizon: Horizon
    acc: float          # directional accuracy (0..1)
    conf: float         # current confidence (0..1)
    n: int              # number of trades used in acc
    direction: Direction
    price_pred: float


@dataclass
class EnsembleParams:
    alpha: float = 2.0
    beta: float = 1.0
    k: float = 20.0        # sample size dampener
    lam: float = 0.3       # minimum consensus share
    tau_1h: float = 0.0001
    tau_24h: float = 0.0003
    a: float = 500.0       # scaling magnitude of return
    b: float = 1.0         # scaling consensus


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_ensemble(
    models: list[ModelSnapshot],
    current_price: float,
    horizon: Horizon,
    params: EnsembleParams = EnsembleParams(),
) -> dict:
    if not models:
        raise ValueError("No models provided.")
    if current_price <= 0:
        raise ValueError("current_price must be positive.")

    tau = params.tau_1h if horizon == "1h" else params.tau_24h
    matching_indices = [idx for idx, model in enumerate(models) if model.horizon == horizon]
    if not matching_indices:
        raise ValueError(f"No models found for horizon={horizon}.")

    q_prime = [0.0] * len(models)
    for idx in matching_indices:
        model = models[idx]
        acc = _clamp(model.acc, 0.5, 1.0)
        conf = _clamp(model.conf, 0.0, 1.0)
        n_obs = max(0.0, float(model.n))

        base_q = (acc ** params.alpha) * (conf ** params.beta)
        scale = (n_obs / (n_obs + params.k)) ** 0.5 if n_obs > 0 else 0.0
        q_prime[idx] = base_q * scale

    total_q = sum(q_prime[idx] for idx in matching_indices)
    if total_q <= 0:
        # If all model quality scores are zero (e.g., cold start),
        # use equal provisional quality so we can still generate an ensemble.
        for idx in matching_indices:
            q_prime[idx] = 1.0
        total_q = float(len(matching_indices))

    consensus = sum(q_prime[idx] * models[idx].direction for idx in matching_indices) / total_q

    scores = [0.0] * len(models)
    for idx in matching_indices:
        model = models[idx]
        c_i = max(0.0, model.direction * consensus)
        scores[idx] = q_prime[idx] * (params.lam + (1.0 - params.lam) * c_i)

    total_s = sum(scores[idx] for idx in matching_indices)
    weights = [0.0] * len(models)
    if total_s <= 0:
        equal_weight = 1.0 / float(len(matching_indices))
        for idx in matching_indices:
            weights[idx] = equal_weight
    else:
        for idx in matching_indices:
            weights[idx] = scores[idx] / total_s

    ensemble_price = sum(weights[idx] * models[idx].price_pred for idx in matching_indices)
    ret_ens = (ensemble_price - current_price) / current_price

    if ret_ens > tau:
        direction: Direction = 1
    elif ret_ens < -tau:
        direction = -1
    else:
        direction = 0

    x = params.a * abs(ret_ens) + params.b * abs(consensus)
    conf_ens = 1.0 / (1.0 + math.exp(-x))

    model_weights = {
        models[idx].name: weights[idx]
        for idx in matching_indices
        if weights[idx] > 0.0
    }

    return {
        "horizon": horizon,
        "ensemble_price": ensemble_price,
        "direction": direction,  # -1 SHORT, 0 FLAT, 1 LONG
        "confidence": conf_ens,
        "weights": model_weights,
        "consensus": consensus,
        "ret_ens": ret_ens,
    }
