from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.train_tft import load_processed_data, prepare_model_frame
from src.settings import ARTIFACTS_DIR, TIMEFRAME_SPECS, ensure_directories

LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _import_ml_dependencies() -> dict[str, Any]:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    return {
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "TimeSeriesDataSet": TimeSeriesDataSet,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_tensor(raw_predictions: Any) -> Any:
    if isinstance(raw_predictions, dict) and "prediction" in raw_predictions:
        return raw_predictions["prediction"]
    if hasattr(raw_predictions, "output"):
        output = raw_predictions.output
        if isinstance(output, dict) and "prediction" in output:
            return output["prediction"]
        if hasattr(output, "prediction"):
            return output.prediction
    if hasattr(raw_predictions, "prediction"):
        return raw_predictions.prediction
    return raw_predictions


def _artifact_dir(timeframe: str, variant: str) -> Path:
    return ARTIFACTS_DIR / timeframe / variant


def _artifact_exists(timeframe: str, variant: str) -> bool:
    directory = _artifact_dir(timeframe, variant)
    return (directory / "model.ckpt").exists() and (directory / "dataset_parameters.pkl").exists()


def _recent_derivative_coverage(frame: pd.DataFrame, window: int) -> float:
    if (
        "funding_rate_available" not in frame.columns
        or "open_interest_available" not in frame.columns
    ):
        return 0.0
    tail = frame.tail(window)
    funding_cov = float(tail["funding_rate_available"].mean())
    oi_cov = float(tail["open_interest_available"].mean())
    return min(funding_cov, oi_cov)


def _select_variant(
    requested_variant: str,
    timeframe: str,
    frame: pd.DataFrame,
    threshold: float,
) -> str:
    lookback = max(200, TIMEFRAME_SPECS[timeframe].max_prediction_length * 5)
    coverage = _recent_derivative_coverage(frame, window=lookback)

    if requested_variant == "price_only":
        return "price_only"
    if requested_variant == "full":
        if coverage < threshold:
            LOGGER.warning(
                "Derivative coverage %.3f < threshold %.3f, falling back to price_only",
                coverage,
                threshold,
            )
            return "price_only"
        return "full"

    # auto
    if coverage >= threshold and _artifact_exists(timeframe, "full"):
        return "full"
    return "price_only"


def run_predict(args: Any) -> None:
    _setup_logging()
    ensure_directories()
    timeframe = str(args.timeframe)
    requested_variant = str(args.variant)
    threshold = float(args.coverage_threshold)

    base_frame = load_processed_data(timeframe)
    selected_variant = _select_variant(requested_variant, timeframe, base_frame, threshold)
    if not _artifact_exists(timeframe, selected_variant):
        raise FileNotFoundError(
            f"Missing artifacts for timeframe={timeframe}, variant={selected_variant}. "
            "Run training first."
        )

    model_frame, _ = prepare_model_frame(base_frame, selected_variant)
    artifact_dir = _artifact_dir(timeframe, selected_variant)
    dataset_param_path = artifact_dir / "dataset_parameters.pkl"
    model_path = artifact_dir / "model.ckpt"

    with dataset_param_path.open("rb") as handle:
        dataset_parameters = pickle.load(handle)

    deps = _import_ml_dependencies()
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]

    prediction_dataset = TimeSeriesDataSet.from_parameters(
        dataset_parameters,
        model_frame,
        predict=True,
        stop_randomization=True,
    )
    prediction_loader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(str(model_path))
    raw_predictions = model.predict(prediction_loader, mode="raw")
    pred = _to_numpy(_prediction_tensor(raw_predictions))
    if pred.ndim != 3:
        raise ValueError(f"Unexpected prediction tensor shape: {pred.shape}")

    quantile_returns = pred[0]
    spec = TIMEFRAME_SPECS[timeframe]
    last_ts = base_frame["timestamp"].max()
    future_ts = pd.date_range(
        start=last_ts + pd.to_timedelta(spec.pandas_freq),
        periods=quantile_returns.shape[0],
        freq=spec.pandas_freq,
        tz="UTC",
    )

    last_close = float(base_frame["close"].iloc[-1])
    q10_prices = last_close * np.exp(np.cumsum(quantile_returns[:, 0]))
    q50_prices = last_close * np.exp(np.cumsum(quantile_returns[:, 1]))
    q90_prices = last_close * np.exp(np.cumsum(quantile_returns[:, 2]))

    forecast = pd.DataFrame(
        {
            "timestamp": future_ts,
            "q10_return": quantile_returns[:, 0],
            "q50_return": quantile_returns[:, 1],
            "q90_return": quantile_returns[:, 2],
            "q10_price": q10_prices,
            "q50_price": q50_prices,
            "q90_price": q90_prices,
            "variant_used": selected_variant,
        }
    )

    output_parquet = artifact_dir / "latest_forecast.parquet"
    output_csv = artifact_dir / "latest_forecast.csv"
    forecast.to_parquet(output_parquet, index=False)
    forecast.to_csv(output_csv, index=False)

    summary = {
        "timeframe": timeframe,
        "requested_variant": requested_variant,
        "selected_variant": selected_variant,
        "coverage_threshold": threshold,
        "forecast_rows": len(forecast),
        "output_parquet": str(output_parquet),
    }
    (artifact_dir / "latest_forecast_meta.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Saved forecast to %s", output_parquet)
    LOGGER.info("Selected variant: %s", selected_variant)
    LOGGER.info("Head:\n%s", forecast.head(10).to_string(index=False))
