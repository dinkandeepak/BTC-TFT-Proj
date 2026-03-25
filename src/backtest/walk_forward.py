from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train_tft import (
    KNOWN_REAL_COLUMNS,
    TARGET_COLUMN,
    load_processed_data,
    prepare_model_frame,
    resolve_lengths,
)
from src.settings import (
    ARTIFACTS_DIR,
    DEFAULT_QUANTILES,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_directories,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _import_ml_dependencies() -> dict[str, Any]:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    return {
        "L": L,
        "EarlyStopping": EarlyStopping,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "TimeSeriesDataSet": TimeSeriesDataSet,
        "GroupNormalizer": GroupNormalizer,
        "QuantileLoss": QuantileLoss,
    }


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def generate_monthly_folds(
    frame: pd.DataFrame,
    train_months: int,
    val_months: int,
    test_months: int,
) -> list[FoldWindow]:
    first_month = _month_start(frame["timestamp"].min())
    last_ts = frame["timestamp"].max()
    month_starts = pd.date_range(
        start=first_month,
        end=last_ts + pd.offsets.MonthBegin(1),
        freq="MS",
        tz="UTC",
    )
    min_offset = train_months + val_months
    folds: list[FoldWindow] = []

    for i in range(min_offset, len(month_starts) - test_months):
        train_start = month_starts[i - min_offset]
        val_start = month_starts[i - val_months]
        test_start = month_starts[i]
        test_end = month_starts[i + test_months] - pd.Timedelta(nanoseconds=1)
        fold = FoldWindow(
            fold_id=len(folds) + 1,
            train_start=train_start,
            train_end=val_start - pd.Timedelta(nanoseconds=1),
            val_start=val_start,
            val_end=test_start - pd.Timedelta(nanoseconds=1),
            test_start=test_start,
            test_end=test_end,
        )
        folds.append(fold)

    if not folds:
        raise ValueError("No monthly folds generated. Increase data range or reduce window sizes.")
    return folds


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


def _extract_x_value(x_payload: Any, key: str) -> Any:
    if isinstance(x_payload, dict):
        return x_payload[key]
    return getattr(x_payload, key)


def _split_prediction_result(prediction_result: Any) -> tuple[Any, Any]:
    if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
        return prediction_result[0], prediction_result[1]
    if hasattr(prediction_result, "output") and hasattr(prediction_result, "x"):
        return prediction_result.output, prediction_result.x
    if (
        isinstance(prediction_result, dict)
        and "output" in prediction_result
        and "x" in prediction_result
    ):
        return prediction_result["output"], prediction_result["x"]
    raise ValueError("Unexpected predict() return type when return_x=True.")


def _predictions_to_frame(
    raw_predictions: Any,
    x_payload: Any,
    index_map: pd.Series,
) -> pd.DataFrame:
    pred = _to_numpy(_prediction_tensor(raw_predictions))
    decoder_target = _to_numpy(_extract_x_value(x_payload, "decoder_target"))
    decoder_time_idx = _to_numpy(_extract_x_value(x_payload, "decoder_time_idx"))

    rows: list[dict[str, float | int]] = []
    for batch_idx in range(pred.shape[0]):
        for step_idx in range(pred.shape[1]):
            rows.append(
                {
                    "time_idx": int(decoder_time_idx[batch_idx, step_idx]),
                    "actual": float(decoder_target[batch_idx, step_idx]),
                    "q10": float(pred[batch_idx, step_idx, 0]),
                    "q50": float(pred[batch_idx, step_idx, 1]),
                    "q90": float(pred[batch_idx, step_idx, 2]),
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame.groupby("time_idx", as_index=False).agg(
        actual=("actual", "mean"),
        q10=("q10", "mean"),
        q50=("q50", "mean"),
        q90=("q90", "mean"),
    )
    frame["timestamp"] = frame["time_idx"].map(index_map)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _pinball_loss(actual: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["actual"].to_numpy()
    median_pred = predictions["q50"].to_numpy()
    q10 = predictions["q10"].to_numpy()
    q90 = predictions["q90"].to_numpy()

    mae = float(np.mean(np.abs(median_pred - actual)))
    rmse = float(np.sqrt(np.mean((median_pred - actual) ** 2)))
    directional_hit_rate = float(np.mean(np.sign(median_pred) == np.sign(actual)))
    pinball_10 = _pinball_loss(actual, q10, 0.1)
    pinball_50 = _pinball_loss(actual, median_pred, 0.5)
    pinball_90 = _pinball_loss(actual, q90, 0.9)
    pinball_mean = float(np.mean([pinball_10, pinball_50, pinball_90]))

    coverage_10 = float(np.mean(actual <= q10))
    coverage_90 = float(np.mean(actual <= q90))
    calibration_error = float(abs(coverage_10 - 0.1) + abs(coverage_90 - 0.9))

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_hit_rate": directional_hit_rate,
        "pinball_loss": pinball_mean,
        "coverage_q10": coverage_10,
        "coverage_q90": coverage_90,
        "calibration_error": calibration_error,
    }


def _plot_fold_predictions(
    predictions: pd.DataFrame,
    timeframe: str,
    variant: str,
    fold_id: int,
) -> Path:
    path = PLOTS_DIR / f"{timeframe}_{variant}_fold_{fold_id}.png"
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(predictions["timestamp"], predictions["actual"], label="Actual return", linewidth=1.0)
    axis.plot(predictions["timestamp"], predictions["q50"], label="Median forecast", linewidth=1.0)
    axis.fill_between(
        predictions["timestamp"],
        predictions["q10"],
        predictions["q90"],
        alpha=0.2,
        label="10-90% band",
    )
    axis.set_title(f"{timeframe} {variant} fold {fold_id}")
    axis.set_xlabel("Timestamp (UTC)")
    axis.set_ylabel("Log return")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _train_fold_model(
    deps: dict[str, Any],
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_columns: list[str],
    min_encoder_length: int,
    max_encoder_length: int,
    prediction_length: int,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
) -> tuple[Any, Any]:
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]
    QuantileLoss = deps["QuantileLoss"]
    L = deps["L"]
    EarlyStopping = deps["EarlyStopping"]

    training = TimeSeriesDataSet(
        train_frame,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["symbol"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=KNOWN_REAL_COLUMNS,
        time_varying_unknown_reals=feature_columns,
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        eval_frame,
        min_prediction_idx=int(train_frame["time_idx"].max()) + 1,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=num_workers,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=len(DEFAULT_QUANTILES),
        loss=QuantileLoss(list(DEFAULT_QUANTILES)),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=4, min_delta=1e-4)],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model, training


def run_single_backtest_variant(
    timeframe: str,
    variant: str,
    args: Any,
) -> dict[str, Any]:
    deps = _import_ml_dependencies()
    min_encoder_length, max_encoder_length, prediction_length = resolve_lengths(timeframe, args)
    base_frame = load_processed_data(timeframe)
    model_frame, feature_columns = prepare_model_frame(base_frame, variant)
    folds = generate_monthly_folds(
        model_frame,
        train_months=int(args.train_months),
        val_months=int(args.val_months),
        test_months=int(args.test_months),
    )

    index_map = model_frame.set_index("time_idx")["timestamp"]
    fold_metrics: list[dict[str, Any]] = []

    for fold in folds:
        train_df = model_frame[
            (model_frame["timestamp"] >= fold.train_start)
            & (model_frame["timestamp"] <= fold.train_end)
        ].copy()
        val_df = model_frame[
            (model_frame["timestamp"] >= fold.val_start)
            & (model_frame["timestamp"] <= fold.val_end)
        ].copy()
        test_df = model_frame[
            (model_frame["timestamp"] >= fold.test_start)
            & (model_frame["timestamp"] <= fold.test_end)
        ].copy()

        if len(train_df) < (max_encoder_length + prediction_length):
            LOGGER.warning("Skipping fold %s due to insufficient training rows", fold.fold_id)
            continue
        if val_df.empty or test_df.empty:
            LOGGER.warning("Skipping fold %s due to empty val/test split", fold.fold_id)
            continue

        LOGGER.info(
            "Fold %s timeframe=%s variant=%s train=%s val=%s test=%s",
            fold.fold_id,
            timeframe,
            variant,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        eval_df = model_frame[model_frame["timestamp"] <= fold.val_end].copy()
        model, training_ds = _train_fold_model(
            deps=deps,
            train_frame=train_df,
            eval_frame=eval_df,
            feature_columns=feature_columns,
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            prediction_length=prediction_length,
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )

        predict_frame = model_frame[model_frame["timestamp"] <= fold.test_end].copy()
        test_dataset = deps["TimeSeriesDataSet"].from_dataset(
            training_ds,
            predict_frame,
            min_prediction_idx=int(test_df["time_idx"].min()),
            stop_randomization=True,
        )
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=int(args.batch_size) * 2,
            num_workers=int(args.num_workers),
        )

        prediction_result = model.predict(test_loader, mode="raw", return_x=True)
        raw_predictions, x_payload = _split_prediction_result(prediction_result)
        predictions = _predictions_to_frame(raw_predictions, x_payload, index_map=index_map)
        predictions = predictions[
            (predictions["timestamp"] >= fold.test_start)
            & (predictions["timestamp"] <= fold.test_end)
        ].copy()
        if predictions.empty:
            LOGGER.warning("Fold %s produced no test predictions", fold.fold_id)
            continue

        metrics = compute_metrics(predictions)
        plot_path = _plot_fold_predictions(predictions, timeframe, variant, fold.fold_id)

        fold_record = {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "val_start": fold.val_start.isoformat(),
            "val_end": fold.val_end.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "plot_path": str(plot_path),
            "n_predictions": int(len(predictions)),
            **metrics,
        }
        fold_metrics.append(fold_record)

    if not fold_metrics:
        raise RuntimeError(f"No completed folds for timeframe={timeframe}, variant={variant}")

    metrics_frame = pd.DataFrame(fold_metrics)
    summary = {
        "timeframe": timeframe,
        "variant": variant,
        "folds": fold_metrics,
        "aggregate": {
            "mae_mean": float(metrics_frame["mae"].mean()),
            "rmse_mean": float(metrics_frame["rmse"].mean()),
            "directional_hit_rate_mean": float(metrics_frame["directional_hit_rate"].mean()),
            "pinball_loss_mean": float(metrics_frame["pinball_loss"].mean()),
            "calibration_error_mean": float(metrics_frame["calibration_error"].mean()),
        },
    }

    variant_dir = ARTIFACTS_DIR / timeframe / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "backtest_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _render_markdown_summary(summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = ["# Walk-Forward Backtest Summary", ""]
    for summary in summaries:
        timeframe = summary["timeframe"]
        variant = summary["variant"]
        aggregate = summary["aggregate"]
        lines.extend(
            [
                f"## {timeframe} - {variant}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| MAE (mean) | {aggregate['mae_mean']:.6f} |",
                f"| RMSE (mean) | {aggregate['rmse_mean']:.6f} |",
                f"| Directional hit rate (mean) | {aggregate['directional_hit_rate_mean']:.4f} |",
                f"| Pinball loss (mean) | {aggregate['pinball_loss_mean']:.6f} |",
                f"| Calibration error (mean) | {aggregate['calibration_error_mean']:.6f} |",
                "",
                "| Fold | Test Start | Test End | MAE | RMSE | Hit Rate | Pinball |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for fold in summary["folds"]:
            lines.append(
                "| {fold_id} | {test_start} | {test_end} | {mae:.6f} | {rmse:.6f} | "
                "{directional_hit_rate:.4f} | {pinball_loss:.6f} |".format(**fold)
            )
        lines.append("")
    return "\n".join(lines)


def run_backtest(args: Any) -> None:
    _setup_logging()
    ensure_directories()

    timeframe = str(args.timeframe)
    variants = ["full", "price_only"] if args.variant == "all" else [str(args.variant)]

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        summary = run_single_backtest_variant(timeframe=timeframe, variant=variant, args=args)
        summaries.append(summary)

    markdown = _render_markdown_summary(summaries)
    summary_path = REPORTS_DIR / "summary.md"
    summary_path.write_text(markdown, encoding="utf-8")
    LOGGER.info("Wrote summary report to %s", summary_path)
