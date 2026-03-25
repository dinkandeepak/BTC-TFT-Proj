from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import (
    ARTIFACTS_DIR,
    DEFAULT_QUANTILES,
    PROCESSED_DATA_DIR,
    TIMEFRAME_SPECS,
    ensure_directories,
)

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "target_return"
KNOWN_REAL_COLUMNS = [
    "time_idx",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_asia_session",
    "is_europe_session",
    "is_us_session",
]
PRICE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "rolling_vol",
    "range_pct",
    "volume_log1p",
    "volume_zscore",
    "ema_slope",
]
DERIVATIVE_FEATURE_COLUMNS = [
    "funding_rate",
    "funding_rate_change",
    "funding_rate_available",
    "open_interest_log",
    "open_interest_change",
    "open_interest_available",
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_processed_data(timeframe: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"btc_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("time_idx").drop_duplicates("time_idx").reset_index(drop=True)
    return frame


def resolve_feature_columns(variant: str) -> list[str]:
    if variant == "price_only":
        return PRICE_FEATURE_COLUMNS.copy()
    if variant == "full":
        return PRICE_FEATURE_COLUMNS + DERIVATIVE_FEATURE_COLUMNS
    raise ValueError(f"Unknown variant: {variant}")


def prepare_model_frame(frame: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = resolve_feature_columns(variant)

    required = {
        "timestamp",
        "symbol",
        "time_idx",
        TARGET_COLUMN,
        *KNOWN_REAL_COLUMNS,
        *feature_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for training: {sorted(missing)}")

    model_frame = frame[list(required)].copy()
    for column in feature_columns:
        model_frame[column] = model_frame[column].fillna(0.0)
    model_frame[TARGET_COLUMN] = model_frame[TARGET_COLUMN].fillna(0.0)
    model_frame["symbol"] = model_frame["symbol"].astype(str)
    model_frame = model_frame.sort_values("time_idx").reset_index(drop=True)
    return model_frame, feature_columns


def compute_training_cutoff(model_frame: pd.DataFrame, max_prediction_length: int) -> int:
    max_time_idx = int(model_frame["time_idx"].max())
    validation_span = max_prediction_length * 3
    training_cutoff = max_time_idx - validation_span
    if training_cutoff <= 0:
        raise ValueError("Not enough rows to create train/validation split.")
    return training_cutoff


def _import_ml_dependencies() -> dict[str, Any]:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    return {
        "L": L,
        "EarlyStopping": EarlyStopping,
        "ModelCheckpoint": ModelCheckpoint,
        "TemporalFusionTransformer": TemporalFusionTransformer,
        "TimeSeriesDataSet": TimeSeriesDataSet,
        "GroupNormalizer": GroupNormalizer,
        "QuantileLoss": QuantileLoss,
    }


def resolve_lengths(timeframe: str, args: Any) -> tuple[int, int, int]:
    spec = TIMEFRAME_SPECS[timeframe]
    min_encoder = int(getattr(args, "min_encoder_length", 0) or spec.min_encoder_length)
    max_encoder = int(getattr(args, "max_encoder_length", 0) or spec.max_encoder_length)
    prediction = int(getattr(args, "prediction_length", 0) or spec.max_prediction_length)

    if min_encoder < 1 or max_encoder < 1 or prediction < 1:
        raise ValueError("Encoder and prediction lengths must be positive integers.")
    if min_encoder > max_encoder:
        raise ValueError("min_encoder_length cannot be greater than max_encoder_length.")
    return min_encoder, max_encoder, prediction


def train_single_variant(timeframe: str, variant: str, args: Any) -> Path:
    deps = _import_ml_dependencies()
    frame = load_processed_data(timeframe)
    model_frame, feature_columns = prepare_model_frame(frame, variant)
    min_encoder_length, max_encoder_length, prediction_length = resolve_lengths(timeframe, args)
    training_cutoff = compute_training_cutoff(model_frame, prediction_length)

    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    GroupNormalizer = deps["GroupNormalizer"]
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]
    QuantileLoss = deps["QuantileLoss"]
    L = deps["L"]
    EarlyStopping = deps["EarlyStopping"]
    ModelCheckpoint = deps["ModelCheckpoint"]

    training = TimeSeriesDataSet(
        model_frame[model_frame["time_idx"] <= training_cutoff],
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
        model_frame,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=int(args.batch_size) * 2,
        num_workers=int(args.num_workers),
    )

    artifact_dir = ARTIFACTS_DIR / timeframe / variant
    artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(artifact_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=6, min_delta=1e-4),
        checkpoint_callback,
    ]

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(args.learning_rate),
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
        max_epochs=int(args.max_epochs),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=False,
        enable_model_summary=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("Training finished without a checkpoint.")

    final_ckpt = artifact_dir / "model.ckpt"
    if Path(best_model_path).resolve() != final_ckpt.resolve():
        shutil.copy2(best_model_path, final_ckpt)

    with (artifact_dir / "dataset_parameters.pkl").open("wb") as handle:
        pickle.dump(training.get_parameters(), handle)

    metadata = {
        "timeframe": timeframe,
        "variant": variant,
        "quantiles": list(DEFAULT_QUANTILES),
        "target_column": TARGET_COLUMN,
        "known_real_columns": KNOWN_REAL_COLUMNS,
        "feature_columns": feature_columns,
        "best_model_path": str(final_ckpt),
        "training_cutoff": training_cutoff,
        "min_encoder_length": min_encoder_length,
        "max_encoder_length": max_encoder_length,
        "prediction_length": prediction_length,
        "rows": len(model_frame),
    }
    (artifact_dir / "model_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved training artifacts to %s", artifact_dir)
    return artifact_dir


def run_train(args: Any) -> None:
    _setup_logging()
    ensure_directories()

    timeframe = str(args.timeframe)
    variants = ["full", "price_only"] if args.variant == "all" else [str(args.variant)]

    for variant in variants:
        LOGGER.info("Training timeframe=%s variant=%s", timeframe, variant)
        train_single_variant(timeframe=timeframe, variant=variant, args=args)
