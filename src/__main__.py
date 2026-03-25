from __future__ import annotations

import argparse
from typing import Sequence

from src.settings import DEFAULT_SYMBOL, DEFAULT_TIMEFRAMES


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(DEFAULT_TIMEFRAMES),
        choices=list(DEFAULT_TIMEFRAMES),
        help="Timeframes to process.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help="Trading symbol (default: BTCUSDT).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC TFT forecasting pipeline")
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser("download", help="Download raw market data")
    _add_shared_data_args(download_parser)
    download_parser.add_argument("--start-date", type=str, default=None, help="UTC start date")
    download_parser.add_argument("--end-date", type=str, default=None, help="UTC end date")

    build_parser_cmd = subparsers.add_parser(
        "build",
        help="Build processed feature datasets from raw data",
    )
    _add_shared_data_args(build_parser_cmd)

    train_parser = subparsers.add_parser("train", help="Train TFT models")
    train_parser.add_argument("--timeframe", choices=list(DEFAULT_TIMEFRAMES), required=True)
    train_parser.add_argument("--variant", choices=["full", "price_only", "all"], default="all")
    train_parser.add_argument("--max-epochs", type=int, default=25)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)

    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtests")
    backtest_parser.add_argument("--timeframe", choices=list(DEFAULT_TIMEFRAMES), required=True)
    backtest_parser.add_argument("--variant", choices=["full", "price_only", "all"], default="all")
    backtest_parser.add_argument("--train-months", type=int, default=12)
    backtest_parser.add_argument("--val-months", type=int, default=1)
    backtest_parser.add_argument("--test-months", type=int, default=1)
    backtest_parser.add_argument("--max-epochs", type=int, default=10)
    backtest_parser.add_argument("--batch-size", type=int, default=128)
    backtest_parser.add_argument("--num-workers", type=int, default=0)

    predict_parser = subparsers.add_parser("predict", help="Generate quantile forecasts")
    predict_parser.add_argument("--timeframe", choices=list(DEFAULT_TIMEFRAMES), required=True)
    predict_parser.add_argument("--variant", choices=["auto", "full", "price_only"], default="auto")
    predict_parser.add_argument("--coverage-threshold", type=float, default=0.7)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        from src.data.download import run_download

        run_download(args)
    elif args.command == "build":
        from src.features.make_features import run_build

        run_build(args)
    elif args.command == "train":
        from src.models.train_tft import run_train

        run_train(args)
    elif args.command == "backtest":
        from src.backtest.walk_forward import run_backtest

        run_backtest(args)
    elif args.command == "predict":
        from src.models.predict import run_predict

        run_predict(args)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
