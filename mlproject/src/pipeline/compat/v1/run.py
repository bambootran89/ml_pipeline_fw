"""
Extended command-line interface (CLI) supporting Training, Evaluation,
Serving, Cross-Validation (CV), and Hyperparameter Tuning.

Usage examples:
    # Standard modes
    python -m mlproject.src.pipeline.run_pipeline train --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline eval --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline serve --config path.yaml \
        --input file.csv

    # Additional modes
    python -m mlproject.src.pipeline.run_pipeline cv --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline tune --config path.yaml
"""

import argparse
import logging
from typing import Any, cast

import pandas as pd

from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.pipeline.compat.v1.cv import CrossValidationPipeline
from mlproject.src.pipeline.compat.v1.eval import EvalPipeline
from mlproject.src.pipeline.compat.v1.serve import ServingPipeline
from mlproject.src.pipeline.compat.v1.training import TrainingPipeline
from mlproject.src.pipeline.compat.v1.tuning import TuningPipeline
from mlproject.src.utils.config_class import ConfigLoader

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["OMP_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)


def run_training(cfg_path: str) -> None:
    """Execute the full training workflow."""
    pipeline = TrainingPipeline(cfg_path)
    pipeline.run()


def run_evaluation(cfg_path: str, alias: str) -> None:
    """Execute evaluation-only workflow."""
    pipeline = EvalPipeline(cfg_path, alias=alias)
    pipeline.run()


def run_serve(
    cfg_path: str, input_path: str | None, alias: str = "latest", time_point="now"
) -> Any:
    """Run inference at serving time using either a CSV input file or auto-loaded
    online features from Feast Feature Store.

    Modes:
    1) CSV Mode  → Load input features from a provided CSV file.
    2) Feast Mode → Ignore CSV and pull the most recent features
        from Feast online store.

    Args:
        cfg_path: Path to the experiment configuration file.
        input_path: Path to the input CSV file (optional when using Feast Mode).
        alias: Model alias in MLflow Model Registry (default: "latest").

    Raises:
        ValueError: If input_path is not provided and config does not specify a valid
                    Feast feature store URI.

    Returns:
        Model predictions as a NumPy array or pandas DataFrame
          depending on pipeline output.
    """
    pipeline = ServingPipeline(cfg_path, alias=alias, time_point=time_point)
    logger.info(
        "[SERVING] Pipeline initialized (config='%s', alias='%s' , time_point='%s')",
        cfg_path,
        alias,
        time_point,
    )

    if input_path:
        logger.info("[SERVING] CSV mode active (input='%s')", input_path)
        print(f"[SERVING] Loading data from CSV: {input_path}")

        df = pd.read_csv(input_path)

        if "date" not in df.columns:
            raise ValueError("Input CSV must contain a 'date' column")

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        logger.info(
            "[SERVING] Input loaded with shape (%d, %d)", df.shape[0], df.shape[1]
        )
        preds = pipeline.run_exp(data=df)

    else:
        uri = pipeline.cfg.data.get("path", "")
        logger.info("[SERVING] Feast mode active (config_uri='%s')", uri)
        print("[SERVING] No CSV provided, loading latest features from Feast")

        if not uri.startswith("feast://"):
            raise ValueError(
                "SERVING requires either a CSV input or a Feast URI in the config"
            )

        preds = pipeline.run_exp(data=None)

    print("\n" + "=" * 70)
    print("SERVING COMPLETED")
    print("=" * 70)
    print(f"Prediction output type: {type(preds)}")

    if hasattr(preds, "shape"):
        print(f"Prediction shape: {preds.shape}")

    print("=" * 70)
    logger.info("[SERVING] Inference completed successfully")

    return preds


def run_cross_validation(cfg_path: str) -> None:
    """
    Execute time-series cross-validation using ExpandingWindowSplitter.
    """
    cfg = ConfigLoader.load(cfg_path)

    splitter: BaseSplitter
    eval_type = cfg.get("data", {}).get("type", "timeseries")
    if eval_type == "timeseries":
        splitter = cast(
            BaseSplitter,
            TimeSeriesFoldSplitter(
                cfg,
                n_splits=cfg.get("tuning", {}).get("n_splits", 3),
            ),
        )
    else:
        splitter = BaseSplitter(
            cfg,
            n_splits=cfg.get("tuning", {}).get("n_splits", 3),
        )

    cv_pipeline = CrossValidationPipeline(cfg, splitter)

    cv_pipeline.run_cv()


def run_tuning(cfg_path: str) -> None:
    """
    Execute hyperparameter tuning workflow.
    """
    tuning_pipeline = TuningPipeline(cfg_path)
    tuning_pipeline.run()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build unified CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified entrypoint for Training, Evaluation, Serving, "
            "Cross-Validation, and Hyperparameter Tuning."
        )
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "serve", "cv", "tune"],
        help="Which workflow pipeline to execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="CSV path to use only for serve mode.",
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="latest",
        help="latest, production, staging",
    )

    parser.add_argument(
        "--time_point",
        type=str,
        default="now",
        help="Reference timestamp for serving (ISO or keyword: 'now').",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.config)
    elif args.mode == "eval":
        run_evaluation(args.config, args.alias)
    elif args.mode == "serve":
        run_serve(
            args.config,
            args.input,
            args.alias,
            args.time_point,
        )
    elif args.mode == "cv":
        run_cross_validation(args.config)
    elif args.mode == "tune":
        run_tuning(args.config)
    else:
        raise ValueError(f"Unknown mode '{args.mode}'.")


def main_run(mode: str, cfg_path: str = "", input_path: str = "") -> None:
    """Programmatic entrypoint."""
    if mode == "train":
        run_training(cfg_path)
    elif mode == "eval":
        run_evaluation(cfg_path, alias="latest")
    elif mode == "serve":
        run_serve(
            cfg_path,
            input_path,
            alias="latest",
        )
    elif mode == "cv":
        run_cross_validation(cfg_path)
    elif mode == "tune":
        run_tuning(cfg_path)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


if __name__ == "__main__":
    main()
