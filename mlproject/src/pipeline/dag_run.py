"""CLI entry point for running ML pipelines.

This module provides a command-line interface for executing different
pipeline types: train, eval, serve, tune, and generate-configs.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.dataio.loaddata import load_csv_data, load_from_feast
from mlproject.src.generator.orchestrator import ConfigGenerator
from mlproject.src.pipeline.flexible_pipeline import FlexiblePipeline
from mlproject.src.utils.config_class import ConfigMerger

# from mlproject.src.generator.orchestrator import ConfigGenerator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


def prepare_serving_data(
    cfg: DictConfig,
    input_path: Optional[str] = None,
    time_point: str = "now",
) -> Dict[str, Any]:
    """Prepare data for serving mode.

    Parameters
    ----------
    cfg : DictConfig
        Configuration.
    input_path : str, optional
        Path to input CSV.
    time_point : str
        Timestamp for Feast.

    Returns
    -------
    Dict[str, Any]
        Context dict for pipeline.
    """
    df = load_csv_data(input_path) if input_path else load_from_feast(cfg, time_point)

    return {
        "df": df.copy(),
        "train_df": pd.DataFrame(),
        "val_df": pd.DataFrame(),
        "test_df": df.copy(),
        "is_splited_input": False,
    }


def _print_separator(title: str) -> None:
    """Print formatted separator with title."""
    print(f"\n{'=' * 60}")
    print(f"[RUN] {title}")
    print(f"{'=' * 60}\n")


def _print_metrics(context: Dict[str, Any], key: str = "evaluate_metrics") -> None:
    """Print metrics from context if available."""
    if key not in context:
        return
    print(f"[RUN] Metrics ({key}):")
    for metric, value in context[key].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.4f}")


def run_training(
    experiment_path: str,
    pipeline_path: str,
) -> None:
    """Run training pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    """
    print("[RUN] mode='train'")

    merged_cfg = ConfigMerger.merge(experiment_path, pipeline_path, mode="train")
    temp_config = f".temp_merged_train_{uuid.uuid4().hex}.yaml"
    ConfigMerger.save(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Training COMPLETE")
        _print_metrics(context)
    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_eval(
    experiment_path: str,
    pipeline_path: str,
    alias: str = "latest",
) -> None:
    """Run evaluation pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    alias : str
        Model alias in MLflow Registry.
    """
    _print_separator("Starting EVALUATION pipeline")
    print(f"[RUN] Model alias: {alias}")

    merged_cfg = ConfigMerger.merge(experiment_path, pipeline_path, mode="eval")

    for step in merged_cfg.pipeline.steps:
        step_type = step.get("type", "")
        if step_type == "mlflow_loader":
            step.alias = alias
            print(f"[CONFIG] Override: mlflow_loader alias='{alias}'")
        elif step_type == "preprocessor" and not step.get("is_train", True):
            step.alias = alias
            print(f"[CONFIG] Override: preprocessor alias='{alias}'")

    temp_config = f".temp_merged_eval_{uuid.uuid4().hex}.yaml"
    ConfigMerger.save(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Evaluation COMPLETE")
        _print_metrics(context)
    finally:
        Path(temp_config).unlink(missing_ok=True)


def _check_initial_context_support(pipeline: FlexiblePipeline) -> bool:
    """Check if pipeline supports initial_context parameter.

    Parameters
    ----------
    pipeline : FlexiblePipeline
        Pipeline instance.

    Returns
    -------
    bool
        True if supports initial_context.

    Raises
    ------
    AttributeError
        If pipeline missing run_exp method.
    """
    if not hasattr(pipeline, "run_exp"):
        raise AttributeError("Pipeline missing run_exp() method")
    sig = inspect.signature(pipeline.run_exp)
    return "initial_context" in sig.parameters


def _run_pipeline_with_context(
    pipeline: FlexiblePipeline,
    initial_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pipeline with initial context.

    Parameters
    ----------
    pipeline : FlexiblePipeline
        Pipeline instance.
    initial_context : Dict[str, Any]
        Pre-initialized context.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution context.

    Raises
    ------
    NotImplementedError
        If pipeline doesn't support context injection.
    """
    if _check_initial_context_support(pipeline):
        return pipeline.run_exp(initial_context=initial_context)

    if hasattr(pipeline, "executor") and hasattr(pipeline.executor, "context"):
        pipeline.executor.context = initial_context
        return pipeline.run_exp()

    raise NotImplementedError(
        "FlexiblePipeline.run_exp() does not support initial_context"
    )


def _preview_predictions(
    predictions: Any,
    main_key: str,
    pred_keys: list[str],
) -> None:
    """Preview model predictions.

    Parameters
    ----------
    predictions : Any
        Prediction data.
    main_key : str
        Primary prediction key.
    pred_keys : list[str]
        All available prediction keys.
    """
    print(f"[RUN] Generated {len(predictions)} predictions [Key: {main_key}]")
    if len(pred_keys) > 1:
        print(f"[RUN] Other available predictions: {pred_keys[1:]}")

    if hasattr(predictions, "shape"):
        print(f"[RUN] Prediction shape: {predictions.shape}")

    # Prediction preview
    try:
        data = np.asarray(predictions)

        # Simple preview: show count and first/last few samples
        print(f"[RUN] Total predictions: {len(data)} [Key: {main_key}]")
        if len(pred_keys) > 1:
            print(f"[RUN] Other available predictions: {pred_keys[1:]}")

        if hasattr(predictions, "shape"):
            print(f"[RUN] Prediction shape: {predictions.shape}")

        preview_len = min(5, len(data))
        if preview_len > 0:
            print(f"[RUN] First {preview_len} values:")
            print(f"{data[:preview_len]}")

            if len(data) > preview_len:
                print("[RUN] Last 5 values:")
                print(f"{data[-5:]}")
        else:
            print("[RUN] WARNING: Prediction array is empty!")

    except Exception as e:
        print(f"[DEBUG] Preview logic failed: {e}")
        preview_len = min(10, len(predictions))
        print(f"[RUN] First {preview_len}: {predictions[:preview_len]}")


def run_serve(
    experiment_path: str,
    pipeline_path: str,
    input_path: Optional[str] = None,
    alias: str = "latest",
    time_point: str = "now",
) -> Any:
    """Run serve pipeline with runtime data injection.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    input_path : str, optional
        Path to input CSV file.
    alias : str
        Model alias in MLflow Registry.
    time_point : str
        Timestamp for Feast.

    Returns
    -------
    Any
        Model predictions.

    Raises
    ------
    RuntimeError
        If no predictions found.
    """
    _print_separator("Starting SERVING pipeline")
    print(f"[RUN] Model alias: {alias}")

    if input_path:
        print(f"[RUN] Data source: CSV ({input_path})")
    else:
        print("[RUN] Data source: Feast Feature Store")
        print(f"[RUN] Time point: {time_point}")

    merged_cfg = ConfigMerger.merge(experiment_path, pipeline_path, mode="serve")

    if alias != "latest":
        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "mlflow_loader":
                step.alias = alias

    temp_config = ".temp_merged_serve.yaml"
    ConfigMerger.save(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        initial_context = prepare_serving_data(
            cfg=merged_cfg,
            input_path=input_path,
            time_point=time_point,
        )

        print(f"[SERVE] Pre-initialized context: {len(initial_context)} keys")
        context = _run_pipeline_with_context(pipeline, initial_context)

        _print_separator("Serving COMPLETE")

        pred_keys = [k for k in context if "_predictions" in k]
        if not pred_keys:
            raise RuntimeError("No predictions found in pipeline context")

        # Use the first key for preview, but inform the user if others exist
        main_key = pred_keys[0]
        predictions = context[main_key]

        _preview_predictions(predictions, main_key, pred_keys)

        return predictions

    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_tune(
    experiment_path: str,
    pipeline_path: str,
    n_trials: Optional[int] = None,
) -> None:
    """Run hyperparameter tuning pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    n_trials : int, optional
        Number of trials to run.
    """
    _print_separator("Starting HYPERPARAMETER TUNING pipeline")
    if n_trials:
        print(f"[RUN] Trials override: {n_trials}")

    merged_cfg = ConfigMerger.merge(experiment_path, pipeline_path, mode="tune")

    if n_trials:
        if "tuning" not in merged_cfg:
            merged_cfg.tuning = {}
        merged_cfg.tuning.n_trials = n_trials

        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "tuner":
                step.n_trials = n_trials
                print(f"[CONFIG] Override: n_trials={n_trials}")
                break

    temp_config = ".temp_merged_tune.yaml"
    ConfigMerger.save(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Tuning COMPLETE")

        for key in context:
            if "_best_params" in key:
                print(f"[RUN] Best parameters ({key}):")
                for param, value in context[key].items():
                    print(f"  - {param}: {value}")

        for key in context:
            if "_study" in key:
                study = context[key]
                print(f"[RUN] Best metric value: {study.best_value:.4f}")
                break

    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_generate_configs(
    train_config: str,
    output_dir: str = "mlproject/configs/generated",
    alias: str = "latest",
    config_type: str = "all",
    experiment_config: Optional[str] = None,
) -> None:
    """Generate eval/serve configs from training config.

    Parameters
    ----------
    train_config : str
        Path to training experiment YAML.
    output_dir : str
        Output directory for generated configs.
    alias : str
        MLflow model alias.
    config_type : str
        Type of config to generate (eval/serve/all).
    experiment_config : str, optional
        Path to experiment config to infer data type.
    """
    _print_separator("GENERATING CONFIGS")
    print(f"[RUN] Source: {train_config}")
    if experiment_config:
        print(f"[RUN] Experiment info: {experiment_config}")
    print(f"[RUN] Output: {output_dir}")

    generator = ConfigGenerator(train_config, experiment_config_path=experiment_config)
    base_name = Path(train_config).stem

    if config_type == "all":
        paths = generator.generate_all(output_dir, alias, include_tune=True)
        print("\nGenerated configs:")
        print(f"  - Eval:  {paths['eval']}")
        print(f"  - Serve: {paths['serve']}")
        print(f"  - Tune: {paths['tune']}")
    elif config_type == "eval":
        out_path = str(Path(output_dir) / f"{base_name}_eval.yaml")
        generator.generate_eval_config(alias=alias, output_path=out_path)
        print(f"Generated: {out_path}")
    else:
        out_path = str(Path(output_dir) / f"{base_name}_serve.yaml")
        generator.generate_serve_config(alias=alias, output_path=out_path)
        print(f"Generated: {out_path}")


def _setup_train_parser(subparsers: Any) -> None:
    """Setup train subcommand parser."""
    parser = subparsers.add_parser("train", help="Run training pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")


def _setup_eval_parser(subparsers: Any) -> None:
    """Setup eval subcommand parser."""
    parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--alias", "-a", default="latest", help="Model alias")


def _setup_serve_parser(subparsers: Any) -> None:
    """Setup serve subcommand parser."""
    parser = subparsers.add_parser("serve", help="Run serve pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--input", "-i", help="Input CSV file")
    parser.add_argument("--alias", "-a", default="latest", help="Model alias")
    parser.add_argument("--time_point", "-t", default="now", help="Timestamp for Feast")


def _setup_tune_parser(subparsers: Any) -> None:
    """Setup tune subcommand parser."""
    parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--trials", "-n", type=int, help="Number of tuning trials")


def _setup_generate_parser(subparsers: Any) -> None:
    """Setup generate subcommand parser."""
    parser = subparsers.add_parser("generate", help="Generate eval/serve configs")
    parser.add_argument(
        "--train-config", "-t", required=True, help="Training config YAML"
    )
    parser.add_argument(
        "--experiment", "-e", help="Experiment config to infer data type"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="mlproject/configs/generated",
        help="Output directory",
    )
    parser.add_argument("--alias", "-a", default="latest", help="MLflow model alias")
    parser.add_argument(
        "--type",
        choices=["eval", "serve", "all"],
        default="all",
        help="Type of config to generate",
    )


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run ML pipelines (train/eval/serve/tune/generate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")

    _setup_train_parser(subparsers)
    _setup_eval_parser(subparsers)
    _setup_serve_parser(subparsers)
    _setup_tune_parser(subparsers)
    _setup_generate_parser(subparsers)

    args = parser.parse_args()

    try:
        if args.command == "train":
            run_training(args.experiment, args.pipeline)
        elif args.command == "eval":
            run_eval(args.experiment, args.pipeline, args.alias)
        elif args.command == "serve":
            run_serve(
                args.experiment,
                args.pipeline,
                args.input,
                args.alias,
                args.time_point,
            )
        elif args.command == "tune":
            run_tune(args.experiment, args.pipeline, args.trials)
        elif args.command == "generate":
            run_generate_configs(
                args.train_config,
                args.output_dir,
                args.alias,
                args.type,
                experiment_config=args.experiment,
            )
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
