"""Enhanced MLflow tracking manager with config logging.

This module extends MLflow tracking to include:
- Full experiment config YAML logging
- Auto-generated eval/serve config artifacts
- Parallel experiment support with unique run names
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from mlproject.src.tracking.pyfunc_wrapper import ArtifactPyFuncWrapper


class MLflowManager:
    """Enhanced MLflow tracking manager.

    This class provides:
    - Centralized MLflow configuration
    - Config YAML artifact logging
    - Auto-generated eval/serve config logging
    - Thread-safe parallel experiment support

    Attributes
    ----------
    cfg : DictConfig
        Full experiment configuration.
    enabled : bool
        Whether MLflow tracking is enabled.
    tracking_uri : str
        MLflow tracking server URI.
    experiment_name : str
        Name of the MLflow experiment.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize MLflow manager.

        Parameters
        ----------
        cfg : DictConfig
            Full experiment configuration.
        """
        self.cfg = cfg
        mlflow_cfg = cfg.get("mlflow", {})

        self.enabled = mlflow_cfg.get("enabled", True)
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        self.experiment_name = mlflow_cfg.get(
            "experiment_name",
            cfg.get("experiment", {}).get("name", "default_experiment"),
        )
        self.run_name_prefix = mlflow_cfg.get("run_name_prefix", "exp")

        # Registry settings
        registry_cfg = mlflow_cfg.get("registry", {})
        self.registry_enabled = registry_cfg.get("enabled", True)
        self.model_name = registry_cfg.get("model_name", "model")

        # Artifact settings
        artifacts_cfg = mlflow_cfg.get("artifacts", {})
        self.log_model = artifacts_cfg.get("log_model", True)
        self.log_config = artifacts_cfg.get("log_config", True)

        self._client: Optional[MlflowClient] = None
        self._active_run: Optional[mlflow.ActiveRun] = None

        if self.enabled:
            self._setup()

    def _setup(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._client = MlflowClient(self.tracking_uri)

    def _generate_unique_run_name(self, base_name: str) -> str:
        """Generate unique run name for parallel execution.

        Parameters
        ----------
        base_name : str
            Base name for the run.

        Returns
        -------
        str
            Unique run name with timestamp and PID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        return f"{base_name}_{timestamp}_{pid}"

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
    ) -> Generator[Optional[mlflow.ActiveRun], None, None]:
        """Start an MLflow run context.

        Parameters
        ----------
        run_name : str, optional
            Name for the run. Auto-generated if None.
        nested : bool, default=False
            Whether this is a nested run.

        Yields
        ------
        Optional[mlflow.ActiveRun]
            Active run context or None if disabled.
        """
        if not self.enabled:
            yield None
            return

        if run_name is None:
            run_name = self._generate_unique_run_name(self.run_name_prefix)

        run = mlflow.start_run(run_name=run_name, nested=nested)
        self._active_run = run

        try:
            yield run
        finally:
            # Log configs before ending run
            if self.log_config:
                self._log_all_configs()
            mlflow.end_run()
            self._active_run = None

    def _log_all_configs(self) -> None:
        """Log all configuration YAMLs as artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log training config
            train_path = Path(tmpdir) / "config_train.yaml"
            self._save_config_yaml(self.cfg, train_path)
            mlflow.log_artifact(str(train_path), "configs")

            # Generate and log eval config
            eval_cfg = self._generate_eval_config()
            eval_path = Path(tmpdir) / "config_eval.yaml"
            self._save_config_yaml(eval_cfg, eval_path)
            mlflow.log_artifact(str(eval_path), "configs")

            # Generate and log serve config
            serve_cfg = self._generate_serve_config()
            serve_path = Path(tmpdir) / "config_serve.yaml"
            self._save_config_yaml(serve_cfg, serve_path)
            mlflow.log_artifact(str(serve_path), "configs")

            # Log config as JSON for easy viewing
            config_json = Path(tmpdir) / "config_summary.json"
            self._save_config_json(config_json)
            mlflow.log_artifact(str(config_json), "configs")

    def _generate_eval_config(self) -> DictConfig:
        """Generate eval config inline without file dependency.

        Returns
        -------
        DictConfig
            Evaluation configuration.
        """
        container = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(container, dict):
            container = {}

        # Remove training-specific keys
        for key in ["pipeline", "tuning"]:
            container.pop(key, None)

        eval_cfg = OmegaConf.create(container)

        # Add eval pipeline
        eval_cfg["pipeline"] = {
            "name": f"{self.experiment_name}_eval",
            "steps": [
                {"id": "load_data", "type": "data_loader", "enabled": True},
                {
                    "id": "preprocess",
                    "type": "preprocessor",
                    "enabled": True,
                    "depends_on": ["load_data"],
                    "is_train": False,
                    "alias": "latest",
                },
                {
                    "id": "load_model",
                    "type": "model_loader",
                    "enabled": True,
                    "depends_on": ["preprocess"],
                    "alias": "latest",
                },
                {
                    "id": "evaluate",
                    "type": "evaluator",
                    "enabled": True,
                    "depends_on": ["load_model", "preprocess"],
                    "model_step_id": "load_model",
                },
            ],
        }

        return eval_cfg

    def _generate_serve_config(self) -> DictConfig:
        """Generate serve config inline.

        Returns
        -------
        DictConfig
            Serving configuration.
        """
        container = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(container, dict):
            container = {}

        for key in ["pipeline", "tuning"]:
            container.pop(key, None)

        serve_cfg = OmegaConf.create(container)

        serve_cfg["pipeline"] = {
            "name": f"{self.experiment_name}_serve",
            "steps": [
                {
                    "id": "preprocess",
                    "type": "preprocessor",
                    "enabled": True,
                    "is_train": False,
                    "alias": "latest",
                },
                {
                    "id": "load_model",
                    "type": "model_loader",
                    "enabled": True,
                    "depends_on": ["preprocess"],
                    "alias": "latest",
                },
                {
                    "id": "inference",
                    "type": "inference",
                    "enabled": True,
                    "depends_on": ["preprocess", "load_model"],
                    "model_step_id": "load_model",
                },
            ],
        }

        return serve_cfg

    def _save_config_yaml(self, cfg: DictConfig, path: Path) -> None:
        """Save config to YAML file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration to save.
        path : Path
            Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(cfg, f)

    def _save_config_json(self, path: Path) -> None:
        """Save config summary as JSON.

        Parameters
        ----------
        path : Path
            Output file path.
        """
        container = OmegaConf.to_container(self.cfg, resolve=True)
        if not isinstance(container, dict):
            container = {}

        summary = {
            "experiment": container.get("experiment", {}),
            "data": container.get("data", {}),
            "preprocessing": container.get("preprocessing", {}),
            "training": container.get("training", {}),
            "evaluation": container.get("evaluation", {}),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def log_metadata(
        self,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log parameters, metrics, and tags.

        Parameters
        ----------
        params : Dict[str, Any], optional
            Parameters to log.
        metrics : Dict[str, float], optional
            Metrics to log.
        tags : Dict[str, str], optional
            Tags to log.
        """
        if not self.enabled:
            return

        if params:
            flat_params = self._flatten_dict(params, max_depth=2)
            mlflow.log_params(flat_params)

        if metrics:
            safe_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            }
            mlflow.log_metrics(safe_metrics)

        if tags:
            mlflow.set_tags(tags)

    def log_component(
        self,
        obj: Any,
        name: str,
        artifact_type: str = "model",
    ) -> Optional[str]:
        """Persist and track a Python component in
            MLflow with PyFunc + registry alias 'latest'.

        Args:
            obj: Component exposing `predict()` or `transform()`.
            name: Unique artifact/registered model name.
            artifact_type: Category for MLflow artifacts.

        Returns:
            model_uri if registered, else None
        """
        if not self.enabled:
            return None

        run = self._active_run or mlflow.active_run()
        if run is None:
            print("[MLflowManager] No active run. Skipping component log.")
            return None

        # Determine logging method for PyFunc
        method_name = "transform" if artifact_type == "preprocess" else "predict"
        if not callable(getattr(obj, method_name, None)):
            raise AttributeError(
                f"Component must implement a callable\
                    `{method_name}()` to be logged via PyFunc."
            )

        # Wrap via PyFunc
        wrapped = ArtifactPyFuncWrapper(obj, predict_method=method_name)

        # Log model artifact
        try:
            mlflow.pyfunc.log_model(
                artifact_path=name,
                python_model=wrapped,
                registered_model_name=name,
            )
        except Exception as exc:
            print(f"[MLflowManager] Warning: Failed to log component '{name}' → {exc}")
            return None

        # Register + set alias latest
        client = self._client or MlflowClient(self.tracking_uri)
        try:
            mv = client.get_model_version_by_alias(name, "latest")
            _ = mv.version
        except Exception:
            try:
                reg = mlflow.register_model(f"runs:/{run.info.run_id}/{name}", name)
                _ = reg.version
            except Exception as reg_exc:
                print(
                    f"[MLflowManager] Warning: Registry registration \
                      failed for '{name}' → {reg_exc}"
                )
                return f"runs:/{run.info.run_id}/{name}"

        return f"models:/{name}@latest"

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow params.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary to flatten.
        parent_key : str, default=""
            Parent key prefix.
        sep : str, default="."
            Separator for nested keys.
        max_depth : int, default=3
            Maximum nesting depth.
        current_depth : int, default=0
            Current nesting level.

        Returns
        -------
        Dict[str, Any]
            Flattened dictionary.
        """
        items: list = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict) and current_depth < max_depth:
                items.extend(
                    self._flatten_dict(
                        v, new_key, sep, max_depth, current_depth + 1
                    ).items()
                )
            else:
                # Truncate long values
                str_v = str(v)
                if len(str_v) > 250:
                    str_v = str_v[:247] + "..."
                items.append((new_key, str_v))

        return dict(items)

    def _fallback_latest(self, name: str, alias: str) -> Optional[Any]:
        """
        Fallback to loading the latest artifact from the most recent run.

        Args:
            name: Model name or artifact path.
            alias: Registry alias attempted before fallback.

        Returns:
            Loaded PyFunc model or None if fallback fails.
        """
        _ = alias
        try:
            runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
            if runs is None or (hasattr(runs, "empty") and runs.empty):  # type: ignore
                print(f"[MLflowManager] No runs found. Cannot fallback for '{name}'.")
                return None

            last_run = runs.iloc[0]  # type: ignore
            # Access run_id from pandas Series, not from .info attribute
            run_id = last_run["run_id"]

            print(f"[MLflowManager] Latest run resolved: run_id='{run_id}'.")
            artifact_uri = f"runs:/{run_id}/{name}"

            print(
                f"[MLflowManager] Loading latest artifact for '{name}' "
                f"from '{artifact_uri}'."
            )

            loaded = mlflow.pyfunc.load_model(artifact_uri)
            print(f"[MLflowManager] Successfully loaded '{name}' from latest run.")
            return loaded

        except Exception as fb_exc:
            print(f"[MLflowManager] Latest fallback also failed for '{name}'.")
            print(f"[MLflowManager] Fallback error: {fb_exc}")
            return None

    def _unwrap(self, loaded: Any, name: str) -> Optional[Any]:
        """
        Unwrap the raw Python artifact from a PyFunc model if wrapped.

        Args:
            loaded: Loaded PyFunc model.
            name: Artifact/model name used during logging.

        Returns:
            The original unwrapped artifact or None if unwrap is not possible.
        """
        model_impl = getattr(loaded, "_model_impl", None)
        if model_impl is None:
            print(f"[MLflowManager] No _model_impl in '{name}'. Cannot unwrap.")
            return None

        python_model = getattr(model_impl, "python_model", None)
        print(
            f"[MLflowManager] PyFunc wrapper type detected: "
            f"'{type(python_model).__name__ if python_model else None}'."
        )

        if isinstance(python_model, ArtifactPyFuncWrapper):
            print(f"[MLflowManager] Unwrapping raw artifact from '{name}'.")
            return python_model.get_raw_artifact()

        print(f"[MLflowManager] '{name}' is not wrapped. Returning None.")
        return None

    def load_component(self, name: str, alias: str = "latest") -> Optional[Any]:
        """
        Load a PyFunc model using registry alias, else fallback to latest run.

        Fallback strategy:
        1. Try specified alias (e.g., "production")
        2. If alias == "latest", resolve latest version dynamically
        3. If failed, fallback to latest run artifact

        Args:
            name: Registry model name.
            alias: Registry alias (default: latest).

        Returns:
            The unwrapped raw artifact or None if loading or unwrapping fails.
        """
        if not self.enabled:
            print(f"[MLflowManager] MLflow disabled. Skipping load for '{name}'.")
            return None

        alias = alias.lower()

        # Special handling for 'latest' since it is a reserved keyword in some
        # MLflow versions and cannot be set as an alias.
        if alias == "latest":
            client = self._client or MlflowClient(self.tracking_uri)
            try:
                # Get latest version (any stage)
                latest_versions = client.get_latest_versions(name, stages=None)
                if latest_versions:
                    # Sort by version just in case, though get_latest_versions usually
                    # returns latest
                    latest_version = sorted(
                        latest_versions, key=lambda x: int(x.version), reverse=True
                    )[0]
                    version = latest_version.version
                    model_uri = f"models:/{name}/{version}"
                    print(
                        f"[MLflowManager] Resolved 'latest' for '{name}' to "
                        f"version {version}."
                    )

                    loaded = mlflow.pyfunc.load_model(model_uri)
                    print(f"[MLflowManager] Loaded '{name}' (v{version}).")
                    return self._unwrap(loaded, name)
                else:
                    print(
                        f"[MLflowManager] No versions found for '{name}' in registry."
                    )
            except Exception as exc:
                print(
                    f"[MLflowManager] Failed to resolve 'latest' version for '{name}'."
                )
                print(f"[MLflowManager] Error: {exc}")

        else:
            # Try with specified alias
            model_uri = f"models:/{name}@{alias}"
            print(f"[MLflowManager] Attempting to load '{name}' using alias '{alias}'.")
            try:
                loaded = mlflow.pyfunc.load_model(model_uri)
                print(f"[MLflowManager] Loaded '{name}' via alias '{alias}'.")
                return self._unwrap(loaded, name)
            except Exception as exc:
                print(f"[MLflowManager] Alias '{alias}' not found for '{name}'.")
                print(f"[MLflowManager] Root error: {exc}")

        # Final fallback: try to load from latest run
        print(f"[MLflowManager] Falling back to latest run for '{name}'...")
        loaded = self._fallback_latest(name, alias)
        if loaded is None:
            return None

        return self._unwrap(loaded, name)
