"""Hyperparameter tuning step using Optuna (BEST PRACTICE VERSION).

This implementation follows the clean separation pattern:
1. Tuning phase: Find best params in nested MLflow runs
2. Retrain phase: Train final model in separate run (handled by pipeline)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import DefaultValues
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.pipeline.steps.core.utils import ConfigAccessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna import OptunaTuner


class TunerStep(BasePipelineStep):
    """Run hyperparameter optimization using Optuna.

    This step performs cross-validation based hyperparameter search
    in a nested MLflow parent run, then stores best params for
    downstream model training.

    Key Design:
    -----------
    - Tuning trials are NESTED under parent run
    - Does NOT train final model (separation of concerns)
    - Best params stored in context for downstream steps
    - Final model training handled by separate ModelTrainingStep
    """

    def __init__(
        self,
        *args,
        n_trials: Optional[int] = None,
        target_model_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize tuning step.

        Parameters
        ----------
        n_trials : int, optional
            Override number of trials from config.
        target_model_id : str, optional
            ID of the specific model step to tune.
            If provided, will extract model info from that step.
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.n_trials_override = n_trials
        self.target_model_id = target_model_id

    def _find_step_recursive(
        self,
        steps: Any,
        target_id: str,
    ) -> Optional[Any]:
        """Recursively find step by ID in pipeline tree.

        Parameters
        ----------
        steps : list
            List of steps to search.
        target_id : str
            Step ID to find.

        Returns
        -------
        Optional[Any]
            Found step or None.
        """
        for step in steps:
            if step.get("id") == target_id:
                return step

            found = self._search_in_step_children(step, target_id)
            if found is not None:
                return found

        return None

    def _search_in_step_children(
        self,
        step: Any,
        target_id: str,
    ) -> Optional[Any]:
        """Search target step inside nested structures of a step."""
        step_type = step.get("type")

        if step_type == "parallel":
            return self._search_in_parallel(step, target_id)

        if step_type == "branch":
            return self._search_in_branch(step, target_id)

        if step_type == "sub_pipeline":
            return self._search_in_sub_pipeline(step, target_id)

        return None

    def _search_in_parallel(
        self,
        step: Any,
        target_id: str,
    ) -> Optional[Any]:
        """Search inside parallel branches."""
        if "branches" not in step:
            return None

        return self._find_step_recursive(step.branches, target_id)

    def _search_in_branch(
        self,
        step: Any,
        target_id: str,
    ) -> Optional[Any]:
        """Search inside branch conditions."""
        if "if_true" in step:
            found = self._find_step_recursive([step.if_true], target_id)
            if found is not None:
                return found

        if "if_false" in step:
            return self._find_step_recursive([step.if_false], target_id)

        return None

    def _search_in_sub_pipeline(
        self,
        step: Any,
        target_id: str,
    ) -> Optional[Any]:
        """Search inside sub-pipeline steps."""
        if "pipeline" not in step:
            return None

        pipeline = step.pipeline
        if "steps" not in pipeline:
            return None

        return self._find_step_recursive(pipeline.steps, target_id)

    def _infer_model_from_id(self, step_id: str) -> Optional[str]:
        """Infer model name from step ID.

        Parameters
        ----------
        step_id : str
            Step ID to analyze.

        Returns
        -------
        Optional[str]
            Inferred model name or None.
        """
        step_id_lower = step_id.lower()

        rules = (
            ("xgboost", "xgboost"),
            ("catboost", "catboost"),
            ("nlinear", "nlinear"),
            ("tft", "tft"),
            ("kmeans", "kmeans"),
            ("kmean", "kmeans"),
            ("cluster", "kmeans"),
        )

        for keyword, model_name in rules:
            if keyword in step_id_lower:
                return model_name

        return None

    def _extract_model_info(self) -> Dict[str, str]:
        """Extract model type and name from target step or experiment config.

        Returns
        -------
        Dict[str, str]
            Dictionary with 'model_name' and 'model_type' keys.
        """
        if not self.target_model_id or not hasattr(self.cfg, "pipeline"):
            return self._extract_from_experiment()

        step = self._find_step_recursive(
            self.cfg.pipeline.steps,
            self.target_model_id,
        )
        if not step:
            return self._extract_from_experiment()

        step_type = step.get("type")

        if step_type in {"trainer", "framework_model"}:
            info = self._extract_from_trainer_step(step)
            if info is not None:
                return info

        if step_type == "dynamic_adapter":
            info = self._extract_from_dynamic_adapter(step)
            if info is not None:
                return info

        if step_type == "clustering":
            return {"model_name": "kmeans", "model_type": "ml"}

        return self._extract_from_experiment()

    def _extract_from_trainer_step(self, step: Any) -> Optional[Dict[str, str]]:
        """Extract model info from trainer-like step."""
        if self.target_model_id is None:
            raise ValueError(f"target_model_id not {self.target_model_id}")
        model_name = self._infer_model_from_id(self.target_model_id)
        if model_name:
            return {
                "model_name": model_name,
                "model_type": self._infer_model_type(model_name),
            }

        if "experiment_config" in step:
            return self._extract_from_experiment_config(step.experiment_config)

        return self._extract_from_experiment()

    def _extract_from_dynamic_adapter(
        self,
        step: Any,
    ) -> Optional[Dict[str, str]]:
        """Extract model info from dynamic adapter step."""
        class_path = step.get("class_path", "").lower()

        if "kmeans" in class_path:
            return {"model_name": "kmeans", "model_type": "ml"}

        if "cluster" in class_path:
            return {"model_name": "clustering", "model_type": "ml"}

        return None

    def _extract_from_experiment_config(
        self,
        exp_config: str,
    ) -> Optional[Dict[str, str]]:
        """Infer model info from experiment_config path."""
        exp = exp_config.lower()

        if "tft" in exp:
            return {"model_name": "tft", "model_type": "dl"}

        if "nlinear" in exp:
            return {"model_name": "nlinear", "model_type": "dl"}

        if "xgboost" in exp or "xgb" in exp:
            return {"model_name": "xgboost", "model_type": "ml"}

        if "catboost" in exp:
            return {"model_name": "catboost", "model_type": "ml"}

        return None

    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name."""
        if model_name in {"tft", "nlinear"}:
            return "dl"

        return "ml"

    def _extract_from_experiment(self) -> Dict[str, str]:
        """Fallback extraction from experiment config."""
        return {
            "model_name": self.cfg.experiment.model.lower(),
            "model_type": self.cfg.experiment.model_type.lower(),
        }

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with tuning results added.
        """
        self.validate_dependencies(context)
        self._print_tuning_header()

        splitter = self._build_splitter()
        mlflow_manager = MLflowManager(self.cfg)
        tuning_cfg = self._extract_tuning_config()

        model_info = self._extract_model_info()
        model_name = model_info["model_name"]
        model_type = model_info["model_type"]

        self._print_tuning_config(
            tuning_cfg=tuning_cfg,
            model_name=model_name,
            model_type=model_type,
        )

        result = self._run_tuning(
            splitter=splitter,
            mlflow_manager=mlflow_manager,
            tuning_cfg=tuning_cfg,
            model_name=model_name,
            model_type=model_type,
        )

        self._store_tuning_result(context, result)
        self._print_tuning_summary(tuning_cfg["metric_name"], result)

        return context

    def _run_tuning(
        self,
        splitter: BaseSplitter,
        mlflow_manager: MLflowManager,
        tuning_cfg: Dict[str, Any],
        model_name: str,
        model_type: str,
    ) -> Dict[str, Any]:
        """Run Optuna tuning inside a parent MLflow run."""
        run_name = DefaultValues.TUNING_RUN_NAME
        with mlflow_manager.start_run(
            run_name=run_name,
        ):
            print(f"\n[MLflow] Started parent run: {run_name}")

            tuner = OptunaTuner(
                cfg=self.cfg,
                splitter=splitter,
                mlflow_manager=mlflow_manager,
                metric_name=tuning_cfg["metric_name"],
                direction=tuning_cfg["direction"],
                model_name=model_name,
                model_type=model_type,
            )

            print(f"\n[{self.step_id}] Running optimization...")
            result = tuner.tune(
                n_trials=tuning_cfg["n_trials"],
                timeout=tuning_cfg["timeout"],
            )

            mlflow_manager.log_metadata(params=result["best_params"])
            print("\n[MLflow] Logged best params to parent run")

            return result

    def _print_tuning_header(self) -> None:
        """Print tuning header."""
        print(f"\n{'=' * 60}")
        print(f"[{self.step_id}] HYPERPARAMETER TUNING")
        print(f"{'=' * 60}\n")

    def _build_splitter(self) -> BaseSplitter:
        """Build cross-validation splitter."""
        config_accessor = ConfigAccessor(self.cfg)
        n_splits = config_accessor.get_n_splits()

        if config_accessor.is_timeseries():
            return TimeSeriesFoldSplitter(self.cfg, n_splits=n_splits)

        return BaseSplitter(self.cfg, n_splits=n_splits)

    def _extract_tuning_config(self) -> Dict[str, Any]:
        """Extract tuning-related configuration."""
        tuning_cfg = self.cfg.get("tuning", {})

        return {
            "metric_name": tuning_cfg.get("optimize_metric", "mae_mean"),
            "direction": tuning_cfg.get("direction", "minimize"),
            "n_trials": self.n_trials_override or tuning_cfg.get("n_trials", 50),
            "timeout": tuning_cfg.get("timeout"),
            "n_splits": tuning_cfg.get("n_splits", DefaultValues.N_SPLITS),
        }

    def _print_tuning_config(
        self,
        tuning_cfg: Dict[str, Any],
        model_name: str,
        model_type: str,
    ) -> None:
        """Print tuning configuration."""
        print(f"[{self.step_id}] Configuration:")
        print(f"  - Trials: {tuning_cfg['n_trials']}")
        print(
            f"  - Metric: {tuning_cfg['metric_name']}" f" ({tuning_cfg['direction']})"
        )
        print(f"  - CV folds: {tuning_cfg['n_splits']}")

        if self.target_model_id:
            print(
                f"  - Target model: {self.target_model_id} "
                f"({model_name}/{model_type})"
            )

    def _store_tuning_result(
        self,
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Store tuning results into pipeline context."""
        context[f"{self.step_id}_best_params"] = result["best_params"]
        context[f"{self.step_id}_best_value"] = result["best_value"]
        context[f"{self.step_id}_study"] = result["study"]

    def _print_tuning_summary(
        self,
        metric_name: str,
        result: Dict[str, Any],
    ) -> None:
        """Print tuning summary."""
        print(f"\n{'=' * 60}")
        print(f"[{self.step_id}] TUNING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Best {metric_name}: {result['best_value']:.6f}")
        print("\n  Best parameters:")
        for param, value in result["best_params"].items():
            print(f"    - {param}: {value}")
        print(f"\n{'=' * 60}\n")


# Register step type
StepFactory.register("tuner", TunerStep)
