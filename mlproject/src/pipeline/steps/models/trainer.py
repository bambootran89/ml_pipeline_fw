"""
Trainer pipeline step for fitting a model wrapper using internal framework
factories and a prepared DataModule. Supports artifact discovery registration
(e.g., MLflow Model Registry) without re-fitting restored components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import ContextKeys
from mlproject.src.pipeline.steps.core.factory import StepFactory
from mlproject.src.trainer.factory import TrainerFactory


class TrainerStep(BasePipelineStep):
    """
    Fit a model wrapper using a prepared DataModule from pipeline context.

    This step performs:
    1. Retrieval of a ready DataModule instance from `context["datamodule"]`.
    2. Dynamic instantiation of model and trainer via framework factories.
    3. Model fitting via `trainer.fit()` using DataModule test/train windows.
    4. Storage of the trained model wrapper back into pipeline context.
    5. Optional registration into `_artifact_registry` for discovery or
       tracking (e.g., MLflow Registry alias binding).

    Expected Context Inputs
    ----------------------
    datamodule : Any
        A prepared DataModule instance containing model-ready data.

    Context Outputs
    ---------------
    model : Any
        Trained model wrapper instance stored via wiring or default key.
    _artifact_registry : Dict[str, Dict[str, Any]], optional
        Discovery registry populated when `log_artifact=True`.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        use_tuned_params: bool = False,
        tune_step_id: Optional[str] = None,
        output_as_feature: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TrainerStep.

        Parameters
        ----------
        step_id : str
            Unique pipeline step identifier.
        cfg : DictConfig or compatible config
            Configuration containing model/training settings.
        enabled : bool, default=True
            Whether the step should execute.
        depends_on : Optional[List[str]], default=None
            Prerequisite step IDs.
        use_tuned_params : bool, default=False
            Whether to use best params from TuningStep.
        tune_step_id : str, optional
            ID of TuningStep to read best_params from.
        output_as_feature : bool, default=False
        **kwargs : Any
            Additional parameters passed to the BasePipelineStep.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.output_as_feature = output_as_feature
        self.use_tuned_params = use_tuned_params
        self.tune_step_id = tune_step_id

    def _get_hyperparams(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters (from config or tuning).

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters to use for training.
        """
        if self.use_tuned_params:
            if self.tune_step_id is None:
                raise ValueError(
                    f"Step '{self.step_id}': use_tuned_params=True requires "
                    f"tune_step_id parameter"
                )

            best_params_key = ContextKeys.step_best_params(self.tune_step_id)
            if best_params_key not in context:
                raise ValueError(
                    f"Step '{self.step_id}': Expected '{best_params_key}' "
                    f"in context. Make sure TuningStep runs before this step."
                )

            best_params = context[best_params_key]

            print(f"\n[{self.step_id}] Using tuned hyperparameters:")
            for param, value in best_params.items():
                print(f"  - {param}: {value}")

            cfg_copy = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            if "args" not in cfg_copy.experiment.hyperparams:
                cfg_copy.experiment.hyperparams.args = {}

            for param, value in best_params.items():
                cfg_copy.experiment.hyperparams.args[param] = value

            return dict(cfg_copy.experiment.hyperparams)
        else:
            params = dict(self.cfg.experiment.hyperparams)

        # Override n_features/input_size if composed features are present in context
        if ContextKeys.COMPOSED_FEATURE_NAMES in context:
            n_features = len(context[ContextKeys.COMPOSED_FEATURE_NAMES])
            if "n_features" in params:
                print(
                    f"[{self.step_id}] Override n_features: "
                    f"{params['n_features']} -> {n_features}"
                )
                params["n_features"] = n_features

            # Also update input_size if it exists (common in RNNs)
            if "input_size" in params:
                print(
                    f"[{self.step_id}] Override input_size: "
                    f"{params['input_size']} -> {n_features}"
                )
                params["input_size"] = n_features

        return params

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit model using trainer and DataModule from pipeline context.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline execution context.

        Returns
        -------
        Dict[str, Any]
            Updated context with trained model wrapper and registry entries.

        Raises
        ------
        ValueError
            If the DataModule is missing in context.
        RuntimeError
            If model fitting fails internally.
        """
        self.validate_dependencies(context)

        datamodule = self._get_datamodule(context)

        # Patch config if composed features are present in context
        # This ensures ModelFactory builds the model with correct input dimensions
        if ContextKeys.COMPOSED_FEATURE_NAMES in context:
            n_features = len(context[ContextKeys.COMPOSED_FEATURE_NAMES])
            print(f"[{self.step_id}] Patching config: n_features -> {n_features}")

            # Update n_features
            if (
                OmegaConf.select(self.cfg, "experiment.hyperparams.n_features")
                is not None
            ):
                OmegaConf.update(
                    self.cfg, "experiment.hyperparams.n_features", n_features
                )

            # Update input_size
            if (
                OmegaConf.select(self.cfg, "experiment.hyperparams.input_size")
                is not None
            ):
                OmegaConf.update(
                    self.cfg, "experiment.hyperparams.input_size", n_features
                )

            # Ensure 'args' reflects this too if present (for some models)
            if (
                OmegaConf.select(self.cfg, "experiment.hyperparams.args.input_size")
                is not None
            ):
                OmegaConf.update(
                    self.cfg, "experiment.hyperparams.args.input_size", n_features
                )

        model_class = self._build_model()
        trainer = self._build_trainer(model_class)

        trained_wrapper = self._fit_model(trainer, datamodule, context)
        self.set_output(context, "model", trained_wrapper)
        if self.log_artifact:
            self.register_for_discovery(context, trained_wrapper)
        if self.output_as_feature:
            x_train = (
                datamodule.get_data()[0]
                if hasattr(datamodule, "get_data")
                else datamodule.get_test_windows()[0]
            )
            preds = trained_wrapper.predict(x_train)
            self.set_output(context, "features", preds)
            print(f"[{self.step_id}] Generated downstream features: {preds.shape}")

        print(f"[{self.step_id}] Model trained and registered successfully.")
        return context

    # Internal helper methods

    def _get_datamodule(self, context: Dict[str, Any]) -> Any:
        """Retrieve the required DataModule instance from pipeline context."""
        dm = self.get_input(context, "datamodule", required=False)
        if dm is None:
            raise ValueError(
                f"Step '{self.step_id}': context['datamodule'] is missing."
            )
        return dm

    def _build_model(
        self,
    ) -> Any:
        """Instantiate the model wrapper using ModelFactory."""
        model_name: str = self.cfg.experiment.model.lower()
        return ModelFactory.create(model_name, self.cfg)

    def _build_trainer(self, model: Any) -> Any:
        """Instantiate the trainer using TrainerFactory."""
        model_name: str = self.cfg.experiment.model.lower()
        model_type: str = self.cfg.experiment.model_type.lower()
        return TrainerFactory.create(
            model_type=model_type,
            model_name=model_name,
            wrapper=model,
            save_dir=self.cfg.training.artifacts_dir,
        )

    def _fit_model(self, trainer: Any, datamodule: Any, context: Dict[str, Any]) -> Any:
        """Perform model fitting and return the trained wrapper."""
        print(f"[{self.step_id}] Starting model training")
        try:
            trained_wrapper = trainer.train(datamodule, self._get_hyperparams(context))
            return trained_wrapper
        except Exception as exc:
            raise RuntimeError(f"Step '{self.step_id}': model fitting failed.") from exc


# Register step type for pipeline factory
StepFactory.register("trainer", TrainerStep)
