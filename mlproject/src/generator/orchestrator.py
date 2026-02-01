"""Pipeline configuration generator for eval/serve/tune modes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from .api_generator import ApiGenerator
from .config import GeneratorConfig
from .constants import STEP_CONSTANTS
from .pipeline.builders.eval import EvalBuilder
from .pipeline.builders.serve import ServeBuilder
from .pipeline.builders.tune import TuneBuilder
from .pipeline.feature_parser import FeaturePipelineParser
from .pipeline.step_analyzer import StepAnalyzer


class ConfigGenerator:
    """Generate pipeline configs for eval/serve/tune from training config.

    Loads a base training YAML configuration and generates transformed
    pipeline configs for evaluation or serving workloads. Does not modify
    training logic or saved MLflow artifacts.
    """

    def __init__(
        self,
        train_config_path: str,
        experiment_config_path: Optional[str] = None,
        generator_config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize config generator.

        Args:
            train_config_path: Path to training config YAML file.
            experiment_config_path: Optional experiment config path.
            generator_config: Optional generator configuration for customization.
        """
        self.generator_config = generator_config or GeneratorConfig()
        loaded = OmegaConf.load(train_config_path)

        if not isinstance(loaded, DictConfig):
            raise TypeError(f"Expected DictConfig but got {type(loaded).__name__}")

        self.train_cfg: DictConfig = loaded

        # If experiment config provided, extract its type and metadata
        self.exp_cfg: Optional[DictConfig] = None
        if experiment_config_path:
            exp_loaded = OmegaConf.load(experiment_config_path)
            if not isinstance(exp_loaded, DictConfig):
                raise TypeError(
                    f"Expected DictConfig but got {type(exp_loaded).__name__}"
                )
            self.exp_cfg = exp_loaded

        self.experiment_name: str = Path(train_config_path).stem

        train_steps = self.train_cfg.pipeline.steps

        # Parse feature pipeline from steps
        self.feature_pipeline = FeaturePipelineParser.parse_from_steps(train_steps)

        # Detect experiment type
        self.experiment_type = self.train_cfg.get("experiment", {}).get(
            "type", self.train_cfg.get("data", {}).get("type")
        )

        if not self.experiment_type and self.exp_cfg:
            self.experiment_type = self.exp_cfg.get("experiment", {}).get(
                "type", self.exp_cfg.get("data", {}).get("type")
            )

        if not self.experiment_type:
            self.experiment_type = StepAnalyzer.infer_experiment_type(train_steps)
            print(f"[ConfigGenerator] Inferred experiment type: {self.experiment_type}")

        if self.feature_pipeline:
            print("[ConfigGenerator] Detected feature pipeline:")
            print(f"  Base: {self.feature_pipeline.base_source}")
            print(f"  Engineered features: {len(self.feature_pipeline.engineered)}")
            for feat in self.feature_pipeline.engineered:
                parent_info = (
                    f" (in {feat.parent_pipeline})" if feat.parent_pipeline else ""
                )
                print(f"    - {feat.source_step_id} -> {feat.output_key}{parent_info}")

        self._eval_builder = EvalBuilder(
            train_steps,
            experiment_type=self.experiment_type,
            config=self.generator_config,
        )
        self._serve_builder = ServeBuilder(
            train_steps,
            experiment_type=self.experiment_type,
            config=self.generator_config,
        )
        self._tune_builder = TuneBuilder(train_steps)
        self._api_generator = ApiGenerator(config=self.generator_config)

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
        alias: str = "production",
    ) -> str:
        """Generate API code from serve configuration.

        Delegates to ApiGenerator.
        """
        return self._api_generator.generate_api(
            serve_config_path,
            output_dir,
            framework,
            experiment_config_path,
            alias=alias,
        )

    def generate_all(
        self,
        output_dir: str,
        alias: str = "latest",
        include_tune: bool = False,
    ) -> Dict[str, str]:
        """Generate eval, serve, and optionally tune configs.

        Args:
            output_dir: Directory to save generated configs.
            alias: Model alias for loading artifacts.
            include_tune: Whether to generate tune config.

        Returns:
            Dict mapping mode name to output file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        eval_file = str(out_path / f"{self.experiment_name}_eval.yaml")
        eval_cfg = self._transform_pipeline("eval", alias)
        self._save_config(eval_cfg, eval_file)
        paths["eval"] = eval_file

        serve_file = str(out_path / f"{self.experiment_name}_serve.yaml")
        serve_cfg = self._transform_pipeline("serve", alias)
        self._save_config(serve_cfg, serve_file)
        paths["serve"] = serve_file

        if include_tune:
            tune_file = str(out_path / f"{self.experiment_name}_tune.yaml")
            tune_cfg = self._transform_pipeline("tune")
            self._save_config(tune_cfg, tune_file)
            paths["tune"] = tune_file

        return paths

    def generate_eval_config(self, alias: str, output_path: str) -> str:
        """Generate evaluation pipeline config.

        Args:
            alias: Model alias for loading artifacts.
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config.

        Args:
            alias: Model alias for loading artifacts.
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_tune_config(self, output_path: str) -> str:
        """Generate tuning pipeline config.

        Args:
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("tune")
        self._save_config(cfg, output_path)
        return output_path

    def _transform_pipeline(self, mode: str, alias: str = "latest") -> DictConfig:
        """Transform training pipeline to target mode.

        Args:
            mode: Target mode - 'eval', 'serve', or 'tune'.
            alias: Model alias for loading artifacts.

        Returns:
            Transformed pipeline config.

        Raises:
            ValueError: If pipeline config is invalid or mode unknown.
        """
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(f"Config {self.experiment_name} missing pipeline.steps")

        train_steps = cfg.pipeline.steps

        if mode == "tune":
            new_steps = self._tune_builder.build()
        else:
            preprocessor_step = self._find_preprocessor(train_steps)
            init_id = STEP_CONSTANTS.INIT_ARTIFACTS_ID

            if mode == "eval":
                new_steps = self._eval_builder.build(
                    alias, init_id, preprocessor_step, self.feature_pipeline
                )
            elif mode == "serve":
                new_steps = self._serve_builder.build(
                    alias, init_id, preprocessor_step, self.feature_pipeline
                )
            else:
                raise ValueError(
                    f"Unknown mode: {mode}. Must be 'eval', 'serve', or 'tune'"
                )

        cfg.pipeline.name = f"{self.experiment_name}_{mode}"
        cfg.pipeline.steps = new_steps

        if "preprocessing" in cfg and mode != "tune":
            cfg.preprocessing.is_train = False

        return cfg

    def _find_preprocessor(self, steps: list) -> Optional[Any]:
        """Find legacy preprocessor step.

        Args:
            steps: List of pipeline steps.

        Returns:
            Preprocessor step if found, None otherwise.
        """
        return next((s for s in steps if s.type == STEP_CONSTANTS.PREPROCESSOR), None)

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save pipeline config to YAML file.

        Args:
            cfg: Config to save.
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as file:
            OmegaConf.save(config=cfg, f=file)

        print(f"[ConfigGenerator] Successfully generated: {path}")
