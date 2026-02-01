"""API code generation mixin for FastAPI and Ray Serve.

This mixin generates API code from serve pipeline configurations,
supporting both FastAPI and Ray Serve frameworks.

Supports:
- Tabular data: batch prediction for multiple rows
- Timeseries data: multi-step prediction with configurable horizon
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from ..config import GeneratorConfig
from ..constants import CONTEXT_KEYS
from .fastapi_generator import ApiGeneratorFastAPIMixin
from .rayserve_generator import ApiGeneratorRayServeMixin


class ApiGeneratorMixin(ApiGeneratorFastAPIMixin, ApiGeneratorRayServeMixin):
    """Mixin for generating API code from serve configurations.

    Provides methods to generate FastAPI and Ray Serve code from
    serve pipeline YAML configs. Uses template-based generation
    to keep complexity low.

    Supports both tabular and timeseries data types with appropriate
    prediction strategies (batch for tabular, multi-step for timeseries).
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize API generator with configuration.

        Args:
            config: Optional GeneratorConfig for customization.
        """
        ApiGeneratorFastAPIMixin.__init__(self, config)
        ApiGeneratorRayServeMixin.__init__(self, config)

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
        alias: str = "production",
    ) -> str:
        """Generate API code from serve configuration."""
        cfg, pipeline_name, steps = self._load_serve_config(serve_config_path)

        if framework not in ["fastapi", "ray"]:
            raise ValueError(f"Unsupported framework: {framework}")

        data_config, feature_generators = self._prepare_data_config(
            cfg, steps, experiment_config_path
        )

        # Extract inference steps
        inference_steps = self._extract_inference_steps_excluding_generators(
            steps, feature_generators
        )

        load_map, preprocessor = self._prepare_inference_context(steps)

        return self._write_api_file(
            output_dir,
            *self._generate_framework_code(
                framework,
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
                data_config,
                alias,
            ),
            framework,
            data_config,
        )

    def _write_api_file(
        self,
        output_dir: str,
        code: str,
        filename: str,
        framework: str,
        data_config: Dict[str, Any],
    ) -> str:
        """Write generated API code to file."""
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[ApiGenerator] Generated {framework} API: {output_path}")
        print(f"[ApiGenerator] Data type: {data_config.get('data_type', 'unknown')}")
        return str(output_path)

    def _generate_framework_code(
        self,
        framework: str,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Dict[str, Any],
        alias: str = "production",
    ) -> tuple[str, str]:
        """Generate API code based on framework."""
        if framework == "fastapi":
            code = self._generate_fastapi_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
                data_config,
                alias=alias,
            )
            filename = f"{pipeline_name}_fastapi.py"
        else:
            code = self._generate_ray_serve_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
                data_config,
                alias=alias,
            )
            filename = f"{pipeline_name}_ray.py"
        return code, filename

    def _load_serve_config(self, serve_config_path: str) -> tuple[DictConfig, str, Any]:
        """Load and validate serve configuration."""
        cfg = OmegaConf.load(serve_config_path)
        assert isinstance(cfg, DictConfig)
        return cfg, cfg.pipeline.name, cfg.pipeline.steps

    def _prepare_data_config(
        self,
        cfg: DictConfig,
        steps: Any,
        experiment_config_path: str,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Prepare data configuration and feature generators."""
        # Extract data config from experiment config (preferred) or serve config
        if experiment_config_path and Path(experiment_config_path).exists():
            exp_cfg = OmegaConf.load(experiment_config_path)
            assert isinstance(exp_cfg, DictConfig)
            data_config = self._extract_data_config(exp_cfg)
        else:
            data_config = self._extract_data_config(cfg)

        # Extract feature generators
        feature_generators = self._extract_feature_generators(list(steps))
        if feature_generators:
            data_config["feature_generators"] = feature_generators
            print(f"[ApiGenerator] Found {len(feature_generators)} feature generators")
            for fg in feature_generators:
                print(f"  - {fg['step_id']} -> {fg['output_key']}")

        return data_config, feature_generators

    def _prepare_inference_context(
        self, steps: List[Any]
    ) -> tuple[Dict[str, str], Optional[Dict[str, Any]]]:
        """Prepare inference context (load map and preprocessor)."""
        load_map = self._extract_load_map(steps)
        preprocessor = self._extract_preprocessor_info(steps)
        return load_map, preprocessor

    def _extract_inference_steps_excluding_generators(self, steps, feature_generators):
        """Extract inference steps, exclude feature generators,
        and sort by dependencies."""
        all_inference_steps = self._extract_inference_steps(steps)
        generator_step_ids = {gen["step_id"] for gen in feature_generators}
        generator_model_keys = {gen["model_key"] for gen in feature_generators}

        # Filter out feature generator steps
        filtered_steps = []
        for inference_step in all_inference_steps:
            model_key = inference_step.get("model_key", "")
            step_id = inference_step.get("id", "").replace(
                CONTEXT_KEYS.INFERENCE_SUFFIX, ""
            )
            if (
                step_id in generator_step_ids
                or model_key in generator_model_keys
                or f"{CONTEXT_KEYS.FITTED_PREFIX}{step_id}" in generator_model_keys
            ):
                continue
            filtered_steps.append(inference_step)

        sorted_steps = self._sort_by_dependencies(filtered_steps)

        if sorted_steps != filtered_steps:
            print("[ApiGenerator] Sorted by dependencies:")
            for index, step in enumerate(sorted_steps):
                additional_keys = step.get("additional_feature_keys", [])
                deps_info = f" (deps: {additional_keys})" if additional_keys else ""
                print(f"  {index+1}. {step.get('id', '?')}{deps_info}")

        return sorted_steps
