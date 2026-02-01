from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigValidator:
    """Validate the structure of an experiment configuration.

    This class ensures that required sections and keys exist in the
    configuration. It does not validate value types or allowed ranges.
    """

    REQUIRED_KEYS = {
        "experiment": ["name", "type", "model"],
        "data": ["features"],
        "preprocessing": ["steps"],
    }

    @staticmethod
    def validate(cfg: DictConfig) -> None:
        """Validate that required config sections and keys exist.

        Args:
            cfg (DictConfig): The configuration object to validate.

        Raises:
            ValueError: If required sections or keys are missing.
        """
        for section, keys in ConfigValidator.REQUIRED_KEYS.items():
            if section not in cfg:
                raise ValueError(f"Missing required section: {section}")

            for key in keys:
                if key not in cfg[section]:
                    raise ValueError(
                        f"Missing required key '{key}' in section '{section}'"
                    )


class ConfigLoader:
    """
    Class-based configuration loader using OmegaConf.

    Features:
    ---------
    - Load experiment YAML
    - Resolve and merge Hydra-style default config blocks
    - Convert ListConfig → DictConfig safely
    - Return a single merged DictConfig without mypy errors

    Example:
    --------
    defaults:
      - data: base/data
      - model: base/model
      - preprocessing: base/preprocessing
      - training: base/training
      - evaluation: base/evaluation
    """

    @staticmethod
    def _ensure_dict(cfg: Union[DictConfig, ListConfig]) -> DictConfig:
        """
        Ensure that an OmegaConf object is returned as DictConfig.
        """
        if isinstance(cfg, DictConfig):
            return cfg

        # ListConfig → DictConfig conversion
        created = OmegaConf.create(cfg)
        return cast(DictConfig, created)

    @staticmethod
    def _load_default_file(
        base_dir: str, entry: Union[str, Dict[str, Any]]
    ) -> DictConfig:
        """
        Load a default YAML config block listed under `defaults`.
        """
        # Case 1: entry: "data.yaml"
        if isinstance(entry, str):
            candidate = os.path.join(base_dir, entry)
            if os.path.exists(candidate):
                return ConfigLoader._ensure_dict(OmegaConf.load(candidate))
            return OmegaConf.create({})

        # Case 2: entry: {"data": "base/data"}
        _, file_stub = next(iter(entry.items()))
        candidate = os.path.join(base_dir, f"{file_stub}.yaml")
        if os.path.exists(candidate):
            return ConfigLoader._ensure_dict(OmegaConf.load(candidate))

        return OmegaConf.create({})

    @staticmethod
    def load(cfg_path: str = "") -> DictConfig:
        """
        Load main experiment config, merge with default config files.

        Args:
            cfg_path:
                Path to experiment YAML.
                Defaults to:
                mlproject/configs/experiments/etth1.yaml

        Returns:
            DictConfig: A fully merged OmegaConf object.
        """
        # Default experiment file
        if not cfg_path:
            raise ValueError("cfg_path must be a non-empty string")

        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        # Load primary config
        cfg_raw = OmegaConf.load(cfg_path)
        cfg: DictConfig = cast(DictConfig, cfg_raw)

        # If no defaults: return raw config directly
        defaults = cfg.get("defaults")  # type: ignore[assignment]
        if defaults is None:
            ConfigValidator.validate(cfg)
            return cfg

        base_dir = os.path.dirname(cfg_path)
        merged: DictConfig = OmegaConf.create({})

        # Merge default blocks
        for entry in cast(ListConfig, defaults):
            sub_cfg = ConfigLoader._load_default_file(base_dir, entry)
            merged = cast(DictConfig, OmegaConf.merge(merged, sub_cfg))

        # Merge main config last (highest priority)
        final_cfg = cast(DictConfig, OmegaConf.merge(merged, cfg))
        # ConfigValidator.validate(final_cfg)
        return final_cfg


class ConfigMerger:
    """Utility class to merge and persist experiment + pipeline configs."""

    @staticmethod
    def merge(
        experiment_path: str,
        pipeline_path: str,
        mode: str = "train",
    ) -> DictConfig:
        """Merge experiment config with pipeline config."""
        _ = mode
        if not experiment_path:
            raise ValueError("experiment_path cannot be None or empty")
        if not pipeline_path:
            raise ValueError("pipeline_path cannot be None or empty")

        exp_cfg = ConfigLoader.load(experiment_path)
        pipe_cfg = OmegaConf.load(pipeline_path)

        merged: Union[DictConfig, ListConfig] = OmegaConf.merge(exp_cfg, pipe_cfg)

        if isinstance(merged, ListConfig):
            return OmegaConf.create({"config": merged})

        return merged

    @staticmethod
    def save(cfg: DictConfig, output_path: str) -> None:
        """Save merged config to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            OmegaConf.save(cfg, f)

        print(f"  - Merged saved: {output_path}\n")
