"""Configuration classes for ML pipeline generator.

This module provides configurable classes that can be customized
via external configuration files or programmatically to override
default constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from .constants import (
    API_DEFAULTS,
    CONTEXT_KEYS,
    DATA_DEFAULTS,
    INFERENCE_METHODS,
    MODEL_PATTERNS,
    STEP_CONSTANTS,
    APIDefaults,
    ContextKeyConstants,
    DataConfigDefaults,
    InferenceMethodDefaults,
    ModelTypePatterns,
    StepConstants,
)


@dataclass
class GeneratorConfig:
    """Main configuration for pipeline generator.

    This class aggregates all configuration sections and can be
    loaded from a YAML file or created programmatically.
    """

    step_constants: StepConstants = field(default_factory=lambda: STEP_CONSTANTS)
    context_keys: ContextKeyConstants = field(default_factory=lambda: CONTEXT_KEYS)
    model_patterns: ModelTypePatterns = field(default_factory=lambda: MODEL_PATTERNS)
    data_defaults: DataConfigDefaults = field(default_factory=lambda: DATA_DEFAULTS)
    api_defaults: APIDefaults = field(default_factory=lambda: API_DEFAULTS)
    inference_methods: InferenceMethodDefaults = field(
        default_factory=lambda: INFERENCE_METHODS
    )

    @classmethod
    def from_file(cls, config_path: str) -> GeneratorConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file.

        Returns:
            GeneratorConfig instance with loaded values.
        """
        if not Path(config_path).exists():
            print(
                f"[GeneratorConfig] Config file not found: {config_path}, "
                f"using defaults"
            )
            return cls()

        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected DictConfig but got {type(cfg).__name__}")

        container = OmegaConf.to_container(cfg)
        if not isinstance(container, dict):
            raise TypeError(
                f"Expected dict from config but got {type(container).__name__}"
            )

        return cls._from_dict(container)  # type: ignore

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> GeneratorConfig:
        """Create configuration from dictionary."""
        # Using top-level imports to avoid redefining outer names
        kwargs = {}

        if "step_constants" in config_dict:
            kwargs["step_constants"] = cls._merge_dataclass(
                StepConstants, STEP_CONSTANTS, config_dict["step_constants"]
            )

        if "context_keys" in config_dict:
            kwargs["context_keys"] = cls._merge_dataclass(
                ContextKeyConstants, CONTEXT_KEYS, config_dict["context_keys"]
            )

        if "model_patterns" in config_dict:
            kwargs["model_patterns"] = cls._merge_dataclass(
                ModelTypePatterns, MODEL_PATTERNS, config_dict["model_patterns"]
            )

        if "data_defaults" in config_dict:
            kwargs["data_defaults"] = cls._merge_dataclass(
                DataConfigDefaults, DATA_DEFAULTS, config_dict["data_defaults"]
            )

        if "api_defaults" in config_dict:
            kwargs["api_defaults"] = cls._merge_dataclass(
                APIDefaults, API_DEFAULTS, config_dict["api_defaults"]
            )

        if "inference_methods" in config_dict:
            kwargs["inference_methods"] = cls._merge_dataclass(
                InferenceMethodDefaults,
                INFERENCE_METHODS,
                config_dict["inference_methods"],
            )

        return cls(**kwargs)

    @staticmethod
    def _merge_dataclass(cls_type: type, default_instance: Any, overrides: Dict):
        """Merge override values into default dataclass instance."""
        # Get current values from default instance
        current_values = {
            k: getattr(default_instance, k)
            for k in default_instance.__dataclass_fields__
        }

        # Update with overrides
        current_values.update(overrides)

        # Create new instance with merged values
        return cls_type(**current_values)

    def save(self, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path where to save configuration.
        """
        config_dict = {
            "step_constants": self._dataclass_to_dict(self.step_constants),
            "context_keys": self._dataclass_to_dict(self.context_keys),
            "model_patterns": self._dataclass_to_dict(self.model_patterns),
            "data_defaults": self._dataclass_to_dict(self.data_defaults),
            "api_defaults": self._dataclass_to_dict(self.api_defaults),
            "inference_methods": self._dataclass_to_dict(self.inference_methods),
        }

        cfg = OmegaConf.create(config_dict)
        with open(output_path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)

        print(f"[GeneratorConfig] Saved configuration to: {output_path}")

    @staticmethod
    def _dataclass_to_dict(instance: Any) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary."""
        return {k: getattr(instance, k) for k in instance.__dataclass_fields__}


class ConfigurablePatternMatcher:
    """Pattern matcher that uses configurable patterns from config.

    This class provides methods to match model types, step types, etc.
    based on patterns defined in GeneratorConfig.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize pattern matcher.

        Args:
            config: GeneratorConfig instance. If None, uses default.
        """
        self.config = config or GeneratorConfig()

    def infer_model_type(self, model_key: str) -> str:
        """Infer model type from model key.

        Args:
            model_key: Model key or class path.

        Returns:
            Model type string ('dl' for deep learning, 'ml' for machine learning).
        """
        key_lower = model_key.lower()

        if any(
            pattern in key_lower
            for pattern in self.config.model_patterns.DEEP_LEARNING_PATTERNS
        ):
            return "dl"

        if any(
            pattern in key_lower
            for pattern in self.config.model_patterns.MACHINE_LEARNING_PATTERNS
        ):
            return "ml"

        return "ml"  # Default fallback

    def is_feature_generator(self, step_id: str, class_path: str = "") -> bool:
        """Check if step is a feature generator based on patterns.

        Args:
            step_id: Step identifier.
            class_path: Step class path if available.

        Returns:
            True if step is identified as feature generator.
        """
        step_lower = step_id.lower()
        class_lower = class_path.lower()

        # Check if it matches feature generator patterns
        if any(
            pattern in step_lower or pattern in class_lower
            for pattern in self.config.model_patterns.FEATURE_GENERATOR_PATTERNS
        ):
            return True

        # Exclude preprocessors
        if any(
            pattern in step_lower or pattern in class_lower
            for pattern in self.config.model_patterns.PREPROCESSOR_PATTERNS
        ):
            return False

        return False

    def should_apply_windowing(self, step_type: str, step_id: str) -> bool:
        """Determine if windowing should be applied to a step.

        Args:
            step_type: Type of the step.
            step_id: Step identifier.

        Returns:
            True if windowing should be applied.
        """
        # Check step types that always use windowing
        if step_type in self.config.step_constants.WINDOWING_STEP_TYPES:
            return True

        # Check step ID keywords that should NOT use windowing
        step_lower = step_id.lower()
        if any(
            keyword in step_lower
            for keyword in self.config.step_constants.NON_WINDOWING_KEYWORDS
        ):
            return False

        # Default for timeseries
        return True

    def get_inference_method(self, step_type: str, step_id: str = "") -> str:
        """Get inference method for a step.

        Args:
            step_type: Type of the step.
            step_id: Step identifier (optional).

        Returns:
            Inference method name.
        """
        step_lower = step_id.lower()

        # Check for clustering
        if step_type == self.config.step_constants.CLUSTERING or any(
            keyword in step_lower
            for keyword in self.config.model_patterns.CLUSTERING_KEYWORDS
        ):
            return self.config.inference_methods.CLUSTERING_METHOD

        # Check for trainer
        if step_type == self.config.step_constants.TRAINER:
            return self.config.inference_methods.TRAINER_METHOD

        # Default to transform
        return self.config.inference_methods.TRANSFORM

    def resolve_feature_key(
        self, wiring_inputs: Dict[str, str], default: Optional[str] = None
    ) -> str:
        """Resolve feature key from wiring inputs based on priority.

        Args:
            wiring_inputs: Dictionary of wiring inputs.
            default: Default key if no match found.

        Returns:
            Resolved feature key.
        """
        for key in self.config.context_keys.FEATURE_KEY_PRIORITY:
            if key in wiring_inputs:
                return wiring_inputs[key]

        return default or self.config.context_keys.PREPROCESSED_DATA

    def resolve_output_key(
        self, wiring_outputs: Dict[str, str], default: Optional[str] = None
    ) -> Optional[str]:
        """Resolve output key from wiring outputs based on priority.

        Args:
            wiring_outputs: Dictionary of wiring outputs.
            default: Default key if no match found.

        Returns:
            Resolved output key or None.
        """
        for key in self.config.context_keys.OUTPUT_KEY_PRIORITY:
            if key in wiring_outputs:
                return wiring_outputs[key]

        return default


# Global default instance
_default_config: Optional[GeneratorConfig] = None


def get_default_config() -> GeneratorConfig:
    """Get or create default generator configuration.

    Returns:
        Default GeneratorConfig instance.
    """
    global _default_config  # pylint: disable=global-statement
    if _default_config is None:
        _default_config = GeneratorConfig()
    return _default_config


def set_default_config(config: GeneratorConfig) -> None:
    """Set default generator configuration.

    Args:
        config: GeneratorConfig to use as default.
    """
    global _default_config  # pylint: disable=global-statement
    _default_config = config


def load_config_from_env() -> GeneratorConfig:
    """Load configuration from environment variable.

    Looks for MLPROJECT_GENERATOR_CONFIG environment variable
    pointing to a YAML configuration file.

    Returns:
        GeneratorConfig loaded from file or default.
    """
    config_path = os.environ.get("MLPROJECT_GENERATOR_CONFIG")
    if config_path:
        return GeneratorConfig.from_file(config_path)
    return GeneratorConfig()


__all__ = [
    "GeneratorConfig",
    "ConfigurablePatternMatcher",
    "get_default_config",
    "set_default_config",
    "load_config_from_env",
]
