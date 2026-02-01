from typing import Any, Dict, Union, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.utils.factory_base import DynamicFactoryBase


class DataModuleFactory(DynamicFactoryBase):
    """Factory for creating DataModule instances based on experiment configuration.

    This factory:
    - Dynamically selects datamodule type (``dl`` or ``ml``)
    - Uses registry lookup instead of hardcoded model lists
    - Loads class objects using :class:`DynamicFactoryBase`
    """

    DATAMODULE_REGISTRY = {
        "dl": {
            "module": "mlproject.src.datamodule.ts_sequence_dm",
            "class": "TSDLDataModule",
        },
        "ml": {
            "module": "mlproject.src.datamodule.base",
            "class": "BaseDataModule",
        },
    }

    @classmethod
    def _get_datamodule_type(cls, cfg: DictConfig) -> str:
        """Return datamodule type (``dl`` or ``ml``) derived from the model registry."""

        model_type = cfg.experiment.get("model", "").lower()
        model_registry = cfg.get("model_registry", {})

        if model_type not in model_registry:
            available = list(model_registry.keys())
            raise ValueError(
                f"Model '{model_type}' not found in model_registry. "
                f"Available models: {available}"
            )

        model_config = model_registry[model_type]
        dm_type = model_config.get("datamodule_type")

        if dm_type is None:
            raise ValueError(
                f"Model '{model_type}' missing 'datamodule_type' in config. "
                "Must specify 'dl' or 'ml'."
            )

        if dm_type not in {"dl", "ml"}:
            raise ValueError(
                f"Invalid datamodule_type '{dm_type}' for model '{model_type}'. "
                "Must be 'dl' or 'ml'."
            )

        return dm_type

    @classmethod
    def build(cls, cfg: DictConfig, df: pd.DataFrame) -> Any:
        """Build and return the appropriate DataModule instance."""
        dm_type = cls._get_datamodule_type(cfg)
        config_entry = cls.DATAMODULE_REGISTRY[dm_type]

        data_module_class = cls._get_class_from_config(config_entry)

        input_chunk = cfg.experiment.hyperparams.get("input_chunk_length", -1)
        output_chunk = cfg.experiment.hyperparams.get("output_chunk_length", -1)

        cfg_init: Union[DictConfig, Dict[str, Any]]

        if dm_type == "dl":
            cfg_init = cfg
        else:
            raw_cfg = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(raw_cfg, dict)
            cfg_init = cast(Dict[str, Any], raw_cfg)

        return data_module_class(
            df=df,
            cfg=cfg_init,
            input_chunk=input_chunk,
            output_chunk=output_chunk,
        )

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Any:
        """Alias for :meth:`build` for API consistency."""
        return cls.build(*args, **kwargs)
