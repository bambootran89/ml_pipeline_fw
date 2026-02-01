from typing import Any, Dict, Optional

from mlproject.src.utils.factory_base import DynamicFactoryBase


class ModelFactory(DynamicFactoryBase):
    """
    Factory for creating and loading model wrappers.

    This class maps a model name to its corresponding wrapper class.
    It supports dynamic creation and loading of model instances
    dynamically through configuration without if/else branching.
    """

    @classmethod
    def create(cls, *args: Any, **kwargs: Any):
        """
        Create a model wrapper instance.

        This method keeps the same signature as ``DynamicFactoryBase.create``
        to comply with pylint rule W0221 (arguments-differ).

        Parameters
        ----------
        *args : Any
            Positional arguments. Expected: (name, config)
        **kwargs : Any
            - model_registry : optional custom registry
            - Extra dynamic options passed to wrapper initializer.

        Returns
        -------
        Any
            Instantiated model wrapper.

        Raises
        ------
        ValueError
            If the model name does not exist in registry.
        """
        if len(args) < 2:
            raise ValueError("ModelFactory.create requires at least (name, config).")

        name = args[0]
        config = args[1]

        model_registry: Optional[Dict[str, Any]] = config.get("model_registry")
        if model_registry is None:
            raise ValueError("Please check config")

        entry = model_registry.get(name)
        if not entry:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available: {list(model_registry.keys())}"
            )

        model_class = cls._get_class_from_config(entry)
        return model_class(cfg=config.get("experiment", {}).get("hyperparams"))

    @classmethod
    def load(cls, *args: Any, **kwargs: Any):
        """
        Load a trained model wrapper from the artifacts directory.

        Parameters
        ----------
        *args : Any
            Positional arguments. Expected: (name, config, artifact_dir)
        **kwargs : Any
            Extra dynamic options such as custom registry.

        Returns
        -------
        Any
            Loaded model wrapper instance.
        """
        if len(args) < 3:
            raise ValueError("ModelFactory.load requires (name, config, artifact_dir).")

        name, config, artifact_dir = args[:3]

        wrapper = cls.create(name, config, **kwargs)
        wrapper.load(artifact_dir)
        return wrapper
