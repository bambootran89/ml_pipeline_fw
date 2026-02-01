from typing import Any, Dict, cast

from mlproject.src.trainer.base import BaseTrainer
from mlproject.src.utils.factory_base import DynamicFactoryBase


class TrainerFactory(DynamicFactoryBase):
    """
    Factory for building Trainer instances dynamically based on model type.
    """

    TRAINER_REGISTRY: Dict[str, Dict[str, str]] = {
        "dl": {
            "module": "mlproject.src.trainer.dl",
            "class": "DeepLearningTrainer",
        },
        "ml": {
            "module": "mlproject.src.trainer.ml",
            "class": "MLTrainer",
        },
    }

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> BaseTrainer:
        """
        Create a Trainer instance.

        This method keeps the same signature as DynamicFactoryBase.create
        to avoid pylint W0221.

        Expected keyword arguments
        --------------------------
        model_name : str
        wrapper : Any
        save_dir : str

        Returns
        -------
        BaseTrainer
            Loaded Trainer instance.

        Raises
        ------
        ValueError
            Missing or invalid required arguments.
        """

        # Extract required arguments
        model_name = kwargs.get("model_name")
        model_type = kwargs.get("model_type")
        wrapper = kwargs.get("wrapper")
        save_dir = kwargs.get("save_dir")

        if not isinstance(model_name, str):
            raise ValueError("TrainerFactory.create requires 'model_name' (str).")
        if wrapper is None:
            raise ValueError("TrainerFactory.create requires 'wrapper'.")
        if not isinstance(save_dir, str):
            raise ValueError("TrainerFactory.create requires 'save_dir' (str).")

        if model_type in ["ml", "dl"]:
            entry = cls.TRAINER_REGISTRY[model_type]
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        trainer_class = cls._get_class_from_config(entry)
        return cast(
            BaseTrainer,
            trainer_class(wrapper=wrapper, save_dir=save_dir, model_type=model_type),
        )
