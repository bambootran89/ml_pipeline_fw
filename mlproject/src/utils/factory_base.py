import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class DynamicFactoryBase(ABC):
    """
    Base class for factories that dynamically load classes from config.
    """

    @staticmethod
    def _get_class_from_config(config_entry: Dict) -> Type:
        """Dynamically imports a class based on module and class names."""
        module_name = config_entry.get("module")
        class_name = config_entry.get("class")

        if not module_name or not class_name:
            raise ValueError("Config entry must contain 'module' and 'class' keys.")

        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Any:
        """Abstract method to define the creation method for subclasses."""
        raise NotImplementedError
