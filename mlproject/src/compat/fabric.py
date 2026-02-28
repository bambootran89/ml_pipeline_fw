"""
Microsoft Fabric runtime environment helper.

This module detects whether the code is running inside Microsoft Fabric
and provides helper utilities to configure paths and MLflow accordingly.

Usage:
    from mlproject.src.compat.fabric import FabricEnv

    env = FabricEnv.detect()
    cfg_overrides = env.get_config_overrides()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_FABRIC_INDICATOR_VARS = [
    "FABRIC_WORKSPACE_ID",
    "FABRIC_CAPACITY_ID",
    "MSSPARKUTILS_VERSION",  # Spark utils available in Fabric notebooks
]

_FABRIC_LAKEHOUSE_ROOT = "/lakehouse/default/Files"


def _is_fabric() -> bool:
    """Return True if running inside a Microsoft Fabric environment."""
    return any(os.environ.get(v) for v in _FABRIC_INDICATOR_VARS)


@dataclass
class FabricEnv:
    """Fabric runtime environment configuration.

    Attributes
    ----------
    enabled : bool
        True when running inside Microsoft Fabric.
    lakehouse_root : str
        Mount path for the attached Lakehouse (Files section).
    workspace_id : Optional[str]
        Fabric workspace ID from environment variables.
    """

    enabled: bool = False
    lakehouse_root: str = _FABRIC_LAKEHOUSE_ROOT
    workspace_id: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> "FabricEnv":
        """Auto-detect the current runtime environment.

        Returns
        -------
        FabricEnv
            Populated instance. ``enabled`` is ``True`` only on Fabric.
        """
        enabled = _is_fabric()
        workspace_id = os.environ.get("FABRIC_WORKSPACE_ID")

        if enabled:
            logger.info(
                "[FabricEnv] Detected Microsoft Fabric runtime "
                f"(workspace_id={workspace_id})"
            )
        else:
            logger.debug("[FabricEnv] Not running in Fabric — using local defaults.")

        return cls(
            enabled=enabled,
            lakehouse_root=_FABRIC_LAKEHOUSE_ROOT,
            workspace_id=workspace_id,
        )

    # ------------------------------------------------------------------ #
    # Config helpers                                                       #
    # ------------------------------------------------------------------ #

    def artifacts_dir(self, sub: str = "models") -> str:
        """Return the Lakehouse-rooted artifacts path for a given sub-folder.

        Parameters
        ----------
        sub : str
            Sub-folder name inside ``Files/`` (e.g. ``"models"``).

        Returns
        -------
        str
            Resolved path. Falls back to ``artifacts/<sub>`` locally.
        """
        if self.enabled:
            return os.path.join(self.lakehouse_root, sub)
        return os.path.join("artifacts", sub)

    def get_config_overrides(self) -> Dict[str, object]:
        """Return a flat dict of OmegaConf-compatible config key overrides.

        These override values can be merged into the experiment config so
        that pipeline steps automatically write to the correct paths when
        running on Fabric.

        Returns
        -------
        Dict[str, object]
            Keys are dot-separated OmegaConf paths; values are the
            overridden settings.
        """
        if not self.enabled:
            return {}

        return {
            # MLflow: use Fabric built-in tracking (no external server needed)
            "mlflow.tracking_uri": "mlflow",
            # Write model/preprocessor artifacts to OneLake
            "training.artifacts_dir": self.artifacts_dir("models"),
            "preprocessing.artifacts_dir": self.artifacts_dir("preprocessors"),
        }

    def apply_env_vars(self) -> None:
        """Set environment variables expected by upstream MLflow calls.

        Call this once at the top of your Fabric notebook before importing
        any pipeline modules.
        """
        if not self.enabled:
            return

        # Fabric manages its own MLflow tracking URI; override if not set
        os.environ.setdefault("MLFLOW_TRACKING_URI", "mlflow")
        logger.info("[FabricEnv] Environment variables applied for Fabric runtime.")
