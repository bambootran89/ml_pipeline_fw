"""Unit tests for the Microsoft Fabric compatibility helper."""

import os
import sys
from unittest.mock import patch

import pytest

from mlproject.src.compat.fabric import FabricEnv, _is_fabric

# ── _is_fabric() ─────────────────────────────────────────────────────────────


def test_is_fabric_false_by_default():
    """Should return False when no Fabric env vars are set."""
    with patch.dict(os.environ, {}, clear=False):
        # Ensure none of the indicator vars are set
        for v in ("FABRIC_WORKSPACE_ID", "FABRIC_CAPACITY_ID", "MSSPARKUTILS_VERSION"):
            os.environ.pop(v, None)
        assert _is_fabric() is False


def test_is_fabric_true_when_var_set():
    """Should return True when FABRIC_WORKSPACE_ID is present."""
    with patch.dict(os.environ, {"FABRIC_WORKSPACE_ID": "test-workspace"}, clear=False):
        assert _is_fabric() is True


# ── FabricEnv.detect() ───────────────────────────────────────────────────────


def test_detect_local():
    """detect() should produce a disabled FabricEnv locally."""
    for v in ("FABRIC_WORKSPACE_ID", "FABRIC_CAPACITY_ID", "MSSPARKUTILS_VERSION"):
        os.environ.pop(v, None)

    env = FabricEnv.detect()
    assert env.enabled is False
    assert env.workspace_id is None


def test_detect_fabric():
    """detect() should produce an enabled FabricEnv when indicator var is set."""
    with patch.dict(
        os.environ,
        {"FABRIC_WORKSPACE_ID": "ws-abc123"},
        clear=False,
    ):
        env = FabricEnv.detect()
        assert env.enabled is True
        assert env.workspace_id == "ws-abc123"


# ── FabricEnv.artifacts_dir() ────────────────────────────────────────────────


def test_artifacts_dir_local():
    """Local fallback should use relative 'artifacts/<sub>' path."""
    env = FabricEnv(enabled=False)
    assert env.artifacts_dir("models") == "artifacts/models"


def test_artifacts_dir_fabric():
    """Fabric path should resolve to OneLake mount root."""
    env = FabricEnv(enabled=True, lakehouse_root="/lakehouse/default/Files")
    assert env.artifacts_dir("models") == "/lakehouse/default/Files/models"


# ── FabricEnv.get_config_overrides() ─────────────────────────────────────────


def test_config_overrides_empty_when_local():
    """Should return empty dict when not on Fabric."""
    env = FabricEnv(enabled=False)
    assert env.get_config_overrides() == {}


def test_config_overrides_fabric():
    """Should return mlflow and artifact path overrides when on Fabric."""
    env = FabricEnv(enabled=True, lakehouse_root="/lakehouse/default/Files")
    overrides = env.get_config_overrides()
    assert overrides["mlflow.tracking_uri"] == "mlflow"
    assert "models" in overrides["training.artifacts_dir"]
    assert "preprocessors" in overrides["preprocessing.artifacts_dir"]


# ── FabricEnv.apply_env_vars() ───────────────────────────────────────────────


def test_apply_env_vars_no_op_when_local():
    """Should not modify environment when not on Fabric."""
    env = FabricEnv(enabled=False)
    original = os.environ.get("MLFLOW_TRACKING_URI")
    env.apply_env_vars()
    assert os.environ.get("MLFLOW_TRACKING_URI") == original


def test_apply_env_vars_sets_tracking_uri():
    """Should set MLFLOW_TRACKING_URI to 'mlflow' when on Fabric."""
    env = FabricEnv(enabled=True)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    env.apply_env_vars()
    assert os.environ.get("MLFLOW_TRACKING_URI") == "mlflow"
    # Cleanup
    os.environ.pop("MLFLOW_TRACKING_URI", None)
