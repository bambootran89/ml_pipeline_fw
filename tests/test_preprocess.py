from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlproject.src.preprocess.transform_manager import TransformManager


@pytest.fixture()
def cfg(tmp_path: Path) -> Any:
    """
    Configuration containing all supported preprocessing steps.
    """
    return OmegaConf.create(
        {
            "preprocessing": {
                "artifacts_dir": str(tmp_path),
                "steps": [
                    {"name": "fill_missing", "columns": ["num"], "method": "mean"},
                    {"name": "label_encoding", "columns": ["cat"]},
                    {"name": "normalize", "columns": ["num"], "method": "zscore"},
                ],
            }
        }
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """
    Sample dataframe covering numerical and categorical cases.
    """
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", "b", "a", None],
        }
    )


def test_fill_missing_mean(cfg: Any, sample_df: pd.DataFrame) -> None:
    """
    Mean imputation should reduce missing values.
    """
    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)
    manager.fit_fillna(sample_df, columns=["num"], method="mean")

    out = manager.transform(sample_df)

    assert out["num"].isna().sum() < sample_df["num"].isna().sum()


def test_label_encoding_applies_transform(cfg: Any, sample_df: pd.DataFrame) -> None:
    """
    Label encoding should apply a transformation without crashing.
    """
    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)
    manager.fit_label_encoding(sample_df, columns=["cat"])

    out = manager.transform(sample_df)

    assert "cat" in out.columns
    assert not out["cat"].equals(sample_df["cat"])


def test_label_encoding_handles_unseen_category(cfg: Any) -> None:
    """
    Unseen category should not crash and should return valid output.
    """
    train_df = pd.DataFrame({"cat": ["a", "b"]})
    test_df = pd.DataFrame({"cat": ["a", "c"]})

    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)
    manager.fit_label_encoding(train_df, columns=["cat"])

    out = manager.transform(test_df)

    assert isinstance(out, pd.DataFrame)
    assert "cat" in out.columns
    assert len(out) == len(test_df)


def test_normalize_zscore_scales_data(cfg: Any, sample_df: pd.DataFrame) -> None:
    """
    Z-score normalization should change scale of numerical data.
    """
    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)
    manager.fit_scaler(sample_df, columns=["num"], method="zscore")

    out = manager.transform(sample_df)

    assert not np.allclose(out["num"].values, sample_df["num"].values, equal_nan=True)


def test_multiple_transforms_pipeline_runs(cfg: Any, sample_df: pd.DataFrame) -> None:
    """
    Multiple preprocessing steps should run sequentially without crashing.
    """
    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)

    manager.fit_fillna(sample_df, columns=["num"], method="mean")
    manager.fit_label_encoding(sample_df, columns=["cat"])
    manager.fit_scaler(sample_df, columns=["num"], method="zscore")

    out = manager.transform(sample_df)

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"num", "cat"}


def test_unknown_transform_raises(cfg: Any, sample_df: pd.DataFrame) -> None:
    """
    Unknown transform step should raise ValueError.
    """
    cfg.preprocessing.steps.append({"name": "unknown_transform"})

    manager = TransformManager(cfg, artifacts_dir=cfg.preprocessing.artifacts_dir)

    with pytest.raises(ValueError):
        manager.transform(sample_df)
