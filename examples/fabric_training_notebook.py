# Microsoft Fabric — Training Notebook
# ─────────────────────────────────────────────────────────────────────────────
# Run this notebook inside a Microsoft Fabric workspace.
# Attach the Lakehouse "ml_artifacts" before running.
# ─────────────────────────────────────────────────────────────────────────────

# ── Cell 1 : Install extra dependencies ──────────────────────────────────────
# Fabric pre-installs: torch, pandas, numpy, scikit-learn, pyarrow, mlflow
# Only install packages not bundled with the Fabric runtime.
# %pip install -q -r /lakehouse/default/Files/requirements/fabric.txt

from omegaconf import OmegaConf

# ── Cell 2 : Detect Fabric environment & apply overrides ─────────────────────
from mlproject.src.compat.fabric import FabricEnv

fabric = FabricEnv.detect()
fabric.apply_env_vars()  # Sets MLFLOW_TRACKING_URI=mlflow if on Fabric

# Load the Fabric-specific config overlay
fabric_cfg_path = "/lakehouse/default/Files/configs/environments/fabric.yaml"
fabric_overlay = OmegaConf.load(fabric_cfg_path)

print(f"Running on Fabric: {fabric.enabled}")
print(f"Lakehouse root  : {fabric.lakehouse_root}")

# ── Cell 3 : Run training pipeline ───────────────────────────────────────────
from mlproject.src.pipeline.dag_run import run_training
from mlproject.src.pipeline.steps.core.utils import ConfigMerger

# Merge your experiment config with the Fabric overlay
experiment_path = "/lakehouse/default/Files/configs/experiments/tabular.yaml"
pipeline_path = "/lakehouse/default/Files/configs/pipelines/standard_train.yaml"

# ConfigMerger.merge handles deep-merge of both YAML files
merged = ConfigMerger.merge(experiment_path, pipeline_path, mode="train")
merged = OmegaConf.merge(merged, fabric_overlay)

import tempfile

# Save merged config to a temp path that FlexiblePipeline can load
import uuid
from pathlib import Path

from mlproject.src.pipeline.steps.core.utils import ConfigMerger as CM

tmp = Path(tempfile.gettempdir()) / f"fabric_merged_{uuid.uuid4().hex}.yaml"
OmegaConf.save(merged, tmp)

try:
    from mlproject.src.pipeline.pipeline import FlexiblePipeline

    pipeline = FlexiblePipeline(str(tmp))
    context = pipeline.run_exp()
    print("Training complete. Context keys:", list(context.keys()))
finally:
    tmp.unlink(missing_ok=True)

# ── Cell 4 : (Optional) Run evaluation ───────────────────────────────────────
# from mlproject.src.pipeline.dag_run import run_eval
# run_eval(experiment_path, pipeline_path, alias="latest")

# ── Cell 5 : (Optional) Batch predict from registered model ──────────────────
# import mlflow, pandas as pd
# model = mlflow.pyfunc.load_model("models:/fabric_model/latest")
# df    = pd.read_csv("/lakehouse/default/Files/data/input.csv")
# preds = model.predict(df)
# print(preds[:5])
