# DAG-Based Pipeline System

## Why DAG Pipelines?

The traditional monolithic pipeline approach has limitations when building complex ML workflows:

| Challenge | Monolithic Approach | DAG Pipeline Approach |
|-----------|--------------------|-----------------------|
| **Reusability** | Copy-paste code between projects | Compose reusable steps via YAML |
| **Flexibility** | Hard-coded execution order | Dynamic DAG with dependencies |
| **Experimentation** | Change code to try new flows | Swap pipeline configs only |
| **Complex Workflows** | Difficult to implement | Native support for parallel, branching, nesting |
| **Separation of Concerns** | Config + Logic mixed | Experiment config vs Pipeline structure |

## Key Design Principle: Separation of Concerns

```
```text
+-------------------------------------------------------------+
|                    EXPERIMENT CONFIG                        |
|              (WHAT to train/evaluate)                       |
|  +-----------------------------------------------------+    |
|  |  * Data source (path, type, columns)                |    |
|  |  * Model selection (xgboost, tft, nlinear)          |    |
|  |  * Hyperparameters (learning_rate, n_estimators)    |    |
|  |  * MLflow settings (tracking, registry)             |    |
|  +-----------------------------------------------------+    |
|    (example) configs/experiments/etth3.yaml                 |
+-------------------------------------------------------------+
                            +
+-------------------------------------------------------------+
|                    PIPELINE CONFIG                          |
|              (HOW to execute steps)                         |
|  +-----------------------------------------------------+    |
|  |  * Step definitions (id, type, depends_on)          |    |
|  |  * Data wiring (input/output key mapping)           |    |
|  |  * Execution flow (sequential, parallel, branch)    |    |
|  |  * Advanced patterns (sub-pipelines, conditions)    |    |
|  +-----------------------------------------------------+    |
|    (example) configs/pipelines/standard_train.yaml          |
+-------------------------------------------------------------+
                            |
                            V
+-------------------------------------------------------------+
|                    MERGED EXECUTION                         |
|  Same experiment config + Different pipeline configs        |
|  = Different workflows without changing code!               |
+-------------------------------------------------------------+
```
```
## Available Manual Pipeline Types

| Category   | Pipeline                     | Description               | Use Case                     |
| ---------- | ---------------------------- | ------------------------- | ----------------------------- |
| Standard   | `standard_train.yaml`       | Training + profiling     | Train model and profile output |
| Standard   | `standard_eval.yaml`        | Evaluation               | Evaluate model on test data    |
| Standard   | `standard_serve.yaml`       | Inference                | Production prediction         |
| Standard   | `standard_tune.yaml`        | Optuna tuning            | Hyperparameter optimization   |
| Advanced   | `kmeans_then_xgboost.yaml`  | 2-stage train            | Clustering + supervised train |
| Advanced   | `kmeans_then_xgboost_eval.yaml` | 2-stage eval        | Evaluate both stages          |
| Advanced   | `parallel_ensemble.yaml`    | Parallel training        | Train models concurrently     |
| Advanced   | `parallel_ensemble_eval.yaml` | Parallel eval         | Evaluate ensemble models      |
| Advanced   | `conditional_branch.yaml`   | Conditional training     | Adaptive model selection      |
| Advanced   | `conditional_branch_eval.yaml` | Conditional eval     | Evaluate selected branch      |
| Advanced   | `feature_engineering.yaml`  | Feature pipeline         | Modular feature engineering   |
| Advanced   | `nested_suppipeline.yaml`   | Nested pipeline          | Sub-pipeline orchestration    |
| Advanced   | `nested_suppipeline_eval.yaml` | Nested eval          | Evaluate nested pipeline      |
| Advanced   | `dynamic_adapter_train.yaml` | Dynamic wiring train   | Auto-wire inputs for training |
| Advanced   | `dynamic_adapter_eval.yaml` | Dynamic wiring eval      | Auto-wire inputs for eval     |

## Auto-Generate Eval/Serve Configs

Automatically generate evaluation and serving configs from your training config:

### Generate Both Configs

```bash
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/pipelines/standard_train.yaml \
    --output-dir mlproject/configs/generated \
    --alias latest

```

**Output:**
```
============================================================
[RUN] GENERATING CONFIGS
============================================================

[RUN] Source: mlproject/configs/pipelines/standard_train.yaml
[RUN] Output: mlproject/configs/generated
[ConfigGenerator] Successfully generated: mlproject/configs/generated/standard_train_eval.yaml
[ConfigGenerator] Successfully generated: mlproject/configs/generated/standard_train_serve.yaml
[ConfigGenerator] Successfully generated: mlproject/configs/generated/standard_train_tune.yaml

Generated configs:
  - Eval:  mlproject/configs/generated/standard_train_eval.yaml
  - Serve: mlproject/configs/generated/standard_train_serve.yaml
  - Tune: mlproject/configs/generated/standard_train_tune.yaml
```

## Basic Usage

### Training
```bash
# Long form, not using feast
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/etth3.yaml \
    --pipeline mlproject/configs/pipelines/standard_train.yaml
```
### Generation eval, serve, tune configs
```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/standard_train.yaml \
    -o mlproject/configs/generated \
    -a latest
```

### Evaluation
```bash
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/standard_train_eval.yaml \
    -a latest
```

### Serving (CSV input)
```bash
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/standard_train_serve.yaml \
    -i ./sample_input.csv \
    -a latest
```

### Hyperparameter Tuning
```bash
python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/standard_train_tune.yaml \
    -n 5  # number of trials
```

### Pipeline Flow with Profiling

```
load_data -> preprocess -> train_model -> evaluate -> profiling -> log_results
                                                       V
                                            [Profile Report]
                                            * Metrics summary
                                            * Cluster distributions
                                            * Prediction statistics
```

## Advanced Pipeline Examples

### 1. Two-Stage Pipeline (KMeans -> XGBoost)
Use clustering labels as additional features for classification.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/kmeans_then_xgboost.yaml

python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/kmeans_then_xgboost.yaml \
    -o mlproject/configs/generated \
    -a latest

python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/kmeans_then_xgboost_eval.yaml \
    -a latest

python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/kmeans_then_xgboost_serve.yaml \
    -i ./sample_input.csv \
    -a latest

python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/kmeans_then_xgboost_tune.yaml \
    -n 5
```

**Flow:**
```
load_data -> preprocess -> kmeans_features -> xgboost_model -> evaluate -> log
                              V
                    [cluster_labels]
```

### 2. Parallel Ensemble
Train multiple models simultaneously and evaluate each.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/parallel_ensemble.yaml

python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/parallel_ensemble.yaml \
    -o mlproject/configs/generated \
    -a latest

python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/parallel_ensemble_eval.yaml \
    -a latest

python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/parallel_ensemble_serve.yaml \
    -i ./sample_input.csv \
    -a latest

python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/parallel_ensemble_tune.yaml \
    -n 5
```

**Flow:**
```
```
load_data -> preprocess -> +-- xgboost_branch --> eval_xgb --+-> log
                         +-- catboost_branch --> eval_cat --+
                         +-- kmeans_branch ---+             |
                              (parallel)                    V
```
```

### 3. Conditional Branching
Automatically select model based on dataset size.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/conditional_branch.yaml

python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/conditional_branch.yaml \
    -o mlproject/configs/generated \
    -a latest

python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/conditional_branch_eval.yaml \
    -a latest

python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/conditional_branch_serve.yaml \
    -i ./sample_input.csv \
    -a latest

python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/conditional_branch_tune.yaml \
    -n 5
```

**Flow:**

load_data -> preprocess -> branch -+- [feature_columns_size < 4] -> TFT (deep learning)
                                 +- [feature_columns_size >= 4] -> XGBoost (ML)
                                          V
                                      evaluate -> log
Encapsulate feature engineering as a reusable sub-pipeline.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/nested_suppipeline.yaml

python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest

python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_eval.yaml \
    -a latest

python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_serve.yaml \
    -i ./sample_input.csv \
    -a latest

python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_tune.yaml \
    -n 5
```

**Flow:**
```
load_data -> feature_pipeline (sub) -> train_model -> evaluate -> log
                   V
           +--------------+
           |  normalize   |
           |      |       |
           |      V       |
           |   cluster    |
           | (features)   |
           +--------------+
```

#### 5. Dynamic Adapter

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/dynamic_adapter_train.yaml

python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/dynamic_adapter_train.yaml \
    -o mlproject/configs/generated \
    -a latest

python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/dynamic_adapter_train_eval.yaml \
    -a latest

python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/dynamic_adapter_train_serve.yaml \
    -i ./sample_input.csv \
    -a latest

python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/dynamic_adapter_train_tune.yaml \
    -n 5
```

## Data Wiring System

The DAG pipeline supports flexible data routing between steps via `wiring` configuration:

```yaml
- id: "train_model"
  type: "trainer"
  depends_on: ["preprocess", "clustering"]
  wiring:
    inputs:
      data: "preprocessed_data"      # Read from context["preprocessed_data"]
      features: "cluster_labels"     # Read from context["cluster_labels"]
    outputs:
      model: "final_model"           # Write to context["final_model"]
      datamodule: "final_dm"         # Write to context["final_dm"]
```

This enables:
- **Custom key mapping**: Override default input/output keys
- **Multi-input steps**: Combine outputs from multiple upstream steps

## Step Types Reference

| Type            | Description                         | Implementation                          |
| :-------------- | :---------------------------------- | :-------------------------------------- |
| `data_loader`   | Data ingestion (CSV/Feast)          | `steps/data/data_loader.py`             |
| `datamodule`    | Data processing and splitting       | `steps/data/datamodule.py`              |
| `preprocessor`  | Feature fitting and transformation  | `steps/data/preprocessor.py`            |
| `trainer`       | Core training execution             | `steps/models/trainer.py`               |
| `model`         | Framework-specific model logic      | `steps/models/framework_model.py`       |
| `mlflow_loader` | Model loading from MLflow registry  | `steps/mlops/mlflow_loader.py`          |
| `tuner`         | Hyperparameter optimization         | `steps/models/tuner.py`                 |
| `profiling`     | Data and output statistics          | `steps/mlops/profiling.py`              |
| `advanced`      | Complex control flow                | `steps/control/advanced.py`             |
| `adapter`       | Dynamic wiring and branching        | `steps/control/dynamic_adapter.py`      |
| `evaluator`     | Metrics and performance evaluation  | `steps/inference/evaluator.py`          |
| `inference`     | Prediction and inference logic      | `steps/inference/inference.py`          |
| `factory`       | Dynamic component generation        | `steps/core/factory.py`                 |
| `logger`        | Pipeline logging and tracking       | `steps/mlops/logger.py`                 |
