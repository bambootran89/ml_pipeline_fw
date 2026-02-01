# Preprocessing & Data Transformation

The **TransformManager** module provides a **stateful preprocessing pipeline**. It ensures that all statistical parameters (mean, median, scaling factors, etc.) are **learned exclusively from the training set** and **consistently applied** to validation, test, and inference data—preventing **data leakage**.

This module is **intentionally limited to simple, generic preprocessing operations** (e.g., imputation, scaling, basic mathematical transforms). **Complex feature engineering, business logic, and domain-specific transformations must be implemented upstream in a dedicated Feature Store**, not in this repository, to ensure reuse, governance, and consistency across models.

Configuration is handled **declaratively** via `preprocessing.yaml`.

---

## Key Features

* **Sequential Pipeline**
  Define steps in a list (`steps`), executed in the **exact order** specified.

* **Stateful Consistency**
  Learned statistics (e.g., mean for imputation, std for z-score) are **serialized (`.pkl`)** and reloaded during serving.

* **No-Code Configuration**
  Manage complex transformations purely through **YAML**, without touching Python code.

---

## Configuration Preprocessing Guide

Define your pipeline in:

```
configs/base/preprocessing.yaml
```

The system supports **two types of transforms**:

* **Stateful** (requires fitting on training data)
* **Stateless** (pure mathematical operations)

---

## Stateful Transforms (Requires Fit)

These steps compute statistics from the **training data** and reuse them during inference.

| Step Name        | Method   | Config Parameters | Description                                                               |
| ---------------- | -------- | ----------------- | ------------------------------------------------------------------------- |
| `fill_missing`   | `mean`   | `columns`         | Imputes missing values with the column mean.                              |
|                  | `median` | `columns`         | Imputes missing values with the column median.                            |
|                  | `mode`   | `columns`         | Imputes missing values with the most frequent value.                      |
| `label_encoding` | *(None)* | `columns`         | Encodes categorical features into integers. Handles unseen labels safely. |
| `normalize`      | `zscore` | `columns`         | StandardScaler: scales data to $\mu = 0$, $\sigma = 1$.                   |
|                  | `minmax` | `columns`         | MinMaxScaler: scales data to the range $[0, 1]$.                          |

---

## Stateless Transforms (Math Functions)

These steps apply **deterministic mathematical transformations** and do not require fitting.

| Step Name     | Config Keys  | Default | Description                                                         |
| ------------- | ------------ | ------- | ------------------------------------------------------------------- |
| `log`         | `offset`     | `0.0`   | Applies $\ln(x + \text{offset})$. Raises error if value $\le 0$. |
| `clip`        | `min`, `max` | `None`  | Clips values to the `[min, max]` range.                             |
| `binary`      | `threshold`  | `0.0`   | Converts to `1` if $x > \text{threshold}$, else `0`.                |
| `round`       | `decimals`   | `0`     | Rounds numbers to fixed decimal places.                             |
| `exponential` | `scale`      | `1.0`   | Applies $e^{(\text{scale} \times x)}$.                              |
| `abs`         | *(None)*     | –       | Computes the absolute value.                                        |

---

## Example Configuration (YAML)

```yaml
preprocessing:
  steps:
    # 1. Impute missing values (Stateful)
    - name: fill_missing
      method: median
      columns: ["age", "income", "household_size"]

    # 2. Clip outliers (Stateless)
    - name: clip
      columns: ["age"]
      min: 18
      max: 90

    # 3. Log transform skewed features (Stateless)
    # Formula: ln(revenue + 1.0)
    - name: log
      columns: ["revenue"]
      offset: 1.0

    # 4. Normalize numerical features (Stateful)
    - name: normalize
      method: zscore
      columns: ["age", "income", "revenue"]
```

---

## Usage in Code

The workflow is divided into **Training (Fit & Save)** and **Inference (Load & Transform)**.

### Python

```python
from mlproject.src.preprocess.transform_manager import TransformManager

# --- Training Phase ---
tm = TransformManager(cfg, artifacts_dir="artifacts/")

# Fit on training data and save state
tm.fit(train_df, config)
tm.save()

# Transform datasets
train_transformed = tm.transform(train_df)
val_transformed = tm.transform(val_df)

# --- Inference Phase ---
# Initialize and load the saved state
prod_tm = TransformManager(artifacts_dir="artifacts/")
prod_tm.load(config)

# Transform new data using training statistics
prediction_input = prod_tm.transform(new_data_df)
```

---

## Technical N

* **Execution Order**
  Steps are executed sequentially as defined in the YAML list. It is strongly recommended to place `fill_missing` **before** any mathematical transforms.

* **Log Transform Safety**
  The `log` transform does **not** automatically clip negative values. Ensure that the input (plus optional offset) is **strictly positive**, otherwise a `ValueError` will be raised.

* **Label Encoding**
  During inference, unseen labels are mapped to the **first known class (index 0)** to prevent runtime failures.
