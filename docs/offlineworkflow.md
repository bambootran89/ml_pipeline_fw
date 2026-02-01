# Offline Workflow (Training Pipeline)

## Overview

The training pipeline orchestrates model development with emphasis on:
- **Anti-leakage**: Preprocessing fitted only on training folds
- **Reproducibility**: All experiments tracked in MLflow
- **Hyperparameter tuning**: Optuna integration
- **Cross-validation**: Time-series aware splitting
- **Dual PyFunc packaging**: Preprocessor + Model as separate but paired artifacts
- **Multi-type ML**: Supports timeseries, tabular, classification, regression

---

## Complete Training Flow

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TD
    %% Node styles
    classDef input fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef orchestrator fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef compute fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef package fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600
    classDef storage fill:#263238,stroke:#000,stroke-width:2.5px,color:#fff,font-weight:600
    classDef hotpath stroke:#2196f3,stroke-width:3.2px
    classDef parampath stroke:#0277bd,stroke-width:2.2px,stroke-dasharray:6 6
    classDef note fill:#fff3e0,stroke:#ef6c00,stroke-width:1.5px,color:#000,font-style:italic,font-size:14px

    %% Phases
    subgraph Phase1["1) Config & Data Loading"]
        direction TB
        Config["Hydra Config<br/>Model/Data/Training"]:::input
        DataIO["DataIO Factory<br/>FeastLoader/FileLoader"]:::input
        DataModule["DataModule Factory<br/>TSDataModule/MLDataModule"]:::input
    end

    subgraph Phase2["2) Data Splitting Strategy"]
        direction TB
        Splitter["Splitter Strategy<br/>TimeSeriesSplitter/KFoldSplitter"]:::orchestrator
        Split["Train / Val / Test Indices"]:::orchestrator
    end

    subgraph Phase3["3) Feature Transform (Per Fold)"]
        direction LR
        TransFit["TransformManager.fit()<br/>on TRAIN only"]:::compute
        TransTrain["transform() TRAIN"]:::compute
        TransVal["transform() VAL"]:::compute
    end

    subgraph Phase4["4) Model Training"]
        direction TB
        ModelFactory["ModelFactory<br/>XGBoost/NLinear/TFT/CatBoost"]:::orchestrator
        Trainer["Trainer (ML/DL)<br/>fit() with early stopping"]:::compute
        Evaluate["Evaluate on VAL<br/>MAE/RMSE/Accuracy"]:::compute
    end

    subgraph Phase5["5) Hyperparameter Tuning (Optional)"]
        direction TB
        Optuna["Optuna Study<br/>Bayesian optimization"]:::orchestrator
        Suggest["Suggest hyperparams"]:::orchestrator
        Report["Report metrics"]:::orchestrator
    end

    subgraph Phase6["6) Serialize Components"]
        direction LR
        SerPrep["Serialize<br/>Preprocessor.pkl"]:::package
        SerModel["Serialize<br/>Model.pkl/.pt"]:::package
    end

    subgraph Phase7["7) MLflow Logging (Dual PyFunc)"]
        direction TB
        LogPrep["Log xgboost_preprocessor<br/>PyFunc v3"]:::storage
        LogModel["Log xgboost_model<br/>PyFunc v3"]:::storage
        Alias["Assign SAME alias<br/>@production -> v3"]:::storage
    end

    %% Main flow
    Config ==> DataIO:::hotpath
    DataIO ==> DataModule:::hotpath
    DataModule ==> Splitter:::hotpath
    Splitter ==> Split:::hotpath

    Split -->|"Train idx"| TransFit:::hotpath
    Split -->|"Val idx"| TransVal:::hotpath
    TransFit ==> TransTrain:::hotpath
    TransFit ==> TransVal:::hotpath

    TransTrain ==> ModelFactory:::hotpath
    ModelFactory ==> Trainer:::hotpath
    TransVal ==> Evaluate:::hotpath
    Trainer ==> Evaluate:::hotpath

    TransFit --> SerPrep:::hotpath
    Trainer --> SerModel:::hotpath

    SerPrep ==> LogPrep:::hotpath
    SerModel ==> LogModel:::hotpath
    LogPrep & LogModel ==> Alias:::hotpath

    %% Optuna flow
    Optuna -.->|"Create study"| Suggest:::parampath
    Suggest -.->|"Hyperparams"| Trainer:::parampath
    Evaluate -.->|"Metrics"| Report:::parampath
    Report -.->|"Next trial"| Suggest:::parampath

    %% Notes
    N1["No Leakage:<br/>Stats only from Train"]:::note
    N2["Dual PyFunc:<br/>Separate but paired"]:::note
    N3["Loop:<br/>Each CV fold"]:::note
    N4["Best Model:<br/>Retrain on full"]:::note

    N1 -.-> TransFit
    N2 -.-> Phase7
    N3 -.-> Splitter
    N4 -.-> Alias

    %% Global edge style
    linkStyle default stroke:#2196f3,stroke-width:5px
```

---

## Phase 1: Configuration & Data Loading

### 1.1 Architecture Overview

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef config fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef factory fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef loader fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef module fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600
    classDef data fill:#fbe9e7,stroke:#d84315,stroke-width:2.2px,color:#000,font-weight:600

    %% Config
    subgraph Config["Configuration"]
        HydraConf["Hydra Config<br/>experiments/etth1_feast.yaml"]:::config
    end

    %% DataIO Layer (Strategy Pattern)
    subgraph DataIO["DataIO Layer (Strategy Pattern)"]
        DataIOFactory["DataLoaderFactory<br/>detect source type"]:::factory

        FeastLoader["FeastDataLoader<br/>feast://repo"]:::loader
        FileLoader["FileDataLoader<br/>.csv/.parquet"]:::loader
        DBLoader["DatabaseLoader<br/>SQL queries"]:::loader
    end

    %% DataModule Layer (Type-specific)
    subgraph DataModule["DataModule Layer (ML Type)"]
        DataModuleFactory["DataModuleFactory<br/>detect ML type"]:::factory

        TSDataModule["TSDataModule<br/>Timeseries:<br/>- Windowing<br/>- Sequences"]:::module
        MLDataModule["MLDataModule<br/>Tabular:<br/>- Feature matrix<br/>- Classification/Regression"]:::module
    end

    %% Output
    subgraph Output["Output"]
        TrainData["Train DataFrame<br/>Features + Targets"]:::data
        ValData["Val DataFrame"]:::data
        TestData["Test DataFrame"]:::data
    end

    %% Flow
    HydraConf ==>|"data.path"| DataIOFactory
    DataIOFactory -->|"feast://"| FeastLoader
    DataIOFactory -->|"*.csv"| FileLoader
    DataIOFactory -->|"postgres://"| DBLoader

    FeastLoader ==> DataModuleFactory
    FileLoader ==> DataModuleFactory
    DBLoader ==> DataModuleFactory

    DataModuleFactory -->|"type: timeseries"| TSDataModule
    DataModuleFactory -->|"type: tabular"| MLDataModule

    TSDataModule ==> TrainData
    TSDataModule ==> ValData
    TSDataModule ==> TestData

    MLDataModule ==> TrainData
    MLDataModule ==> ValData
    MLDataModule ==> TestData

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

### 1.2 DataIO Pattern (Source Abstraction)

**Purpose:** Abstract away data source complexity using Strategy Pattern.

**Directory Structure:**
```
src/datamodule/loaders/
|-- base.py              # BaseDataLoader interface
|-- feast_loader.py      # Feast integration
|-- file_loader.py       # CSV/Parquet
|-- database_loader.py   # SQL databases
+-- factory.py           # DataLoaderFactory
```

**Config Examples:**
```yaml
# Feast source
data:
  path: "feast://weather_repo"
  featureview: "hourly_features"
  features: ["HUFL", "MUFL", "mobility_inflow"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

# File source
data:
  path: "mlproject/data/ETTh1.csv"
  index_col: "date"
  features: ["HUFL", "MUFL", "HULL"]
  target_columns: ["OT"]

# Database source
data:
  path: "postgresql://user:pass@host/db"
  query: "SELECT * FROM timeseries WHERE date > '2023-01-01'"
```

**Factory Logic:**
```python
class DataLoaderFactory:
    @staticmethod
    def create(config):
        path = config.data.path

        if path.startswith("feast://"):
            return FeastDataLoader(config)
        elif path.endswith((".csv", ".parquet")):
            return FileDataLoader(config)
        elif path.startswith(("postgresql://", "mysql://")):
            return DatabaseLoader(config)
        else:
            raise ValueError(f"Unsupported path: {path}")
```

**Benefits:**
- Single interface for multiple sources
- Easy to add new sources
- Config-driven selection

### 1.3 DataModule Pattern (ML Type Abstraction)

**Purpose:** Handle different ML types with appropriate data transformations.

**Directory Structure:**
```
src/datamodule/
|-- base.py              # BaseDataModule interface
|-- ts_datamodule.py     # Time-series
|-- ml_datamodule.py     # Tabular (classification/regression)
|-- splitter.py          # Splitting strategies
+-- factory.py           # DataModuleFactory
```

**ML Type Support:**

| ML Type | DataModule | Models | Use Case |
|---------|-----------|--------|----------|
| **timeseries** | TSDataModule | NLinear, TFT, LSTM | Forecasting, sequences |
| **tabular** | MLDataModule | XGBoost, CatBoost, RF | Classification, regression |

**DataModule Responsibilities:**

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart LR
    %% Styles
    classDef input fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef ts fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef ml fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef output fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600

    %% Input
    RawData["Raw DataFrame<br/>From DataLoader"]:::input

    %% TSDataModule
    subgraph TS["TSDataModule (Timeseries)"]
        TSWindow["Create windows<br/>input_chunk: 24<br/>output_chunk: 6"]:::ts
        TSSeq["Create sequences<br/>[t-24:t] -> [t+1:t+6]"]:::ts
    end

    %% MLDataModule
    subgraph ML["MLDataModule (Tabular)"]
        MLFeature["Feature matrix<br/>X: features<br/>y: target"]:::ml
        MLSplit["Standard split<br/>Train/Val/Test"]:::ml
    end

    %% Output
    TSOutput["3D Tensors<br/>(batch, time, features)"]:::output
    MLOutput["2D Arrays<br/>(batch, features)"]:::output

    %% Flow
    RawData -->|"type: timeseries"| TSWindow
    TSWindow --> TSSeq
    TSSeq --> TSOutput

    RawData -->|"type: tabular"| MLFeature
    MLFeature --> MLSplit
    MLSplit --> MLOutput

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Code Example:**
```python
# Factory creates appropriate DataModule
datamodule_factory = DataModuleFactory()
datamodule = datamodule_factory.create(
    data_type=config.data.type,  # "timeseries" or "tabular"
    config=config
)

# TSDataModule output (timeseries)
if config.data.type == "timeseries":
    # Shape: (batch, time, features)
    X_train.shape  # (1000, 24, 3)  # 24 timesteps, 3 features
    y_train.shape  # (1000, 6, 1)   # Predict 6 future steps

# MLDataModule output (tabular)
else:
    # Shape: (batch, features)
    X_train.shape  # (1000, 10)  # 10 features
    y_train.shape  # (1000,)     # Single target
```

---

## Phase 2: Data Splitting Strategy

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef data fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef splitter fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef fold fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600

    %% Input
    FullData["Full Dataset<br/>From DataModule"]:::data

    %% Splitter Strategy
    subgraph Splitter["Splitter Strategy"]
        TSSplitter["TimeSeriesSplitter<br/>Expanding window<br/>No future leakage"]:::splitter
        KFoldSplitter["KFoldSplitter<br/>Stratified splits<br/>For tabular"]:::splitter
    end

    %% Folds (Timeseries)
    subgraph TSFolds["Time-Series Folds (Expanding)"]
        F1["Fold 1<br/>Train: Day 1-100<br/>Val: Day 101-130"]:::fold
        F2["Fold 2<br/>Train: Day 1-130<br/>Val: Day 131-160"]:::fold
        F3["Fold 3<br/>Train: Day 1-160<br/>Val: Day 161-190"]:::fold
    end

    %% Folds (KFold)
    subgraph KFFolds["K-Fold (Tabular)"]
        K1["Fold 1<br/>80% Train / 20% Val"]:::fold
        K2["Fold 2<br/>80% Train / 20% Val"]:::fold
        K3["Fold 3<br/>80% Train / 20% Val"]:::fold
    end

    %% Flow
    FullData -->|"type: timeseries"| TSSplitter
    FullData -->|"type: tabular"| KFoldSplitter

    TSSplitter --> F1 & F2 & F3
    KFoldSplitter --> K1 & K2 & K3

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Timeline Visual (Timeseries):**
```
Timeline: =========================================>
          Day 1                            Day 365

Fold 1:   [========Train========][==Val==]
Fold 2:   [===========Train===========][==Val==]
Fold 3:   [===============Train===============][==Val==]

Note: No overlap between folds
Note: Validation always uses future data
Note: Train size increases (model sees more data)
```

**Directory Structure:**
```
src/datamodule/
|-- splitter.py
    |-- TimeSeriesSplitter      # Expanding window
    |-- KFoldSplitter           # Standard K-Fold
    +-- StratifiedKFoldSplitter # For classification
```

---

## Phase 3: Feature Engineering (Per Fold)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef data fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef fit fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:700
    classDef transform fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef stats fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600
    classDef warning fill:#ffebee,stroke:#c62828,stroke-width:2.2px,color:#000,font-weight:700

    %% Raw Data
    TrainRaw["Train Data (Raw)<br/>From split indices"]:::data
    ValRaw["Val Data (Raw)<br/>From split indices"]:::data

    %% Fit Phase
    subgraph FitPhase["Fit Phase (TRAIN ONLY)"]
        Fit["TransformManager.fit(train_df)<br/>Learn statistics"]:::fit

        subgraph Stats["Learned Statistics"]
            Mean["Mean: [25.5, 60.2, ...]"]:::stats
            Std["Std: [5.1, 10.3, ...]"]:::stats
            Encoders["LabelEncoders<br/>{A: 0, B: 1}"]:::stats
        end
    end

    %% Transform Phase
    subgraph TransformPhase["Transform Phase"]
        TransTrain["transform(train_df)<br/>Apply learned stats"]:::transform
        TransVal["transform(val_df)<br/>Apply SAME stats"]:::transform
    end

    %% Transformed Data
    TrainClean["Train (Transformed)<br/>Scaled, encoded"]:::data
    ValClean["Val (Transformed)<br/>Using TRAIN stats"]:::data

    %% Flow
    TrainRaw ==> Fit
    Fit ==> Mean & Std & Encoders

    Mean & Std & Encoders ==> TransTrain
    Mean & Std & Encoders ==> TransVal

    TrainRaw --> TransTrain
    ValRaw --> TransVal

    TransTrain ==> TrainClean
    TransVal ==> ValClean

    %% Warning
    Warning["CRITICAL:<br/>Val NEVER in fit()<br/>Prevents leakage"]:::warning
    Warning -.-> Fit

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Directory Structure:**
```
src/preprocess/
|-- transform_manager.py     # Orchestrator
|-- transforms/
|   |-- fill_missing.py      # Imputation (stateful)
|   |-- normalize.py         # Scaling (stateful)
|   |-- label_encoding.py    # Encoding (stateful)
|   +-- math_transforms.py   # Log, clip (stateless)
+-- offline.py               # Offline preprocessor
```

**Anti-Leakage Guarantee:**
```python
# CORRECT: Fit only on train
transform_manager.fit(train_df)  # Learn mean, std from train
train_transformed = transform_manager.transform(train_df)
val_transformed = transform_manager.transform(val_df)  # Use TRAIN stats

# WRONG: Fit on all data (LEAKAGE!)
all_df = pd.concat([train_df, val_df])
transform_manager.fit(all_df)  # This is data leakage!
```

---

## Phase 4: Model Training (Multi-Type Support)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef config fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef factory fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef trainer fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600

    %% Config
    Config["Hydra Config<br/>experiment.model: xgboost"]:::config

    %% Model Factory
    subgraph Factory["ModelFactory"]
        ModelFactory["create(model_name, hyperparams)"]:::factory
    end

    %% Model Wrappers
    subgraph Models["Model Wrappers (MLModelWrapper)"]
        subgraph TSModels["Time-Series Models"]
            NLinear["NLinear<br/>Linear baseline"]:::model
            TFT["TFT<br/>Temporal Fusion<br/>Transformer"]:::model
            LSTM["LSTM<br/>Recurrent NN"]:::model
        end

        subgraph MLModels["Tabular Models"]
            XGBoost["XGBoost<br/>Gradient boosting"]:::model
            CatBoost["CatBoost<br/>Categorical support"]:::model
            RF["RandomForest<br/>Ensemble"]:::model
        end
    end

    %% Trainer Factory
    subgraph TrainerF["TrainerFactory"]
        TrainerFactory["create(model_type)"]:::factory

        MLTrainer["MLTrainer<br/>Sklearn-style<br/>fit(X, y)"]:::trainer
        DLTrainer["DLTrainer<br/>PyTorch<br/>Training loops"]:::trainer
    end

    %% Flow
    Config ==> ModelFactory

    ModelFactory -->|"timeseries"| NLinear & TFT & LSTM
    ModelFactory -->|"tabular"| XGBoost & CatBoost & RF

    NLinear & TFT & LSTM ==> TrainerFactory
    XGBoost & CatBoost & RF ==> TrainerFactory

    TrainerFactory -->|"DL models"| DLTrainer
    TrainerFactory -->|"ML models"| MLTrainer

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Directory Structure:**
```
src/models/
|-- base.py                  # MLModelWrapper interface
|-- xgboost_wrapper.py       # XGBoost (tabular)
|-- catboost_wrapper.py      # CatBoost (tabular)
|-- nlinear_wrapper.py       # NLinear (timeseries)
|-- tft_wrapper.py           # TFT (timeseries)
+-- model_factory.py         # ModelFactory

src/trainer/
|-- base.py                  # BaseTrainer
|-- ml_trainer.py            # Sklearn-style
|-- dl_trainer.py            # PyTorch
+-- trainer_factory.py       # TrainerFactory
```

**Unified Interface:**
```python
class MLModelWrapper(ABC):
    """Unified interface for all models"""

    @abstractmethod
    def build(self, model_type: str) -> None:
        """Initialize model architecture"""
        pass

    @abstractmethod
    def fit(self, x, y, x_val=None, y_val=None, **kwargs) -> None:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, x, **kwargs) -> np.ndarray:
        """Run inference"""
        pass
```

---

## Phase 5: Hyperparameter Tuning (Optuna)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart TB
    %% Styles
    classDef optuna fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef training fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef eval fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef best fill:#f3e5f5,stroke:#6a1b9a,stroke-width:3px,color:#000,font-weight:700

    %% Optuna Setup
    subgraph OptunaSetup["Optuna Study"]
        Study["create_study()<br/>optimize_metric: MAE<br/>direction: minimize<br/>n_trials: 30"]:::optuna
    end

    %% Trial Loop
    subgraph TrialLoop["Trial Loop (30 iterations)"]
        Suggest["trial.suggest_int()<br/>trial.suggest_float()<br/>n_estimators, learning_rate, etc."]:::optuna

        subgraph Training["Training"]
            Build["ModelFactory.create()<br/>with suggested params"]:::training
            Fit["Trainer.fit(train, val)"]:::training
        end

        subgraph Evaluation["Evaluation"]
            Predict["model.predict(val)"]:::eval
            Metrics["Compute MAE, RMSE"]:::eval
            Report["trial.report(mae)"]:::eval
        end
    end

    %% Best Model
    subgraph BestModel["Best Model Selection"]
        SelectBest["study.best_trial<br/>lowest MAE"]:::best
        Retrain["Retrain on full data<br/>(Train + Val)"]:::best
        Final["Final Model<br/>Ready for deployment"]:::best
    end

    %% Flow
    Study ==> Suggest
    Suggest ==> Build
    Build ==> Fit
    Fit ==> Predict
    Predict ==> Metrics
    Metrics ==> Report

    Report -.->|"Next trial"| Suggest
    Report ==>|"After 30 trials"| SelectBest

    SelectBest ==> Retrain
    Retrain ==> Final

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Config Example:**
```yaml
tuning:
  n_trials: 30
  n_splits: 3
  optimize_metric: "mae_mean"
  direction: "minimize"
  n_jobs: 1

  # Search space for XGBoost
  xgboost:
    n_estimators:
      type: "int"
      range: [50, 300]
      step: 50
    learning_rate:
      type: "float"
      range: [0.01, 0.1]
      log: true
    max_depth:
      type: "int"
      range: [3, 10]
```

**Directory Structure:**
```
src/tuning/
|-- optuna_tuner.py          # Optuna integration
|-- search_space.py          # Parameter definitions
+-- callbacks.py             # Early stopping, pruning
```

---

## Phase 6 & 7: Serialization & MLflow Logging

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 10,
    "rankSpacing": 50,
    "curve": "basis",
    "padding": 10
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "12px"
  }
}}%%

flowchart LR
    %% Styles
    classDef component fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef serialized fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef pyfunc fill:#f3e5f5,stroke:#6a1b9a,stroke-width:3px,color:#000,font-weight:700
    classDef mlflow fill:#263238,stroke:#000,stroke-width:2.5px,color:#fff,font-weight:600

    %% Components
    subgraph Components["Trained Components"]
        Preprocessor["TransformManager<br/>Fitted scalers, encoders"]:::component
        Model["Trained Model<br/>XGBoost/TFT weights"]:::component
    end

    %% Serialization
    subgraph Serialization["Serialization"]
        PrepPkl["preprocessor.pkl<br/>pickle.dump()"]:::serialized
        ModelPkl["model.pkl or model.pt"]:::serialized
    end

    %% PyFunc Wrappers (Separate)
    subgraph PyFunc["Separate PyFunc Models"]
        PrepPyFunc["PreprocessorPyFunc<br/>predict() calls transform()"]:::pyfunc
        ModelPyFunc["ModelPyFunc<br/>predict() calls model.predict()"]:::pyfunc
    end

    %% MLflow
    subgraph MLflow["MLflow Registry"]
        PrepReg["xgboost_preprocessor<br/>Version: 3<br/>Alias: production"]:::mlflow
        ModelReg["xgboost_model<br/>Version: 3<br/>Alias: production"]:::mlflow
    end

    %% Flow
    Preprocessor ==> PrepPkl
    Model ==> ModelPkl

    PrepPkl ==> PrepPyFunc
    ModelPkl ==> ModelPyFunc

    PrepPyFunc ==> PrepReg
    ModelPyFunc ==> ModelReg

    %% Global styling
    linkStyle default stroke:#2196f3,stroke-width:5px
```

**Logging Code:**
```python
# 1. Serialize components
import pickle

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(transform_manager, f)

with open("model.pkl", "wb") as f:
    pickle.dump(trained_model, f)

# 2. Wrap as separate PyFunc models
prep_pyfunc = PreprocessorPyFunc(preprocessor_path="preprocessor.pkl")
model_pyfunc = ModelPyFunc(model_path="model.pkl")

# 3. Log as separate registered models
model_name = "xgboost"

mlflow.pyfunc.log_model(
    artifact_path="preprocessor",
    python_model=prep_pyfunc,
    registered_model_name=f"{model_name}_preprocessor"
)
# Creates: xgboost_preprocessor v3

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=model_pyfunc,
    registered_model_name=f"{model_name}_model"
)
# Creates: xgboost_model v3

# 4. Assign SAME alias to BOTH (pairing!)
version = 3
alias = "production"

mlflow.set_registered_model_alias(
    name=f"{model_name}_preprocessor",
    alias=alias,
    version=version
)

mlflow.set_registered_model_alias(
    name=f"{model_name}_model",
    alias=alias,
    version=version
)
```

**MLflow UI Structure:**
```
Models/
|-- xgboost_preprocessor
|   |-- Version 3 (production, latest)
|   |-- Version 2 (staging)
|   +-- Version 1
|
+-- xgboost_model
    |-- Version 3 (production, latest)  # Paired with preprocessor v3
    |-- Version 2 (staging)             # Paired with preprocessor v2
    +-- Version 1                       # Paired with preprocessor v1
```

---

## Summary: Complete Pipeline

**Command:**
```bash
python -m mlproject.src.pipeline.run train \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

**Execution Steps:**

1. **Load Config** (Hydra composition)
   - Base configs + experiment overrides

2. **Load Data** (DataIO + DataModule)
   - DataLoaderFactory detects source (Feast/CSV/DB)
   - DataModuleFactory creates appropriate module (TS/ML)

3. **Split Data** (Splitter Strategy)
   - TimeSeriesSplitter for timeseries
   - KFoldSplitter for tabular

4. **For Each Fold:**
   - Fit preprocessor on TRAIN indices only
   - Transform TRAIN and VAL using fitted preprocessor
   - ModelFactory creates model (XGBoost/NLinear/TFT/etc.)
   - TrainerFactory creates appropriate trainer (ML/DL)
   - Train model with early stopping on VAL
   - Evaluate metrics (MAE, RMSE, Accuracy)
   - Optuna suggests next hyperparameters (if tuning)

5. **Select Best** (After all folds/trials)
   - Choose best hyperparameters
   - Retrain on full dataset (Train + Val)

6. **Serialize**
   - Save preprocessor.pkl
   - Save model.pkl or model.pt

7. **Log to MLflow** (Dual PyFunc)
   - Log xgboost_preprocessor v3 (PyFunc)
   - Log xgboost_model v3 (PyFunc)
   - Assign SAME alias to both (production/staging/latest)

**Output:**
- Two registered models in MLflow
- Paired by version and alias
- Ready for deployment
- Fully reproducible

**Key Guarantees:**
- **No data leakage**: Val never seen during preprocessing fit
- **Dual PyFunc**: Separate but paired models
- **Reproducible**: Config + Version -> Exact result
- **Multi-type**: Supports timeseries and tabular ML
- **Flexible**: Easy to add new models, data sources, splitters

**Directory Structure Utilized:**
```
mlproject/src/
|-- datamodule/
|   |-- loaders/        # DataIO (Feast/File/DB)
|   |-- ts_datamodule   # Timeseries handling
|   +-- ml_datamodule   # Tabular handling
|-- preprocess/         # TransformManager
|-- models/             # Model wrappers
|-- trainer/            # Training loops
|-- tuning/             # Optuna integration
+-- tracking/           # MLflow integration
```
