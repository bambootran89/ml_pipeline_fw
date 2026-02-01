# Directory Structure

## Overview

The project structure is designed for **scalability, modularity, and feature-store integration**. Each directory has a clear responsibility following **separation of concerns** principle.


## Complete Directory Tree

```text
mlproject/
|
|-- configs/                     # Hydra configuration hub
|   |-- base/                    # Base configs (shared)
|   |   |-- model.yaml
|   |   |-- training.yaml
|   |   |-- preprocessing.yaml
|   |   |-- evaluation.yaml
|   |   |-- mlflow.yaml
|   |   +-- tuning.yaml
|   +-- experiments/             # Experiment-specific overrides
|       |-- etth1.yaml
|       |-- etth1_feast.yaml
|       |-- etth2_tuning.yaml
|       +-- titanic_feast.yaml
|
|-- data/                        # Raw data (git-ignored)
|   |-- ETTh1.csv
|   |-- titanic.csv
|   +-- README.md
|
|-- artifacts/                    # Generated artifacts (git-ignored)
|   |-- models/
|   |   |-- model.pkl
|   |   +-- model.pt
|   +-- preprocessing/
|       |-- fillna_stats.pkl
|       |-- label_encoders.pkl
|       +-- scaler.pkl
|
|-- serve/                        # Serving layer
|   |-- api.py                    # FastAPI app
|   |-- ray_deploy.py             # Ray Serve deployment
|   |-- models_service.py         # Inference logic
|   +-- schemas.py                # Request/response schemas
|
|-- src/                          # Core ML logic
|   |-- generator/                # Config & API generation
|   |   |-- orchestrator.py       # Main entry point (ConfigGenerator)
|   |   |-- config.py             # Generator configuration
|   |   |-- constants.py          # Generator constants
|   |   |-- api_generator.py      # API generation orchestrator
|   |   |-- apis/                 # API code generators
|   |   |   |-- generator.py      # Core generator logic
|   |   |   |-- fastapi_generator.py
|   |   |   |-- rayserve_generator.py
|   |   |   |-- extractors.py     # Feature extraction utilities
|   |   |   +-- types.py          # Type definitions
|   |   +-- pipeline/             # Pipeline transformation
|   |       |-- builders/         # Pipeline builders (grouped)
|   |       |   |-- base.py       # BasePipelineBuilder
|   |       |   |-- eval.py       # EvalBuilder
|   |       |   |-- serve.py      # ServeBuilder
|   |       |   |-- tune.py       # TuneBuilder
|   |       |   +-- loader.py     # LoaderBuilder
|   |       |-- dependency.py     # Dependency resolution
|   |       |-- feature_parser.py # Feature pipeline parser
|   |       |-- step_analyzer.py  # Step analysis
|   |       +-- step_transformer.py
|   |
|   |-- pipeline/                 # DAG-based pipeline execution
|   |   |-- dag_run.py            # Main DAG runner
|   |   |-- executor.py           # Pipeline executor
|   |   |-- compat/               # Backward compatibility
|   |   |   +-- v1/               # Legacy v1.run interface
|   |   |-- feature_ops/          # Feature store operations
|   |   +-- steps/                # Pipeline step implementations
|   |       |-- core/             # Core framework
|   |       |   |-- base.py       # BasePipelineStep
|   |       |   |-- factory.py    # StepFactory
|   |       |   |-- constants.py  # Step constants (ContextKeys, DataTypes, etc.)
|   |       |   +-- utils.py      # Utilities (ConfigAccessor, WindowBuilder, etc.)
|   |       |-- handlers/         # Data type handlers
|   |       |   +-- data_handlers.py  # TimeseriesHandler, TabularHandler
|   |       |-- data/             # Data pipeline steps
|   |       |   |-- data_loader.py
|   |       |   |-- preprocessor.py
|   |       |   +-- datamodule.py
|   |       |-- features/         # Feature engineering steps
|   |       |   |-- composer.py
|   |       |   +-- inference.py
|   |       |-- models/           # Model training steps
|   |       |   |-- framework_model.py
|   |       |   |-- trainer.py
|   |       |   +-- tuner.py
|   |       |-- inference/        # Inference steps
|   |       |   |-- inference.py
|   |       |   +-- evaluator.py
|   |       |-- mlops/            # MLOps steps
|   |       |   |-- logger.py
|   |       |   |-- mlflow_loader.py
|   |       |   +-- profiling.py
|   |       +-- control/          # Control flow steps
|   |           |-- advanced.py   # ParallelStep, BranchStep, SubPipelineStep
|   |           +-- dynamic_adapter.py
|   |
|   |-- datamodule/               # Data loading & splitting
|   |-- eval/                     # Evaluation metrics & runners
|   |-- features/                 # Feature store (Feast)
|   |   |-- definitions/          # Feature definitions
|   |   |-- transformers/         # Feature transformers
|   |   +-- examples/             # Usage examples
|   |-- models/                   # Model implementations
|   |-- preprocess/               # Data transformation & preprocessing
|   |-- tracking/                 # MLflow integration
|   |-- trainer/                  # Training loops (ML & DL)
|   |-- tuning/                   # Hyperparameter tuning (Optuna)
|   +-- utils/                    # Helper functions
|
|-- tests/                        # Test suite
|   |-- test_datamodule.py
|   |-- test_models.py
|   |-- test_pipeline.py
|   +-- ...
|
|-- docs/                         # Documentation
|   |-- architecture.md
|   |-- offlineworkflow.md
|   |-- onlineworkflow.md
|   +-- preprocessing.md
|
|-- Docker & Deployment
|   |-- Dockerfile
|   |-- docker-compose.yml
|   +-- .dockerignore
|
|-- Project Files
|   |-- requirements/prod.txt
|   |-- requirements/dev.txt
|   |-- setup.py
|   |-- pyproject.toml
|   |-- Makefile
|   |-- .gitignore
|   +-- README.md
|
+-- Generated runtime
    |-- mlruns/                   # MLflow tracking
    +-- .ray/                     # Ray runtime files
```
