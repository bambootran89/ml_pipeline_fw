# Feature Store Examples Guide

This directory provides **end-to-end demonstrations** of using the Feast-backed Feature Store framework within a **modern, standardized MLOps architecture**.
The examples illustrate best practices for organizing feature **definitions**, **transformations**, **ingestion**, and **retrieval** for both **offline training workloads** and **online inference APIs**.

---

## 1. Problem Domains Demonstrated

The Feature Store framework is applied across four widely-used machine learning domains:

| Domain | Example Script | Key Focus |
|---|---|---|
| **Fraud Detection** | `fraud.py` | Real-time risk scoring on high-velocity transaction data, PIT-consistent training joins |
| **Recommendation System** | `recsys.py` | User-item interaction features with ultra-low latency for ranking models |
| **Time-Series Forecasting** | `forecast.py` | Sensor telemetry transformations including lag and rolling window aggregations |
| **Station Sequence Modeling** | `station.py` | Sequence retrieval using a specialized wrapper (`TimeSeriesFeatureStore`) for LSTM/Transformer models |

---

## 2. Design Philosophy

All examples follow the **Modern MLOps — Separation of Concerns** principle:

### 📌 Component Responsibilities

| Layer | Directory / File Location | Responsibility |
|---|---|---|
| **Feature Definitions** | `src/features/definitions/` | Declares *what* the features are (schema, entities, TTL, metadata). No transformation logic. |
| **Feature Transformers** | `src/features/transformers/` | Pure functions defining *how* features are computed (lag, rolling stats, time encoding). Independent of Feast and fully unit-testable. |
| **Feature Ingestion Pipelines** | `ingest_*.py` | Executes the orchestration: simulate raw data, apply transformers, persist to Parquet (offline store), register metadata to Feast. |
| **Usage Demos** | `fraud.py`, `recsys.py`, `forecast.py`, `station.py` | Serves as executable documentation for feature retrieval in offline and online workloads. |

---

## 3. Supported Workloads

Each example demonstrates two primary retrieval paths:

- **Offline Retrieval** → Used during model training with **Point-in-Time (PIT) safe joins**
- **Online Retrieval** → Used for **real-time inference APIs** after **materialization** to the online store

---

## 4. How to Run

Make sure you are in the **project root directory** and your Python environment is activated.

### Step 1: Ingest & Register Features

Populate the Offline + Online stores using the provided ingestion pipelines:

```bash
# Ingest Fraud data
python -m mlproject.src.features.examples.ingest_fraud

# Ingest RecSys data
python -m mlproject.src.features.examples.ingest_recsys

# Ingest Forecasting data
python -m mlproject.src.features.examples.ingest_forecast

# Ingest Station Sequence data
python -m mlproject.src.features.examples.ingest_station
```
### Step 2: Run Retrieval Demonstrations

After ingestion is complete, run the demo scripts to test feature lookup:
```bash
# Run Fraud demo (Historical PIT join + Online lookup)
python -m mlproject.src.features.examples.fraud

# Run RecSys demo (Materialization + Online serving)
python -m mlproject.src.features.examples.recsys

# Run Forecast demo (PIT-safe offline retrieval)
python -m mlproject.src.features.examples.forecast

# Run Station Sequence demo (Latest N points + Time range retrieval)
python -m mlproject.src.features.examples.station
```
## 5. Directory Structure Summary
```pgsql
mlproject/src/features/examples/
│
├── ingest_*.py       # Feature ingestion and metadata registration pipelines
├── fraud.py          # Fraud Detection feature retrieval demo
├── recsys.py         # Recommendation System online feature demo
├── forecast.py       # Time-Series forecasting PIT-safe retrieval
└── station.py        # Sequence retrieval demo using TimeSeriesFeatureStore wrapper
```
