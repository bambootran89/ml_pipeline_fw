"""Pipeline profiling step for output analysis.

This module provides comprehensive profiling of pipeline outputs including:
- Cluster distribution analysis
- Prediction statistics
- Model performance summaries
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from mlproject.src.pipeline.steps.core.base import BasePipelineStep
from mlproject.src.pipeline.steps.core.constants import ContextKeys
from mlproject.src.pipeline.steps.core.factory import StepFactory


def pretty_print(key: str, data: Any) -> None:
    """Print pipeline outputs in a human-friendly format."""
    print(f"\n=== {key.upper()} OUTPUT ===")

    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False))

    elif isinstance(data, pd.DataFrame):
        print(data.to_markdown(index=True, tablefmt="grid"))

    elif isinstance(data, np.ndarray):
        arr = data.tolist()
        print(json.dumps(arr, indent=2, ensure_ascii=False))

    elif isinstance(data, (DictConfig, ListConfig)):
        print(OmegaConf.to_yaml(data))

    else:
        print(data)

    print("=" * 30)


class ProfilingStep(BasePipelineStep):
    """Profile and summarize pipeline outputs.

    Context Inputs
    --------------
    All keys from previous steps are analyzed automatically.

    Context Outputs
    ---------------
    <step_id>_profile : Dict[str, Any]
        Profiling report with statistics.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize profiling step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration object.
        enabled : bool
            Whether step is active.
        depends_on : Optional[List[str]]
            Prerequisite steps.
        include_keys : Optional[List[str]]
            Specific context keys to profile.
        exclude_keys : Optional[List[str]]
            Context keys to exclude from profiling.
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.include_keys = include_keys or []
        self.exclude_keys = exclude_keys or ["cfg", "preprocessor"]

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Profile pipeline outputs and generate report.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context with all step outputs.

        Returns
        -------
        Dict[str, Any]
            Context with profiling report added.
        """
        self.validate_dependencies(context)

        profile: Dict[str, Any] = {
            "summary": {},
            "clusters": {},
            "predictions": {},
            "metrics": {},
            "data_quality": {},
        }

        keys_to_profile = self._get_keys_to_profile(context)

        for key in keys_to_profile:
            self._profile_value(key, context[key], profile)

        self._print_profile(profile)
        self.set_output(context, "profile", profile)

        return context

    def _get_keys_to_profile(self, context: Dict[str, Any]) -> List[str]:
        """Determine which context keys to profile.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        List[str]
            Keys to include in profiling.
        """
        if self.include_keys:
            return [k for k in self.include_keys if k in context]
        return [k for k in context if k not in self.exclude_keys]

    def _profile_value(
        self,
        key: str,
        value: Any,
        profile: Dict[str, Any],
    ) -> None:
        """Profile a single context value.

        Parameters
        ----------
        key : str
            Context key name.
        value : Any
            Value to profile.
        profile : Dict[str, Any]
            Profile dict to update.
        """

        if key.endswith(ContextKeys.METRICS):
            profile["metrics"][key] = self._summarize_metrics(value)
        elif "prediction" in key.lower() or "pred" in key.lower():
            profile["predictions"][key] = self._analyze_predictions(value)
        elif isinstance(value, pd.DataFrame):
            profile["data_quality"][key] = self._analyze_dataframe(value)
        elif isinstance(value, np.ndarray):
            # Only analyze as clusters if 1D integer-like array
            if np.issubdtype(value.dtype, np.integer) and (
                "cluster" in key.lower() or "label" in key.lower()
            ):
                profile["clusters"][key] = self._analyze_clusters(value)
            else:
                profile["summary"][key] = self._analyze_array(value)

    def _summarize_metrics(self, metrics: Any) -> Dict[str, Any]:
        """Summarize evaluation metrics.

        Parameters
        ----------
        metrics : Any
            Raw metrics dict.

        Returns
        -------
        Dict[str, Any]
            Summarized metrics with rankings.
        """
        if not isinstance(metrics, dict):
            return {"raw": str(metrics)}

        summary: Dict[str, Any] = {"values": {}, "best": None, "worst": None}
        numeric_metrics: Dict[str, float] = {}

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                summary["values"][k] = round(float(v), 6)
                numeric_metrics[k] = float(v)

        if numeric_metrics:
            sorted_keys = sorted(
                numeric_metrics.keys(), key=lambda x: numeric_metrics[x]
            )
            summary["best"] = sorted_keys[0]
            summary["worst"] = sorted_keys[-1]

        return summary

    def _analyze_clusters(self, labels: Any) -> Dict[str, Any]:
        """Analyze cluster label distribution.

        Parameters
        ----------
        labels : Any
            Cluster labels array.

        Returns
        -------
        Dict[str, Any]
            Cluster distribution statistics.
        """
        if labels is None:
            return {"error": "No labels provided"}

        # Skip non-array types
        if isinstance(labels, dict):
            return {"error": "Expected array, got dict"}
        if not hasattr(labels, "__iter__") or isinstance(labels, str):
            return {"error": f"Expected array, got {type(labels).__name__}"}

        try:
            arr = np.asarray(labels).flatten()
        except (ValueError, TypeError) as exc:
            return {"error": f"Cannot convert to array: {exc}"}
        unique, counts = np.unique(arr, return_counts=True)
        distribution = {int(u): int(c) for u, c in zip(unique, counts)}
        total = len(arr)
        percentages = {k: round(v / total * 100, 2) for k, v in distribution.items()}

        return {
            "n_clusters": len(unique),
            "n_samples": total,
            "distribution": distribution,
            "percentages": percentages,
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "std_cluster_size": round(float(counts.std()), 2),
            "balance_ratio": round(float(counts.min() / counts.max()), 4),
        }

    def _analyze_predictions(
        self,
        preds: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute summary statistics from model prediction arrays.

        This method supports sliding window forecasting outputs with 1D, 2D,
        or 3D numpy arrays. Statistics include mean, standard deviation,
        min, max, median, and interquartile range (25th/75th percentiles).

        Parameters
        ----------
        preds : Optional[np.ndarray]
            Model prediction array. Accepted shapes:
            - 1D: [samples]
            - 2D: [samples, targets]
            - 3D: [samples, steps, targets]

        Returns
        -------
        Dict[str, Any]
            A dictionary containing statistics. For 2D/3D inputs, each target
            is summarized under a unique key. If preds is None or empty-like,
            an error dictionary is returned.

        Raises
        ------
        ValueError
            If the array has more than 3 dimensions.
        """

        def _stats(arr: np.ndarray) -> Dict[str, Any]:
            return {
                "n_samples": arr.size,
                "mean": (
                    round(float(np.nanmean(arr)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "std": (
                    round(float(np.nanstd(arr)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "min": (
                    round(float(np.nanmin(arr)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "max": (
                    round(float(np.nanmax(arr)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "median": (
                    round(float(np.nanmedian(arr)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "q25": (
                    round(float(np.nanpercentile(arr, 25)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
                "q75": (
                    round(float(np.nanpercentile(arr, 75)), 6)
                    if not np.all(np.isnan(arr))
                    else float("nan")
                ),
            }

        if preds.ndim == 1:
            return _stats(preds)

        if preds.ndim == 2:
            out: Dict[str, Any] = {}
            for i, col in enumerate(preds.T):
                out[f"target_{i}"] = _stats(col)
            return out

        if preds.ndim == 3:
            result: Dict[str, Any] = {}
            n_steps, n_targets = preds.shape[1], preds.shape[2]
            for t in range(n_targets):
                for s in range(n_steps):
                    key = f"target({t})_step({s})"
                    result[key] = _stats(preds[:, s, t])
            return result

        raise ValueError(f"Only 1D, 2D, 3D arrays supported, got {preds.ndim}D")

    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame quality.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.

        Returns
        -------
        Dict[str, Any]
            Data quality metrics.
        """
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_per_column": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        }

    def _analyze_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Analyze numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array to analyze.

        Returns
        -------
        Dict[str, Any]
            Array statistics.
        """
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
        }

    def _print_profile(self, profile: Dict[str, Any]) -> None:
        """
        Converts and prints profiling results as clean, structured DataFrames.

        This method flattens complex nested dictionaries (Data Quality, Clusters,
        Metrics) into readable tables while preserving all essential information.
        """
        print(f"\n>>> [PROFILING REPORT: {self.step_id}]")

        # 1. Data Quality Summary & Schema
        if profile.get("data_quality"):
            self._display_data_quality(profile["data_quality"])

        # 2. Cluster Analysis Summary & Distribution
        if profile.get("clusters"):
            self._display_clusters(profile["clusters"])

        # 3. Metrics & Predictions
        self._display_performance_stats(profile)

    def _display_data_quality(self, dq_data: Dict[str, Any]) -> None:
        """Processes and displays Data Quality statistics."""
        summary_rows = []
        for key, data in dq_data.items():
            shape = data.get("shape", [0, 0])
            if shape[0] == 0 and shape[1] == 0:
                continue

            summary_rows.append(
                {
                    "Object": key,
                    "Rows": shape[0],
                    "Cols": shape[1],
                    "Missing": data.get("missing_total", 0),
                    "Dtypes": ", ".join(set(data.get("dtypes", {}).values())),
                }
            )

        if summary_rows:
            print("\n[Data Quality Summary]")
            print(pd.DataFrame(summary_rows).to_string(index=False))

            # Print Schema for non-empty objects
            for key, data in dq_data.items():
                if data.get("shape", [0, 0])[0] > 0:
                    print(f"\n  > Schema: {key}")
                    schema_df = pd.DataFrame(
                        {
                            "Column": data.get("columns", []),
                            "Dtype": [
                                data.get("dtypes", {}).get(c)
                                for c in data.get("columns", [])
                            ],
                            "Missing": [
                                data.get("missing_per_column", {}).get(c)
                                for c in data.get("columns", [])
                            ],
                        }
                    )
                    print(textwrap.indent(schema_df.to_string(index=False), "    "))

    def _display_clusters(self, cluster_data: Dict[str, Any]) -> None:
        """Processes and displays Cluster Analysis statistics."""
        for key, data in cluster_data.items():
            print(f"\n[Cluster Analysis: {key}]")

            # Summary Metrics
            summary = pd.DataFrame(
                [
                    {
                        "N_Clusters": data.get("n_clusters"),
                        "N_Samples": data.get("n_samples"),
                        "Std_Size": f"{data.get('std_cluster_size', 0):.2f}",
                        "Balance": f"{data.get('balance_ratio', 0):.4f}",
                    }
                ]
            )
            print(summary.to_string(index=False))

            # Distribution Table
            dist = data.get("distribution", {})
            perc = data.get("percentages", {})
            dist_df = pd.DataFrame(
                {
                    "Cluster": list(dist.keys()),
                    "Count": list(dist.values()),
                    "Percentage (%)": [f"{p:.2f}%" for p in perc.values()],
                }
            )
            print("  Distribution:")
            print(textwrap.indent(dist_df.to_string(index=False), "    "))

    def _display_performance_stats(self, profile: Dict[str, Any]) -> None:
        """
        Processes and displays metrics and prediction statistics.

        Detects statistical dictionaries in predictions and expands them into
        individual columns for improved readability and analysis.
        """
        # 1. Handle Simple Metrics (Scalars)
        metric_rows = []
        for key, data in profile.get("metrics", {}).items():
            for m_name, m_val in data.get("values", {}).items():
                metric_rows.append({"Group": key, "Metric": m_name, "Value": m_val})

        if metric_rows:
            print("\n[ Performance Metrics ]")
            print(pd.DataFrame(metric_rows).to_string(index=False))

        # 2. Handle Prediction Statistics (Distributions)
        self._display_prediction_stats(profile.get("predictions", {}))

    def _display_prediction_stats(self, pred_data: Dict[str, Any]) -> None:
        """
        Formats prediction distribution statistics into a tabular summary.

        Args:
            pred_data: Dictionary containing prediction groups and target stats.
        """
        rows: List[Dict[str, Any]] = []

        for group_name, targets in pred_data.items():
            if not isinstance(targets, dict):
                continue

            for target_name, stats in targets.items():
                if not isinstance(stats, dict):
                    continue

                # Flatten statistical dictionary into a row
                rows.append(
                    {
                        "Group": group_name,
                        "Target": target_name,
                        "Samples": stats.get("n_samples"),
                        "Mean": f"{stats.get('mean', 0):.4f}",
                        "Std": f"{stats.get('std', 0):.4f}",
                        "Min": f"{stats.get('min', 0):.4f}",
                        "Max": f"{stats.get('max', 0):.4f}",
                        "Median": f"{stats.get('median', 0):.4f}",
                    }
                )

        if rows:
            print("\n[ Prediction Statistics ]")
            # Display the stats in a clean table without index
            stats_df = pd.DataFrame(rows)
            print(stats_df.to_string(index=False, justify="left"))


StepFactory.register("profiling", ProfilingStep)
