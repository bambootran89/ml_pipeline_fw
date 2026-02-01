#!/usr/bin/env python3
"""Example script for generating serve APIs from configs.

This script demonstrates how to use ConfigGenerator to automatically
generate FastAPI and Ray Serve code from serve pipeline configurations.
"""

from pathlib import Path

from mlproject.src.generator.orchestrator import ConfigGenerator


def generate_apis_for_pipeline(
    train_config: str,
    serve_config: str,
    output_dir: str = "mlproject/serve/generated",
) -> None:
    """Generate both FastAPI and Ray Serve APIs for a pipeline.

    Args:
        train_config: Path to training pipeline YAML.
        serve_config: Path to serve pipeline YAML.
        output_dir: Output directory for generated APIs.
    """
    print(f"\n{'='*60}")
    print(f"Generating APIs for: {Path(train_config).stem}")
    print(f"{'='*60}\n")

    # Initialize generator
    generator = ConfigGenerator(train_config)

    # Generate FastAPI
    print("1. Generating FastAPI...")
    try:
        fastapi_path = generator.generate_api(
            serve_config_path=serve_config,
            output_dir=output_dir,
            framework="fastapi",
            experiment_config_path=train_config,
        )
        print(f"   ✓ Generated: {fastapi_path}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Generate Ray Serve
    print("\n2. Generating Ray Serve...")
    try:
        ray_path = generator.generate_api(
            serve_config_path=serve_config,
            output_dir=output_dir,
            framework="ray",
            experiment_config_path=train_config,
        )
        print(f"   ✓ Generated: {ray_path}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print()


def main() -> None:
    """Generate APIs for all example pipelines."""
    print("\n" + "=" * 60)
    print("Auto-Generating Serve APIs")
    print("=" * 60)

    # Example 1: Standard single-model pipeline
    generate_apis_for_pipeline(
        train_config="mlproject/configs/pipelines/standard_train.yaml",
        serve_config="mlproject/configs/generated/standard_train_serve.yaml",
    )

    # Example 2: Conditional branch with multiple models
    generate_apis_for_pipeline(
        train_config="mlproject/configs/pipelines/conditional_branch.yaml",
        serve_config="mlproject/configs/generated/conditional_branch_serve.yaml",
    )

    # Example 3: K-means then XGBoost pipeline
    generate_apis_for_pipeline(
        train_config="mlproject/configs/pipelines/kmeans_then_xgboost.yaml",
        serve_config="mlproject/configs/generated/kmeans_then_xgboost_serve.yaml",
    )

    print("=" * 60)
    print("API Generation Complete!")
    print("=" * 60)
    print("\nGenerated files are in: mlproject/serve/generated/")
    print("\nTo run FastAPI:")
    print("  python mlproject/serve/generated/standard_train_serve_fastapi.py")
    print("\nTo run Ray Serve:")
    print("  python mlproject/serve/generated/standard_train_serve_ray.py")
    print()


if __name__ == "__main__":
    main()
