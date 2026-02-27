#!/usr/bin/env python3
"""Example script for generating serve APIs from configs.

This script demonstrates how to use ConfigGenerator to automatically
generate FastAPI and Ray Serve code from serve pipeline configurations.
"""

import logging

logger = logging.getLogger(__name__)

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
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating APIs for: {Path(train_config).stem}")
    logger.info(f"{'='*60}\n")

    # Initialize generator
    generator = ConfigGenerator(train_config)

    # Generate FastAPI
    logger.info("1. Generating FastAPI...")
    try:
        fastapi_path = generator.generate_api(
            serve_config_path=serve_config,
            output_dir=output_dir,
            framework="fastapi",
            experiment_config_path=train_config,
        )
        logger.info(f"   ✓ Generated: {fastapi_path}")
    except Exception as e:
        logger.info(f"   ✗ Failed: {e}")

    # Generate Ray Serve
    logger.info("\n2. Generating Ray Serve...")
    try:
        ray_path = generator.generate_api(
            serve_config_path=serve_config,
            output_dir=output_dir,
            framework="ray",
            experiment_config_path=train_config,
        )
        logger.info(f"   ✓ Generated: {ray_path}")
    except Exception as e:
        logger.info(f"   ✗ Failed: {e}")

    logger.info()


def main() -> None:
    """Generate APIs for all example pipelines."""
    logger.info("\n" + "=" * 60)
    logger.info("Auto-Generating Serve APIs")
    logger.info("=" * 60)

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

    logger.info("=" * 60)
    logger.info("API Generation Complete!")
    logger.info("=" * 60)
    logger.info("\nGenerated files are in: mlproject/serve/generated/")
    logger.info("\nTo run FastAPI:")
    logger.info("  python mlproject/serve/generated/standard_train_serve_fastapi.py")
    logger.info("\nTo run Ray Serve:")
    logger.info("  python mlproject/serve/generated/standard_train_serve_ray.py")
    logger.info()


if __name__ == "__main__":
    main()
