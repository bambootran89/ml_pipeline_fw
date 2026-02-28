#!/usr/bin/env python3
"""Auto-generate and run serve API from config.

This script automatically generates API code from serve configuration
and runs it immediately. No manual steps needed!

Usage:
    python -m mlproject.serve.run_generated_api \
        --serve-config mlproject/configs/generated/standard_train_serve.yaml \
        --experiment-config mlproject/configs/experiments/etth1.yaml \
        --framework fastapi \
        --port 8000

    # Or with all options:
    python -m mlproject.serve.run_generated_api \
        --serve-config mlproject/configs/generated/standard_train_serve.yaml \
        --experiment-config mlproject/configs/experiments/etth1.yaml \
        --framework ray \
        --port 8000 \
        --host 0.0.0.0 \
        --alias latest
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

from mlproject.src.generator.api_generator import ApiGenerator

logger = logging.getLogger(__name__)


# Suppress OpenMP conflicting dynamic runtime libraries error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


def _configure_server_settings(
    api_path: str, framework: str, host: str, port: int
) -> None:
    """Modify generated code to use custom host/port.

    Args:
        api_path: Path to generated API file.
        framework: 'fastapi' or 'ray'.
        host: Host to bind to.
        port: Port to bind to.
    """
    with open(api_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Inject host/port for FastAPI
    if framework == "fastapi":
        code = code.replace(
            'uvicorn.run(\n        "mlproject.serve.api:app",\n        '
            'host="0.0.0.0",\n        port=8000,',
            f'uvicorn.run(\n        app,\n        host="{host}",\n        port={port},',
        )
        code = code.replace(
            'uvicorn.run(app, host="0.0.0.0", port=8000)',
            f'uvicorn.run(app, host="{host}", port={port})',
        )
        code = code.replace(
            "uvicorn.run(app, host=API_DEFAULTS.FASTAPI_HOST, "
            "port=API_DEFAULTS.FASTAPI_PORT)",
            f'uvicorn.run(app, host="{host}", port={port})',
        )
    elif framework == "ray":
        code = code.replace(
            "serve.run(app_builder({}))",
            f'serve.start(http_options={{"host": "{host}", "port": {port}}})\n'
            f"    serve.run(app_builder({{}}))",
        )

    with open(api_path, "w", encoding="utf-8") as f:
        f.write(code)


def _run_server(api_path: str, framework: str, host: str, port: int) -> None:
    """Run the generated API server.

    Args:
        api_path: Path to generated API file.
        framework: Framework name for display.
        host: Host to bind to.
        port: Port to bind to.
    """
    logger.info(f"\n[3/3] Starting {framework.upper()} server...")
    logger.info(f"\n{'='*60}")
    logger.info(f"API starting at: http://{host}:{port}")
    logger.info(f"API docs: http://{host}:{port}/docs")
    logger.info(f"Health check: http://{host}:{port}/health")
    logger.info(f"{'='*60}")
    logger.info("\nPress Ctrl+C to stop the server\n")
    logger.info("-" * 60)

    try:
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent.parent
        env["PYTHONPATH"] = str(project_root)
        subprocess.run(
            [sys.executable, api_path],
            check=False,
            env=env,
            cwd=str(project_root),
        )
    except KeyboardInterrupt:
        logger.info("\n\n" + "=" * 60)
        logger.info("Server stopped by user")
        logger.info("=" * 60)
    except Exception as e:
        logger.info(f"\nServer failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def generate_and_run(
    serve_config_path: str,
    experiment_config_path: str,
    framework: str = "fastapi",
    host: str = "0.0.0.0",
    port: int = 8000,
    alias: str = "production",
) -> None:
    """Generate API code and run it immediately.

    Args:
        serve_config_path: Path to serve YAML config.
        experiment_config_path: Path to experiment YAML config.
        framework: 'fastapi' or 'ray'.
        host: Host to bind to.
        port: Port to bind to.
        alias: MLflow model alias.
    """
    logger.info("=" * 60)
    logger.info(f"Auto-Generate & Run {framework.upper()} API")
    logger.info("=" * 60)
    logger.info(f"Experiment config: {experiment_config_path}")
    logger.info(f"Serve config: {serve_config_path}")
    logger.info(f"Framework: {framework}")
    logger.info(f"Address: {host}:{port}")
    logger.info(f"Alias: {alias}")
    logger.info("=" * 60)

    # Step 1: Generate API code
    logger.info("\n[1/3] Generating API code...")
    generator = ApiGenerator()

    try:
        api_path = generator.generate_api(
            serve_config_path=serve_config_path,
            output_dir="mlproject/serve/generated",
            framework=framework,
            experiment_config_path=experiment_config_path,
            alias=alias,
        )
        logger.info(f"Generated: {api_path}")
    except Exception as e:
        logger.info(f"Generation failed: {e}")
        sys.exit(1)

    # Step 2: Configure server settings
    logger.info("\n[2/3] Configuring server settings...")
    _configure_server_settings(api_path, framework, host, port)
    logger.info(f"Configured: {host}:{port}")

    # Step 3: Run server
    _run_server(api_path, framework, host, port)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-generate and run serve API from config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FastAPI on port 8000
  python -m mlproject.serve.run_generated_api \
      --serve-config mlproject/configs/generated/standard_train_serve.yaml \
      --experiment-config mlproject/configs/experiments/etth1.yaml \
      --framework fastapi

  # Ray Serve on port 9000
  python -m mlproject.serve.run_generated_api \
      --serve-config mlproject/configs/generated/standard_train_serve.yaml \
      --experiment-config mlproject/configs/experiments/etth1.yaml \
      --framework ray \
      --port 9000

  # Conditional branch pipeline
  python -m mlproject.serve.run_generated_api \
      --serve-config mlproject/configs/generated/conditional_branch_serve.yaml \
      --experiment-config mlproject/configs/experiments/etth1.yaml \
      --framework fastapi
        """,
    )

    parser.add_argument(
        "--serve-config",
        required=True,
        help="Path to serve YAML config (e.g., standard_train_serve.yaml)",
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment YAML config (e.g., etth1.yaml)",
    )
    parser.add_argument(
        "--framework",
        choices=["fastapi", "ray"],
        default="fastapi",
        help="Framework to use (default: fastapi)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="MLflow model alias to load (default: production)",
    )

    args = parser.parse_args()

    # Verify experiment config exists
    if not Path(args.experiment_config).exists():
        logger.info(f"Error: Experiment config not found: {args.experiment_config}")
        sys.exit(1)

    # Generate and run
    generate_and_run(
        serve_config_path=args.serve_config,
        experiment_config_path=args.experiment_config,
        framework=args.framework,
        host=args.host,
        port=args.port,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
