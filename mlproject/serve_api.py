#!/usr/bin/env python3
"""Quick launcher for serve APIs - Simple wrapper around run_generated_api."""

import logging
import sys

from mlproject.serve.run_generated_api import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nStopped by user")
        sys.exit(0)
    except Exception as e:
        logger.info(f"\nError: {e}")
        sys.exit(1)
