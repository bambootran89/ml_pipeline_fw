#!/usr/bin/env python3
"""Quick launcher for serve APIs - Simple wrapper around run_generated_api."""

import sys

from mlproject.serve.run_generated_api import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
