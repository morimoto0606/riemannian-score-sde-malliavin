#!/usr/bin/env python3
"""Compatibility entry point for the shared S2 earth-data postprocessor."""

from pathlib import Path
import sys

# Support both direct CLI execution and importlib-based legacy callers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_s2_earth_data import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
