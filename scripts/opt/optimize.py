#!/usr/bin/env python3
"""Thin CLI entry point for the SQuID optimiser.

Implementation: squid/cli/optimize.py. The legacy flat path
``scripts/optimize.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.optimize import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
