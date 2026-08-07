#!/usr/bin/env python3
"""Grouped entry point for the SQuID diagnostic CLI."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.diagnose import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
