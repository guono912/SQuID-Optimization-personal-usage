#!/usr/bin/env python3
"""Grouped entry point for Boozer-to-NEO-RT conversion."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.nc_to_neort import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
