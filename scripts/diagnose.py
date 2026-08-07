#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`squid.cli.diagnose`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.diagnose import *  # noqa: F401,F403,E402
from squid.cli.diagnose import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
