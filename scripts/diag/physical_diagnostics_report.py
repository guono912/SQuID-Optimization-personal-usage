#!/usr/bin/env python3
"""Thin CLI entry point for the physics-first diagnostic report.

Implementation: squid/cli/diag/physical_diagnostics_report.py. The legacy
flat path ``scripts/physical_diagnostics_report.py`` remains as a
compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.diag.physical_diagnostics_report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
