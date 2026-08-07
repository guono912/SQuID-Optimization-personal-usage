#!/usr/bin/env python3
"""Thin CLI entry point for the MHD gate.

Implementation: squid/cli/gates/mhd.py. The gate decision logic itself lives
in squid/evaluation/gates.py. This module exists so the new grouped path
``python scripts/gate/mhd_gate.py`` works; the legacy flat path
``scripts/mhd_gate.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.gates.mhd import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
