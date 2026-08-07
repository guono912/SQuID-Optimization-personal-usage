#!/usr/bin/env python3
"""Thin CLI entry point for the coil feasibility gate.

Implementation: squid/cli/gates/coil_feasibility.py. The reusable winding
surface and current-spectrum helpers live in squid/evaluation/coil_metrics.py
and squid/objectives/coil_proxy.py. The legacy flat path
``scripts/coil_feasibility_gate.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.gates.coil_feasibility import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
