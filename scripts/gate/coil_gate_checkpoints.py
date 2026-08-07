#!/usr/bin/env python3
"""Thin CLI entry point for the batch coil feasibility + contour gate.

Implementation: squid/cli/gates/coil_gate_checkpoints.py, which drives the
per-checkpoint coil feasibility and contour checks as subprocesses through
scripts/gate/coil_feasibility_gate.py and scripts/util/coil_contour_metrics.py.
The legacy flat path ``scripts/coil_gate_checkpoints.py`` remains as a
compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.gates.coil_gate_checkpoints import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
