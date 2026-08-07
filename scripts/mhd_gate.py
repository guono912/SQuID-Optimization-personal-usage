#!/usr/bin/env python3
"""Legacy-compatible entry point for the MHD gate.

The gate implementation moved to squid/cli/gates/mhd.py (CLI) and
squid/evaluation/gates.py (decision logic); the new grouped entry point is
scripts/gate/mhd_gate.py. This module keeps the flat path working, including
``from scripts.mhd_gate import gate`` style imports from older tooling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.gates.mhd import main  # noqa: E402
from squid.evaluation.gates import (  # noqa: E402
    _compute_ballooning_envelope,
    _compute_effective_ripple,
    gate,
)

if __name__ == "__main__":
    sys.exit(main())
