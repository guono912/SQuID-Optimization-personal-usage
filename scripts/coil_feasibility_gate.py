#!/usr/bin/env python3
"""Legacy-compatible entry point for the coil feasibility gate.

The implementation moved to squid/cli/gates/coil_feasibility.py; the reusable
helpers live in squid/evaluation/coil_metrics.py and
squid/objectives/coil_proxy.py. The new grouped entry point is
scripts/gate/coil_feasibility_gate.py. This module keeps the flat path
working, including ``from scripts.coil_feasibility_gate import
make_winding_surface`` style imports from older tooling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.gates.coil_feasibility import main, make_plots, run_gate  # noqa: E402
from squid.evaluation.coil_metrics import (  # noqa: E402
    make_scaled_winding_surface,
    make_winding_surface,
)
from squid.objectives.coil_proxy import (  # noqa: E402
    _float,
    _list_item,
    current_spectrum_metrics,
)

if __name__ == "__main__":
    sys.exit(main())
