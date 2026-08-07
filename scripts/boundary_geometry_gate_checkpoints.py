#!/usr/bin/env python3
"""Legacy-compatible entry point for the boundary-geometry gate.

The implementation moved to squid/cli/gates/boundary_geometry.py; the new
grouped entry point is scripts/gate/boundary_geometry_gate_checkpoints.py.
This module keeps the flat path working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.gates.boundary_geometry import (  # noqa: E402
    _eval_index,
    _stat,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
