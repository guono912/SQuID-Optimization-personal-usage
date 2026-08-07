#!/usr/bin/env python3
"""Thin CLI entry point for the boundary-geometry gate.

Implementation: squid/cli/gates/boundary_geometry.py. The metrics come from
squid/diagnostics/boundary_geometry.py and squid/objectives/pdrot_residual.py.
The legacy flat path ``scripts/boundary_geometry_gate_checkpoints.py`` remains
as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.gates.boundary_geometry import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
