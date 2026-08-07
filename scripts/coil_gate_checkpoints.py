#!/usr/bin/env python3
"""Legacy-compatible entry point for the batch coil gate.

The implementation moved to squid/cli/gates/coil_gate_checkpoints.py; the new
grouped entry point is scripts/gate/coil_gate_checkpoints.py. This module
keeps the flat path working, including ``from scripts.coil_gate_checkpoints
import gate_checkpoints`` style imports from older tooling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.gates.coil_gate_checkpoints import (  # noqa: E402
    _ensure_checkpoint_wouts,
    _latest_row,
    _load_summary,
    gate_checkpoints,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
