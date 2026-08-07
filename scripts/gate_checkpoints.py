#!/usr/bin/env python3
"""Legacy-compatible entry point for the batch MHD gate over checkpoints.

The implementation moved to squid/cli/gates/gate_checkpoints.py; the new
grouped entry point is scripts/gate/gate_checkpoints.py. This module keeps
the flat path working, including ``from scripts.gate_checkpoints import
gate_checkpoints`` style imports from older tooling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.gates.gate_checkpoints import (  # noqa: E402
    gate_checkpoints,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
