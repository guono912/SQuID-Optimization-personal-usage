#!/usr/bin/env python3
"""Thin CLI entry point for batch MHD gate over checkpoint wouts.

Implementation: squid/cli/gates/gate_checkpoints.py. The promotion gate
itself lives in squid/evaluation/gates.py. The legacy flat path
``scripts/gate_checkpoints.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.gates.gate_checkpoints import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
