#!/usr/bin/env python3
"""Thin CLI entry point for recomposing a VMEC input boundary.

Implementation: squid/cli/seed/recompose_vmec_input.py. The legacy flat path
``scripts/recompose_vmec_input.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.seed.recompose_vmec_input import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
