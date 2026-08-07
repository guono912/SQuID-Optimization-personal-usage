#!/usr/bin/env python3
"""Thin CLI entry point for iterative QI refinement.

Implementation: squid/cli/refine_qi.py. The legacy flat path
``scripts/refine_qi.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.refine_qi import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
