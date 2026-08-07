#!/usr/bin/env python3
"""Thin CLI entry point for generating QI VMEC input seeds.

Implementation: squid/cli/seed/generate_seed.py. The legacy flat path
``scripts/generate_seed.py`` remains as a compatibility wrapper.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from squid.cli.seed.generate_seed import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
