#!/usr/bin/env python3
"""Legacy-compatible entry point for generating QI VMEC input seeds.

The implementation moved to squid/cli/seed/generate_seed.py; the new grouped
entry point is scripts/seed/generate_seed.py. This module keeps the flat path
working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.seed.generate_seed import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
