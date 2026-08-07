#!/usr/bin/env python3
"""Legacy-compatible entry point for iterative QI refinement.

The implementation moved to squid/cli/refine_qi.py; the new grouped entry
point is scripts/opt/refine_qi.py. The optimization loop now runs inside
main() instead of at module import time. This module keeps the flat path
working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.refine_qi import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
