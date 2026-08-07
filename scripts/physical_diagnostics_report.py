#!/usr/bin/env python3
"""Legacy-compatible entry point for the physics-first diagnostic report.

The implementation moved to squid/cli/diag/physical_diagnostics_report.py;
the new grouped entry point is scripts/diag/physical_diagnostics_report.py.
This module keeps the flat path working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.diag.physical_diagnostics_report import (  # noqa: E402
    _array,
    _dangerous_rationals,
    _desc_ballooning,
    _desc_effective_ripple,
    _desc_newcomb_metric,
    _scalar,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
