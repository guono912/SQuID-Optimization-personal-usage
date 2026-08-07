#!/usr/bin/env python3
"""Compatibility wrapper: implementation moved to scripts/util/.

Old invocation ``python scripts/coil_contour_metrics.py`` keeps working;
the CLI lives in ``scripts/util/coil_contour_metrics.py``. The shared curve
metrics now live in ``squid/evaluation/coil_metrics.py`` and are re-exported.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.util.coil_contour_metrics import main  # noqa: E402
from squid.evaluation.coil_metrics import curve_metrics_xyz  # noqa: E402,F401

__all__ = ["curve_metrics_xyz", "main"]

if __name__ == "__main__":
    sys.exit(main())
