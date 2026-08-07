#!/usr/bin/env python3
"""Compatibility wrapper: implementation moved to scripts/util/.

Old invocation ``python scripts/compare_desc_profiles.py`` keeps working;
the CLI lives in ``scripts/util/compare_desc_profiles.py``. Public helper
functions are re-exported so old imports keep resolving.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.util.compare_desc_profiles import (  # noqa: E402
    get_B_range,
    get_profiles,
    get_total_current,
    get_volume_averaged_beta,
    load_eq,
    main,
)

__all__ = [
    "get_B_range",
    "get_profiles",
    "get_total_current",
    "get_volume_averaged_beta",
    "load_eq",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
