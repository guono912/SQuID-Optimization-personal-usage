#!/usr/bin/env python3
"""Compatibility wrapper: implementation moved to scripts/diag/.

Old invocation ``python scripts/check_mercier_normalization.py`` keeps
working; the CLI lives in ``scripts/diag/check_mercier_normalization.py``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.diag.check_mercier_normalization import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
