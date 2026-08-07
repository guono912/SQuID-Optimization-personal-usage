#!/usr/bin/env python3
"""Compatibility wrapper: implementation moved to scripts/transform/.

Old invocation ``python scripts/rescale_and_gate.py`` keeps working; the CLI
lives in ``scripts/transform/rescale_and_gate.py``. ``rescale`` and ``gate``
are re-exported for old callers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.transform.rescale_and_gate import (  # noqa: E402
    _run_in_directory,
    gate,
    main,
    rescale,
)

__all__ = ["_run_in_directory", "gate", "main", "rescale"]

if __name__ == "__main__":
    sys.exit(main())
