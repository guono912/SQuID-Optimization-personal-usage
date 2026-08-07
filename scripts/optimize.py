#!/usr/bin/env python3
"""Legacy-compatible entry point for the SQuID optimiser.

The implementation moved to squid/cli/optimize.py; the new grouped entry
point is scripts/opt/optimize.py. This module keeps the flat path working,
including ``from scripts.optimize import MODE_PRESETS, _build_parser`` style
imports from older tooling and tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.optimize import (  # noqa: E402
    HAS_DESC,
    HAS_VMEC,
    MODE_PRESETS,
    _build_parser,
    _flatten_config,
    _jsonify,
    _load_config,
    _reject_legacy_mercier_config,
    _write_resolved_parameters,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
