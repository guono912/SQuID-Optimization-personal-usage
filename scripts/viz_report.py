#!/usr/bin/env python3
"""Legacy-compatible entry point for the SQuID configuration diagnostic report.

The implementation moved to squid/cli/viz.py; the new grouped entry point is
scripts/viz/viz_report.py. This module keeps the flat path working, including
``from scripts.viz_report import generate_report`` style imports from older
tooling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.cli.viz import (  # noqa: E402
    build_scalar_json,
    generate_report,
    main,
    plot_equilibrium_overview,
    plot_itg_flux_compression,
    plot_mhd_stability,
    plot_neoclassical,
)

if __name__ == "__main__":
    sys.exit(main())
