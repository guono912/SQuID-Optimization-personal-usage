"""Compatibility wrapper: implementation moved to squid/utils/nc_to_neort.py.

Old imports such as ``import nc_to_neort`` inside scripts keep working:
``convert_boozmn_to_neort`` is re-exported from the library module.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from squid.utils.nc_to_neort import MU0, convert_boozmn_to_neort  # noqa: E402
from squid.cli.nc_to_neort import main  # noqa: E402

__all__ = ["MU0", "convert_boozmn_to_neort"]


if __name__ == "__main__":
    sys.exit(main())
