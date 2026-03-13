"""
Consolidated plotting utilities for SQuID diagnostics.

Re-exports the plot functions from evaluation modules so they
can be accessed as squid.utils.plotting.plot_boozer_surface etc.
"""

from ..evaluation.evaluate import (
    plot_boozer_surface,
    plot_squash_stretch,
    plot_gradient_diagnostics,
    plot_J_contours,
)
from ..evaluation.axis_geometry import plot_axis_geometry

__all__ = [
    "plot_boozer_surface",
    "plot_squash_stretch",
    "plot_gradient_diagnostics",
    "plot_J_contours",
    "plot_axis_geometry",
]
