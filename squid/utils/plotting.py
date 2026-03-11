"""
Consolidated plotting utilities for SQuID diagnostics.

Re-exports the plot functions from evaluation.evaluate so they
can be accessed as squid.utils.plotting.plot_boozer_surface etc.
"""

from ..evaluation.evaluate import (
    plot_boozer_surface,
    plot_squash_stretch,
    plot_gradient_diagnostics,
)

__all__ = [
    "plot_boozer_surface",
    "plot_squash_stretch",
    "plot_gradient_diagnostics",
]
