"""Diagnostic evaluation and plotting for SQuID equilibria."""

from .evaluate import (
    evaluate_squid,
    evaluate_itg,
    plot_boozer_surface,
    plot_squash_stretch,
    plot_gradient_diagnostics,
    plot_J_contours,
)
from .axis_geometry import (
    axis_curvature_torsion,
    axis_geometry_from_vmec,
    plot_axis_geometry,
)
from .available_energy import (
    available_energy,
    ae_surface,
    ae_diagnostics,
)

__all__ = [
    "evaluate_squid",
    "evaluate_itg",
    "plot_boozer_surface",
    "plot_squash_stretch",
    "plot_gradient_diagnostics",
    "plot_J_contours",
    "axis_curvature_torsion",
    "axis_geometry_from_vmec",
    "plot_axis_geometry",
    "available_energy",
    "ae_surface",
    "ae_diagnostics",
]
