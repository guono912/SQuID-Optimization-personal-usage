"""Diagnostic evaluation and plotting for SQuID equilibria.

Names are exposed lazily (PEP 562) so importing a single submodule such as
``squid.evaluation.gates`` does not eagerly pull the heavy evaluation chain
(scipy/simsopt). ``from squid.evaluation import evaluate_squid`` still works
when the optional dependencies are installed.
"""

import importlib

_LAZY_EXPORTS = {
    "evaluate_squid": ".evaluate",
    "evaluate_itg": ".evaluate",
    "plot_boozer_surface": ".evaluate",
    "plot_squash_stretch": ".evaluate",
    "plot_gradient_diagnostics": ".evaluate",
    "plot_J_contours": ".evaluate",
    "axis_curvature_torsion": ".axis_geometry",
    "axis_geometry_from_vmec": ".axis_geometry",
    "plot_axis_geometry": ".axis_geometry",
    "available_energy": ".available_energy",
    "ae_surface": ".available_energy",
    "ae_diagnostics": ".available_energy",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = importlib.import_module(module, __name__)
    value = getattr(owner, name)
    globals()[name] = value
    return value
