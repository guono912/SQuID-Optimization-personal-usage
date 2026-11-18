from types import SimpleNamespace

import numpy as np

from squid.cli.diagnose import (
    _compute_equilibrium_sanity,
    _first_nonzero_radial_index,
    _nc_scalar,
    _plot_axis_geometry_summary,
)


def test_equilibrium_sanity_uses_vmec_wout():
    wout = SimpleNamespace(
        aspect=9.0,
        Aminor_p=0.13,
        phi=np.array([0.0, 0.01]),
        iotaf=np.array([-0.38, -0.49]),
        presf=np.array([100.0, 0.0]),
        ier_flag=0,
        fsqr=1.0e-12,
        fsqz=1.0e-12,
        fsql=1.0e-12,
    )
    squid_info = {
        "f_QI": 0.01,
        "f_maxJ": 0.02,
        "mirror_ratio": 0.1,
        "common_B_valid": True,
    }
    result = _compute_equilibrium_sanity(
        SimpleNamespace(wout=wout), squid_info
    )
    assert result["status"] == "OK"


def test_first_nonzero_radial_index_returns_start_when_all_zero():
    values = np.zeros((4, 3))
    assert _first_nonzero_radial_index(values, start=2) == 2


def test_nc_scalar_returns_declared_default_for_missing_variable():
    dataset = SimpleNamespace(variables={})
    assert _nc_scalar(dataset, "missing", default=7.5) == 7.5


def test_axis_geometry_plot_uses_pyplot_alias():
    axis_info = {
        "phi": np.array([0.0, 0.5, 1.0]),
        "curvature": np.array([0.8, 1.0, 1.2]),
        "torsion": np.array([-0.2, 0.0, 0.2]),
        "axis_length": 8.0,
    }
    figure = _plot_axis_geometry_summary(axis_info)
    assert len(figure.axes) == 4
