"""Surface-current coil-realizability proxy.

This module implements the lightweight quasi-single-stage proxy used for
triage during fixed-boundary optimisation: build a uniformly offset winding
surface, solve a regularized surface-current subproblem, and penalize only when
the residual normal field or current density exceeds conservative targets.
"""

import os

import numpy as np


INVALID_COIL_PROXY = 1.0e6


def _float(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else float("nan")


def _list_item(data, key, index):
    value = data[key]
    if isinstance(value, (list, tuple)):
        return value[index]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 1:
        return arr[0]
    return arr[index]


def current_spectrum_metrics(phi_mn, modes):
    phi_mn = np.asarray(phi_mn, dtype=float)
    modes = np.asarray(modes, dtype=int)
    total = float(np.linalg.norm(phi_mn))
    if total <= 0:
        return {
            "phi_norm": 0.0,
            "phi_high_mode_fraction": 0.0,
            "phi_weighted_mode_rms": 0.0,
            "phi_max_abs": 0.0,
        }
    abs_phi = np.abs(phi_mn)
    mode_radius = np.sqrt(modes[:, 0] ** 2 + modes[:, 1] ** 2)
    high = mode_radius >= max(4.0, 0.5 * np.max(mode_radius))
    return {
        "phi_norm": total,
        "phi_high_mode_fraction": float(np.linalg.norm(phi_mn[high]) / total),
        "phi_weighted_mode_rms": float(
            np.sqrt(np.sum((abs_phi * mode_radius) ** 2) / np.sum(abs_phi ** 2))
        ),
        "phi_max_abs": float(np.max(abs_phi)),
    }


def evaluate_coil_proxy(
    wout,
    *,
    offset_fraction=0.35,
    lambda_regularization=1e-8,
    desc_L=4,
    desc_M=4,
    desc_N=4,
    M_Phi=4,
    N_Phi=4,
    source_M=16,
    source_N=16,
    eval_M=16,
    eval_N=16,
    current_helicity=(1, 0),
    regularization_type="regcoil",
    vacuum=False,
    chunk_size=None,
    verbose=0,
):
    """Evaluate a single DESC surface-current proxy solve.

    Returns a metrics dictionary. The normalized normal-field metrics are
    computed on the plasma boundary as ``Bn_total / |B|``.
    """
    if not wout or not os.path.exists(wout):
        raise FileNotFoundError(f"VMEC wout not found: {wout}")

    from desc.grid import LinearGrid
    from desc.geometry import FourierRZToroidalSurface  # noqa: F401
    from desc.magnetic_fields import FourierCurrentPotentialField
    from desc.magnetic_fields import solve_regularized_surface_current
    from desc.vmec import VMECIO

    eq = VMECIO.load(wout, L=desc_L, M=desc_M, N=desc_N)
    aminor = _float(eq.compute("a")["a"])
    surface = eq.surface.constant_offset_surface(
        float(offset_fraction) * aminor,
        M=max(eq.surface.M, eq.M),
        N=max(eq.surface.N, eq.N),
    )
    field = FourierCurrentPotentialField(
        R_lmn=surface.R_lmn,
        Z_lmn=surface.Z_lmn,
        modes_R=surface.R_basis.modes[:, 1:],
        modes_Z=surface.Z_basis.modes[:, 1:],
        NFP=surface.NFP,
        sym=surface.sym,
        M_Phi=M_Phi,
        N_Phi=N_Phi,
    )
    source_grid = LinearGrid(
        M=max(3 * M_Phi, source_M),
        N=max(3 * N_Phi, source_N),
        NFP=eq.NFP,
        sym=False,
    )
    eval_grid = LinearGrid(
        M=eval_M,
        N=eval_N,
        NFP=eq.NFP,
        sym=False,
        rho=np.array([1.0]),
    )
    bmag_eval = np.asarray(eq.compute("|B|", grid=eval_grid)["|B|"], dtype=float)
    lambdas = np.array([float(lambda_regularization)], dtype=float)
    fields, data = solve_regularized_surface_current(
        field,
        eq,
        lambda_regularization=lambdas,
        current_helicity=tuple(current_helicity),
        vacuum=vacuum,
        regularization_type=regularization_type,
        source_grid=source_grid,
        eval_grid=eval_grid,
        verbose=verbose,
        chunk_size=chunk_size,
        B_plasma_chunk_size=chunk_size,
    )

    k_key = "||K||" if "||K||" in data else "|K|"
    k_norm = np.asarray(_list_item(data, k_key, 0), dtype=float)
    bn_total = np.asarray(_list_item(data, "Bn_total", 0), dtype=float)
    phi = np.asarray(_list_item(data, "Phi_mn", 0), dtype=float)
    bn_unitless = bn_total / np.maximum(np.abs(bmag_eval), 1e-300)
    out = {
        "offset_fraction": float(offset_fraction),
        "offset_m_est": float(offset_fraction * aminor),
        "lambda_regularization": float(lambda_regularization),
        "chi2_B": float(_list_item(data, "chi^2_B", 0)),
        "chi2_K": float(_list_item(data, "chi^2_K", 0)),
        "Bn_rms_unitless": float(np.sqrt(np.mean(bn_unitless**2))),
        "Bn_max_abs_unitless": float(np.max(np.abs(bn_unitless))),
        "Bn_avg_abs_unitless": float(np.mean(np.abs(bn_unitless))),
        "K_mean": float(np.mean(k_norm)),
        "K_rms": float(np.sqrt(np.mean(k_norm**2))),
        "K_max": float(np.max(k_norm)),
        "K_p95": float(np.percentile(k_norm, 95)),
    }
    out.update(current_spectrum_metrics(phi, fields[0].Phi_basis.modes))
    return out


def coil_proxy_residuals(
    metrics,
    *,
    bn_max_target=5e-3,
    k_rms_target_MApm=2.2,
    phi_high_target=0.50,
):
    """Return hinge residuals for Bn_max, K_rms and high-mode current spectrum."""
    bn = float(metrics.get("Bn_max_abs_unitless", np.inf))
    k_mamp = float(metrics.get("K_rms", np.inf)) / 1e6
    phi_high = float(metrics.get("phi_high_mode_fraction", np.inf))
    return {
        "bn": max(bn - bn_max_target, 0.0),
        "k": max(k_mamp - k_rms_target_MApm, 0.0),
        "phi": max(phi_high - phi_high_target, 0.0),
    }
