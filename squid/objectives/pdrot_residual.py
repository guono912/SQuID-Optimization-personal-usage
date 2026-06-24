"""SQuID pdrot objective — principal-direction rotation rate on VMEC boundary.

This module wraps the standalone ``pdrot_target.py`` algorithm into a
SQuID-compatible objective that computes::

    pdrot = |grad_S(alpha)|

where *alpha* is the angle between the first principal-curvature direction and
the local poloidal tangent direction.  The residual vector consists of
pointwise area-weighted inequality penalties::

    max(a_eff * pdrot - q_target, 0)
    max(pdrot - rho_target, 0)

plus optional area-mean penalties.

Unlike the existing ``w_boundary_curvature`` proxy (which reads a wout file),
this objective works directly on the SIMSOPT ``SurfaceRZFourier``, so it does
**not** require an extra VMEC run.

Notes
-----
* The surface grid must cover one complete field period or the full torus,
  and must be uniform and endpoint-free in both angles.
* There is no analytic Jacobian; the optimiser must use finite-difference
  gradients for this term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

Array = np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# Core differential-geometry computation
# ═══════════════════════════════════════════════════════════════════════════


def _dot(a: Array, b: Array) -> Array:
    """Pointwise dot product over the final Cartesian axis."""
    return np.einsum("...i,...i->...", a, b)


def _uniform_periodic_spacing(
    points: Array,
    *,
    expected_periods: Tuple[float, ...],
    name: str,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
) -> float:
    """Validate an endpoint-free uniform periodic grid and return its spacing."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 1 or points.size < 5:
        raise ValueError(f"{name} must be a 1-D grid containing at least 5 points.")

    diffs = np.diff(points)
    h = float(np.mean(diffs))
    if h <= 0.0 or not np.allclose(diffs, h, rtol=rtol, atol=atol):
        raise ValueError(f"{name} must be uniformly spaced and strictly increasing.")

    inferred_period = h * points.size
    if not any(
        np.isclose(inferred_period, period, rtol=rtol, atol=atol)
        for period in expected_periods
    ):
        allowed = ", ".join(f"{p:.16g}" for p in expected_periods)
        raise ValueError(
            f"{name} has inferred period {inferred_period:.16g}; expected one of "
            f"[{allowed}]. Use a full-torus or one-field-period surface grid, "
            "not a half-period grid."
        )
    return h


def _periodic_central_difference(values: Array, spacing: float, axis: int) -> Array:
    """Second-order central difference on an endpoint-free periodic grid."""
    return (
        np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
    ) / (2.0 * spacing)


def _positive_part(x: Array, smooth_eps: float) -> Array:
    """Hard or smooth positive part [x]_+."""
    if smooth_eps < 0.0:
        raise ValueError("smooth_eps must be non-negative.")
    if smooth_eps == 0.0:
        return np.maximum(x, 0.0)
    return 0.5 * (x + np.sqrt(x * x + smooth_eps * smooth_eps))


def principal_direction_rotation_rate(
    surface,
    *,
    delta_kappa_a: float = 1.0e-3,
    metric_floor: float = 1.0e-28,
) -> Dict[str, Array | float]:
    """Compute the regularized principal-direction rotation rate on a surface.

    Parameters
    ----------
    surface:
        A SIMSOPT Surface, normally ``vmec.boundary``.
    delta_kappa_a:
        Dimensionless regularization scale ``delta_kappa * a_eff`` for nearly
        umbilic points.
    metric_floor:
        Positive floor used only to detect a degenerate surface metric.

    Returns
    -------
    dict
        Contains ``pdrot`` [1/m], ``q=a_eff*pdrot``, ``area_weights``,
        ``area_element``, ``a_eff``, ``mean_pdrot``, ``mean_q``, ``p95_pdrot``,
        ``p95_q``, ``kappa_gap``, and intermediate angle derivatives.
    """
    if delta_kappa_a < 0.0:
        raise ValueError("delta_kappa_a must be non-negative.")

    r_phi = np.asarray(surface.gammadash1(), dtype=float)
    r_theta = np.asarray(surface.gammadash2(), dtype=float)
    r_phiphi = np.asarray(surface.gammadash1dash1(), dtype=float)
    r_thetatheta = np.asarray(surface.gammadash2dash2(), dtype=float)
    r_phitheta = np.asarray(surface.gammadash1dash2(), dtype=float)

    expected_shape = r_phi.shape
    if (
        r_phi.ndim != 3
        or r_phi.shape[-1] != 3
        or r_theta.shape != expected_shape
        or r_phiphi.shape != expected_shape
        or r_thetatheta.shape != expected_shape
        or r_phitheta.shape != expected_shape
    ):
        raise ValueError(
            "Unexpected SIMSOPT surface derivative shapes. Expected all arrays "
            "to have shape (nphi, ntheta, 3)."
        )

    phi = np.asarray(surface.quadpoints_phi, dtype=float)
    theta = np.asarray(surface.quadpoints_theta, dtype=float)
    nfp = int(surface.nfp)

    dphi = _uniform_periodic_spacing(
        phi,
        expected_periods=(1.0, 1.0 / nfp),
        name="surface.quadpoints_phi",
    )
    dtheta = _uniform_periodic_spacing(
        theta,
        expected_periods=(1.0,),
        name="surface.quadpoints_theta",
    )

    ru = r_theta
    rv = r_phi
    ruu = r_thetatheta
    ruv = r_phitheta
    rvv = r_phiphi

    E = _dot(ru, ru)
    F = _dot(ru, rv)
    G = _dot(rv, rv)
    detg = E * G - F * F

    if np.any(~np.isfinite(detg)) or np.min(detg) <= metric_floor:
        raise FloatingPointError(
            "The surface metric is singular or non-finite at one or more grid points."
        )

    cross = np.cross(ru, rv)
    jac = np.linalg.norm(cross, axis=2)
    nhat = cross / jac[..., None]

    L = _dot(ruu, nhat)
    M = _dot(ruv, nhat)
    N = _dot(rvv, nhat)

    sqrt_detg = np.sqrt(detg)

    # Second fundamental form in orthonormal tangent frame
    b11 = L / E
    b12 = (E * M - F * L) / (E * sqrt_detg)
    b22 = (E * E * N - 2.0 * E * F * M + F * F * L) / (E * detg)

    X = b11 - b22
    Y = 2.0 * b12
    gap2 = X * X + Y * Y
    kappa_gap = np.sqrt(gap2)

    a_eff = float(surface.minor_radius())
    if not np.isfinite(a_eff) or a_eff <= 0.0:
        raise FloatingPointError(f"Invalid effective minor radius: {a_eff}")

    delta_kappa = delta_kappa_a / a_eff
    regularized_denominator = gap2 + delta_kappa * delta_kappa

    dX_dphi = _periodic_central_difference(X, dphi, axis=0)
    dY_dphi = _periodic_central_difference(Y, dphi, axis=0)
    dX_dtheta = _periodic_central_difference(X, dtheta, axis=1)
    dY_dtheta = _periodic_central_difference(Y, dtheta, axis=1)

    alpha_phi = 0.5 * (X * dY_dphi - Y * dX_dphi) / regularized_denominator
    alpha_theta = (
        0.5 * (X * dY_dtheta - Y * dX_dtheta) / regularized_denominator
    )

    pdrot2 = (
        G * alpha_theta * alpha_theta
        - 2.0 * F * alpha_theta * alpha_phi
        + E * alpha_phi * alpha_phi
    ) / detg
    pdrot = np.sqrt(np.maximum(pdrot2, 0.0))
    q = a_eff * pdrot

    area_weights = jac / np.sum(jac)

    mean_pdrot = float(np.sum(area_weights * pdrot))
    mean_q = float(np.sum(area_weights * q))
    p95_pdrot = float(np.percentile(pdrot, 95.0))
    p95_q = float(np.percentile(q, 95.0))

    return {
        "pdrot": pdrot,
        "q": q,
        "area_weights": area_weights,
        "area_element": jac,
        "a_eff": a_eff,
        "mean_pdrot": mean_pdrot,
        "mean_q": mean_q,
        "p95_pdrot": p95_pdrot,
        "p95_q": p95_q,
        "kappa_gap": kappa_gap,
        "alpha_phi": alpha_phi,
        "alpha_theta": alpha_theta,
        "delta_kappa": delta_kappa,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Penalty settings dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PDROTPenaltySettings:
    """Settings for the dimensionless + absolute pdrot inequality objective."""

    q_target: Optional[float] = None
    rho_target: Optional[float] = None
    mean_q_target: Optional[float] = None
    mean_rho_target: Optional[float] = None
    weight_q: float = 1.0
    weight_rho: float = 1.0
    weight_mean_q: float = 0.0
    weight_mean_rho: float = 0.0
    delta_kappa_a: float = 1.0e-3
    smooth_eps_q: float = 0.0
    smooth_eps_rho: float = 0.0

    def validate(self) -> None:
        thresholds = {
            "q_target": self.q_target,
            "rho_target": self.rho_target,
            "mean_q_target": self.mean_q_target,
            "mean_rho_target": self.mean_rho_target,
        }
        for _name, value in thresholds.items():
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{_name} must be finite and non-negative or None.")

        weights = {
            "weight_q": self.weight_q,
            "weight_rho": self.weight_rho,
            "weight_mean_q": self.weight_mean_q,
            "weight_mean_rho": self.weight_mean_rho,
        }
        for _name, value in weights.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{_name} must be finite and non-negative.")

        if self.delta_kappa_a < 0.0:
            raise ValueError("delta_kappa_a must be non-negative.")
        if self.smooth_eps_q < 0.0 or self.smooth_eps_rho < 0.0:
            raise ValueError("Smoothing epsilons must be non-negative.")

        active = (
            self.q_target is not None and self.weight_q > 0.0,
            self.rho_target is not None and self.weight_rho > 0.0,
            self.mean_q_target is not None and self.weight_mean_q > 0.0,
            self.mean_rho_target is not None and self.weight_mean_rho > 0.0,
        )
        if not any(active):
            raise ValueError("At least one pdrot penalty term must be active.")


def pdrot_residuals(surface, settings: PDROTPenaltySettings) -> Array:
    """Return a 1-D least-squares residual vector.

    Use with goal=0 and outer weight=1.  Weights in ``settings`` already appear
    inside the residual vector as square roots.
    """
    settings.validate()
    data = principal_direction_rotation_rate(
        surface, delta_kappa_a=settings.delta_kappa_a
    )

    pdrot = np.asarray(data["pdrot"])
    q = np.asarray(data["q"])
    area_weights = np.asarray(data["area_weights"])
    residual_blocks = []

    if settings.q_target is not None and settings.weight_q > 0.0:
        excess_q = _positive_part(q - settings.q_target, settings.smooth_eps_q)
        residual_blocks.append(
            np.sqrt(settings.weight_q * area_weights).ravel() * excess_q.ravel()
        )

    if settings.rho_target is not None and settings.weight_rho > 0.0:
        excess_rho = _positive_part(
            pdrot - settings.rho_target, settings.smooth_eps_rho
        )
        residual_blocks.append(
            np.sqrt(settings.weight_rho * area_weights).ravel()
            * excess_rho.ravel()
        )

    if settings.mean_q_target is not None and settings.weight_mean_q > 0.0:
        mean_q_excess = _positive_part(
            np.asarray([float(data["mean_q"]) - settings.mean_q_target]),
            settings.smooth_eps_q,
        )
        residual_blocks.append(np.sqrt(settings.weight_mean_q) * mean_q_excess)

    if settings.mean_rho_target is not None and settings.weight_mean_rho > 0.0:
        mean_rho_excess = _positive_part(
            np.asarray([float(data["mean_pdrot"]) - settings.mean_rho_target]),
            settings.smooth_eps_rho,
        )
        residual_blocks.append(
            np.sqrt(settings.weight_mean_rho) * mean_rho_excess
        )

    return np.concatenate(residual_blocks).astype(float, copy=False)


# ═══════════════════════════════════════════════════════════════════════════
# SQuID-compatible wrapper
# ═══════════════════════════════════════════════════════════════════════════


def compute_pdrot_from_vmec(vmec, args):
    """Compute pdrot residuals and diagnostics from a VMEC boundary.

    Parameters
    ----------
    vmec: simsopt.mhd.Vmec instance (after ``vmec.run()``).
    args: argparse.Namespace or similar object with weight/target attributes.

    Returns
    -------
    residuals: np.ndarray (1-D, for LeastSquaresProblem)
    metrics: dict with scalar diagnostics for history/logging.
    """
    surf = vmec.boundary

    q_target = getattr(args, "pdrot_q_target", None) or None
    rho_target = getattr(args, "pdrot_rho_target", None) or None

    # Fast path: if no targets are set, return zero residual with diagnostics only
    if q_target is None and rho_target is None:
        try:
            diag = principal_direction_rotation_rate(surf)
        except Exception:
            diag = {}
        metrics = _build_metrics(diag)
        return np.zeros(1, dtype=float), metrics

    settings = PDROTPenaltySettings(
        q_target=q_target,
        rho_target=rho_target,
        weight_q=max(float(getattr(args, "pdrot_weight_q", 1.0) or 1.0), 0.0),
        weight_rho=max(float(getattr(args, "pdrot_weight_rho", 1.0) or 1.0), 0.0),
        mean_q_target=getattr(args, "pdrot_mean_q_target", None) or None,
        mean_rho_target=getattr(args, "pdrot_mean_rho_target", None) or None,
        weight_mean_q=max(float(getattr(args, "pdrot_weight_mean_q", 0.0) or 0.0), 0.0),
        weight_mean_rho=max(float(getattr(args, "pdrot_weight_mean_rho", 0.0) or 0.0), 0.0),
        delta_kappa_a=float(getattr(args, "pdrot_delta_kappa_a", 0.001)),
        smooth_eps_q=float(getattr(args, "pdrot_smooth_eps_q", 0.0)),
        smooth_eps_rho=float(getattr(args, "pdrot_smooth_eps_rho", 0.0)),
    )

    try:
        residuals = pdrot_residuals(surf, settings)
        diag = principal_direction_rotation_rate(
            surf, delta_kappa_a=settings.delta_kappa_a
        )
    except Exception:
        residuals = np.array([1.0])
        diag = {}

    metrics = _build_metrics(diag)
    return residuals, metrics


def _build_metrics(diag):
    """Extract scalar diagnostics from a pdrot computation result."""
    pdrot_arr = np.asarray(diag.get("pdrot", []))
    return {
        "pdrot_mean": float(diag.get("mean_pdrot", np.nan)),
        "pdrot_max": float(np.max(pdrot_arr)) if pdrot_arr.size else np.nan,
        "pdrot_p95": float(diag.get("p95_pdrot", np.nan)),
        "pdrot_p99": float(np.percentile(pdrot_arr, 99)) if pdrot_arr.size else np.nan,
        "pdrot_q_mean": float(diag.get("mean_q", np.nan)),
        "pdrot_q_p95": float(diag.get("p95_q", np.nan)),
        "pdrot_a_eff": float(diag.get("a_eff", np.nan)),
        "pdrot_kappa_gap_max": float(np.max(diag["kappa_gap"])) if diag else np.nan,
    }
