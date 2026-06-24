"""Boundary curvature diagnostics for coil-engineering screening."""

from __future__ import annotations

import numpy as np


def _periodic_gradient(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


def _tail_mean_high(flat: np.ndarray, frac: float) -> float:
    if flat.size == 0:
        return np.nan
    n = max(1, int(np.ceil(flat.size * frac)))
    return float(np.mean(np.partition(flat, flat.size - n)[-n:]))


def _field_stats(field: np.ndarray, weight: np.ndarray | None = None) -> dict:
    arr = np.asarray(field, dtype=float)
    flat = arr[np.isfinite(arr)]
    out = {
        "min": float(np.min(flat)) if flat.size else np.nan,
        "max": float(np.max(flat)) if flat.size else np.nan,
        "mean": float(np.mean(flat)) if flat.size else np.nan,
        "p95": float(np.percentile(flat, 95)) if flat.size else np.nan,
        "p99": float(np.percentile(flat, 99)) if flat.size else np.nan,
        "cvar_top1": _tail_mean_high(flat, 0.01),
        "cvar_top3": _tail_mean_high(flat, 0.03),
    }
    if weight is not None:
        w = np.asarray(weight, dtype=float)
        mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
        total = float(np.sum(w[mask]))
        out["mean_area_weighted"] = (
            float(np.sum(arr[mask] * w[mask]) / total) if total > 0 else np.nan
        )
    return out


def _tangent_vectors(surface):
    # Simsopt quadpoints are normalized to [0, 1). Convert derivatives to
    # derivatives with respect to radian-valued (phi, theta).
    r_phi = np.asarray(surface.gammadash1(), dtype=float) / (2.0 * np.pi)
    r_theta = np.asarray(surface.gammadash2(), dtype=float) / (2.0 * np.pi)
    nphi, ntheta = r_phi.shape[:2]
    return r_phi, r_theta, 2.0 * np.pi / nphi, 2.0 * np.pi / ntheta


def _area_weights(surface) -> np.ndarray:
    r_phi, r_theta, dphi, dtheta = _tangent_vectors(surface)
    return np.linalg.norm(np.cross(r_theta, r_phi, axis=-1), axis=-1) * dtheta * dphi


def _surface_curvature_fields(surface):
    if hasattr(surface, "surface_curvatures"):
        curv = np.asarray(surface.surface_curvatures(), dtype=float)
        if curv.ndim != 3 or curv.shape[-1] < 4:
            raise ValueError(f"Unexpected surface_curvatures shape: {curv.shape}")
        # Simsopt returns [H, K, k1, k2] for current SurfaceRZFourier.
        return {
            "K": curv[:, :, 1],
            "H": curv[:, :, 0],
            "k1": curv[:, :, 2],
            "k2": curv[:, :, 3],
        }
    if hasattr(surface, "principal_curvatures"):
        k1, k2 = surface.principal_curvatures()
        return {
            "K": np.asarray(surface.gaussian_curvature(), dtype=float),
            "H": np.asarray(surface.mean_curvature(), dtype=float),
            "k1": np.asarray(k1, dtype=float),
            "k2": np.asarray(k2, dtype=float),
        }
    raise AttributeError("Surface does not expose simsopt curvature methods")


def _pdrot(surface, area_weight: np.ndarray) -> tuple[np.ndarray, dict]:
    r_phi, r_theta, dphi, dtheta = _tangent_vectors(surface)
    normal = np.asarray(surface.unitnormal(), dtype=float)

    g_phi_phi = np.sum(r_phi * r_phi, axis=-1)
    g_phi_theta = np.sum(r_phi * r_theta, axis=-1)
    g_theta_theta = np.sum(r_theta * r_theta, axis=-1)

    r_phi_phi = _periodic_gradient(r_phi, dphi, axis=0)
    r_phi_theta = _periodic_gradient(r_phi, dtheta, axis=1)
    r_theta_theta = _periodic_gradient(r_theta, dtheta, axis=1)

    h_phi_phi = np.sum(r_phi_phi * normal, axis=-1)
    h_phi_theta = np.sum(r_phi_theta * normal, axis=-1)
    h_theta_theta = np.sum(r_theta_theta * normal, axis=-1)

    nphi, ntheta = g_phi_phi.shape
    metric = np.zeros((nphi, ntheta, 2, 2))
    metric[..., 0, 0] = g_phi_phi
    metric[..., 0, 1] = g_phi_theta
    metric[..., 1, 0] = g_phi_theta
    metric[..., 1, 1] = g_theta_theta

    second = np.zeros_like(metric)
    second[..., 0, 0] = h_phi_phi
    second[..., 0, 1] = h_phi_theta
    second[..., 1, 0] = h_phi_theta
    second[..., 1, 1] = h_theta_theta

    metric_inv = np.linalg.inv(metric)
    shape = np.matmul(metric_inv, second)
    eigvals, eigvecs = np.linalg.eig(shape)
    principal_idx = np.argmax(eigvals, axis=-1)
    ii, jj = np.indices((nphi, ntheta))
    a = eigvecs[ii, jj, 0, principal_idx]
    b = eigvecs[ii, jj, 1, principal_idx]

    principal = a[..., None] * r_phi + b[..., None] * r_theta
    principal /= np.linalg.norm(principal, axis=-1, keepdims=True) + 1e-14

    e1 = r_phi / (np.sqrt(g_phi_phi)[..., None] + 1e-14)
    e2 = np.cross(normal, e1)
    cos_a = np.sum(principal * e1, axis=-1)
    sin_a = np.sum(principal * e2, axis=-1)

    # Double-angle representation removes the pi ambiguity of principal axes.
    cos_2a = cos_a**2 - sin_a**2
    sin_2a = 2.0 * cos_a * sin_a

    dcos_dphi = _periodic_gradient(cos_2a, dphi, axis=0)
    dcos_dtheta = _periodic_gradient(cos_2a, dtheta, axis=1)
    dsin_dphi = _periodic_gradient(sin_2a, dphi, axis=0)
    dsin_dtheta = _periodic_gradient(sin_2a, dtheta, axis=1)

    d2a_dphi = cos_2a * dsin_dphi - sin_2a * dcos_dphi
    d2a_dtheta = cos_2a * dsin_dtheta - sin_2a * dcos_dtheta

    g11 = metric_inv[..., 0, 0]
    g12 = metric_inv[..., 0, 1]
    g22 = metric_inv[..., 1, 1]
    norm_grad_sq = (
        g11 * d2a_dphi**2
        + 2.0 * g12 * d2a_dphi * d2a_dtheta
        + g22 * d2a_dtheta**2
    )
    pdrot = 0.5 * np.sqrt(np.maximum(norm_grad_sq, 0.0))
    return pdrot, _field_stats(pdrot, area_weight)


def _surface_twist(surface, area_weight: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return |M|/sqrt(|L*N|), a coordinate-vs-principal-direction twist proxy."""
    r_phi, r_theta, dphi, dtheta = _tangent_vectors(surface)
    normal = np.asarray(surface.unitnormal(), dtype=float)

    r_phi_phi = _periodic_gradient(r_phi, dphi, axis=0)
    r_phi_theta = _periodic_gradient(r_phi, dtheta, axis=1)
    r_theta_theta = _periodic_gradient(r_theta, dtheta, axis=1)

    l_coef = np.sum(r_phi_phi * normal, axis=-1)
    m_coef = np.sum(r_phi_theta * normal, axis=-1)
    n_coef = np.sum(r_theta_theta * normal, axis=-1)

    denom = np.sqrt(np.maximum(np.abs(l_coef * n_coef), 1e-300))
    tau = np.abs(m_coef) / denom
    tau[~np.isfinite(tau)] = np.nan
    return tau, _field_stats(tau, area_weight)


def boundary_geometry_metrics(
    wout_file: str,
    ntheta: int = 128,
    nphi: int = 128,
    torus_range: str = "full torus",
) -> dict:
    """Compute LCFS curvature statistics and principal-direction rotation.

    `pdrot` is |grad principal-axis angle| on the surface. Large area-weighted
    means or localized maxima indicate rapidly rotating principal curvature
    directions, which can correlate with coil non-planarity and coil curvature.
    """
    from simsopt.geo import SurfaceRZFourier

    surface = SurfaceRZFourier.from_wout(
        wout_file, range=torus_range, nphi=int(nphi), ntheta=int(ntheta)
    )
    area = _area_weights(surface)
    fields = _surface_curvature_fields(surface)
    pdrot_field, pdrot_stats = _pdrot(surface, area)
    tau_field, tau_stats = _surface_twist(surface, area)

    stats = {name: _field_stats(field, area) for name, field in fields.items()}
    stats["pdrot"] = pdrot_stats
    stats["tau_surf"] = tau_stats

    return {
        "source": "simsopt SurfaceRZFourier LCFS geometry",
        "grid": {"ntheta": int(ntheta), "nphi": int(nphi), "range": torus_range},
        "surface_area_m2_simsopt": float(np.sum(area)),
        "K_1_per_m2": stats["K"],
        "H_1_per_m": stats["H"],
        "k1_1_per_m": stats["k1"],
        "k2_1_per_m": stats["k2"],
        "pdrot_1_per_m": stats["pdrot"],
        "tau_surf": stats["tau_surf"],
        "pdrot_note": (
            "pdrot is the surface gradient magnitude of the principal curvature "
            "axis angle; larger values indicate faster principal-direction "
            "rotation and possible coil non-planarity/curvature pressure."
        ),
        "tau_surf_note": (
            "tau_surf = |M|/sqrt(|L*N|) measures how far the parameter-line "
            "directions are from principal curvature directions; use as a "
            "diagnostic, not a hard optimizer gate by default."
        ),
        "pdrot_field_max_1_per_m": float(np.nanmax(pdrot_field)),
        "tau_surf_field_p95": float(np.nanpercentile(tau_field, 95)),
    }
