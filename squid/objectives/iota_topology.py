"""Residuals that protect self-consistent iota topology during optimization."""

import numpy as np


def iota_topology_residuals(
    iota,
    *,
    reference_iota,
    sample_s,
    profile_tolerance=0.004,
    profile_scale=0.004,
    shear_absmin=0.005,
    shear_scale=0.0025,
    reference_direction=1.0,
    monotonic_scale=0.0025,
):
    """Return fixed-size iota profile, shear, and monotonicity residuals."""
    iota = np.asarray(iota, dtype=float).ravel()
    reference_iota = np.asarray(reference_iota, dtype=float).ravel()
    sample_s = np.asarray(sample_s, dtype=float).ravel()
    if iota.size < 3:
        raise ValueError("iota must contain at least 3 radial points")
    if sample_s.size < 3 or reference_iota.shape != sample_s.shape:
        raise ValueError("reference_iota and sample_s must have the same length >= 3")
    if not np.all(np.isfinite(iota)) or not np.all(np.isfinite(reference_iota)):
        raise ValueError("iota profiles must be finite")
    if not np.all(np.diff(sample_s) > 0):
        raise ValueError("sample_s must be strictly increasing")
    numeric_parameters = (
        profile_tolerance, profile_scale, shear_absmin, shear_scale,
        reference_direction, monotonic_scale,
    )
    if not np.all(np.isfinite(np.asarray(numeric_parameters, dtype=float))):
        raise ValueError("iota topology parameters must be finite")

    source_s = np.linspace(0.0, 1.0, iota.size)
    current = np.interp(sample_s, source_s, iota)
    source_gradient = np.gradient(iota, source_s)
    gradient = np.interp(sample_s, source_s, source_gradient)

    norm = np.sqrt(sample_s.size)
    profile_scale = max(float(profile_scale), 1e-12)
    shear_scale = max(float(shear_scale), 1e-12)
    monotonic_scale = max(float(monotonic_scale), 1e-12)
    direction = float(np.sign(reference_direction)) or 1.0

    deviation = np.abs(current - reference_iota)
    profile_residuals = (
        np.maximum(deviation - max(float(profile_tolerance), 0.0), 0.0)
        / (profile_scale * norm)
    )
    shear_residuals = (
        np.maximum(max(float(shear_absmin), 0.0) - np.abs(gradient), 0.0)
        / (shear_scale * norm)
    )
    signed_gradient = direction * gradient
    monotonic_residuals = (
        np.maximum(-signed_gradient, 0.0) / (monotonic_scale * norm)
    )

    i_min = int(np.argmin(np.abs(gradient)))
    metrics = {
        "profile_rms_dev": float(np.sqrt(np.mean((current - reference_iota) ** 2))),
        "profile_max_dev": float(np.max(deviation)),
        "shear_absmin": float(np.abs(gradient[i_min])),
        "shear_absmin_s": float(sample_s[i_min]),
        "monotonic_violation_count": int(np.sum(signed_gradient < 0.0)),
        "reference_direction": direction,
    }
    return {
        "profile_residuals": profile_residuals,
        "shear_residuals": shear_residuals,
        "monotonic_residuals": monotonic_residuals,
        "metrics": metrics,
    }
