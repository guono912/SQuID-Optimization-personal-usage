"""Canonical Mercier normalization for VMEC equilibria.

VMEC's ``DMerc`` is defined using derivatives with respect to physical
toroidal flux, so its raw magnitude carries the inverse-square flux scale.
SQuID's paper-facing convention is therefore

    dmerc_flux_normalized = Phi_edge**2 * dmerc_vmec_raw

where ``Phi_edge`` is read directly from ``wout.phi[-1]`` in webers. The
sign is unchanged. This module is the only place where that conversion
should be implemented.
"""

from __future__ import annotations

import numpy as np


MERCIER_CONVENTION_VERSION = "squid-flux-normalized-v1"
MERCIER_FLUX_NORMALIZATION = "Phi_edge^2 * VMEC.DMerc"


def _read_array(source, name):
    """Read a named VMEC variable from a wout object or netCDF dataset."""
    if hasattr(source, "variables") and name in source.variables:
        value = source.variables[name][:]
    elif hasattr(source, name):
        value = getattr(source, name)
    else:
        raise KeyError(f"VMEC variable {name!r} is not available")
    return np.asarray(value, dtype=float)


def edge_toroidal_flux_wb(source):
    """Return ``abs(phi[-1])``, VMEC's physical edge toroidal flux in Wb."""
    phi = _read_array(source, "phi").reshape(-1)
    if phi.size == 0 or not np.isfinite(phi[-1]) or abs(phi[-1]) <= 0.0:
        raise ValueError("VMEC wout.phi[-1] is missing, non-finite, or zero")
    return float(abs(phi[-1]))


def flux_normalize_mercier(values, source_or_flux):
    """Multiply a VMEC Mercier profile or component by ``Phi_edge**2``."""
    if np.isscalar(source_or_flux):
        phi_edge = float(abs(source_or_flux))
    else:
        phi_edge = edge_toroidal_flux_wb(source_or_flux)
    if not np.isfinite(phi_edge) or phi_edge <= 0.0:
        raise ValueError("edge toroidal flux must be finite and positive")
    return np.asarray(values, dtype=float) * phi_edge**2


def vmec_half_grid_profile(values, ns=None):
    """Return a VMEC radial diagnostic on its physical half-grid.

    VMEC stores Mercier profiles with a leading non-physical placeholder, so
    an array of length ``ns`` represents ``ns - 1`` samples at
    ``(j - 1/2) / (ns - 1)``, for ``j = 1, ..., ns - 1``.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    ns = int(values.size if ns is None else ns)
    if ns < 2:
        raise ValueError("VMEC half-grid profiles require ns >= 2")
    if values.size == ns:
        values = values[1:]
    if values.size != ns - 1:
        raise ValueError(
            f"expected VMEC half-grid profile of length {ns - 1}, "
            f"got {values.size}"
        )
    s_half = (np.arange(ns - 1, dtype=float) + 0.5) / (ns - 1)
    return s_half, values


def mercier_profiles(source):
    """Return raw and flux-normalized VMEC Mercier profiles and components."""
    phi_edge = edge_toroidal_flux_wb(source)
    profiles = {}
    for name in ("DMerc", "DShear", "DWell", "DCurr", "DGeod"):
        try:
            raw = _read_array(source, name).reshape(-1)
        except KeyError:
            continue
        profiles[name] = {
            "vmec_raw": raw,
            "flux_normalized": flux_normalize_mercier(raw, phi_edge),
        }
    if "DMerc" not in profiles:
        raise KeyError("VMEC variable 'DMerc' is not available")
    return {
        "convention": MERCIER_CONVENTION_VERSION,
        "formula": MERCIER_FLUX_NORMALIZATION,
        "edge_toroidal_flux_wb": phi_edge,
        "profiles": profiles,
    }


def mercier_summary(source, s_min=0.1, s_max=0.95):
    """Summarize Mercier stability on a declared normalized-flux interval."""
    result = mercier_profiles(source)
    raw_full = result["profiles"]["DMerc"]["vmec_raw"]
    normalized_full = result["profiles"]["DMerc"]["flux_normalized"]
    s, raw = vmec_half_grid_profile(raw_full)
    _, normalized = vmec_half_grid_profile(normalized_full, ns=raw_full.size)
    mask = (
        np.isfinite(raw)
        & np.isfinite(normalized)
        & (s >= float(s_min))
        & (s <= float(s_max))
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("no finite DMerc samples in the requested radial interval")
    index = int(indices[np.argmin(normalized[indices])])
    result.update(
        s=s,
        s_min=float(s_min),
        s_max=float(s_max),
        minimum_s=float(s[index]),
        dmerc_vmec_raw_min=float(raw[index]),
        dmerc_flux_normalized_min=float(normalized[index]),
        dmerc_negative_count=int(np.count_nonzero(raw[mask] < 0.0)),
        sample_count=int(indices.size),
    )
    return result
