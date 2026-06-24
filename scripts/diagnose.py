#!/usr/bin/env python3
"""
SQuID Diagnostic Tool — evaluate an equilibrium without optimisation.

Computes all SQuID target function components and generates
diagnostic plots, including stability and transport metrics.

Usage:
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --plot
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended \
        --extended_surfaces 0.1 0.3 0.5
    python scripts/diagnose.py --nc_file path/to/wout_xxx.nc --extended \
        --ae --ae_surfaces 0.2 0.5 0.8 --plot

Tiers:
    default      -> core SQuID + equilibrium sanity + geometry/stability preview
    --extended   -> add third-tier transport diagnostics (ITG)
    --ae         -> add Available Energy on top of --extended
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import argparse
import json
from pathlib import Path
from fractions import Fraction
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _evaluate_effective_ripple(vmec, s_val, nc_file_path, output_dir="."):
    """
    Evaluate effective ripple eps_eff at normalised flux *s_val*.

    Graceful degradation:
      1. DESC (Nemov formula via bounce integrals) — most reliable.
      2. NEO-RT — Boozer transform + Fortran solver.
      3. Boozer proxy fallback — (B_max - B_min) / (2*B00).

    Returns
    -------
    (value, source_str) : (float | None, str)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Priority 1: DESC effective ripple (Nemov formula) ────────────
    try:
        import numpy as _np
        from desc.vmec import VMECIO
        from desc.grid import LinearGrid
        from desc.objectives import EffectiveRipple

        print(f"      [DESC] Computing effective ripple at s={s_val} ...")
        eq = VMECIO.load(nc_file_path)
        rho = _np.array([_np.sqrt(s_val)])
        grid = LinearGrid(
            rho=rho, M=eq.M_grid, N=eq.N_grid,
            NFP=eq.NFP, sym=False,
        )
        obj = EffectiveRipple(eq, grid=grid, num_transit=10, num_pitch=31)
        obj.build(verbose=0)
        f = obj.compute(eq.params_dict)
        eps_eff = float(f[0])
        if not _np.isfinite(eps_eff):
            raise ValueError("DESC returned NaN/Inf")
        return (eps_eff, "DESC (Nemov bounce-integral)")

    except ImportError:
        print("      [DESC] Not available, trying NEO-RT ...")
    except Exception as e:
        print(f"      [DESC] Failed: {e}, trying NEO-RT ...")

    # ── Priority 2: NEO-RT ──────────────────────────────────────────
    try:
        from simsopt.mhd.boozer import Boozer
        import nc_to_neort
        import subprocess

        # Register enough surfaces for stable cubic spline interpolation.
        # Particle drift orbits can wander radially; a narrow [s±0.1]
        # range causes spline extrapolation → NaN.
        raw = [s_val - 0.2, s_val - 0.1, s_val, s_val + 0.1, s_val + 0.2]
        s_surfaces = sorted(set(max(0.01, min(0.99, s)) for s in raw))
        print(f"      [NEO-RT] Running Boozer transform at s={s_surfaces} ...")
        boozer = Boozer(vmec, mpol=16, ntor=16)
        for s in s_surfaces:
            boozer.register(s)
        boozer.run()

        latest_boozmn = output_dir / "boozmn_squid_diag.nc"
        try:
            boozer.write_boozmn(str(latest_boozmn))
        except AttributeError:
            boozer.bx.write_boozmn(str(latest_boozmn))
        if not latest_boozmn.is_file():
            raise FileNotFoundError(
                f"write_boozmn() did not produce {latest_boozmn}")
        print(f"      [NEO-RT] Using Boozer file: {latest_boozmn}")

        # Convert boozmn netCDF → NEO-RT ASCII "in_file"
        # phi_b and aspect_b are "not implemented" in this booz_xform,
        # so we supply the real values from VMEC.
        phi_edge = float(vmec.wout.phi[-1])         # total toroidal flux [Tm²]
        a_minor  = float(vmec.wout.Aminor_p)        # minor radius [m]
        result = nc_to_neort.convert_boozmn_to_neort(
            str(latest_boozmn), output_path=str(output_dir / "in_file"),
            s_values=s_surfaces,
            flux_override=phi_edge, a_override=a_minor,
        )
        if result is None:
            raise RuntimeError("boozmn → in_file conversion failed")
        in_file_path, epsmn, pert_m0, pert_mph = result

        # Locate neo_rt.x
        neort_exe = os.environ.get("NEORT_EXECUTABLE", "")
        if not neort_exe and os.environ.get("NEO_RT_ROOT"):
            neort_exe = os.path.join(
                os.environ["NEO_RT_ROOT"], "build", "neo_rt.x")
        if not neort_exe or not os.path.isfile(neort_exe):
            raise FileNotFoundError(
                f"neo_rt.x not found (NEORT_EXECUTABLE={neort_exe!r}). "
                "Set NEORT_EXECUTABLE or NEO_RT_ROOT.")

        # Write a minimal NEO-RT namelist for transport evaluation.
        # NOTE: run_driftorbit.run_single_flux_surface() is NOT used here
        # because it prepends './' to the executable path, which breaks
        # absolute paths.  We call neo_rt.x via subprocess directly.
        runname = "squid_diag"
        with open(output_dir / f"{runname}.in", "w") as fh:
            fh.write(
                "&params\n"
                f"    s = {s_val}\n"
                "    m_t = 1.0d-2\n"
                "    qs = 1.0\n"
                "    ms = 2.014\n"
                "    vth = 1.0d8\n"
                f"    epsmn = {epsmn}\n"
                f"    m0 = {pert_m0}\n"
                f"    mph = {max(pert_mph, 1)}\n"
                "    magdrift = .true.\n"
                "    nopassing = .false.\n"
                "    noshear = .false.\n"
                "    pertfile = .false.\n"
                "    nonlin = .false.\n"
                "    bfac = 1.0\n"
                "    efac = 1.0\n"
                "    inp_swi = 9\n"
                "    vsteps = 512\n"
                "    log_level = -1\n"
                "/\n"
            )

        print(f"      [NEO-RT] Running {os.path.basename(neort_exe)} {runname} ...")
        result = subprocess.run(
            [neort_exe, runname],
            capture_output=True, text=True, timeout=300,
            cwd=str(output_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"neo_rt.x exited with code {result.returncode}: "
                f"{result.stderr[:300]}")

        # Parse D11 from {runname}.out
        # Header: "# M_t D11co D11ctr D11t D11 D12co D12ctr D12t D12"
        out_file = output_dir / f"{runname}.out"
        if not out_file.is_file():
            raise FileNotFoundError(f"{out_file} not written by neo_rt.x")
        with open(out_file) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                cols = stripped.split()
                if len(cols) >= 5:
                    D11 = float(cols[4])
                    if not np.isfinite(D11):
                        raise ValueError(
                            "NEO-RT returned NaN/Inf — likely spline "
                            "extrapolation or extreme ripple "
                            f"(eps≈{epsmn:.2f})")
                    return (D11,
                            "NEO-RT D11 (neoclassical transport coeff)")
        raise RuntimeError(f"Could not parse D11 from {out_file}")

    except ImportError as e:
        print(f"      [Fallback] NEO-RT dependencies not available: {e}")
    except FileNotFoundError as e:
        print(f"      [Fallback] File not found: {e}")
    except Exception as e:
        print(f"      [Fallback] NEO-RT evaluation failed: {e}")

    # ── Priority 2: Boozer proxy (fallback) ─────────────────────────
    try:
        from squid.core.boozer_utils import run_boozer

        _, surface_data = run_boozer(vmec, [s_val], mpol=16, ntor=16)
        data = surface_data[0]
        B_min, B_max = data["B_min"], data["B_max"]
        m_arr = np.array(data["m"])
        n_arr = np.array(data["n"])
        bmnc = np.array(data["bmnc"])
        idx_00 = np.where((m_arr == 0) & (n_arr == 0))[0]
        B00 = float(np.abs(bmnc[idx_00[0]])) if len(idx_00) > 0 else (B_max + B_min) / 2.0
        if B00 > 1e-30:
            eps_proxy = (B_max - B_min) / (2.0 * B00)
            return (float(eps_proxy), "Boozer proxy (B_max-B_min)/(2*B00)")
    except Exception as e:
        return (None, f"Boozer proxy failed: {e}")

    return (None, "No method available")


def _evaluate_effective_ripple_series(vmec, s_vals, nc_file_path, output_dir="."):
    """Evaluate the ripple metric on several representative surfaces."""
    results = []
    for s_val in s_vals:
        value, source = _evaluate_effective_ripple(
            vmec, float(s_val), nc_file_path, output_dir=output_dir)
        results.append({
            "s": float(s_val),
            "value": value,
            "source": source,
        })
    return results


def _jsonify(obj):
    """Convert numpy-rich diagnostic objects to JSON-serialisable values."""
    if isinstance(obj, np.ndarray):
        return [_jsonify(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    return obj


def _compute_equilibrium_sanity(vmec, squid_info):
    """Basic numerical sanity checks for a single equilibrium."""
    fails, warns = [], []
    wout = vmec.wout

    scalar_checks = {
        "aspect": float(getattr(wout, "aspect", np.nan)),
        "Aminor_p": float(getattr(wout, "Aminor_p", np.nan)),
        "phi_edge": float(np.array(getattr(wout, "phi", [np.nan]))[-1]),
    }
    for name, value in scalar_checks.items():
        if not np.isfinite(value):
            fails.append(f"{name} is not finite")
    if np.isfinite(scalar_checks["Aminor_p"]) and scalar_checks["Aminor_p"] <= 0:
        fails.append("Aminor_p <= 0")

    iotaf = np.array(getattr(wout, "iotaf", []), dtype=float)
    if iotaf.size < 2:
        fails.append("iota profile missing or too short")
    elif not np.all(np.isfinite(iotaf[[0, -1]])):
        fails.append("iota axis/edge is not finite")

    for key in ("f_QI", "f_maxJ", "mirror_ratio"):
        value = float(squid_info.get(key, np.nan))
        if not np.isfinite(value):
            fails.append(f"{key} is not finite")

    if not squid_info.get("common_B_valid", True):
        fails.append("no common B* range across requested diagnostic surfaces")

    if not np.all(np.isfinite(iotaf)):
        warns.append("interior iota profile contains non-finite values")

    status = "OK"
    if fails:
        status = "FAIL"
    elif warns:
        status = "WARN"
    return {"status": status, "fails": fails, "warns": warns}


def _print_equilibrium_sanity(sanity):
    print(f"  Status: {sanity['status']}")
    for msg in sanity["fails"]:
        print(f"    FAIL: {msg}")
    for msg in sanity["warns"]:
        print(f"    WARN: {msg}")


def _sorted_surface_dict(metric_dict):
    """Return sorted (s, value) arrays from a {surface: value} dict."""
    items = [
        (float(k), float(v))
        for k, v in metric_dict.items()
        if np.isscalar(k)
    ]
    items.sort(key=lambda kv: kv[0])
    s = np.array([float(k) for k, _ in items], dtype=float)
    values = np.array([float(v) for _, v in items], dtype=float)
    return s, values


def _low_order_rationals(max_denominator, max_value):
    """Return unique positive p/q values up to *max_denominator*."""
    rationals = set()
    max_value = max(float(max_value), 0.0)
    for q in range(1, int(max_denominator) + 1):
        p_max = int(np.ceil(max_value * q)) + 1
        for p in range(1, p_max + 1):
            rationals.add(Fraction(p, q))
    return sorted(rationals, key=lambda x: (float(x), x.denominator))


def _scan_iota_rationals(vmec, max_denominator=8, warn_distance=0.01,
                         s_min=0.05, s_max=0.95):
    """
    Scan the VMEC iota profile against low-order rational surfaces.

    This function only reports geometric facts relative to the supplied
    threshold. It does not decide whether a configuration should be accepted.
    """
    iotaf = np.array(getattr(vmec.wout, "iotaf", []), dtype=float)
    if iotaf.size < 2 or not np.any(np.isfinite(iotaf)):
        return {
            "available": False,
            "reason": "iota profile missing or non-finite",
        }

    s_grid = np.linspace(0.0, 1.0, iotaf.size)
    mask = (
        (s_grid >= float(s_min)) &
        (s_grid <= float(s_max)) &
        np.isfinite(iotaf)
    )
    if not np.any(mask):
        return {
            "available": False,
            "reason": "no finite iota points in requested scan range",
        }

    s_scan = s_grid[mask]
    iota_scan = iotaf[mask]
    abs_iota = np.abs(iota_scan)
    sign = -1.0 if np.nanmedian(iota_scan) < 0 else 1.0

    rationals = _low_order_rationals(
        max_denominator=max_denominator,
        max_value=np.nanmax(abs_iota) + abs(warn_distance),
    )

    nearest = None
    near_points = []
    crossings = []

    for rat in rationals:
        rat_value = float(rat)
        distances = np.abs(abs_iota - rat_value)
        idx = int(np.nanargmin(distances))
        distance = float(distances[idx])
        record = {
            "s": float(s_scan[idx]),
            "iota": float(iota_scan[idx]),
            "rational": f"{int(sign * rat.numerator)}/{rat.denominator}",
            "rational_abs": rat_value,
            "denominator": int(rat.denominator),
            "distance": distance,
        }
        if nearest is None or distance < nearest["distance"]:
            nearest = record
        if distance <= warn_distance:
            near_points.append(record)

        diff = abs_iota - rat_value
        for j in range(len(diff) - 1):
            if not (np.isfinite(diff[j]) and np.isfinite(diff[j + 1])):
                continue
            if diff[j] == 0:
                s_cross = float(s_scan[j])
            elif diff[j] * diff[j + 1] > 0:
                continue
            else:
                denom = diff[j + 1] - diff[j]
                if abs(denom) < 1e-30:
                    continue
                t = -diff[j] / denom
                if t < 0 or t > 1:
                    continue
                s_cross = float(s_scan[j] + t * (s_scan[j + 1] - s_scan[j]))
            crossings.append({
                "s": s_cross,
                "rational": f"{int(sign * rat.numerator)}/{rat.denominator}",
                "rational_abs": rat_value,
                "denominator": int(rat.denominator),
            })

    near_points.sort(key=lambda item: item["distance"])
    crossings.sort(key=lambda item: (item["s"], item["denominator"]))
    shear = np.gradient(iota_scan, s_scan) if len(s_scan) > 1 else np.array([np.nan])

    return {
        "available": True,
        "settings": {
            "max_denominator": int(max_denominator),
            "warn_distance": float(warn_distance),
            "s_min": float(s_min),
            "s_max": float(s_max),
        },
        "s": s_scan,
        "iota": iota_scan,
        "iota_min": float(np.nanmin(iota_scan)),
        "iota_max": float(np.nanmax(iota_scan)),
        "abs_iota_min": float(np.nanmin(abs_iota)),
        "abs_iota_max": float(np.nanmax(abs_iota)),
        "shear_min": float(np.nanmin(shear)),
        "shear_max": float(np.nanmax(shear)),
        "nearest_low_order": nearest,
        "near_low_order": near_points,
        "crossings": crossings,
    }


def _mercier_summary(mercier_data):
    if mercier_data is None:
        return {"available": False}
    values = np.asarray(mercier_data["values"], dtype=float)
    s_vals = np.asarray(mercier_data["s"], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"available": False, "reason": "no finite D_Merc values"}
    finite_values = values[finite]
    finite_s = s_vals[finite]
    min_idx = int(np.argmin(finite_values))
    neg = finite_values < 0
    return {
        "available": True,
        "min": float(finite_values[min_idx]),
        "s_at_min": float(finite_s[min_idx]),
        "negative_count": int(np.count_nonzero(neg)),
        "negative_fraction": float(np.count_nonzero(neg) / finite_values.size),
    }


def _well_summary(well_data, well_depth):
    if well_data is None or not np.isfinite(well_depth):
        return {"available": False}
    values = np.asarray(well_data["values"], dtype=float)
    s_vals = np.asarray(well_data["s"], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"available": False, "reason": "no finite magnetic-well values"}
    finite_values = values[finite]
    finite_s = s_vals[finite]
    min_idx = int(np.argmin(finite_values))
    return {
        "available": True,
        "edge": float(well_depth),
        "min": float(finite_values[min_idx]),
        "s_at_min": float(finite_s[min_idx]),
        "negative_count": int(np.count_nonzero(finite_values < 0)),
    }


def _nc_scalar(ds, name, default=float("nan")):
    if name not in ds.variables:
        return default
    try:
        arr = np.asarray(ds.variables[name][:])
        return float(arr.reshape(-1)[0])
    except Exception:
        return default


def _fourier_cos(coeff, m, n, theta, zeta):
    angle = m[None, None, :] * theta[:, :, None] - n[None, None, :] * zeta[:, :, None]
    return np.sum(coeff[None, None, :] * np.cos(angle), axis=2)


def _fourier_sin(coeff, m, n, theta, zeta):
    angle = m[None, None, :] * theta[:, :, None] - n[None, None, :] * zeta[:, :, None]
    return np.sum(coeff[None, None, :] * np.sin(angle), axis=2)


def _surface_area_from_rz(R, Z, zeta):
    theta_vals = np.linspace(0.0, 2.0 * np.pi, R.shape[0], endpoint=False)
    zeta_vals = zeta[0, :]
    x = R * np.cos(zeta)
    y = R * np.sin(zeta)
    dtheta = theta_vals[1] - theta_vals[0]
    dzeta = zeta_vals[1] - zeta_vals[0]
    rx_t = np.gradient(x, dtheta, axis=0, edge_order=2)
    ry_t = np.gradient(y, dtheta, axis=0, edge_order=2)
    rz_t = np.gradient(Z, dtheta, axis=0, edge_order=2)
    rx_z = np.gradient(x, dzeta, axis=1, edge_order=2)
    ry_z = np.gradient(y, dzeta, axis=1, edge_order=2)
    rz_z = np.gradient(Z, dzeta, axis=1, edge_order=2)
    cx = ry_t * rz_z - rz_t * ry_z
    cy = rz_t * rx_z - rx_t * rz_z
    cz = rx_t * ry_z - ry_t * rx_z
    jac = np.sqrt(cx * cx + cy * cy + cz * cz)
    return float(np.sum(jac) * dtheta * dzeta)


def _cross_section_area(R, Z):
    # Shoelace area in each constant-zeta poloidal cut.
    x = R
    y = Z
    xp = np.roll(x, -1, axis=0)
    yp = np.roll(y, -1, axis=0)
    return 0.5 * np.abs(np.sum(x * yp - xp * y, axis=0))


def _first_nonzero_radial_index(arr, start=0, atol=1e-14):
    for idx in range(start, arr.shape[0]):
        if np.nanmax(np.abs(arr[idx])) > atol:
            return idx
    return start


def _basic_configuration_summary(
        vmec, nc_file, info=None, qi_r2_info=None, mercier_facts=None,
        well_facts=None, iota_scan=None, eps_eff_results=None):
    """Build a compact single-configuration engineering/physics summary.

    These quantities are intended for quick CLI inspection. Surface/area
    metrics are reconstructed from VMEC Fourier coefficients on a grid, so
    they are diagnostic estimates, not a replacement for dedicated CAD/mesh
    post-processing.
    """
    import netCDF4

    out = {}
    with netCDF4.Dataset(nc_file) as ds:
        nfp = int(round(_nc_scalar(ds, "nfp", 0)))
        rmajor = _nc_scalar(ds, "Rmajor_p")
        aminor = _nc_scalar(ds, "Aminor_p")
        aspect = _nc_scalar(ds, "aspect")
        volume = _nc_scalar(ds, "volume_p")
        beta = _nc_scalar(ds, "betatotal")
        ctor = _nc_scalar(ds, "ctor", 0.0)

        phi = np.asarray(ds.variables["phi"][:], dtype=float) if "phi" in ds.variables else np.array([])
        chi = np.asarray(ds.variables["chi"][:], dtype=float) if "chi" in ds.variables else np.array([])
        iotaf = np.asarray(ds.variables["iotaf"][:], dtype=float) if "iotaf" in ds.variables else np.array([])
        presf = np.asarray(ds.variables["presf"][:], dtype=float) if "presf" in ds.variables else np.array([])

        theta_1d = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        zeta_1d = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        theta, zeta = np.meshgrid(theta_1d, zeta_1d, indexing="ij")

        boundary_R = boundary_Z = None
        area_avg = surface_area = float("nan")
        if all(name in ds.variables for name in ["rmnc", "zmns", "xm", "xn"]):
            rmnc = np.asarray(ds.variables["rmnc"][:], dtype=float)
            zmns = np.asarray(ds.variables["zmns"][:], dtype=float)
            xm = np.asarray(ds.variables["xm"][:], dtype=float)
            xn = np.asarray(ds.variables["xn"][:], dtype=float)
            boundary_R = _fourier_cos(rmnc[-1], xm, xn, theta, zeta)
            boundary_Z = _fourier_sin(zmns[-1], xm, xn, theta, zeta)
            area_avg = float(np.mean(_cross_section_area(boundary_R, boundary_Z)))
            surface_area = _surface_area_from_rz(boundary_R, boundary_Z, zeta)

        B_axis = B_lcfs = None
        b_axis_idx = 0
        if all(name in ds.variables for name in ["bmnc", "xm_nyq", "xn_nyq"]):
            bmnc = np.asarray(ds.variables["bmnc"][:], dtype=float)
            xm_nyq = np.asarray(ds.variables["xm_nyq"][:], dtype=float)
            xn_nyq = np.asarray(ds.variables["xn_nyq"][:], dtype=float)
            b_axis_idx = _first_nonzero_radial_index(bmnc, start=0)
            B_axis = _fourier_cos(bmnc[b_axis_idx], xm_nyq, xn_nyq, theta, zeta)
            B_lcfs = _fourier_cos(bmnc[-1], xm_nyq, xn_nyq, theta, zeta)

        b0_method = "B_axis_abs_mean_fallback"
        b0_iss04 = float(np.nanmean(np.abs(B_axis))) if B_axis is not None else float("nan")
        if "bvco" in ds.variables and np.isfinite(rmajor) and abs(rmajor) > 1e-30:
            bvco = np.asarray(ds.variables["bvco"][:], dtype=float)
            nz = np.flatnonzero(np.isfinite(bvco) & (np.abs(bvco) > 1e-30))
            if nz.size:
                b0_iss04 = float(abs(bvco[nz[0]]) / abs(rmajor))
                b0_method = "abs_bvco_first_nonzero_over_Rmajor"

    out["NFP_field_periods"] = float(nfp)
    out["R0_major_radius_average"] = rmajor
    out["R0_over_a_aspect_ratio"] = aspect
    out["minor_radius_effective_a"] = aminor
    out["B0_toroidal_ISS04_B_phi_or_fallback_T"] = b0_iss04
    out["B0_ISS04_extraction_method"] = b0_method
    if B_axis is not None:
        b_axis_abs = np.abs(B_axis)
        out["B_axis_absB_mean_over_theta_zeta"] = float(np.nanmean(b_axis_abs))
        out["B_axis_absB_min_over_theta_zeta"] = float(np.nanmin(b_axis_abs))
        out["B_axis_absB_max_over_theta_zeta"] = float(np.nanmax(b_axis_abs))
        out["B_axis_absB_sampling_rho"] = float(np.sqrt(b_axis_idx / max(bmnc.shape[0] - 1, 1)))
        out["B_axis_absB_sampling_radial_index"] = int(b_axis_idx)
        out["B_axis_flux_surface_absB_mean_DESC_or_VMEC"] = float(np.nanmean(b_axis_abs))
    out["B0_for_ISS04_0D_code_recommended"] = b0_iss04
    out["A_cross_section_area_avg_extrapolated_LCFS"] = area_avg
    out["V_plasma_volume_enclosed_LCFS"] = volume
    out["S_outer_flux_surface_area_extrapolated_LCFS"] = surface_area
    if boundary_R is not None and boundary_Z is not None:
        out["boundary_R_max"] = float(np.nanmax(boundary_R))
        out["boundary_R_min"] = float(np.nanmin(boundary_R))
        out["boundary_Z_max"] = float(np.nanmax(boundary_Z))
        out["boundary_Z_min"] = float(np.nanmin(boundary_Z))
        r_span = out["boundary_R_max"] - out["boundary_R_min"]
        z_span = out["boundary_Z_max"] - out["boundary_Z_min"]
        out["elongation_kappa_Zspan_over_Rspan"] = float(z_span / r_span) if r_span > 0 else None
        out["elongation_vs_2a_Zspan_over_2a"] = float(z_span / (2.0 * aminor)) if aminor > 0 else None
    out["Phi_t_toroidal_flux_Psi"] = float(phi[-1]) if phi.size else None
    out["Phi_p_poloidal_flux_true_2pi_chi"] = float(2.0 * np.pi * chi[-1]) if chi.size else None
    out["chi_poloidal_flux_normalized_by_2pi"] = float(chi[-1]) if chi.size else None
    if B_lcfs is not None:
        b_lcfs_abs = np.abs(B_lcfs)
        bmin = float(np.nanmin(b_lcfs_abs))
        bmax = float(np.nanmax(b_lcfs_abs))
        bavg = float(np.nanmean(b_lcfs_abs))
        out["mirror_ratio_lcfs_rho1_simplified"] = float((bmax - bmin) / (bmax + bmin)) if (bmax + bmin) > 0 else None
        out["Bmin_lcfs"] = bmin
        out["Bmax_lcfs"] = bmax
        out["Bavg_lcfs_flux_surface_absB_mean"] = bavg
        out["Bavg_lcfs_arithmetic_on_grid"] = bavg

    if B_axis is not None and B_lcfs is not None:
        samples = np.concatenate([np.ravel(np.abs(B_axis)), np.ravel(np.abs(B_lcfs))])
        out["principal_abs_max_over_all_rho_samples"] = float(np.nanmax(samples))
        out["principal_abs_min_over_all_rho_samples"] = float(np.nanmin(samples))
        out["principal_abs_mean_over_all_rho_samples"] = float(np.nanmean(samples))

    out["total_toroidal_current_A_VMEC_ctor"] = ctor
    out["volume_averaged_beta"] = beta
    if iotaf.size:
        out["iota_max"] = float(np.nanmax(iotaf))
        out["iota_min"] = float(np.nanmin(iotaf))
        out["iota_axis"] = float(iotaf[0])
        out["iota_edge"] = float(iotaf[-1])
    if presf.size:
        out["pressure_Pa_max"] = float(np.nanmax(presf))
        out["pressure_Pa_min"] = float(np.nanmin(presf))
        out["pressure_edge_Pa"] = float(presf[-1])

    if info is not None:
        out["f_QI"] = float(info.get("f_QI", np.nan))
        out["f_maxJ"] = float(info.get("f_maxJ", np.nan))
        out["f_Bmin"] = float(info.get("f_Bmin", np.nan))
        out["mirror_ratio_core_diagnostic"] = float(info.get("mirror_ratio", np.nan))
        out["maxJ_global_pass_ratio"] = float(info.get("maxj_global_pass_ratio", np.nan))
        out["maxJ_global_violation_fraction"] = float(info.get("maxj_global_violation_fraction", np.nan))
        if len(info.get("qi_surface_rms", [])):
            out["QI_worst_surface_RMS"] = float(np.nanmax(info["qi_surface_rms"]))
        centers = info.get("interval_centers", [])
        idx = int(info.get("maxj_worst_interval_idx", -1))
        if idx >= 0 and idx < len(centers):
            out["maxJ_worst_interval_center_s"] = float(centers[idx])
            out["maxJ_worst_lambda_N"] = float(info.get("maxj_worst_lambda", np.nan))
    if qi_r2_info and "f_QI_R2" in qi_r2_info:
        out["f_QI_R2"] = float(qi_r2_info["f_QI_R2"])
    if mercier_facts and mercier_facts.get("available"):
        out["Mercier_DMerc_min"] = float(mercier_facts["min"])
        out["Mercier_DMerc_s_at_min"] = float(mercier_facts["s_at_min"])
        out["Mercier_DMerc_negative_count"] = int(mercier_facts["negative_count"])
    if well_facts and well_facts.get("available"):
        out["magnetic_well_depth_edge"] = float(well_facts["edge"])
        out["magnetic_well_depth_min"] = float(well_facts["min"])
        out["magnetic_well_negative_count"] = int(well_facts["negative_count"])
    if iota_scan and iota_scan.get("available"):
        nearest = iota_scan["nearest_low_order"]
        out["nearest_low_order_iota_rational"] = nearest["rational"]
        out["nearest_low_order_iota_s"] = float(nearest["s"])
        out["nearest_low_order_iota_distance"] = float(nearest["distance"])
        out["low_order_iota_near_count"] = len(iota_scan.get("near_low_order", []))
        out["low_order_iota_crossing_count"] = len(iota_scan.get("crossings", []))
    if eps_eff_results:
        finite = [x for x in eps_eff_results if x.get("value") is not None]
        if finite:
            worst = max(finite, key=lambda x: float(x["value"]))
            out["effective_ripple_or_proxy_worst_s"] = float(worst["s"])
            out["effective_ripple_or_proxy_worst_value"] = float(worst["value"])
            out["effective_ripple_or_proxy_source"] = worst["source"]
    return out


def _compact_profile(x, y, x_name, y_name):
    if x is None or y is None:
        return []
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    return [
        {x_name: float(x[i]), y_name: float(y[i])}
        for i in range(n)
        if np.isfinite(x[i]) and np.isfinite(y[i])
    ]


def _compact_diagnostic_report(
        nc_file, s_vals, info, qi_r2_info, basic_summary, sanity,
        threshold_facts, iota_scan, mercier_data, mercier_facts,
        well_data, well_facts, well_depth, eps_eff_results,
        itg_info, ae_info, notes, args):
    """Report intended for humans: scalar facts and short profiles only."""
    core = {
        "f_QI": float(info.get("f_QI", np.nan)),
        "f_maxJ": float(info.get("f_maxJ", np.nan)),
        "f_Bmin": float(info.get("f_Bmin", np.nan)),
        "mirror_ratio": float(info.get("mirror_ratio", np.nan)),
        "maxJ_global_pass_ratio": float(info.get("maxj_global_pass_ratio", np.nan)),
        "maxJ_global_violation_fraction": float(info.get("maxj_global_violation_fraction", np.nan)),
        "QI_worst_surface_RMS": float(np.nanmax(info.get("qi_surface_rms", [np.nan]))),
    }
    centers = info.get("interval_centers", [])
    idx = int(info.get("maxj_worst_interval_idx", -1))
    if idx >= 0 and idx < len(centers):
        core["maxJ_worst_interval_center_s"] = float(centers[idx])
        core["maxJ_worst_lambda_N"] = float(info.get("maxj_worst_lambda", np.nan))

    qi_surface_profile = []
    for s_val, rms, p95 in zip(info.get("s_vals", []),
                               info.get("qi_surface_rms", []),
                               info.get("qi_surface_p95", [])):
        qi_surface_profile.append({
            "s": float(s_val),
            "QI_RMS": float(rms),
            "QI_p95": float(p95),
        })

    maxj_profile = []
    for center, passed, violated, worst_lambda in zip(
            info.get("interval_centers", []),
            info.get("maxj_interval_pass_ratio", []),
            info.get("maxj_interval_violation_fraction", []),
            info.get("maxj_interval_worst_lambda", [])):
        maxj_profile.append({
            "s_center": float(center),
            "pass_ratio": float(passed),
            "violation_fraction": float(violated),
            "worst_lambda_N": float(worst_lambda),
        })

    if qi_r2_info:
        core["f_QI_R2"] = qi_r2_info.get("f_QI_R2")
        core["f_QI_R2_rms"] = qi_r2_info.get("rms")

    report = {
        "input": nc_file,
        "summary": basic_summary,
        "core_physics_targets": core,
        "profiles": {
            "QI_by_surface": qi_surface_profile,
            "maxJ_by_radial_interval": maxj_profile,
            "Mercier_DMerc": _compact_profile(
                None if mercier_data is None else mercier_data.get("s"),
                None if mercier_data is None else mercier_data.get("values"),
                "s", "DMerc"),
            "magnetic_well_depth": _compact_profile(
                None if well_data is None else well_data.get("s"),
                None if well_data is None else well_data.get("values"),
                "s", "well_depth"),
            "effective_ripple_or_proxy": eps_eff_results or [],
        },
        "mhd": {
            "Mercier": mercier_facts,
            "magnetic_well": well_facts,
            "well_depth_edge": well_depth,
        },
        "iota_scan": {
            "nearest_low_order": iota_scan.get("nearest_low_order") if iota_scan else None,
            "near_low_order_count": len(iota_scan.get("near_low_order", [])) if iota_scan else None,
            "crossing_count": len(iota_scan.get("crossings", [])) if iota_scan else None,
            "scan_settings": {
                "max_denominator": args.rational_max_denominator,
                "warn_distance": args.rational_warn_distance,
                "s_min": args.rational_s_min,
                "s_max": args.rational_s_max,
            },
        },
        "transport": {
            "ITG": itg_info,
            "available_energy": ae_info,
        },
        "sanity": sanity,
        "diagnostic_facts": threshold_facts,
        "notes": notes,
        "settings": {
            "num_alpha": args.num_alpha,
            "num_pitch": args.num_pitch,
            "num_surfaces": args.num_surfaces,
            "surfaces": [float(x) for x in np.asarray(s_vals, dtype=float)],
            "qi_r2": bool(args.qi_r2),
            "skip_ripple": bool(args.skip_ripple),
            "mhd_s_min": args.mhd_s_min,
        },
    }
    return report


def _main_issue_summary(info, ripple_results=None, itg_info=None, ae_info=None):
    notes = []

    if len(info["s_vals"]) > 0:
        worst_qi_s = float(info["s_vals"][int(info["qi_worst_surface_idx"])])
        notes.append(f"QI worst at s={worst_qi_s:.2f}")

    if len(info["interval_centers"]) > 0 and info["maxj_worst_interval_idx"] >= 0:
        worst_interval = float(info["interval_centers"][int(info["maxj_worst_interval_idx"])])
        notes.append(
            f"max-J weakest near s~{worst_interval:.2f}, "
            f"shallow trapped depth λ_N~{info['maxj_worst_lambda']:.03f}"
        )

    if ripple_results:
        finite = [item for item in ripple_results
                  if item["value"] is not None and np.isfinite(item["value"])]
        if finite:
            worst = max(finite, key=lambda item: float(item["value"]))
            notes.append(f"ripple rises outward; worst checked at s={worst['s']:.2f}")

    if itg_info is not None and len(itg_info.get("per_surface", {})) > 0:
        itg_s, itg_vals = _sorted_surface_dict(itg_info["per_surface"])
        if len(itg_vals) > 0 and np.any(np.isfinite(itg_vals)):
            worst_idx = int(np.nanargmax(itg_vals))
            notes.append(f"ITG target peaks at s={itg_s[worst_idx]:.2f}")

    if ae_info is not None:
        ae_s, ae_vals = _sorted_surface_dict({k: v for k, v in ae_info.items() if k != "total"})
        if len(ae_vals) > 0 and np.any(np.isfinite(ae_vals)):
            worst_idx = int(np.nanargmax(ae_vals))
            notes.append(f"AE peaks at s={ae_s[worst_idx]:.2f}")

    return notes


def _plot_axis_geometry_summary(ax_info, mercier_data=None, well_data=None,
                                ripple_results=None):
    """Compact axis/stability summary figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    phi = ax_info["phi"]
    curvature = ax_info["curvature"]
    torsion = ax_info["torsion"]

    ax = axes[0, 0]
    ax.plot(phi, curvature, "k-", lw=1.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("Axis curvature")

    ax = axes[0, 1]
    ax.plot(phi, torsion, "k-", lw=1.5)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\tau$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Axis torsion")

    ax = axes[1, 0]
    handles, labels = [], []
    if mercier_data is not None:
        line = ax.plot(
            mercier_data["s"],
            mercier_data["values"],
            marker="o",
            lw=1.8,
            color="tab:blue",
            label=r"$D_{\mathrm{Merc}}$",
        )[0]
        ax.axhline(0, color="tab:blue", ls=":", alpha=0.4)
        handles.append(line)
        labels.append(line.get_label())
    ax.set_xlabel("s")
    ax.set_ylabel(r"$D_{\mathrm{Merc}}$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Mercier and magnetic well")

    if well_data is not None:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            well_data["s"],
            100.0 * well_data["values"],
            marker="s",
            lw=1.5,
            ls="--",
            color="tab:orange",
            label="well depth [%]",
        )[0]
        ax2.axhline(0, color="tab:orange", ls=":", alpha=0.4)
        ax2.set_ylabel("Well depth [%]")
        handles.append(line2)
        labels.append(line2.get_label())
    if handles:
        ax.legend(handles, labels, loc="best")

    ax = axes[1, 1]
    if ripple_results:
        s = np.array([item["s"] for item in ripple_results], dtype=float)
        y = np.array([
            np.nan if item["value"] is None else float(item["value"])
            for item in ripple_results
        ])
        ax.plot(s, y, marker="o", lw=1.8, color="tab:green")
        for idx, item in enumerate(ripple_results):
            src = item["source"]
            short = "DESC" if "DESC" in src else (
                "NEO-RT" if "NEO-RT" in src else (
                    "proxy" if "proxy" in src else "n/a"
                )
            )
            ax.annotate(short, (s[idx], y[idx]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)
    ax.set_xlabel("s")
    ax.set_ylabel("Ripple metric")
    ax.grid(True, alpha=0.3)
    ax.set_title("Effective ripple / proxy")

    fig.suptitle(
        f"Axis and stability summary (L_axis={ax_info['axis_length']:.3f} m)"
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="SQuID diagnostic evaluation (no optimisation)"
    )
    parser.add_argument("--nc_file", type=str, required=True,
                        help="VMEC wout .nc file to evaluate")
    parser.add_argument("--num_alpha", type=int, default=8)
    parser.add_argument("--num_pitch", type=int, default=50)
    parser.add_argument("--num_surfaces", type=int, default=5)
    parser.add_argument("--s_center", type=float, default=0.5)
    parser.add_argument("--ds", type=float, default=0.1)
    parser.add_argument("--itg_method", choices=["drift_curvature", "vacuum_dBds"],
                        default="drift_curvature",
                        help="Bad-curvature detection method for f_nabla_s")
    parser.add_argument("--extended", action="store_true",
                        help="Run third-tier transport diagnostics (ITG, optional AE)")
    parser.add_argument("--extended_surfaces", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5],
                        help="Flux surfaces for third-tier ITG diagnostics")
    parser.add_argument("--eps_eff_surface", type=float, default=None,
                        help="Deprecated single-surface effective ripple evaluation")
    parser.add_argument("--eps_eff_surfaces", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75],
                        help="Normalised flux surfaces for effective ripple evaluation")
    parser.add_argument("--skip_ripple", action="store_true",
                        help="Skip effective-ripple/proxy evaluation for fast target scans")
    parser.add_argument("--rational_max_denominator", type=int, default=8,
                        help="Largest denominator q in low-order iota p/q scan")
    parser.add_argument("--rational_warn_distance", type=float, default=0.01,
                        help="Report iota points within this absolute distance of p/q")
    parser.add_argument("--rational_s_min", type=float, default=0.05,
                        help="Inner s boundary for low-order iota scan")
    parser.add_argument("--rational_s_max", type=float, default=0.95,
                        help="Outer s boundary for low-order iota scan")
    parser.add_argument("--mhd_s_min", type=float, default=0.1,
                        help="Minimum s used for Mercier/MHD gate summaries")
    parser.add_argument("--qi_r2", action="store_true",
                        help="Evaluate optional R2 squash-stretch-shuffle QI diagnostic")
    parser.add_argument("--qi_r2_nphi", type=int, default=201)
    parser.add_argument("--qi_r2_nalpha", type=int, default=16)
    parser.add_argument("--qi_r2_nbj", type=int, default=201)
    parser.add_argument("--qi_r2_mpol", type=int, default=12)
    parser.add_argument("--qi_r2_ntor", type=int, default=12)
    parser.add_argument("--qi_r2_arr_out", action="store_true",
                        help="Use full (surface, alpha, phi) R2 residuals instead of per-alpha RMS")
    parser.add_argument("--ae_surfaces", type=float, nargs="+",
                        default=[0.2, 0.5, 0.8],
                        help="Flux surfaces for AE diagnostics when --ae is enabled")
    parser.add_argument("--ae", action="store_true",
                        help="Compute Available Energy (TEM turbulence metric)")
    parser.add_argument("--ae_omn", type=float, default=1.0,
                        help="AE: -d ln n / d s")
    parser.add_argument("--ae_omt", type=float, default=3.0,
                        help="AE: -d ln T / d s")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--output_dir", type=str, default="runs/diagnose_latest",
                        help="Directory for report JSON, plots, and solver scratch files")
    parser.add_argument("--report_json", type=str, default=None,
                        help="Optional report filename inside output_dir")
    args = parser.parse_args()

    from simsopt.mhd import Vmec
    from squid.evaluation.evaluate import (
        evaluate_squid_detailed,
        evaluate_itg,
        plot_J_contours,
        plot_squid_core_diagnostics,
        plot_transport_diagnostics,
    )
    from squid.evaluation.axis_geometry import axis_geometry_from_vmec
    from squid.evaluation.available_energy import ae_diagnostics
    from squid.objectives.qi_residual import compute_qi_residual_r2

    nc_file = os.path.abspath(args.nc_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = args.report_json or f"{Path(nc_file).stem}_diagnostics.json"

    print(f"\n{'=' * 60}")
    print("SQuID Diagnostic Evaluation")
    print(f"{'=' * 60}")
    print(f"  Input: {nc_file}")
    print(f"  Output dir: {output_dir}")

    vmec = Vmec(nc_file)
    vmec.run()
    print(f"  nfp = {vmec.wout.nfp}, ns = {vmec.wout.ns}")

    half = (args.num_surfaces - 1) * args.ds / 2
    s_vals = np.linspace(
        max(0.05, args.s_center - half),
        min(0.95, args.s_center + half),
        args.num_surfaces,
    )
    print(f"  Surfaces: {s_vals}")

    print(f"\n--- Core SQuID Targets ---")
    info = evaluate_squid_detailed(
        vmec, s_vals=s_vals,
        num_alpha=args.num_alpha,
        num_pitch=args.num_pitch,
        verbose=False,
    )

    print(f"  f_QI             = {info['f_QI']:.6e}")
    print(f"  f_maxJ           = {info['f_maxJ']:.6e}")
    print(f"  f_Bmin           = {info['f_Bmin']:.6e}")
    print(f"  mirror ratio     = {info['mirror_ratio']:.4f}")
    print(f"  QI worst surface = s={info['s_vals'][info['qi_worst_surface_idx']]:.2f}")
    for s_val, rms, p95 in zip(info["s_vals"], info["qi_surface_rms"], info["qi_surface_p95"]):
        print(f"    QI @ s={s_val:.2f}: RMS={rms:.3e}, p95={p95:.3e}")

    qi_r2_info = None
    if args.qi_r2:
        print(f"  Evaluating R2 QI diagnostic ...")
        try:
            _, qi_r2_residuals = compute_qi_residual_r2(
                vmec, s_vals,
                nphi=args.qi_r2_nphi,
                nalpha=args.qi_r2_nalpha,
                nBj=args.qi_r2_nbj,
                mpol=args.qi_r2_mpol,
                ntor=args.qi_r2_ntor,
                arr_out=args.qi_r2_arr_out,
            )
            qi_r2_info = {
                "f_QI_R2": float(np.sum(qi_r2_residuals ** 2)),
                "rms": float(np.sqrt(np.mean(qi_r2_residuals ** 2))),
                "max_abs": float(np.max(np.abs(qi_r2_residuals))) if qi_r2_residuals.size else 0.0,
                "num_residuals": int(qi_r2_residuals.size),
                "settings": {
                    "nphi": args.qi_r2_nphi,
                    "nalpha": args.qi_r2_nalpha,
                    "nBj": args.qi_r2_nbj,
                    "mpol": args.qi_r2_mpol,
                    "ntor": args.qi_r2_ntor,
                    "arr_out": bool(args.qi_r2_arr_out),
                },
            }
            print(
                f"  f_QI_R2          = {qi_r2_info['f_QI_R2']:.6e} "
                f"(rms={qi_r2_info['rms']:.3e}, N={qi_r2_info['num_residuals']})"
            )
        except Exception as e:
            qi_r2_info = {"failed": str(e)}
            print(f"  f_QI_R2          = [failed: {e}]")

    if len(info["interval_centers"]) > 0:
        worst_interval_idx = int(info["maxj_worst_interval_idx"])
        print(f"  max-J pass ratio = {info['maxj_global_pass_ratio']:.3f}")
        print(f"  max-J violation  = {info['maxj_global_violation_fraction']:.3f}")
        print("  max-J per interval:")
        for center, pass_ratio, frac, worst_lambda in zip(
            info["interval_centers"],
            info["maxj_interval_pass_ratio"],
            info["maxj_interval_violation_fraction"],
            info["maxj_interval_worst_lambda"],
        ):
            print(
                f"    s~{center:.2f}: pass={pass_ratio:.3f}, "
                f"violation={frac:.3f}, worst λ_N≈{worst_lambda:.3f}"
            )
        print(
            "  max-J worst interval = "
            f"s~{info['interval_centers'][worst_interval_idx]:.2f}"
        )

    print(f"\n--- Equilibrium Sanity ---")
    sanity = _compute_equilibrium_sanity(vmec, info)
    _print_equilibrium_sanity(sanity)

    print(f"\n--- Basic Geometry ---")
    print(f"  Aspect ratio = {vmec.wout.aspect:.4f}")
    print(f"  Iota (core)  = {vmec.wout.iotaf[0]:.4f}")
    print(f"  Iota (edge)  = {vmec.wout.iotaf[-1]:.4f}")
    iota_scan = _scan_iota_rationals(
        vmec,
        max_denominator=args.rational_max_denominator,
        warn_distance=args.rational_warn_distance,
        s_min=args.rational_s_min,
        s_max=args.rational_s_max,
    )
    if iota_scan.get("available"):
        nearest = iota_scan["nearest_low_order"]
        print(
            "  Iota range   = "
            f"[{iota_scan['iota_min']:.4f}, {iota_scan['iota_max']:.4f}] "
            f"for s in [{args.rational_s_min:.2f}, {args.rational_s_max:.2f}]"
        )
        print(
            "  Nearest low-order iota: "
            f"{nearest['rational']} at s={nearest['s']:.3f}, "
            f"iota={nearest['iota']:.4f}, |Δι|={nearest['distance']:.4e}"
        )
        print(
            "  Low-order crossings: "
            f"{len(iota_scan['crossings'])} for q<={args.rational_max_denominator}; "
            f"near points within {args.rational_warn_distance:g}: "
            f"{len(iota_scan['near_low_order'])}"
        )
    else:
        print(f"  Iota rational scan: unavailable ({iota_scan.get('reason', 'unknown')})")

    if sanity["status"] == "FAIL":
        print("\nAborting further diagnostics: equilibrium failed sanity checks.")
        raise SystemExit(1)

    # ---------------------------------------------------------
    # NEW: Stability & Transport Diagnostics
    # ---------------------------------------------------------
    print(f"\n--- Stability & Transport Diagnostics ---")
    mercier_data = None
    well_data = None
    well_depth = float("nan")

    # 1. Mercier Stability Criterion
    try:
        if hasattr(vmec.wout, "DMerc"):
            dmerc_raw = vmec.wout.DMerc
        else:
            dmerc_raw = vmec.wout.Dmerc
        dmerc_full = np.array(dmerc_raw[1:], dtype=float)
        s_half_full = np.array(vmec.s_half_grid, dtype=float)
        mhd_mask = s_half_full >= args.mhd_s_min
        dmerc = dmerc_full[mhd_mask]
        s_half = s_half_full[mhd_mask]
        mercier_data = {"s": s_half, "values": dmerc}
        mercier_facts = _mercier_summary(mercier_data)
        print(
            f"  Mercier D_Merc (s>={args.mhd_s_min:g}): "
            f"min={mercier_facts['min']:.4e} at s={mercier_facts['s_at_min']:.2f}, "
            f"negative_points={mercier_facts['negative_count']}, "
            f"negative_fraction={mercier_facts['negative_fraction']:.3f}"
        )
    except Exception:
        mercier_facts = {"available": False}
        print("  Mercier Stability: [Not available in this nc file]")

    # 2. Magnetic Well Depth
    try:
        vp = np.array(vmec.wout.vp[1:], dtype=float)
        well_profile = (vp[0] - vp) / max(vp[0], 1e-30)
        well_data = {"s": np.array(vmec.s_half_grid, dtype=float), "values": well_profile}
        well_depth = float(well_profile[-1])
        well_facts = _well_summary(well_data, well_depth)
        print(
            "  Magnetic Well Depth: "
            f"edge={well_depth * 100:.2f}%, "
            f"min={well_facts['min'] * 100:.2f}% at "
            f"s={well_facts['s_at_min']:.2f}, "
            f"negative_points={well_facts['negative_count']}"
        )
    except Exception:
        well_facts = {"available": False}
        print("  Magnetic Well Depth: [Not available in this nc file]")

    # 3. Effective Ripple (eps_eff)
    print("  Effective Ripple / Ripple Proxy:")
    if args.skip_ripple:
        eps_eff_results = []
        print("    skipped by --skip_ripple")
    else:
        eps_eff_surfaces = args.eps_eff_surfaces
        if args.eps_eff_surface is not None:
            eps_eff_surfaces = [args.eps_eff_surface]
        eps_eff_surfaces = sorted(set(max(0.01, min(0.99, float(s))) for s in eps_eff_surfaces))
        eps_eff_results = _evaluate_effective_ripple_series(
            vmec, eps_eff_surfaces, nc_file, output_dir=output_dir)
        for item in eps_eff_results:
            if item["value"] is None:
                print(f"    s={item['s']:.2f}: [Could not evaluate. {item['source']}]")
            else:
                print(f"    s={item['s']:.2f}: {item['value']:.4e}  ({item['source']})")
                if "Boozer proxy" in item["source"]:
                    print("      [Note] This value is a ripple amplitude proxy, not the true ε_eff.")
                if "NEO-RT D11" in item["source"]:
                    print("      [Note] This value is the NEO-RT transport coefficient D11, not the direct ε_eff.")

    # 4. Axis Geometry
    print(f"\n--- Axis Geometry ---")
    ax_info = None
    try:
        ax_info = axis_geometry_from_vmec(vmec)
        kappa = ax_info["curvature"]
        tau = ax_info["torsion"]
        print(f"  Axis length:    {ax_info['axis_length']:.4f} m")
        print(f"  Curvature κ:    min={np.min(kappa):.4f}, "
              f"max={np.max(kappa):.4f}, mean={np.mean(kappa):.4f}")
        print(f"  Torsion   τ:    min={np.min(tau):.4f}, "
              f"max={np.max(tau):.4f}, mean={np.mean(tau):.4f}")
    except Exception as e:
        print(f"  [Failed] {e}")

    # ---------------------------------------------------------
    # Third-tier / extended diagnostics
    # ---------------------------------------------------------
    run_extended = args.extended or args.ae
    itg_info = None
    ae_info = None

    if run_extended:
        print(f"\n--- Extended Transport Diagnostics (reference only) ---")

        itg_surfaces = np.array(
            [max(0.01, min(0.99, float(s))) for s in args.extended_surfaces],
            dtype=float,
        )
        itg_surfaces = np.unique(itg_surfaces)
        itg_info = evaluate_itg(
            vmec,
            snorms=itg_surfaces,
            method=args.itg_method,
            verbose=False,
        )
        itg_s, itg_vals = _sorted_surface_dict(itg_info["per_surface"])
        print(f"  ITG method       = {args.itg_method}")
        print(f"  Total f_nabla_s  = {itg_info['total']:.6e}")
        if len(itg_vals) > 0 and np.any(np.isfinite(itg_vals)):
            worst_itg_idx = int(np.nanargmax(itg_vals))
            print(f"  Worst ITG surface = s={itg_s[worst_itg_idx]:.2f} ({itg_vals[worst_itg_idx]:.6e})")
        for s_itg, val_itg in zip(itg_s, itg_vals):
            print(f"    ITG @ s={s_itg:.2f}: {val_itg:.6e}")

        if args.ae:
            ae_surfaces = np.array(
                [max(0.01, min(0.99, float(s))) for s in args.ae_surfaces],
                dtype=float,
            )
            ae_surfaces = np.unique(ae_surfaces)
            print(f"\n  Available Energy (gradient-dependent reference diagnostic)")
            try:
                ae_info = ae_diagnostics(
                    vmec, s_vals=ae_surfaces,
                    omn=args.ae_omn, omt=args.ae_omt,
                    n_alpha=min(args.num_alpha, 4),
                    n_turns=3, lam_res=200, gridpoints=512,
                    verbose=False,
                )
                ae_s, ae_vals = _sorted_surface_dict({
                    k: v for k, v in ae_info.items() if k != "total"
                })
                if np.any(np.isfinite(ae_vals)):
                    worst_ae_idx = int(np.nanargmax(ae_vals))
                    print(
                        f"  Mean AE         = {ae_info['total']:.6e}\n"
                        f"  Worst AE surface = s={ae_s[worst_ae_idx]:.2f} ({ae_vals[worst_ae_idx]:.6e})"
                    )
                for s_ae, val_ae in zip(ae_s, ae_vals):
                    print(f"    AE @ s={s_ae:.2f}: {val_ae:.6e}")
            except Exception as e:
                print(f"  [AE failed] {e}")
        else:
            print("  AE skipped. Use --ae to include TEM-oriented diagnostics.")

    notes = _main_issue_summary(info, ripple_results=eps_eff_results,
                                itg_info=itg_info, ae_info=ae_info)

    threshold_facts = {
        "qi_worst_surface_rms": float(np.nanmax(info["qi_surface_rms"]))
        if len(info["qi_surface_rms"]) else float("nan"),
        "maxj_global_pass_ratio": float(info["maxj_global_pass_ratio"]),
        "maxj_global_violation_fraction": float(info["maxj_global_violation_fraction"]),
        "bmin_penalty": float(info["f_Bmin"]),
        "mirror_ratio": float(info["mirror_ratio"]),
        "mercier_min": mercier_facts.get("min"),
        "mercier_negative_count": mercier_facts.get("negative_count"),
        "well_depth_edge": well_facts.get("edge"),
        "well_min": well_facts.get("min"),
        "well_negative_count": well_facts.get("negative_count"),
        "nearest_low_order_iota": iota_scan.get("nearest_low_order"),
        "low_order_iota_near_count": len(iota_scan.get("near_low_order", [])),
        "low_order_iota_crossing_count": len(iota_scan.get("crossings", [])),
        "thresholds": {
            "rational_max_denominator": args.rational_max_denominator,
            "rational_warn_distance": args.rational_warn_distance,
        },
    }

    print(f"\n--- Diagnostic Facts ---")
    if sanity['status'] != 'OK':
        print(f"  Sanity status    = {sanity['status']}")
    print(f"  QI worst RMS     = {threshold_facts['qi_worst_surface_rms']:.6e}")
    print(f"  max-J pass ratio = {threshold_facts['maxj_global_pass_ratio']:.3f}")
    print(f"  Bmin penalty     = {threshold_facts['bmin_penalty']:.6e}")
    if threshold_facts["mercier_min"] is not None:
        print(
            f"  Mercier min      = {threshold_facts['mercier_min']:.6e} "
            f"({threshold_facts['mercier_negative_count']} negative points)"
        )
    if threshold_facts["well_depth_edge"] is not None:
        print(f"  Well edge depth  = {threshold_facts['well_depth_edge']:.6e}")
    for note in notes:
        print(f"  Note            = {note}")

    basic_summary = _basic_configuration_summary(
        vmec, nc_file,
        info=info,
        qi_r2_info=qi_r2_info,
        mercier_facts=mercier_facts,
        well_facts=well_facts,
        iota_scan=iota_scan,
        eps_eff_results=eps_eff_results,
    )
    print(f"\n--- Basic Configuration Summary JSON ---")
    print(json.dumps(_jsonify(basic_summary), indent=2))

    report = _compact_diagnostic_report(
        nc_file, s_vals, info, qi_r2_info, basic_summary, sanity,
        threshold_facts, iota_scan, mercier_data, mercier_facts,
        well_data, well_facts, well_depth, eps_eff_results,
        itg_info, ae_info, notes, args,
    )
    report_path = output_dir / report_name
    with open(report_path, "w") as fh:
        json.dump(_jsonify(report), fh, indent=2)
    print(f"  Report          = {report_path}")

    # Plots
    if args.plot:
        import matplotlib.pyplot as plt
        print(f"\n--- Generating plots ---")

        # ---------------------------------------------------------
        # 替换后的 Boozer Surface 画图代码
        # ---------------------------------------------------------
        boozer_s_vals = [0.25, 0.5, 0.75, 1.0] # 对应图中的4个面
        fig1, axes1 = plt.subplots(2, 2, figsize=(8, 5.5)) # 改为 2x2 布局
        axes_flat = axes1.flatten()
        from squid.evaluation.evaluate import run_boozer, reconstruct_B
        
        nfp = int(vmec.wout.nfp)
        safe_s = [max(0.01, min(0.99, s)) for s in boozer_s_vals]
        _, all_surf = run_boozer(vmec, safe_s, mpol=20, ntor=20)
        
        ntheta, nphi = 100, 100
        th = np.linspace(0, 2 * np.pi, ntheta)
        ze = np.linspace(0, 2 * np.pi / nfp, nphi)
        TH, ZE = np.meshgrid(th, ze, indexing="ij")
        b_all = [reconstruct_B(d["m"], d["n"], d["bmnc"], TH, ZE) for d in all_surf]
        
        for i, (s, data, B_2d) in enumerate(zip(boozer_s_vals, all_surf, b_all)):
            ax = axes_flat[i]
            
            # 动态计算 level，大约18条线，防止过于密集
            levels = np.linspace(B_2d.min(), B_2d.max(), 18)
            
            # 使用 contour 代替 contourf，cmap 采用 'plasma' 以匹配原图紫色到黄色的渐变
            cs = ax.contour(ZE, TH, B_2d, levels=levels, cmap="plasma", linewidths=1.2)
            
            # 添加左上角的标签文本框 (例如: |B| @ s=0.25)
            # 根据图中s=1没有小数位的情况做一点格式化
            s_label = f"{s:g}" if s == 1.0 else f"{s}"
            ax.text(0.03, 0.95, f"|B| @ s={s_label}", transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
            # 设置刻度以匹配原图
            ax.set_xticks([0, 2 * np.pi / nfp])
            # 图中 x 轴右侧刻度为 \pi/2，这通常对应 nfp=4 的情况。这里做个动态适配
            if nfp == 4:
                ax.set_xticklabels(['0', r'$\pi/2$'], fontsize=11)
            else:
                ax.set_xticklabels(['0', rf'$2\pi/{nfp}$'], fontsize=11)
                
            ax.set_yticks([0, 2 * np.pi])
            ax.set_yticklabels(['0', r'$2\pi$'], fontsize=11)
            
            # 坐标轴标签
            ax.set_xlabel(r"$\phi$", fontsize=12, labelpad=-8, fontweight='bold')
            ax.set_ylabel(r"$\theta$", fontsize=12, labelpad=-5, fontweight='bold')
            
            # 为每个子图添加单独的 colorbar
            cbar = fig1.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)
            
        fig1.tight_layout()
        fig1_path = output_dir / "boozer_surface.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fig1_path}")

        fig2 = plot_squid_core_diagnostics(
            info,
            metadata=dict(
                num_surfaces=len(s_vals),
                num_alpha=args.num_alpha,
                num_pitch=args.num_pitch,
            ),
        )
        fig2_path = output_dir / "squid_core_diagnostics.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fig2_path}")

        if run_extended:
            fig_ext = plot_transport_diagnostics(
                itg_info=itg_info,
                ae_info=ae_info,
                itg_method=args.itg_method,
                metadata=dict(
                    itg_surfaces=",".join(f"{s:.2f}" for s in itg_surfaces),
                    ae_surfaces=",".join(f"{s:.2f}" for s in ae_surfaces) if args.ae else "off",
                ),
            )
            if fig_ext is not None:
                fig_ext_path = output_dir / "transport_diagnostics.png"
                fig_ext.savefig(fig_ext_path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {fig_ext_path}")

        print("  Computing J contour polar plot (Fig. 9) ...")
        fig3 = plot_J_contours(vmec, lambda_N=0.3)
        fig3_path = output_dir / "j_contours_polar.png"
        fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fig3_path}")

        try:
            if ax_info is None:
                raise RuntimeError("axis geometry unavailable")
            fig4 = _plot_axis_geometry_summary(
                ax_info,
                mercier_data=mercier_data,
                well_data=well_data,
                ripple_results=eps_eff_results,
            )
            fig4_path = output_dir / "axis_geometry.png"
            fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fig4_path}")
        except Exception as e:
            print(f"  [axis_geometry plot failed: {e}]")

        plt.close("all")

    print(f"\n{'=' * 60}")
    print("Diagnostic evaluation complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
