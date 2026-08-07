"""
VMEC backend for SQuID optimisation.

Uses simsopt + VMEC2000 Fortran extension to run the equilibrium solver
and the LeastSquaresProblem interface.
"""

import os

import time
import glob
import csv
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
import numpy as np
import netCDF4

from simsopt.mhd import Vmec
from simsopt._core import Optimizable
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.qi_residual import QIResidual
from ..objectives.itg_residual import ITGResidual
from ..objectives.coil_proxy import coil_proxy_residuals, evaluate_coil_proxy
from ..objectives.oops_residual import oops_harmonics_residuals
from ..objectives.pdrot_residual import compute_pdrot_from_vmec
from ..objectives.penalties import bmin_slope_residuals, hinge_loss
from ..objectives.iota_topology import iota_topology_residuals
from ..diagnostics.mercier_normalization import (
    MERCIER_CONVENTION_VERSION,
    flux_normalize_mercier,
    mercier_profiles,
    vmec_half_grid_profile,
)


INVALID_OBJECTIVE = 1.0e6


class EvaluationLimitReached(RuntimeError):
    pass


@contextmanager
def _run_in_directory(path):
    """Run a VMEC call where its legacy relative outputs are isolated."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def wout_to_input(wout_path, input_path, ns=31,
                  prescribed_iota_ax=None, prescribed_iota_edge=None,
                  free_iota=True, zero_current=False):
    """Convert wout_*.nc to VMEC input file.

    Parameters
    ----------
    free_iota : bool
        If True, use NCURR=1 so that the rotational transform is computed
        self-consistently. The source current profile is preserved when the
        wout contains one.
    zero_current : bool
        Explicitly replace the source current profile with zero current.
        When ``free_iota`` is true and this flag is false, preserve the source
        current representation whenever it is available.
    prescribed_iota_ax, prescribed_iota_edge : float or None
        Only used when free_iota=False.  Sets a linear iota profile
        instead of fitting the iotaf from the wout file.
    """
    ds = netCDF4.Dataset(wout_path, "r")
    nfp = int(ds.variables["nfp"][:])
    mpol = int(ds.variables["mpol"][:])
    ntor = int(ds.variables["ntor"][:])
    ns_w = int(ds.variables["ns"][:])

    phi_arr = np.array(ds.variables["phi"][:])
    phiedge = float(phi_arr[-1])
    iotaf = np.array(ds.variables["iotaf"][:])
    presf = np.array(ds.variables["presf"][:])
    rmnc = np.array(ds.variables["rmnc"][:])
    zmns = np.array(ds.variables["zmns"][:])
    xm = np.array(ds.variables["xm"][:], dtype=int)
    xn_full = np.array(ds.variables["xn"][:], dtype=int)
    ac = np.asarray(ds.variables["ac"][:], dtype=float) if "ac" in ds.variables else None
    ac_aux_s = (
        np.asarray(ds.variables["ac_aux_s"][:], dtype=float)
        if "ac_aux_s" in ds.variables else None
    )
    ac_aux_f = (
        np.asarray(ds.variables["ac_aux_f"][:], dtype=float)
        if "ac_aux_f" in ds.variables else None
    )
    source_ctor = (
        float(np.asarray(ds.variables["ctor"][:]).squeeze())
        if "ctor" in ds.variables else 0.0
    )
    if "pcurr_type" in ds.variables:
        chars = np.asarray(ds.variables["pcurr_type"][:]).ravel()
        pcurr_type = b"".join(chars.tolist()).decode("ascii", "ignore").strip()
    else:
        pcurr_type = "power_series"
    ds.close()

    lower_pcurr = pcurr_type.lower()
    if lower_pcurr.endswith("_ip"):
        pcurr_type = pcurr_type[:-3] + "_Ip"
    elif lower_pcurr.endswith("_i"):
        pcurr_type = pcurr_type[:-2] + "_I"

    current_aux = []
    if ac_aux_s is not None and ac_aux_f is not None:
        valid = (
            np.isfinite(ac_aux_s) & np.isfinite(ac_aux_f)
            & (ac_aux_s >= 0.0) & (ac_aux_s <= 1.0)
        )
        current_aux = list(zip(ac_aux_s[valid], ac_aux_f[valid]))

    s_full = np.linspace(0, 1, ns_w)

    # Clamp negative edge pressure (polynomial fitting artifact)
    presf = np.maximum(presf, 0.0)
    p_max = presf.max()
    pn = presf / p_max if p_max > 0 else presf
    am = np.polynomial.polynomial.polyfit(s_full, pn, min(10, ns_w - 1))

    xn_in = xn_full // nfp
    rbc, zbs = {}, {}
    for i in range(len(xm)):
        m, n = int(xm[i]), int(xn_in[i])
        if m >= mpol or abs(n) > ntor:
            continue
        rc, zs = float(rmnc[-1, i]), float(zmns[-1, i])
        if abs(rc) > 1e-16 or abs(zs) > 1e-16:
            rbc[(n, m)] = rc
            zbs[(n, m)] = zs

    with open(input_path, "w") as f:
        f.write("&INDATA\n")
        f.write("  DELT = 0.9\n  TCON0 = 1.0\n")
        f.write(f"  NFP = {nfp}\n  MPOL = {mpol}\n  NTOR = {ntor}\n")
        f.write(f"  NS_ARRAY = {ns}\n  NITER_ARRAY = 5000\n")
        #f.write("  NSTEP = 200\n  FTOL_ARRAY = 1.0E-12\n")
        f.write("  NSTEP = 200\n  FTOL_ARRAY = 1.0E-09\n")
        f.write(f"  PHIEDGE = {phiedge:.15e}\n")
        f.write("  GAMMA = 0.0\n  LFREEB = F\n")

        if free_iota:
            f.write("  NCURR = 1\n")
            if zero_current:
                f.write("  PCURR_TYPE = 'power_series'\n")
                f.write("  AC(0) = 0.000000000000000e+00\n")
            elif len(current_aux) >= 2:
                f.write(f"  PCURR_TYPE = '{pcurr_type}'\n")
                f.write(f"  CURTOR = {source_ctor:.15e}\n")
                for i, (s_val, f_val) in enumerate(current_aux, start=1):
                    f.write(f"  AC_AUX_S({i}) = {s_val:.15e}\n")
                    f.write(f"  AC_AUX_F({i}) = {f_val:.15e}\n")
            elif ac is not None and np.any(np.abs(ac) > 0.0):
                f.write(f"  PCURR_TYPE = '{pcurr_type}'\n")
                f.write(f"  CURTOR = {source_ctor:.15e}\n")
                for i, c in enumerate(ac):
                    f.write(f"  AC({i}) = {c:.15e}\n")
            else:
                f.write("  PCURR_TYPE = 'power_series'\n")
                f.write("  AC(0) = 0.000000000000000e+00\n")
        else:
            f.write("  NCURR = 0\n")
            f.write("  PIOTA_TYPE = 'power_series'\n")
            if prescribed_iota_ax is not None and prescribed_iota_edge is not None:
                ai = np.array([prescribed_iota_ax,
                               prescribed_iota_edge - prescribed_iota_ax])
            else:
                ai = np.polynomial.polynomial.polyfit(
                    s_full, iotaf, min(10, ns_w - 1))
            for i, c in enumerate(ai):
                f.write(f"  AI({i}) = {c:.15e}\n")

        f.write("  PMASS_TYPE = 'power_series'\n")
        f.write(f"  PRES_SCALE = {p_max:.15e}\n")
        for i, c in enumerate(am):
            f.write(f"  AM({i}) = {c:.15e}\n")
        for (n, m) in sorted(rbc.keys()):
            f.write(f"  RBC({n:d},{m:d}) = {rbc[(n, m)]:.15e}\n")
            zs = zbs.get((n, m), 0.0)
            if abs(zs) > 1e-16:
                f.write(f"  ZBS({n:d},{m:d}) = {zs:.15e}\n")
        f.write("/\n")


def _profile_error(reference, candidate, scale_floor=1.0e-12):
    """Compare radial profiles after interpolation to a common normalized grid."""
    reference = np.asarray(reference, dtype=float).reshape(-1)
    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    if reference.size < 2 or candidate.size < 2:
        raise ValueError("profile comparison requires at least two radial samples")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("profile comparison received non-finite values")

    s_reference = np.linspace(0.0, 1.0, reference.size)
    s_candidate = np.linspace(0.0, 1.0, candidate.size)
    candidate_on_reference = np.interp(s_reference, s_candidate, candidate)
    delta = candidate_on_reference - reference
    scale = max(float(np.max(np.abs(reference))), float(scale_floor))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "max_relative_to_reference_peak": float(np.max(np.abs(delta)) / scale),
        "rms_relative_to_reference_peak": float(np.sqrt(np.mean(delta**2)) / scale),
    }


def _validate_reconstructed_equilibrium(source_wout, rebuilt_wout):
    """Fail before optimization if wout-to-input reconstruction drifted."""
    with netCDF4.Dataset(source_wout, "r") as ds:
        ref_iota = np.asarray(ds.variables["iotaf"][:], dtype=float)
        ref_pressure = np.asarray(ds.variables["presf"][:], dtype=float)
        ref_current_density = (
            np.asarray(ds.variables["jcurv"][:], dtype=float)
            if "jcurv" in ds.variables else None
        )
        ref_ctor = float(np.asarray(ds.variables["ctor"][:]).squeeze())
        ref_beta = float(np.asarray(ds.variables["betatotal"][:]).squeeze())

    got_iota = np.asarray(rebuilt_wout.iotaf, dtype=float)
    got_pressure = np.asarray(rebuilt_wout.presf, dtype=float)
    iota_profile = _profile_error(ref_iota, got_iota)
    pressure_profile = _profile_error(ref_pressure, got_pressure)
    current_profile = None
    if ref_current_density is not None and hasattr(rebuilt_wout, "jcurv"):
        current_profile = _profile_error(
            ref_current_density,
            np.asarray(rebuilt_wout.jcurv, dtype=float),
        )
    iota_endpoint_error = max(
        abs(float(got_iota[0]) - float(ref_iota[0])),
        abs(float(got_iota[-1]) - float(ref_iota[-1])),
    )
    ctor_scale = max(abs(ref_ctor), 1.0)
    beta_scale = max(abs(ref_beta), 1e-8)
    ctor_rel_error = abs(float(rebuilt_wout.ctor) - ref_ctor) / ctor_scale
    beta_rel_error = abs(float(rebuilt_wout.betatotal) - ref_beta) / beta_scale

    profile_drift = (
        iota_profile["max_abs"] > 0.02
        or pressure_profile["max_relative_to_reference_peak"] > 0.02
        or (
            current_profile is not None
            and current_profile["max_relative_to_reference_peak"] > 0.05
        )
    )
    if (
        iota_endpoint_error > 0.02
        or ctor_rel_error > 0.05
        or beta_rel_error > 0.10
        or profile_drift
    ):
        raise RuntimeError(
            "wout-to-input reconstruction changed the operating point: "
            f"max endpoint |delta iota|={iota_endpoint_error:.3e}, "
            f"max profile |delta iota|={iota_profile['max_abs']:.3e}, "
            "pressure profile peak-relative error="
            f"{pressure_profile['max_relative_to_reference_peak']:.3e}, "
            "current-density profile peak-relative error="
            f"{None if current_profile is None else current_profile['max_relative_to_reference_peak']}, "
            f"current relative error={ctor_rel_error:.3e}, "
            f"beta relative error={beta_rel_error:.3e}. "
            "Pass the original --vmec_input_file when exact profile metadata "
            "cannot be reconstructed."
        )
    return {
        "iota_endpoint_max_abs_error": iota_endpoint_error,
        "iota_profile_max_abs_error": iota_profile["max_abs"],
        "pressure_profile_peak_relative_error": pressure_profile[
            "max_relative_to_reference_peak"
        ],
        "current_density_profile_peak_relative_error": (
            None if current_profile is None
            else current_profile["max_relative_to_reference_peak"]
        ),
        "current_relative_error": ctor_rel_error,
        "beta_relative_error": beta_rel_error,
    }


def _compute_mirror_penalty(mirror_ratio, mirror_target):
    """Penalise mirror ratio above the configured upper bound."""
    return hinge_loss(mirror_ratio, mirror_target)


def _compute_beta_penalty(beta, beta_target):
    """Penalise total beta above the configured upper bound."""
    if not np.isfinite(beta):
        return INVALID_OBJECTIVE
    return hinge_loss(beta, beta_target)


def _compute_iota_penalty(iota_axis, iota_edge, iota_ax_target,
                          iota_edge_target, tolerance=0.01,
                          edge_mode="target"):
    axis_penalty = 0.0
    if iota_ax_target is not None:
        axis_penalty = hinge_loss(abs(iota_axis - iota_ax_target), tolerance)
    if edge_mode == "upper":
        edge_penalty = hinge_loss(iota_edge - iota_edge_target, tolerance)
    elif edge_mode == "lower":
        edge_penalty = hinge_loss(iota_edge_target - iota_edge, tolerance)
    else:
        edge_penalty = hinge_loss(abs(iota_edge - iota_edge_target), tolerance)
    return axis_penalty + edge_penalty


def _compute_well_penalty(vmec_ro, target_well):
    """Penalise magnetic hill (well_depth < target_well).

    well_depth = (V'_axis - V'_edge) / V'_axis.
    Positive → magnetic well (stable), negative → hill (unstable).
    """
    try:
        vp = np.array(vmec_ro.wout.vp)
        vp_axis = float(vp[1])
        vp_edge = float(vp[-1])
        if abs(vp_axis) < 1e-30:
            return 0.0
        well_depth = (vp_axis - vp_edge) / vp_axis
        return hinge_loss(target_well - well_depth, 0.0)
    except Exception:
        return INVALID_OBJECTIVE


def _mercier_residuals(vmec_ro, s_min=0.1, s_max=0.95):
    try:
        dmerc_full = flux_normalize_mercier(vmec_ro.wout.DMerc, vmec_ro.wout)
        s, dmerc = vmec_half_grid_profile(dmerc_full)
        mask = np.isfinite(dmerc) & (s >= s_min) & (s <= s_max)
        vals = dmerc[mask]
        if vals.size == 0:
            return np.array([np.sqrt(INVALID_OBJECTIVE)])
        return np.maximum(-vals, 0.0)
    except Exception:
        return np.array([np.sqrt(INVALID_OBJECTIVE)])


def _parse_rational(value):
    """Parse rationals supplied as strings like '4/5' or numeric values."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), str(value)
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        if abs(den_f) < 1e-15:
            raise ValueError(f"Invalid rational denominator: {value}")
        return float(num) / den_f, text
    return float(text), text


def _rational_placement_residuals(vmec_ro, args, skip_hard=False):
    """
    Low-order iota rational placement guard.

    Soft residuals penalize closest approach inside [s_min, s_max].
    If hard_rational_crossing is enabled, crossings of rational_hard_targets
    return a large residual to reject the trial.

    Parameters
    ----------
    skip_hard : bool
        If True, the hard crossing penalty is suppressed (used for the
        starting-point evaluation so that the initial objective is not
        dominated by a pre-existing rational crossing outside the
        optimiser's control).
    """
    targets_raw = getattr(args, "rational_targets", []) or []
    if isinstance(targets_raw, (str, int, float)):
        targets_raw = [targets_raw]
    hard_raw = getattr(args, "rational_hard_targets", []) or []
    if isinstance(hard_raw, (str, int, float)):
        hard_raw = [hard_raw]

    targets = [_parse_rational(x) for x in targets_raw]
    hard_values = {round(_parse_rational(x)[0], 12) for x in hard_raw}
    if not targets:
        return np.zeros(1), {
            "nearest_label": "",
            "nearest_s": np.nan,
            "nearest_distance": np.nan,
            "crossing_count": 0,
            "hard_crossing_count": 0,
        }

    iota = np.asarray(vmec_ro.wout.iotaf, dtype=float)
    ns = iota.size
    if ns < 2:
        return np.array([np.sqrt(INVALID_OBJECTIVE)]), {
            "nearest_label": "",
            "nearest_s": np.nan,
            "nearest_distance": np.nan,
            "crossing_count": 0,
            "hard_crossing_count": 0,
        }

    s_native = np.linspace(0.0, 1.0, ns)
    s_min = float(getattr(args, "rational_s_min", 0.1))
    s_max = float(getattr(args, "rational_s_max", 0.95))
    n_scan = max(int(getattr(args, "rational_scan_ns", 96)), 2)
    s_scan = np.linspace(s_min, s_max, n_scan)
    i_scan = np.interp(s_scan, s_native, iota)

    min_distance = float(getattr(args, "rational_min_distance", 0.003))
    scale = max(float(getattr(args, "rational_distance_scale", 0.001)), 1e-12)
    residuals = []
    nearest = {"label": "", "s": np.nan, "distance": np.inf}
    crossing_count = 0
    hard_crossing_count = 0

    for val, label in targets:
        delta = i_scan - val
        abs_delta = np.abs(delta)
        idx = int(np.argmin(abs_delta))
        dist = float(abs_delta[idx])
        residuals.append(max((min_distance - dist) / scale, 0.0))
        signs = np.sign(delta)
        exact = np.any(abs_delta < 1e-10)
        sign_cross = np.any(signs[:-1] * signs[1:] < 0)
        crosses = bool(exact or sign_cross)
        if crosses:
            crossing_count += 1
            if round(val, 12) in hard_values:
                hard_crossing_count += 1
        if dist < nearest["distance"]:
            nearest = {"label": label, "s": float(s_scan[idx]), "distance": dist}

    if getattr(args, "hard_rational_crossing", False):
        if hard_crossing_count > 0 and not skip_hard:
            residuals.append(np.sqrt(INVALID_OBJECTIVE))
        else:
            # Keep dimension fixed for finite-difference Jacobian
            residuals.append(0.0)

    metrics = {
        "nearest_label": nearest["label"],
        "nearest_s": nearest["s"],
        "nearest_distance": nearest["distance"],
        "crossing_count": crossing_count,
        "hard_crossing_count": hard_crossing_count,
    }
    return np.asarray(residuals, dtype=float), metrics


def run_vmec(args):
    """Optimisation using SIMSOPT + VMEC (LeastSquaresProblem)."""
    print("\n  Backend: VMEC (SIMSOPT + VMEC2000)")

    requested_run_dir = getattr(args, "run_dir", None)
    if not requested_run_dir:
        raise ValueError(
            "VMEC backend requires an explicit args.run_dir; "
            "refusing to write products into the current working directory"
        )
    run_dir = os.path.abspath(requested_run_dir)
    os.makedirs(run_dir, exist_ok=True)
    input_path = os.path.join(run_dir, "input.squid_init")
    free_iota = getattr(args, 'free_iota', True)
    vmec_input_file = getattr(args, "vmec_input_file", None)
    if vmec_input_file:
        print(f"  Copying VMEC input -> {input_path} ...")
        print("  Mode: direct VMEC input (preserve pressure/current/profile settings)")
        shutil.copy2(vmec_input_file, input_path)
    else:
        print(f"  Converting wout -> {input_path} ...")
        if free_iota:
            print("  Mode: NCURR=1 (iota self-consistent with boundary)")
        else:
            print("  Mode: NCURR=0 (iota prescribed by AI coefficients)")
        wout_to_input(args.nc_file, input_path, ns=args.ns_vmec,
                      prescribed_iota_ax=args.iota_ax,
                      prescribed_iota_edge=args.iota_edge,
                      free_iota=free_iota,
                      zero_current=getattr(args, "zero_current", False))

    with _run_in_directory(run_dir):
        # VMEC opens some legacy Fortran output units during initialization.
        # Constructing here keeps those persistent handles inside the run.
        vmec = Vmec(input_path)
        vmec.run()
    if args.nc_file and not vmec_input_file and free_iota and not getattr(args, "zero_current", False):
        reconstruction = _validate_reconstructed_equilibrium(args.nc_file, vmec.wout)
        print(
            "  Reconstruction check: "
            f"|delta iota|={reconstruction['iota_endpoint_max_abs_error']:.2e}, "
            "profile |delta iota|="
            f"{reconstruction['iota_profile_max_abs_error']:.2e}, "
            "profile delta p/p="
            f"{reconstruction['pressure_profile_peak_relative_error']:.2e}, "
            f"delta I/I={reconstruction['current_relative_error']:.2e}, "
            f"delta beta/beta={reconstruction['beta_relative_error']:.2e}"
        )
    A0 = vmec.aspect()
    if args.aspect_target is None:
        args.aspect_target = round(A0, 1)
    print(f"  Aspect: {A0:.2f}  (upper bound {args.aspect_target})")

    # --- DoFs ---
    surf = vmec.boundary
    surf.fix_all()
    freed = []
    m_max = int(getattr(args, "free_m_max", 4))
    n_max = int(getattr(args, "free_n_max", 3))
    m_order = [1, 0] + [m for m in range(2, m_max + 1)]
    n_order = [0]
    for n in range(1, n_max + 1):
        n_order.extend([n, -n])
    for m in m_order:
        for n in n_order:
            if m == 0 and n == 0:
                continue
            if len(freed) >= args.max_dofs:
                break
            for coeff in ("rc", "zs"):
                if len(freed) >= args.max_dofs:
                    break
                name = f"{coeff}({m},{n})"
                try:
                    surf.unfix(name)
                    val = (surf.get_rc(m, n) if coeff == "rc"
                           else surf.get_zs(m, n))
                    freed.append((name, val))
                except Exception:
                    pass

    n_dofs = len(freed)
    print(f"\n  Free DoFs ({n_dofs}):")
    for name, val in freed:
        print(f"    {name:>12s} = {val:+.6f}")

    s_vals = np.linspace(0.2, 0.8, args.num_surfaces)
    alphas = np.linspace(0, 2 * np.pi, args.num_alpha, endpoint=False)
    x0_vmec = np.array(vmec.x, dtype=float)
    s_grad = np.linspace(args.grad_s_smin, args.grad_s_smax, args.grad_s_ns)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if getattr(args, "checkpoint_every", 0) > 0 or getattr(args, "html_every", 0) > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(run_dir, "history.csv")
    history_fields = [
        "eval", "objective_terms_ready", "f_QI", "f_maxJ", "f_Bmin",
        "mirror_ratio", "iota_axis", "iota_edge", "beta", "f_grad_s",
        "well_depth", "f_mercier", "dmerc_convention",
        "dmerc_edge_toroidal_flux_wb", "dmerc_vmec_raw_min",
        "dmerc_flux_normalized_min", "dmerc_min_s", "dmerc_negative_count",
        "dmerc_axis_vmec_raw_min", "dmerc_axis_flux_normalized_min",
        "dmerc_axis_min_s", "dmerc_axis_negative_count",
        "dshear_vmec_raw_min", "dshear_flux_normalized_min", "dshear_min_s",
        "dwell_vmec_raw_min", "dwell_flux_normalized_min", "dwell_min_s",
        "dcurr_vmec_raw_min", "dcurr_flux_normalized_min", "dcurr_min_s",
        "dgeod_vmec_raw_min", "dgeod_flux_normalized_min", "dgeod_min_s",
        "iota_prime_absmin", "iota_prime_absmin_s", "pprime_min",
        "pprime_min_s",
        "iota_profile_rms_dev", "iota_profile_max_dev",
        "iota_topology_shear_absmin", "iota_topology_shear_absmin_s",
        "iota_monotonic_violation_count", "iota_reference_direction",
        "iota_profile_residual_rms", "iota_shear_residual_rms",
        "iota_monotonic_residual_rms", "iota_topology_hard_violation",
        "balloon_n_pos", "balloon_lam_max",
        "coil_Bn_max", "coil_Bn_rms", "coil_K_rms", "coil_K_max",
        "coil_phi_high", "coil_r_bn", "coil_r_k", "coil_r_phi",
        "coil_hard_bn",
        "oops_scalar_mean", "oops_scalar_max", "oops_residual_rms",
        "local_maxj_rms", "local_maxj_max", "local_maxj_count",
        "maxj_guard_residual", "oops_guard_residual",
        "boundary_pdrot_aw", "boundary_pdrot_max", "boundary_pdrot_p99",
        "boundary_pdrot_cvar1", "boundary_k2_min", "boundary_H_min",
        "boundary_r_pdrot_aw", "boundary_r_pdrot_max", "boundary_r_pdrot_p99",
        "boundary_r_pdrot_cvar1", "boundary_r_k2", "boundary_r_H",
        "rational_nearest", "rational_nearest_s", "rational_nearest_distance",
        "rational_crossing_count", "rational_hard_crossing_count",
        "pdrot_mean", "pdrot_rms", "pdrot_median", "pdrot_max",
        "pdrot_p95", "pdrot_p99", "pdrot_p999", "pdrot_cvar1",
        "pdrot_q_mean", "pdrot_q_p95",
        "pdrot_a_eff", "pdrot_kappa_gap_max",
        "aspect", "elapsed_s",
    ]
    with open(history_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=history_fields).writeheader()

    def _append_history(row):
        with open(history_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=history_fields)
            writer.writerow({k: row.get(k, "") for k in history_fields})

    def _write_checkpoint(eval_id, make_html=False):
        stem = f"squid_eval_{eval_id:06d}"
        input_out = os.path.join(checkpoint_dir, f"input.{stem}")
        try:
            vmec.write_input(input_out)
        except Exception as exc:
            print(f"    [SQuID #{eval_id}] checkpoint input failed: {exc}")
            return
        if not make_html:
            return
        try:
            with _run_in_directory(run_dir):
                vmec.run()
            wout_name = getattr(vmec, "output_file", None)
            if wout_name is None:
                wout_name = f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc"
            if not os.path.isabs(wout_name):
                wout_name = os.path.join(run_dir, wout_name)
            if os.path.exists(wout_name):
                wout_out = os.path.join(checkpoint_dir, f"wout_{stem}.nc")
                shutil.copy2(wout_name, wout_out)
                html_out = os.path.join(checkpoint_dir, f"{stem}.html")
                script = getattr(args, "wout_to_html_script", None) or ""
                if script and os.path.exists(script):
                    subprocess.run(
                        [sys.executable, script, "--wout", wout_out,
                         "--output", html_out, "--style", "auto"],
                        check=False, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception as exc:
            print(f"    [SQuID #{eval_id}] checkpoint HTML failed: {exc}")

    # Auto-detect iota targets
    iota_arr = vmec.wout.iotaf
    if args.iota_ax is None:
        args.iota_ax = round(float(iota_arr[0]), 3)
    if args.iota_edge is None:
        args.iota_edge = round(float(iota_arr[-1]), 3)
    print(f"  Iota targets: axis={args.iota_ax:.3f}  edge={args.iota_edge:.3f}")

    iota_profile_ns = max(int(getattr(args, "iota_profile_ns", 17)), 3)
    iota_profile_s_min = float(getattr(args, "iota_profile_s_min", 0.1))
    iota_profile_s_max = float(getattr(args, "iota_profile_s_max", 0.95))
    if not 0.0 <= iota_profile_s_min < iota_profile_s_max <= 1.0:
        raise ValueError("Require 0 <= iota_profile_s_min < iota_profile_s_max <= 1")
    iota_profile_s = np.linspace(iota_profile_s_min, iota_profile_s_max, iota_profile_ns)
    iota_initial_s = np.linspace(0.0, 1.0, len(iota_arr))
    iota_profile_reference = np.interp(iota_profile_s, iota_initial_s, iota_arr)
    iota_reference_gradient = np.gradient(iota_profile_reference, iota_profile_s)
    iota_reference_direction = float(np.sign(np.nanmedian(iota_reference_gradient)))
    if iota_reference_direction == 0.0:
        iota_reference_direction = float(np.sign(iota_profile_reference[-1] - iota_profile_reference[0]))
    if iota_reference_direction == 0.0:
        iota_reference_direction = 1.0
    print(
        f"  Iota topology reference: s=[{iota_profile_s_min:.2f},{iota_profile_s_max:.2f}] "
        f"N={iota_profile_ns} direction={iota_reference_direction:+.0f}"
    )

    class SQuIDObjective(Optimizable):
        def __init__(self):
            self._cache_x = None
            self._fmaxj = 0.0
            self._fqi = 0.0
            self._fbmin = 0.0
            self._qi_residuals = np.zeros(1)
            self._maxj_residuals = np.zeros(1)
            self._local_maxj_residuals = np.zeros(1)
            self._local_maxj_metrics = {"rms": 0.0, "max": 0.0, "count": 0}
            self._bmin_residuals = np.zeros(1)
            self._fgs = 0.0
            self._mirror = 0.0
            self._beta = 0.0
            self._iota_ax = 0.0
            self._iota_ed = 0.0
            self._well_depth = 0.0
            self._mercier_residuals = np.zeros(1)
            self._mercier_margin_residuals = np.zeros(1)
            self._hard_mhd_residuals = np.zeros(1)
            self._ballooning_residuals = np.zeros(80)  # 10 rhos × 8 alphas
            self._ballooning_lam_raw = np.zeros(80)
            self._coil_proxy_residual_bn = np.zeros(1)
            self._coil_proxy_residual_k = np.zeros(1)
            self._coil_proxy_residual_phi = np.zeros(1)
            self._coil_proxy_hard_bn_residual = np.zeros(1)
            self._coil_proxy_metrics = {}
            self._oops_residuals = np.zeros(1)
            self._oops_metrics = {}
            self._initial_fmaxj = None
            self._initial_oops_scalar = None
            self._maxj_guard_residual = np.zeros(1)
            self._oops_guard_residual = np.zeros(1)
            self._boundary_curvature_residuals = np.zeros(6)
            self._boundary_curvature_metrics = {}
            self._rational_residuals = np.zeros(1)
            self._rational_metrics = {}
            self._pdrot_residuals = np.zeros(1)
            self._pdrot_metrics = {}
            self._fmercier = 0.0
            self._dmerc_vmec_raw_min = np.nan
            self._dmerc_flux_normalized_min = np.nan
            self._dmerc_min_s = np.nan
            self._dmerc_edge_toroidal_flux_wb = np.nan
            self._dmerc_negative_count = 0
            self._dmerc_axis_vmec_raw_min = np.nan
            self._dmerc_axis_flux_normalized_min = np.nan
            self._dmerc_axis_min_s = np.nan
            self._dmerc_axis_negative_count = 0
            self._mhd_profile_metrics = {}
            self._iota_profile_residuals = np.zeros(iota_profile_ns)
            self._iota_shear_residuals = np.zeros(iota_profile_ns)
            self._iota_monotonic_residuals = np.zeros(iota_profile_ns)
            self._hard_iota_topology_residuals = np.zeros(1)
            self._iota_topology_metrics = {}
            self._n = 0
            super().__init__(depends_on=[vmec])

        @staticmethod
        def _large_like(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return np.array([np.sqrt(INVALID_OBJECTIVE)])
            return np.full_like(arr, np.sqrt(INVALID_OBJECTIVE), dtype=float)

        @staticmethod
        def _profile_min(values, s, mask):
            values = np.asarray(values, dtype=float)
            s = np.asarray(s, dtype=float)
            idxs = np.where(mask & np.isfinite(values))[0]
            if idxs.size == 0:
                return np.nan, np.nan
            i = int(idxs[np.nanargmin(values[idxs])])
            return float(values[i]), float(s[i])

        @staticmethod
        def _profile_absmin(values, s, mask):
            values = np.asarray(values, dtype=float)
            s = np.asarray(s, dtype=float)
            idxs = np.where(mask & np.isfinite(values))[0]
            if idxs.size == 0:
                return np.nan, np.nan
            i = int(idxs[np.nanargmin(np.abs(values[idxs]))])
            return float(abs(values[i])), float(s[i])

        def _mark_hard_mhd_violation(self):
            current = np.asarray(self._hard_mhd_residuals, dtype=float).ravel()
            if current.size == 1 and current[0] == 0.0:
                current = np.zeros(0, dtype=float)
            self._hard_mhd_residuals = np.append(
                current, np.sqrt(INVALID_OBJECTIVE)
            )

        def _update_mhd_profile_metrics(self):
            metrics = {}
            try:
                mercier = mercier_profiles(vmec.wout)
                phi_edge = mercier["edge_toroidal_flux_wb"]
                dmerc_raw_full = mercier["profiles"]["DMerc"]["vmec_raw"]
                dmerc_fluxnorm_full = mercier["profiles"]["DMerc"]["flux_normalized"]
                s, dmerc_raw = vmec_half_grid_profile(dmerc_raw_full)
                _, dmerc_fluxnorm = vmec_half_grid_profile(
                    dmerc_fluxnorm_full, ns=dmerc_raw_full.size
                )
                mask_mid = (
                    (s >= float(args.mercier_s_min))
                    & (s <= float(args.mercier_s_max))
                    & np.isfinite(dmerc_raw)
                )
                mask_axis = (s > 0.0) & (s < 0.1) & np.isfinite(dmerc_raw)
                dmin_raw, dmin_s = self._profile_min(dmerc_raw, s, mask_mid)
                dmin_fluxnorm, _ = self._profile_min(dmerc_fluxnorm, s, mask_mid)
                amin_raw, amin_s = self._profile_min(dmerc_raw, s, mask_axis)
                amin_fluxnorm, _ = self._profile_min(dmerc_fluxnorm, s, mask_axis)
                self._dmerc_vmec_raw_min = dmin_raw
                self._dmerc_flux_normalized_min = dmin_fluxnorm
                self._dmerc_min_s = dmin_s
                self._dmerc_edge_toroidal_flux_wb = phi_edge
                self._dmerc_negative_count = int(np.sum(dmerc_raw[mask_mid] < 0)) if np.any(mask_mid) else 0
                self._dmerc_axis_vmec_raw_min = amin_raw
                self._dmerc_axis_flux_normalized_min = amin_fluxnorm
                self._dmerc_axis_min_s = amin_s
                self._dmerc_axis_negative_count = int(np.sum(dmerc_raw[mask_axis] < 0)) if np.any(mask_axis) else 0
                metrics.update(
                    dmerc_convention=MERCIER_CONVENTION_VERSION,
                    dmerc_edge_toroidal_flux_wb=self._dmerc_edge_toroidal_flux_wb,
                    dmerc_vmec_raw_min=self._dmerc_vmec_raw_min,
                    dmerc_flux_normalized_min=self._dmerc_flux_normalized_min,
                    dmerc_min_s=self._dmerc_min_s,
                    dmerc_negative_count=self._dmerc_negative_count,
                    dmerc_axis_vmec_raw_min=self._dmerc_axis_vmec_raw_min,
                    dmerc_axis_flux_normalized_min=self._dmerc_axis_flux_normalized_min,
                    dmerc_axis_min_s=self._dmerc_axis_min_s,
                    dmerc_axis_negative_count=self._dmerc_axis_negative_count,
                )
                for name, out_prefix in (
                    ("DShear", "dshear"),
                    ("DWell", "dwell"),
                    ("DCurr", "dcurr"),
                    ("DGeod", "dgeod"),
                ):
                    if name in mercier["profiles"]:
                        component = mercier["profiles"][name]
                        _, component_raw = vmec_half_grid_profile(
                            component["vmec_raw"], ns=dmerc_raw_full.size
                        )
                        _, component_fluxnorm = vmec_half_grid_profile(
                            component["flux_normalized"], ns=dmerc_raw_full.size
                        )
                        raw_min, vmin_s = self._profile_min(
                            component_raw, s, mask_mid
                        )
                        fluxnorm_min, _ = self._profile_min(
                            component_fluxnorm, s, mask_mid
                        )
                        metrics[f"{out_prefix}_vmec_raw_min"] = raw_min
                        metrics[f"{out_prefix}_flux_normalized_min"] = fluxnorm_min
                        metrics[f"{out_prefix}_min_s"] = vmin_s
                iota = np.asarray(vmec.wout.iotaf, dtype=float)
                si = np.linspace(0.0, 1.0, len(iota))
                ip = np.gradient(iota, si)
                mask_i = (
                    (si >= float(args.mercier_s_min))
                    & (si <= float(args.mercier_s_max))
                    & np.isfinite(ip)
                )
                ip_min, ip_min_s = self._profile_absmin(ip, si, mask_i)
                metrics["iota_prime_absmin"] = ip_min
                metrics["iota_prime_absmin_s"] = ip_min_s
                pres = np.asarray(vmec.wout.presf, dtype=float)
                sp = np.linspace(0.0, 1.0, len(pres))
                pp = np.gradient(pres, sp)
                mask_p = (
                    (sp >= float(args.mercier_s_min))
                    & (sp <= float(args.mercier_s_max))
                    & np.isfinite(pp)
                )
                pp_min, pp_min_s = self._profile_min(pp, sp, mask_p)
                metrics["pprime_min"] = pp_min
                metrics["pprime_min_s"] = pp_min_s
            except Exception as exc:
                print(f"    [SQuID #{self._n}] MHD profile diagnostics FAILED: {exc}")
            self._mhd_profile_metrics = metrics

        def _update_iota_topology(self):
            try:
                result = iota_topology_residuals(
                    np.asarray(vmec.wout.iotaf, dtype=float),
                    reference_iota=iota_profile_reference, sample_s=iota_profile_s,
                    profile_tolerance=float(getattr(args, "iota_profile_tolerance", 0.004)),
                    profile_scale=float(getattr(args, "iota_profile_scale", 0.004)),
                    shear_absmin=float(getattr(args, "iota_shear_absmin", 0.005)),
                    shear_scale=float(getattr(args, "iota_shear_scale", 0.0025)),
                    reference_direction=iota_reference_direction,
                    monotonic_scale=float(getattr(args, "iota_monotonic_scale", 0.0025)),
                )
                self._iota_profile_residuals = result["profile_residuals"]
                self._iota_shear_residuals = result["shear_residuals"]
                self._iota_monotonic_residuals = result["monotonic_residuals"]
                self._iota_topology_metrics = result["metrics"]
                hard_violation = False
                if getattr(args, "hard_gate_iota_topology", False):
                    metrics = self._iota_topology_metrics
                    hard_violation = (
                        (not np.isfinite(metrics["profile_max_dev"]))
                        or metrics["profile_max_dev"] > float(getattr(args, "hard_iota_profile_max_dev", np.inf))
                        or (not np.isfinite(metrics["shear_absmin"]))
                        or metrics["shear_absmin"] < float(getattr(args, "hard_iota_shear_absmin", 0.0))
                        or metrics["monotonic_violation_count"] > int(getattr(args, "hard_iota_monotonic_violation_max", 0))
                    )
                self._iota_topology_metrics["hard_violation"] = int(hard_violation)
                self._hard_iota_topology_residuals = np.array(
                    [np.sqrt(INVALID_OBJECTIVE) if hard_violation else 0.0]
                )
            except Exception as exc:
                print(f"    [SQuID #{self._n}] Iota topology FAILED: {exc}")
                self._iota_profile_residuals = self._large_like(self._iota_profile_residuals)
                self._iota_shear_residuals = self._large_like(self._iota_shear_residuals)
                self._iota_monotonic_residuals = self._large_like(self._iota_monotonic_residuals)
                self._hard_iota_topology_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                self._iota_topology_metrics = {"hard_violation": 1}

        def _compute(self):
            try:
                cx = tuple(vmec.x)
            except Exception:
                cx = None
            if cx is not None and cx == self._cache_x:
                return
            max_total_evals = int(getattr(args, "max_total_evals", 0) or 0)
            if max_total_evals > 0 and self._n >= max_total_evals:
                raise EvaluationLimitReached(
                    f"Reached --max_total_evals={max_total_evals}"
                )
            self._n += 1
            t0 = time.time()
            try:
                info = _evaluate_squid(
                    vmec, s_vals, alphas,
                    args.num_pitch, T_J=-0.06, mboz=args.mboz, nboz=args.nboz,
                )
            except Exception as exc:
                print(f"    [SQuID #{self._n}] FAILED: {exc}")
                info = dict(f_maxJ=INVALID_OBJECTIVE, f_QI=INVALID_OBJECTIVE,
                            f_Bmin=INVALID_OBJECTIVE,
                            qi_residuals=self._large_like(self._qi_residuals),
                            maxj_residuals=self._large_like(self._maxj_residuals),
                            bmin_residuals=self._large_like(self._bmin_residuals),
                            mirror_ratio=0.0,
                            iota_axis=0.0, iota_edge=0.0,
                            B_min=0.0, B_max=0.0,
                            surface_Bmin=np.full_like(s_vals, np.nan))
            self._fmaxj = info["f_maxJ"]
            self._fqi = info["f_QI"]
            self._fbmin = info.get("f_Bmin", 0.0)
            self._qi_residuals = info["qi_residuals"]
            self._maxj_residuals = info["maxj_residuals"]
            self._local_maxj_residuals = np.zeros(1)
            self._local_maxj_metrics = {"rms": 0.0, "max": 0.0, "count": 0}
            if getattr(args, "w_local_maxj", 0.0) > 0:
                try:
                    blocks = np.asarray(
                        info.get("maxj_exceedance_blocks", np.zeros((0, 0, 0))),
                        dtype=float,
                    )
                    interval_centers = np.asarray(
                        info.get("maxj_interval_centers", []), dtype=float
                    )
                    lambda_grid = np.asarray(
                        info.get("maxj_lambda_grid", []), dtype=float
                    )
                    if blocks.ndim != 3 or blocks.size == 0:
                        selected = np.zeros(0)
                    else:
                        s_center = float(getattr(args, "local_maxj_s_center", 0.35))
                        s_width = max(float(getattr(args, "local_maxj_s_width", 0.35)), 0.0)
                        lam_min = float(getattr(args, "local_maxj_lambda_min", 0.0))
                        lam_max = float(getattr(args, "local_maxj_lambda_max", 1.0))
                        mask_s = np.abs(interval_centers - s_center) <= 0.5 * s_width
                        mask_l = (lambda_grid >= lam_min) & (lambda_grid <= lam_max)
                        if np.any(mask_s) and np.any(mask_l):
                            selected = blocks[mask_s][:, mask_l, :].ravel()
                        else:
                            selected = np.zeros(0)
                    target = max(float(getattr(args, "local_maxj_exceedance_target", 0.0)), 0.0)
                    selected = np.maximum(selected - target, 0.0)
                    scale = max(float(getattr(args, "local_maxj_scale", 1.0)), 1e-12)
                    if selected.size:
                        self._local_maxj_residuals = selected / (np.sqrt(selected.size) * scale)
                        self._local_maxj_metrics = {
                            "rms": float(np.sqrt(np.mean(selected ** 2))),
                            "max": float(np.max(selected)),
                            "count": int(selected.size),
                        }
                    else:
                        self._local_maxj_residuals = np.zeros(1)
                        self._local_maxj_metrics = {"rms": 0.0, "max": 0.0, "count": 0}
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] local maxJ target FAILED: {exc}")
                    self._local_maxj_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._local_maxj_metrics = {
                        "rms": INVALID_OBJECTIVE, "max": INVALID_OBJECTIVE, "count": 0
                    }
            try:
                self._bmin_residuals = bmin_slope_residuals(
                    s_vals, info["surface_Bmin"], args.bmin_slope_target)
                self._fbmin = float(np.sum(self._bmin_residuals ** 2))
            except Exception as exc:
                print(f"    [SQuID #{self._n}] B_min target FAILED: {exc}")
                self._fbmin = INVALID_OBJECTIVE
                self._bmin_residuals = self._large_like(self._bmin_residuals)
            self._mirror = info["mirror_ratio"]
            self._iota_ax = info["iota_axis"]
            self._iota_ed = info["iota_edge"]

            # Always read beta for history (even when w_beta=0)
            try:
                self._beta = float(vmec.wout.betatotal)
            except Exception:
                self._beta = np.inf
            self._update_mhd_profile_metrics()
            self._update_iota_topology()

            if getattr(args, 'w_grad_s', 0.0) > 0:
                try:
                    self._fgs = ITGResidual(
                        vmec, s_grad, method=args.itg_method).total()
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] ITG target FAILED: {exc}")
                    self._fgs = INVALID_OBJECTIVE

            if getattr(args, 'w_well', 0.0) > 0:
                try:
                    vp = np.array(vmec.wout.vp)
                    vp_ax = float(vp[1])
                    vp_ed = float(vp[-1])
                    self._well_depth = ((vp_ax - vp_ed) / vp_ax
                                        if abs(vp_ax) > 1e-30 else 0.0)
                except Exception:
                    self._well_depth = -np.inf

            if getattr(args, 'w_mercier', 0.0) > 0:
                try:
                    dmerc_full = flux_normalize_mercier(vmec.wout.DMerc, vmec.wout)
                    s, dmerc = vmec_half_grid_profile(dmerc_full)
                    mask = ((s >= args.mercier_s_min)
                            & (s <= args.mercier_s_max)
                            & np.isfinite(dmerc))
                    vals = dmerc[mask]
                    if vals.size == 0:
                        self._mercier_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                        self._fmercier = INVALID_OBJECTIVE
                        self._dmerc_flux_normalized_min = np.nan
                        self._dmerc_negative_count = 0
                    else:
                        self._mercier_residuals = np.maximum(-vals, 0.0)
                        self._fmercier = float(np.mean(self._mercier_residuals ** 2))
                        self._dmerc_flux_normalized_min = float(np.nanmin(vals))
                        self._dmerc_negative_count = int(np.sum(vals < 0))
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Mercier target FAILED: {exc}")
                    self._mercier_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._fmercier = INVALID_OBJECTIVE
                    self._dmerc_flux_normalized_min = np.nan
                    self._dmerc_negative_count = 0

            if getattr(args, 'w_mercier_margin', 0.0) > 0:
                try:
                    dmerc_full = flux_normalize_mercier(vmec.wout.DMerc, vmec.wout)
                    s, dmerc = vmec_half_grid_profile(dmerc_full)
                    mask = ((s >= args.mercier_s_min)
                            & (s <= args.mercier_s_max)
                            & np.isfinite(dmerc))
                    vals = dmerc[mask]
                    if vals.size == 0:
                        self._mercier_margin_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                        if getattr(args, 'w_mercier', 0.0) <= 0:
                            self._dmerc_flux_normalized_min = np.nan
                            self._dmerc_negative_count = 0
                    else:
                        target = float(getattr(
                            args, 'mercier_flux_normalized_margin_target', 0.0
                        ))
                        self._mercier_margin_residuals = np.maximum(target - vals, 0.0)
                        if getattr(args, 'w_mercier', 0.0) <= 0:
                            self._dmerc_flux_normalized_min = float(np.nanmin(vals))
                            self._dmerc_negative_count = int(np.sum(vals < 0))
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Mercier margin target FAILED: {exc}")
                    self._mercier_margin_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    if getattr(args, 'w_mercier', 0.0) <= 0:
                        self._dmerc_flux_normalized_min = np.nan
                        self._dmerc_negative_count = 0

            if getattr(args, 'hard_gate_mhd', False):
                hard_res = []
                try:
                    if not np.isfinite(self._dmerc_flux_normalized_min):
                        dmerc_full = flux_normalize_mercier(
                            vmec.wout.DMerc, vmec.wout
                        )
                        s, dmerc = vmec_half_grid_profile(dmerc_full)
                        mask = ((s >= args.mercier_s_min)
                                & (s <= args.mercier_s_max)
                                & np.isfinite(dmerc))
                        vals = dmerc[mask]
                        if vals.size == 0:
                            self._dmerc_flux_normalized_min = np.nan
                            self._dmerc_negative_count = 0
                        else:
                            self._dmerc_flux_normalized_min = float(np.nanmin(vals))
                            self._dmerc_negative_count = int(np.sum(vals < 0))
                    hard_dmerc_fluxnorm_min = float(getattr(
                        args, 'hard_dmerc_flux_normalized_min', 0.0
                    ))
                    hard_dmerc_neg_max = int(getattr(args, 'hard_dmerc_neg_max', 0))
                    if (
                        (not np.isfinite(self._dmerc_flux_normalized_min))
                        or self._dmerc_flux_normalized_min < hard_dmerc_fluxnorm_min
                        or self._dmerc_negative_count > hard_dmerc_neg_max
                    ):
                        hard_res.append(np.sqrt(INVALID_OBJECTIVE))
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Hard MHD gate FAILED: {exc}")
                    hard_res.append(np.sqrt(INVALID_OBJECTIVE))
                hard_beta_max = float(getattr(args, 'hard_beta_max', float('inf')))
                if np.isfinite(hard_beta_max):
                    if (not np.isfinite(self._beta)) or self._beta > hard_beta_max:
                        hard_res.append(np.sqrt(INVALID_OBJECTIVE))
                self._hard_mhd_residuals = np.array(hard_res or [0.0], dtype=float)
            else:
                self._hard_mhd_residuals = np.zeros(1)

            # ── Ballooning penalty (DESC, no force balance) ──
            # Match the post-run gate/viz convention: use the shifted
            # BallooningStability objective and convert back to raw lambda.
            b_rhos = np.array(getattr(args, 'ballooning_rhos', [0.5, 0.65, 0.8, 0.9]))
            # Match scripts/mhd_gate.py default angular sampling. The previous
            # 6-alpha grid missed marginal edge modes at alpha=3*pi/8.
            b_n_alpha = int(getattr(args, "ballooning_n_alpha", 8))
            b_alphas = np.linspace(0, np.pi, b_n_alpha, endpoint=False)
            b_zeta0 = np.linspace(-0.5*np.pi, 0.5*np.pi, 5)
            b_n_expected = len(b_rhos) * len(b_alphas)  # shape after zeta0 max
            self._ballooning_residuals = np.full(b_n_expected, np.nan)
            self._ballooning_lam_raw = np.full(b_n_expected, np.nan)
            if (
                getattr(args, 'w_ballooning', 0.0) > 0
                or getattr(args, 'hard_gate_mhd', False)
            ):
                try:
                    from desc.vmec import VMECIO
                    from desc.objectives import BallooningStability
                    from desc.objectives import get_fixed_boundary_constraints
                    # Use the VMEC object's own wout — reliable, no glob needed.
                    wout_temp = getattr(vmec, "output_file", None)
                    if wout_temp is None or not os.path.exists(wout_temp):
                        wout_temp = os.path.join(
                            run_dir,
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    eq_b = VMECIO.load(wout_temp, L=6, M=6, N=6)
                    if getattr(args, "ballooning_force_balance", False):
                        eq_b.solve(
                            objective="force",
                            constraints=get_fixed_boundary_constraints(
                                eq=eq_b, profiles=True, normalize=True),
                            optimizer="lsq-exact",
                            maxiter=int(getattr(args, "ballooning_fb_maxiter", 100)),
                            verbose=0,
                        )
                    lam_residuals = []
                    lam_raw_values = []
                    # Hinge residual: stable modes below target are not penalized.
                    lam_target = float(getattr(args, "ballooning_lambda_target", 0.0))
                    lam_shift = 1.0
                    for a in b_alphas:
                        obj = BallooningStability(
                            eq_b, rho=b_rhos, alpha=np.array([a]),
                            nturns=2, nzetaperturn=80, zeta0=b_zeta0,
                            Neigvals=1, lambda0=-lam_shift, w0=0.0, w1=1.0,
                        )
                        obj.build(use_jit=False, verbose=0)
                        v = np.asarray(obj.compute(eq_b.params_dict), dtype=float)
                        lam_raw = v.ravel() - lam_shift
                        lam_raw_values.extend(lam_raw.tolist())
                        lam_residuals.extend(
                            np.maximum(lam_raw - lam_target, 0.0).tolist())
                    self._ballooning_lam_raw = np.array(lam_raw_values, dtype=float)
                    self._ballooning_residuals = np.array(lam_residuals, dtype=float)
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Ballooning FAILED: {exc}")
                    self._ballooning_residuals = np.full(len(b_rhos) * len(b_alphas), np.sqrt(INVALID_OBJECTIVE))
                    self._ballooning_lam_raw = np.full(len(b_rhos) * len(b_alphas), np.nan)

            if getattr(args, 'hard_gate_mhd', False):
                finite_lam = self._ballooning_lam_raw[
                    np.isfinite(self._ballooning_lam_raw)
                ]
                balloon_failed = finite_lam.size != self._ballooning_lam_raw.size
                if not balloon_failed:
                    balloon_failed = (
                        int(np.sum(finite_lam > 0.0))
                        > int(getattr(args, 'hard_ballooning_n_max', 0))
                        or float(np.max(finite_lam))
                        > float(getattr(args, 'hard_ballooning_lambda_max', 0.0))
                    )
                if balloon_failed:
                    self._mark_hard_mhd_violation()

            # ── DESC force-balance residual (no solve, just FB eval on fit) ──
            self._fb_residuals = np.zeros(1)
            self._fb_rms = np.nan
            hard_fb_limit = float(getattr(args, 'hard_fb_rms_max', np.inf))
            if (
                getattr(args, 'w_force_balance', 0.0) > 0
                or (
                    getattr(args, 'hard_gate_mhd', False)
                    and np.isfinite(hard_fb_limit)
                )
            ):
                try:
                    from desc.vmec import VMECIO as _VMECIO_fb
                    from desc.objectives import ForceBalance as _ForceBalance
                    _wout_fb = getattr(vmec, "output_file", None)
                    if _wout_fb is None or not os.path.exists(_wout_fb):
                        _wout_fb = os.path.join(
                            run_dir,
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    _eq_fb = _VMECIO_fb.load(_wout_fb, L=6, M=6, N=6)
                    _fb_obj = _ForceBalance(eq=_eq_fb)
                    _fb_obj.build(use_jit=False, verbose=0)
                    _fb_vals = np.asarray(_fb_obj.compute(_eq_fb.params_dict), dtype=float)
                    _fb_rms = float(np.sqrt(np.mean(_fb_vals**2)))
                    self._fb_rms = _fb_rms
                    # Normalize: residual = sqrt(FB_RMS / FB_RMS0 - 1), ~0 at start
                    if not hasattr(self, '_fb_rms0'):
                        self._fb_rms0 = max(_fb_rms, 1.0)
                    self._fb_residuals = np.array([np.sqrt(max(_fb_rms / self._fb_rms0 - 1.0, 0.0))])
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Force-balance FAILED: {exc}")
                    self._fb_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])

            if (
                getattr(args, 'hard_gate_mhd', False)
                and np.isfinite(hard_fb_limit)
                and (
                    not np.isfinite(self._fb_rms)
                    or self._fb_rms > hard_fb_limit
                )
            ):
                self._mark_hard_mhd_violation()

            # ── Quasi-single-stage coil-realizability proxy ──
            if (
                getattr(args, 'w_coil_proxy_bn', 0.0) > 0
                or getattr(args, 'w_coil_proxy_k', 0.0) > 0
                or getattr(args, 'w_coil_proxy_phi', 0.0) > 0
                or np.isfinite(float(getattr(args, "hard_coil_proxy_bn_max", np.inf)))
            ):
                try:
                    _wout_cp = getattr(vmec, "output_file", None)
                    if _wout_cp is None or not os.path.exists(_wout_cp):
                        _wout_cp = os.path.join(
                            run_dir,
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    self._coil_proxy_metrics = evaluate_coil_proxy(
                        _wout_cp,
                        offset_fraction=getattr(args, "coil_proxy_offset_fraction", 0.35),
                        lambda_regularization=getattr(args, "coil_proxy_lambda_regularization", 1e-8),
                        desc_L=getattr(args, "coil_proxy_desc_L", 4),
                        desc_M=getattr(args, "coil_proxy_desc_M", 4),
                        desc_N=getattr(args, "coil_proxy_desc_N", 4),
                        M_Phi=getattr(args, "coil_proxy_M_Phi", 4),
                        N_Phi=getattr(args, "coil_proxy_N_Phi", 4),
                        source_M=getattr(args, "coil_proxy_source_M", 16),
                        source_N=getattr(args, "coil_proxy_source_N", 16),
                        eval_M=getattr(args, "coil_proxy_eval_M", 16),
                        eval_N=getattr(args, "coil_proxy_eval_N", 16),
                        current_helicity=getattr(args, "coil_proxy_current_helicity", [1, 0]),
                        regularization_type=getattr(args, "coil_proxy_regularization_type", "regcoil"),
                        vacuum=getattr(args, "coil_proxy_vacuum", False),
                        chunk_size=getattr(args, "coil_proxy_chunk_size", None),
                        verbose=getattr(args, "coil_proxy_verbose", 0),
                    )
                    _r = coil_proxy_residuals(
                        self._coil_proxy_metrics,
                        bn_max_target=getattr(args, "coil_proxy_bn_max_target", 5e-3),
                        k_rms_target_MApm=getattr(args, "coil_proxy_k_rms_target_MApm", 2.2),
                        phi_high_target=getattr(args, "coil_proxy_phi_high_target", 0.50),
                    )
                    self._coil_proxy_residual_bn = np.array([_r["bn"]])
                    self._coil_proxy_residual_k = np.array([_r["k"]])
                    self._coil_proxy_residual_phi = np.array([_r["phi"]])
                    hard_bn = float(getattr(args, "hard_coil_proxy_bn_max", np.inf))
                    bn_max = float(self._coil_proxy_metrics.get("Bn_max_abs_unitless", np.nan))
                    if np.isfinite(hard_bn) and np.isfinite(bn_max) and bn_max > hard_bn:
                        self._coil_proxy_hard_bn_residual = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    else:
                        self._coil_proxy_hard_bn_residual = np.zeros(1)
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Coil proxy FAILED: {exc}")
                    bad = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._coil_proxy_residual_bn = bad
                    self._coil_proxy_residual_k = bad
                    self._coil_proxy_residual_phi = bad
                    self._coil_proxy_hard_bn_residual = bad
                    self._coil_proxy_metrics = {}

            if (
                getattr(args, "w_oops", 0.0) > 0
                or getattr(args, "w_oops_guard", 0.0) > 0
            ):
                try:
                    _wout_oops = getattr(vmec, "output_file", None)
                    if _wout_oops is None or not os.path.exists(_wout_oops):
                        _wout_oops = os.path.join(
                            run_dir,
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    self._oops_residuals, self._oops_metrics = (
                        oops_harmonics_residuals(_wout_oops, args)
                    )
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] OOPS residual FAILED: {exc}")
                    self._oops_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._oops_metrics = {"ok": False, "error": repr(exc)}

            if self._initial_fmaxj is None and np.isfinite(self._fmaxj):
                self._initial_fmaxj = float(self._fmaxj)
            oops_scalar = self._oops_metrics.get("scalar_mean", np.nan)
            try:
                oops_scalar = float(oops_scalar)
            except Exception:
                oops_scalar = np.nan
            if self._initial_oops_scalar is None and np.isfinite(oops_scalar):
                self._initial_oops_scalar = oops_scalar

            if getattr(args, "w_maxj_guard", 0.0) > 0:
                maxj_tol = max(
                    float(getattr(args, "maxj_guard_abs_tol", 0.0) or 0.0),
                    float(getattr(args, "maxj_guard_rel_tol", 0.0) or 0.0)
                    * abs(float(self._initial_fmaxj or 0.0)),
                )
                maxj_scale = max(float(getattr(args, "maxj_guard_scale", 1e-4)), 1e-12)
                if self._initial_fmaxj is None or not np.isfinite(self._fmaxj):
                    self._maxj_guard_residual = np.array([np.sqrt(INVALID_OBJECTIVE)])
                else:
                    self._maxj_guard_residual = np.array([
                        max((float(self._fmaxj) - self._initial_fmaxj - maxj_tol) / maxj_scale, 0.0)
                    ])
            else:
                self._maxj_guard_residual = np.zeros(1)

            if getattr(args, "w_oops_guard", 0.0) > 0:
                oops_tol = max(
                    float(getattr(args, "oops_guard_abs_tol", 0.0) or 0.0),
                    float(getattr(args, "oops_guard_rel_tol", 0.0) or 0.0)
                    * abs(float(self._initial_oops_scalar or 0.0)),
                )
                oops_scale = max(float(getattr(args, "oops_guard_scale", 1e-3)), 1e-12)
                if self._initial_oops_scalar is None or not np.isfinite(oops_scalar):
                    self._oops_guard_residual = np.array([np.sqrt(INVALID_OBJECTIVE)])
                else:
                    self._oops_guard_residual = np.array([
                        max((oops_scalar - self._initial_oops_scalar - oops_tol) / oops_scale, 0.0)
                    ])
            else:
                self._oops_guard_residual = np.zeros(1)

            # ── Boundary curvature/twist proxy for coil non-planarity margin ──
            if getattr(args, "w_boundary_curvature", 0.0) > 0:
                try:
                    from simsopt.geo import SurfaceRZFourier
                    from squid.diagnostics.boundary_geometry import boundary_geometry_metrics
                    from squid.objectives.pdrot_residual import (
                        pdrot_area_weighted_stats,
                        principal_direction_rotation_rate,
                    )

                    _wout_bg = getattr(vmec, "output_file", None)
                    if _wout_bg is None or not os.path.exists(_wout_bg):
                        _wout_bg = os.path.join(
                            run_dir,
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    ntheta_bg = int(getattr(args, "boundary_curvature_ntheta", 48))
                    nphi_bg = int(getattr(args, "boundary_curvature_nphi", 48))
                    bg = boundary_geometry_metrics(
                        _wout_bg,
                        ntheta=ntheta_bg,
                        nphi=nphi_bg,
                        torus_range="full torus",
                    )
                    pdrot_surface = SurfaceRZFourier.from_wout(
                        _wout_bg,
                        range="full torus",
                        ntheta=ntheta_bg,
                        nphi=nphi_bg,
                    )
                    pdrot_diag = principal_direction_rotation_rate(
                        pdrot_surface,
                        delta_kappa_a=float(getattr(args, "pdrot_delta_kappa_a", 0.001)),
                    )
                    pdrot_stats = pdrot_area_weighted_stats(pdrot_diag)
                    pdrot_aw = float(pdrot_stats.get("pdrot_mean", np.nan))
                    pdrot_max = float(pdrot_stats.get("pdrot_max", np.nan))
                    pdrot_p99 = float(pdrot_stats.get("pdrot_p99", np.nan))
                    pdrot_cvar1 = float(pdrot_stats.get("pdrot_cvar1", np.nan))
                    k2_min = float(bg["k2_1_per_m"].get("min", np.nan))
                    H_min = float(bg["H_1_per_m"].get("min", np.nan))
                    r_pdrot_aw = max(
                        (pdrot_aw - float(getattr(args, "boundary_pdrot_aw_target", 1.45)))
                        / max(float(getattr(args, "boundary_pdrot_aw_scale", 0.05)), 1e-12),
                        0.0,
                    )
                    r_pdrot_max = max(
                        (pdrot_max - float(getattr(args, "boundary_pdrot_max_target", 16.0)))
                        / max(float(getattr(args, "boundary_pdrot_max_scale", 1.0)), 1e-12),
                        0.0,
                    )
                    r_pdrot_p99 = max(
                        (pdrot_p99 - float(getattr(args, "boundary_pdrot_p99_target", 1e99)))
                        / max(float(getattr(args, "boundary_pdrot_p99_scale", 1.0)), 1e-12),
                        0.0,
                    )
                    r_pdrot_cvar1 = max(
                        (pdrot_cvar1 - float(getattr(args, "boundary_pdrot_cvar1_target", 1e99)))
                        / max(float(getattr(args, "boundary_pdrot_cvar1_scale", 1.0)), 1e-12),
                        0.0,
                    )
                    r_k2 = max(
                        ((-k2_min) - float(getattr(args, "boundary_k2_abs_max_target", 100.0)))
                        / max(float(getattr(args, "boundary_k2_abs_scale", 5.0)), 1e-12),
                        0.0,
                    )
                    r_H = max(
                        ((-H_min) - float(getattr(args, "boundary_H_abs_max_target", 50.0)))
                        / max(float(getattr(args, "boundary_H_abs_scale", 3.0)), 1e-12),
                        0.0,
                    )
                    self._boundary_curvature_metrics = {
                        "pdrot_aw": pdrot_aw,
                        "pdrot_max": pdrot_max,
                        "pdrot_p99": pdrot_p99,
                        "pdrot_cvar1": pdrot_cvar1,
                        "k2_min": k2_min,
                        "H_min": H_min,
                    }
                    self._boundary_curvature_residuals = np.array(
                        [r_pdrot_aw, r_pdrot_max, r_pdrot_p99, r_pdrot_cvar1, r_k2, r_H],
                        dtype=float,
                    )
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Boundary curvature FAILED: {exc}")
                    self._boundary_curvature_residuals = np.full(6, np.sqrt(INVALID_OBJECTIVE))
                    self._boundary_curvature_metrics = {}

            if (
                getattr(args, "w_rational", 0.0) > 0
                or getattr(args, "hard_rational_crossing", False)
            ):
                try:
                    self._rational_residuals, self._rational_metrics = (
                        _rational_placement_residuals(
                            vmec, args, skip_hard=(self._n == 1))
                    )
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Rational placement FAILED: {exc}")
                    self._rational_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._rational_metrics = {}

            # ── pdrot objective (principal-direction rotation rate) ──
            if getattr(args, "w_pdrot", 0.0) > 0:
                try:
                    self._pdrot_residuals, self._pdrot_metrics = (
                        compute_pdrot_from_vmec(vmec, args)
                    )
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] pdrot FAILED: {exc}")
                    self._pdrot_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._pdrot_metrics = {}

            self._cache_x = cx
            dt = time.time() - t0
            try:
                aspect = float(vmec.aspect())
            except Exception:
                aspect = np.nan
            _ballooning_lam_finite = self._ballooning_lam_raw[
                np.isfinite(self._ballooning_lam_raw)
            ]
            _balloon_n_pos = int(np.sum(_ballooning_lam_finite > 0))
            _balloon_lam_max = (
                float(np.max(_ballooning_lam_finite))
                if _ballooning_lam_finite.size
                else np.nan
            )
            _append_history(dict(
                eval=self._n,
                objective_terms_ready=1,
                f_QI=self._fqi,
                f_maxJ=self._fmaxj,
                f_Bmin=self._fbmin,
                mirror_ratio=self._mirror,
                iota_axis=self._iota_ax,
                iota_edge=self._iota_ed,
                beta=self._beta,
                f_grad_s=self._fgs,
                well_depth=self._well_depth,
                f_mercier=self._fmercier,
                dmerc_convention=MERCIER_CONVENTION_VERSION,
                dmerc_edge_toroidal_flux_wb=self._dmerc_edge_toroidal_flux_wb,
                dmerc_vmec_raw_min=self._dmerc_vmec_raw_min,
                dmerc_flux_normalized_min=self._dmerc_flux_normalized_min,
                dmerc_min_s=self._dmerc_min_s,
                dmerc_negative_count=self._dmerc_negative_count,
                dmerc_axis_vmec_raw_min=self._dmerc_axis_vmec_raw_min,
                dmerc_axis_flux_normalized_min=self._dmerc_axis_flux_normalized_min,
                dmerc_axis_min_s=self._dmerc_axis_min_s,
                dmerc_axis_negative_count=self._dmerc_axis_negative_count,
                dshear_vmec_raw_min=self._mhd_profile_metrics.get("dshear_vmec_raw_min", ""),
                dshear_flux_normalized_min=self._mhd_profile_metrics.get("dshear_flux_normalized_min", ""),
                dshear_min_s=self._mhd_profile_metrics.get("dshear_min_s", ""),
                dwell_vmec_raw_min=self._mhd_profile_metrics.get("dwell_vmec_raw_min", ""),
                dwell_flux_normalized_min=self._mhd_profile_metrics.get("dwell_flux_normalized_min", ""),
                dwell_min_s=self._mhd_profile_metrics.get("dwell_min_s", ""),
                dcurr_vmec_raw_min=self._mhd_profile_metrics.get("dcurr_vmec_raw_min", ""),
                dcurr_flux_normalized_min=self._mhd_profile_metrics.get("dcurr_flux_normalized_min", ""),
                dcurr_min_s=self._mhd_profile_metrics.get("dcurr_min_s", ""),
                dgeod_vmec_raw_min=self._mhd_profile_metrics.get("dgeod_vmec_raw_min", ""),
                dgeod_flux_normalized_min=self._mhd_profile_metrics.get("dgeod_flux_normalized_min", ""),
                dgeod_min_s=self._mhd_profile_metrics.get("dgeod_min_s", ""),
                iota_prime_absmin=self._mhd_profile_metrics.get("iota_prime_absmin", ""),
                iota_prime_absmin_s=self._mhd_profile_metrics.get("iota_prime_absmin_s", ""),
                pprime_min=self._mhd_profile_metrics.get("pprime_min", ""),
                pprime_min_s=self._mhd_profile_metrics.get("pprime_min_s", ""),
                iota_profile_rms_dev=self._iota_topology_metrics.get("profile_rms_dev", ""),
                iota_profile_max_dev=self._iota_topology_metrics.get("profile_max_dev", ""),
                iota_topology_shear_absmin=self._iota_topology_metrics.get("shear_absmin", ""),
                iota_topology_shear_absmin_s=self._iota_topology_metrics.get("shear_absmin_s", ""),
                iota_monotonic_violation_count=self._iota_topology_metrics.get("monotonic_violation_count", ""),
                iota_reference_direction=iota_reference_direction,
                iota_profile_residual_rms=float(np.sqrt(np.mean(self._iota_profile_residuals ** 2))),
                iota_shear_residual_rms=float(np.sqrt(np.mean(self._iota_shear_residuals ** 2))),
                iota_monotonic_residual_rms=float(np.sqrt(np.mean(self._iota_monotonic_residuals ** 2))),
                iota_topology_hard_violation=self._iota_topology_metrics.get("hard_violation", ""),
                balloon_n_pos=_balloon_n_pos,
                balloon_lam_max=_balloon_lam_max,
                coil_Bn_max=self._coil_proxy_metrics.get("Bn_max_abs_unitless", ""),
                coil_Bn_rms=self._coil_proxy_metrics.get("Bn_rms_unitless", ""),
                coil_K_rms=self._coil_proxy_metrics.get("K_rms", ""),
                coil_K_max=self._coil_proxy_metrics.get("K_max", ""),
                coil_phi_high=self._coil_proxy_metrics.get("phi_high_mode_fraction", ""),
                coil_r_bn=float(self._coil_proxy_residual_bn[0]),
                coil_r_k=float(self._coil_proxy_residual_k[0]),
                coil_r_phi=float(self._coil_proxy_residual_phi[0]),
                coil_hard_bn=float(self._coil_proxy_hard_bn_residual[0]),
                oops_scalar_mean=self._oops_metrics.get("scalar_mean", ""),
                oops_scalar_max=self._oops_metrics.get("scalar_max", ""),
                oops_residual_rms=self._oops_metrics.get("residual_rms", ""),
                local_maxj_rms=self._local_maxj_metrics.get("rms", 0.0),
                local_maxj_max=self._local_maxj_metrics.get("max", 0.0),
                local_maxj_count=self._local_maxj_metrics.get("count", 0),
                maxj_guard_residual=float(self._maxj_guard_residual[0]),
                oops_guard_residual=float(self._oops_guard_residual[0]),
                boundary_pdrot_aw=self._boundary_curvature_metrics.get("pdrot_aw", ""),
                boundary_pdrot_max=self._boundary_curvature_metrics.get("pdrot_max", ""),
                boundary_pdrot_p99=self._boundary_curvature_metrics.get("pdrot_p99", ""),
                boundary_pdrot_cvar1=self._boundary_curvature_metrics.get("pdrot_cvar1", ""),
                boundary_k2_min=self._boundary_curvature_metrics.get("k2_min", ""),
                boundary_H_min=self._boundary_curvature_metrics.get("H_min", ""),
                boundary_r_pdrot_aw=float(self._boundary_curvature_residuals[0]),
                boundary_r_pdrot_max=float(self._boundary_curvature_residuals[1]),
                boundary_r_pdrot_p99=float(self._boundary_curvature_residuals[2]),
                boundary_r_pdrot_cvar1=float(self._boundary_curvature_residuals[3]),
                boundary_r_k2=float(self._boundary_curvature_residuals[4]),
                boundary_r_H=float(self._boundary_curvature_residuals[5]),
                rational_nearest=self._rational_metrics.get("nearest_label", ""),
                rational_nearest_s=self._rational_metrics.get("nearest_s", ""),
                rational_nearest_distance=self._rational_metrics.get("nearest_distance", ""),
                rational_crossing_count=self._rational_metrics.get("crossing_count", ""),
                rational_hard_crossing_count=self._rational_metrics.get("hard_crossing_count", ""),
                pdrot_mean=self._pdrot_metrics.get("pdrot_mean", ""),
                pdrot_rms=self._pdrot_metrics.get("pdrot_rms", ""),
                pdrot_median=self._pdrot_metrics.get("pdrot_median", ""),
                pdrot_max=self._pdrot_metrics.get("pdrot_max", ""),
                pdrot_p95=self._pdrot_metrics.get("pdrot_p95", ""),
                pdrot_p99=self._pdrot_metrics.get("pdrot_p99", ""),
                pdrot_p999=self._pdrot_metrics.get("pdrot_p999", ""),
                pdrot_cvar1=self._pdrot_metrics.get("pdrot_cvar1", ""),
                pdrot_q_mean=self._pdrot_metrics.get("pdrot_q_mean", ""),
                pdrot_q_p95=self._pdrot_metrics.get("pdrot_q_p95", ""),
                pdrot_a_eff=self._pdrot_metrics.get("pdrot_a_eff", ""),
                pdrot_kappa_gap_max=self._pdrot_metrics.get("pdrot_kappa_gap_max", ""),
                aspect=aspect,
                elapsed_s=dt,
            ))
            cp_every = int(getattr(args, "checkpoint_every", 0) or 0)
            html_every = int(getattr(args, "html_every", 0) or 0)
            if cp_every > 0 and self._n % cp_every == 0:
                _write_checkpoint(self._n, make_html=False)
            if html_every > 0 and self._n % html_every == 0:
                _write_checkpoint(self._n, make_html=True)
            parts = [f"f_QI={self._fqi:.3e}",
                     f"f_maxJ={self._fmaxj:.3e}",
                     f"f_Bmin={self._fbmin:.3e}",
                     f"delta={self._mirror:.4f}",
                     f"iota=[{self._iota_ax:.3f},{self._iota_ed:.3f}]"]
            if (getattr(args, "w_iota_profile", 0.0) > 0
                    or getattr(args, "w_iota_shear", 0.0) > 0
                    or getattr(args, "hard_gate_iota_topology", False)):
                parts.append(f"shear_min={self._iota_topology_metrics.get('shear_absmin', np.nan):.3e}")
            if getattr(args, 'w_beta', 0.0) > 0:
                parts.append(f"beta={self._beta:.4f}")
            if getattr(args, 'w_grad_s', 0.0) > 0:
                parts.append(f"f_nabla_s={self._fgs:.3e}")
            if getattr(args, 'w_well', 0.0) > 0:
                well_pct = self._well_depth * 100
                well_tag = "well" if self._well_depth > 0 else "HILL"
                parts.append(f"well={well_pct:+.2f}%({well_tag})")
            if getattr(args, 'w_mercier', 0.0) > 0:
                parts.append(
                    f"PhiEdge^2*DMerc_min={self._dmerc_flux_normalized_min:.3e}"
                )
                parts.append(f"raw={self._dmerc_vmec_raw_min:.3e}")
                parts.append(f"DMerc_neg={self._dmerc_negative_count}")
            if getattr(args, 'w_ballooning', 0.0) > 0:
                bpos = int(np.sum(self._ballooning_lam_raw > 0))
                bmax = float(np.nanmax(self._ballooning_lam_raw))
                parts.append(f"balloon_n+={bpos} λmax={bmax:.2e}")
            if (
                getattr(args, 'w_coil_proxy_bn', 0.0) > 0
                or getattr(args, 'w_coil_proxy_k', 0.0) > 0
                or getattr(args, 'w_coil_proxy_phi', 0.0) > 0
            ):
                bn = self._coil_proxy_metrics.get("Bn_max_abs_unitless", np.nan)
                kr = self._coil_proxy_metrics.get("K_rms", np.nan) / 1e6
                ph = self._coil_proxy_metrics.get("phi_high_mode_fraction", np.nan)
                parts.append(f"coilBn={bn:.2e} K_rms={kr:.2f}MA/m phi_hi={ph:.2f}")
            if getattr(args, "w_boundary_curvature", 0.0) > 0:
                bg = self._boundary_curvature_metrics
                parts.append(
                    f"pdrot={bg.get('pdrot_aw', np.nan):.2f}/{bg.get('pdrot_max', np.nan):.1f} "
                    f"k2min={bg.get('k2_min', np.nan):.1f}")
            if (
                getattr(args, "w_rational", 0.0) > 0
                or getattr(args, "hard_rational_crossing", False)
            ):
                rm = self._rational_metrics
                parts.append(
                    f"rat={rm.get('nearest_label', '')}@s={rm.get('nearest_s', np.nan):.3f} "
                    f"d={rm.get('nearest_distance', np.nan):.2e} "
                    f"cross={rm.get('crossing_count', 0)}")
            if getattr(args, "w_pdrot", 0.0) > 0:
                pm = self._pdrot_metrics
                parts.append(
                    f"pdr_m={pm.get('pdrot_mean', np.nan):.2f} "
                    f"max={pm.get('pdrot_max', np.nan):.1f} "
                    f"p95={pm.get('pdrot_p95', np.nan):.1f}")
            if getattr(args, "w_oops", 0.0) > 0:
                parts.append(
                    f"oops={self._oops_metrics.get('scalar_mean', np.nan):.3e}")
            parts.append(f"({dt:.1f}s)")
            print(f"    [SQuID #{self._n}]  {'  '.join(parts)}")

        def maxJ_residuals(self):
            self._compute()
            return self._maxj_residuals

        def local_maxj_penalty(self):
            self._compute()
            return self._local_maxj_residuals

        def qi_residuals(self):
            self._compute()
            return self._qi_residuals

        def bmin_residuals(self):
            self._compute()
            return self._bmin_residuals

        def mirror_penalty(self):
            self._compute()
            return np.array([np.sqrt(_compute_mirror_penalty(
                self._mirror, args.mirror_target))])

        def beta_penalty(self):
            self._compute()
            return np.array([np.sqrt(_compute_beta_penalty(
                self._beta, args.beta_target))])

        def aspect_penalty(self):
            return np.array([np.sqrt(hinge_loss(vmec.aspect(), args.aspect_target))])

        def iota_penalty(self):
            self._compute()
            return np.array([np.sqrt(_compute_iota_penalty(
                self._iota_ax, self._iota_ed,
                args.iota_ax, args.iota_edge,
                args.iota_tolerance,
                getattr(args, "iota_edge_mode", "target")))])

        def iota_profile_penalty(self):
            self._compute()
            return self._iota_profile_residuals

        def iota_shear_penalty(self):
            self._compute()
            return self._iota_shear_residuals

        def iota_monotonic_penalty(self):
            self._compute()
            return self._iota_monotonic_residuals

        def hard_iota_topology_penalty(self):
            self._compute()
            return self._hard_iota_topology_residuals

        def grad_s_penalty(self):
            self._compute()
            return np.array([np.sqrt(max(self._fgs, 0.0))])

        def well_penalty(self):
            self._compute()
            return np.array([np.sqrt(_compute_well_penalty(
                vmec, args.target_well))])

        def mercier_penalty(self):
            self._compute()
            return self._mercier_residuals

        def mercier_margin_penalty(self):
            self._compute()
            return self._mercier_margin_residuals

        def hard_mhd_penalty(self):
            self._compute()
            return self._hard_mhd_residuals

        def ballooning_penalty(self):
            self._compute()
            return self._ballooning_residuals

        def force_balance_penalty(self):
            self._compute()
            return self._fb_residuals

        def coil_proxy_bn_penalty(self):
            self._compute()
            return self._coil_proxy_residual_bn

        def coil_proxy_k_penalty(self):
            self._compute()
            return self._coil_proxy_residual_k

        def coil_proxy_phi_penalty(self):
            self._compute()
            return self._coil_proxy_residual_phi

        def coil_proxy_hard_bn_penalty(self):
            self._compute()
            return self._coil_proxy_hard_bn_residual

        def oops_penalty(self):
            self._compute()
            return self._oops_residuals

        def maxj_guard_penalty(self):
            self._compute()
            return self._maxj_guard_residual

        def oops_guard_penalty(self):
            self._compute()
            return self._oops_guard_residual

        def boundary_curvature_penalty(self):
            self._compute()
            return self._boundary_curvature_residuals

        def rational_penalty(self):
            self._compute()
            return self._rational_residuals

        def pdrot_penalty(self):
            self._compute()
            return self._pdrot_residuals

        def reg_penalty(self):
            x_cur = np.array(vmec.x, dtype=float)
            delta = x_cur - x0_vmec
            return delta / np.sqrt(max(delta.size, 1))

        def shape_anchor_penalty(self):
            x_cur = np.array(vmec.x, dtype=float)
            delta = np.abs(x_cur - x0_vmec)
            allowed = np.maximum(
                float(getattr(args, "shape_anchor_abs", 1e-3)),
                float(getattr(args, "shape_anchor_frac", 0.015)) * np.maximum(np.abs(x0_vmec), 1e-3),
            )
            residuals = np.maximum(delta - allowed, 0.0)
            return residuals / np.sqrt(max(residuals.size, 1))

    squid = SQuIDObjective()
    qi_r2 = None
    if args.w_qi_r2 > 0:
        qi_r2 = QIResidual(
            vmec, s_vals,
            nphi=args.qi_r2_nphi,
            nalpha=args.qi_r2_nalpha,
            nBj=args.qi_r2_nbj,
            mpol=args.qi_r2_mpol,
            ntor=args.qi_r2_ntor,
            arr_out=args.qi_r2_arr_out,
        )

    # Core objectives (always active)
    tuples = [
        (squid.qi_residuals, 0.0, args.w_qi),
        (squid.maxJ_residuals, 0.0, args.w_maxj),
        (squid.bmin_residuals, 0.0, args.w_bmin),
        (squid.aspect_penalty, 0.0, args.w_ar),
        (squid.reg_penalty, 0.0, args.w_reg),
    ]
    active = [f"w_QI={args.w_qi}", f"w_maxJ={args.w_maxj}",
              f"w_Bmin={args.w_bmin}", f"w_AR={args.w_ar}",
              f"w_reg={args.w_reg}"]
    if qi_r2 is not None:
        tuples.append((qi_r2.residuals, 0.0, args.w_qi_r2))
        active.append(f"w_QI_R2={args.w_qi_r2}")

    # Optional objectives (only added when weight > 0)
    optional = [
        (args.w_mirror, squid.mirror_penalty, "w_mirror"),
        (getattr(args, 'w_beta', 0.0), squid.beta_penalty, "w_beta"),
        (args.w_iota, squid.iota_penalty, "w_iota"),
        (getattr(args, "w_iota_profile", 0.0), squid.iota_profile_penalty, "w_iota_profile"),
        (getattr(args, "w_iota_shear", 0.0), squid.iota_shear_penalty, "w_iota_shear"),
        (getattr(args, "w_iota_monotonic", 0.0), squid.iota_monotonic_penalty, "w_iota_monotonic"),
        (1.0 if getattr(args, "hard_gate_iota_topology", False) else 0.0,
         squid.hard_iota_topology_penalty, "hard_iota_topology"),
        (getattr(args, 'w_grad_s', 0.0), squid.grad_s_penalty, "w_grad_s"),
        (getattr(args, 'w_well', 0.0), squid.well_penalty, "w_well"),
        (getattr(args, 'w_mercier', 0.0), squid.mercier_penalty, "w_mercier"),
        (getattr(args, 'w_mercier_margin', 0.0), squid.mercier_margin_penalty,
         "w_mercier_margin"),
        (1.0 if getattr(args, 'hard_gate_mhd', False) else 0.0,
         squid.hard_mhd_penalty, "hard_mhd_gate"),
        (getattr(args, 'w_ballooning', 0.0), squid.ballooning_penalty, "w_ballooning"),
        (getattr(args, 'w_force_balance', 0.0), squid.force_balance_penalty, "w_force_balance"),
        (getattr(args, 'w_coil_proxy_bn', 0.0), squid.coil_proxy_bn_penalty, "w_coil_proxy_bn"),
        (getattr(args, 'w_coil_proxy_k', 0.0), squid.coil_proxy_k_penalty, "w_coil_proxy_k"),
        (getattr(args, 'w_coil_proxy_phi', 0.0), squid.coil_proxy_phi_penalty, "w_coil_proxy_phi"),
        (
            1.0 if np.isfinite(float(getattr(args, "hard_coil_proxy_bn_max", np.inf))) else 0.0,
            squid.coil_proxy_hard_bn_penalty,
            "hard_coil_proxy_bn",
        ),
        (getattr(args, 'w_oops', 0.0), squid.oops_penalty, "w_oops"),
        (getattr(args, 'w_local_maxj', 0.0), squid.local_maxj_penalty, "w_local_maxj"),
        (getattr(args, 'w_maxj_guard', 0.0), squid.maxj_guard_penalty, "w_maxj_guard"),
        (getattr(args, 'w_oops_guard', 0.0), squid.oops_guard_penalty, "w_oops_guard"),
        (getattr(args, 'w_shape_anchor', 0.0), squid.shape_anchor_penalty, "w_shape_anchor"),
        (getattr(args, 'w_boundary_curvature', 0.0), squid.boundary_curvature_penalty, "w_boundary_curvature"),
        (getattr(args, 'w_pdrot', 0.0), squid.pdrot_penalty, "w_pdrot"),
        (
            max(
                getattr(args, 'w_rational', 0.0),
                1.0 if getattr(args, "hard_rational_crossing", False) else 0.0,
            ),
            squid.rational_penalty,
            "w_rational",
        ),
    ]
    for w, func, name in optional:
        if w > 0:
            tuples.append((func, 0.0, w))
            active.append(f"{name}={w}")

    prob = LeastSquaresProblem.from_tuples(tuples)

    print(f"\n  Active objectives ({len(tuples)}):")
    print(f"    {', '.join(active)}")

    print("\n  Evaluating initial state ...")
    with _run_in_directory(run_dir):
        obj0 = prob.objective()
    print(f"  Initial objective = {obj0:.4e}")

    # Random perturbation to escape local minima
    perturb = getattr(args, 'perturb', 0.0)
    if perturb > 0:
        x_cur = np.array(vmec.x, dtype=float)
        rng = np.random.default_rng()
        scale = perturb * np.maximum(np.abs(x_cur), 1e-3)
        vmec.x = x_cur + rng.normal(0, scale)
        x0_vmec[:] = np.array(vmec.x, dtype=float)
        print(f"\n  Applied random perturbation (amplitude={perturb})")

    max_nfev = args.maxiter * (n_dofs + 1)
    max_total_evals = int(getattr(args, "max_total_evals", 0) or 0)
    cap_reached = False
    solve_start = np.array(vmec.x, dtype=float)
    abs_step = getattr(args, 'abs_step', 1e-4)
    rel_step = getattr(args, 'rel_step', 0.0)
    if max_nfev > 0:
        cap_msg = (f", max_total_evals={max_total_evals}"
                   if max_total_evals > 0 else "")
        print(f"\n  Starting optimisation (max_nfev={max_nfev}{cap_msg}, "
              f"abs_step={abs_step:.0e}) ...")
        t_start = time.time()
        solve_kwargs = {}
        dof_bound_frac = float(getattr(args, "dof_bound_frac", 0.0) or 0.0)
        if dof_bound_frac > 0:
            x_start = np.array(vmec.x, dtype=float)
            bound_abs = float(getattr(args, "shape_anchor_abs", 1e-3))
            half_width = np.maximum(
                bound_abs,
                dof_bound_frac * np.maximum(np.abs(x_start), 1e-3),
            )
            solve_kwargs["bounds"] = (x_start - half_width, x_start + half_width)
            print(
                f"  DoF bounds active: frac={dof_bound_frac:.3g}, "
                f"abs_floor={bound_abs:.1e}"
            )
        try:
            with _run_in_directory(run_dir):
                least_squares_serial_solve(
                    prob, max_nfev=max_nfev, grad=True,
                    abs_step=abs_step, rel_step=rel_step,
                    **solve_kwargs,
                )
        except EvaluationLimitReached as exc:
            cap_reached = True
            vmec.x = solve_start
            squid._cache_x = None
            print(f"\n  Stopped by evaluation cap: {exc}")
            print("  Restored the pre-solve state; the last trial is not accepted.")
        t_total = time.time() - t_start
        print(f"\n  Finished in {t_total / 60:.1f} min  ({squid._n} evals)")
    else:
        print("\n  maxiter <= 0: initial evaluation only; optimisation skipped.")

    try:
        if cap_reached:
            raise EvaluationLimitReached("evaluation cap already reached")
        with _run_in_directory(run_dir):
            obj_f = prob.objective()
    except EvaluationLimitReached:
        obj_f = np.nan
        print("  Final objective skipped after evaluation-cap recovery.")
    print(f"\n  Initial objective = {obj0:.4e}")
    if np.isfinite(obj_f):
        print(f"  Final objective   = {obj_f:.4e}")
    else:
        print("  Final objective   = unavailable after evaluation-cap stop")

    output_name = (
        "input.squid_unaccepted_cap" if cap_reached else "input.squid_optimized"
    )
    out_path = os.path.join(run_dir, output_name)
    try:
        vmec.write_input(out_path)
        print(f"\n  Saved: {out_path}")
    except Exception as e:
        print(f"\n  Could not save: {e}")

    status = {
        "accepted": not cap_reached,
        "state": (
            "evaluation_cap_recovered_start"
            if cap_reached else ("initial_only" if max_nfev <= 0 else "completed")
        ),
        "input_file": output_name,
        "initial_objective": float(obj0),
        "final_objective": float(obj_f) if np.isfinite(obj_f) else None,
        "evaluations": int(squid._n),
    }
    with open(os.path.join(run_dir, "run_status.json"), "w") as handle:
        json.dump(status, handle, indent=2)

    for dat in glob.glob(os.path.join(run_dir, "simsopt_*.dat")):
        try:
            os.remove(dat)
        except OSError:
            pass
    return status
