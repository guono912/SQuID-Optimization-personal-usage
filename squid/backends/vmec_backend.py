"""
VMEC backend for SQuID optimisation.

Uses simsopt + VMEC2000 Fortran extension to run the equilibrium solver
and the LeastSquaresProblem interface.
"""

import os

import time
import glob
import csv
import shutil
import subprocess
import sys
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
from ..objectives.pdrot_residual import compute_pdrot_from_vmec
from ..objectives.penalties import bmin_slope_residuals, hinge_loss


INVALID_OBJECTIVE = 1.0e6


class EvaluationLimitReached(RuntimeError):
    pass


def wout_to_input(wout_path, input_path, ns=31,
                  prescribed_iota_ax=None, prescribed_iota_edge=None,
                  free_iota=True):
    """Convert wout_*.nc to VMEC input file.

    Parameters
    ----------
    free_iota : bool
        If True, use NCURR=1 with zero current so that the rotational
        transform is computed self-consistently from the boundary shape.
        The iota penalty in the optimizer then actively steers iota.
        If False, use NCURR=0 with prescribed AI coefficients.
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
    ds.close()

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
        dmerc = np.array(vmec_ro.wout.DMerc, dtype=float)
        ns = len(dmerc)
        s = np.linspace(0.0, 1.0, ns)
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

    run_dir = os.path.abspath(getattr(args, "run_dir", "") or os.getcwd())
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
                      free_iota=free_iota)

    vmec = Vmec(input_path)
    vmec.run()

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
        "well_depth", "f_mercier", "dmerc_min", "dmerc_negative_count",
        "balloon_n_pos", "balloon_lam_max",
        "coil_Bn_max", "coil_Bn_rms", "coil_K_rms", "coil_K_max",
        "coil_phi_high", "coil_r_bn", "coil_r_k", "coil_r_phi",
        "boundary_pdrot_aw", "boundary_pdrot_max", "boundary_pdrot_p99",
        "boundary_pdrot_cvar1", "boundary_k2_min", "boundary_H_min",
        "boundary_r_pdrot_aw", "boundary_r_pdrot_max", "boundary_r_pdrot_p99",
        "boundary_r_pdrot_cvar1", "boundary_r_k2", "boundary_r_H",
        "rational_nearest", "rational_nearest_s", "rational_nearest_distance",
        "rational_crossing_count", "rational_hard_crossing_count",
        "pdrot_mean", "pdrot_max", "pdrot_p95", "pdrot_p99",
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
            vmec.run()
            wout_name = getattr(vmec, "output_file", None)
            if wout_name is None:
                wout_name = f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc"
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

    class SQuIDObjective(Optimizable):
        def __init__(self):
            self._cache_x = None
            self._fmaxj = 0.0
            self._fqi = 0.0
            self._fbmin = 0.0
            self._qi_residuals = np.zeros(1)
            self._maxj_residuals = np.zeros(1)
            self._bmin_residuals = np.zeros(1)
            self._fgs = 0.0
            self._mirror = 0.0
            self._beta = 0.0
            self._iota_ax = 0.0
            self._iota_ed = 0.0
            self._well_depth = 0.0
            self._mercier_residuals = np.zeros(1)
            self._mercier_margin_residuals = np.zeros(1)
            self._ballooning_residuals = np.zeros(80)  # 10 rhos × 8 alphas
            self._coil_proxy_residual_bn = np.zeros(1)
            self._coil_proxy_residual_k = np.zeros(1)
            self._coil_proxy_residual_phi = np.zeros(1)
            self._coil_proxy_metrics = {}
            self._boundary_curvature_residuals = np.zeros(6)
            self._boundary_curvature_metrics = {}
            self._rational_residuals = np.zeros(1)
            self._rational_metrics = {}
            self._pdrot_residuals = np.zeros(1)
            self._pdrot_metrics = {}
            self._fmercier = 0.0
            self._dmerc_min = np.nan
            self._dmerc_negative_count = 0
            self._n = 0
            super().__init__(depends_on=[vmec])

        @staticmethod
        def _large_like(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return np.array([np.sqrt(INVALID_OBJECTIVE)])
            return np.full_like(arr, np.sqrt(INVALID_OBJECTIVE), dtype=float)

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
                    dmerc = np.array(vmec.wout.DMerc, dtype=float)
                    ns = len(dmerc)
                    s = np.linspace(0.0, 1.0, ns)
                    mask = ((s >= args.mercier_s_min)
                            & (s <= args.mercier_s_max)
                            & np.isfinite(dmerc))
                    vals = dmerc[mask]
                    if vals.size == 0:
                        self._mercier_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                        self._fmercier = INVALID_OBJECTIVE
                        self._dmerc_min = np.nan
                        self._dmerc_negative_count = 0
                    else:
                        self._mercier_residuals = np.maximum(-vals, 0.0)
                        self._fmercier = float(np.mean(self._mercier_residuals ** 2))
                        self._dmerc_min = float(np.nanmin(vals))
                        self._dmerc_negative_count = int(np.sum(vals < 0))
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Mercier target FAILED: {exc}")
                    self._mercier_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._fmercier = INVALID_OBJECTIVE
                    self._dmerc_min = np.nan
                    self._dmerc_negative_count = 0

            if getattr(args, 'w_mercier_margin', 0.0) > 0:
                try:
                    dmerc = np.array(vmec.wout.DMerc, dtype=float)
                    ns = len(dmerc)
                    s = np.linspace(0.0, 1.0, ns)
                    mask = ((s >= args.mercier_s_min)
                            & (s <= args.mercier_s_max)
                            & np.isfinite(dmerc))
                    vals = dmerc[mask]
                    if vals.size == 0:
                        self._mercier_margin_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                        if getattr(args, 'w_mercier', 0.0) <= 0:
                            self._dmerc_min = np.nan
                            self._dmerc_negative_count = 0
                    else:
                        target = float(getattr(args, 'mercier_margin_target', 0.0))
                        self._mercier_margin_residuals = np.maximum(target - vals, 0.0)
                        if getattr(args, 'w_mercier', 0.0) <= 0:
                            self._dmerc_min = float(np.nanmin(vals))
                            self._dmerc_negative_count = int(np.sum(vals < 0))
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Mercier margin target FAILED: {exc}")
                    self._mercier_margin_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    if getattr(args, 'w_mercier', 0.0) <= 0:
                        self._dmerc_min = np.nan
                        self._dmerc_negative_count = 0

            # ── Ballooning penalty (DESC, no force balance) ──
            # Match the post-run gate/viz convention: use the shifted
            # BallooningStability objective and convert back to raw lambda.
            b_rhos = np.array(getattr(args, 'ballooning_rhos', [0.5, 0.65, 0.8, 0.9]))
            b_alphas = np.linspace(0, np.pi, 6, endpoint=False)
            b_zeta0 = np.linspace(-0.5*np.pi, 0.5*np.pi, 5)
            b_n_expected = len(b_rhos) * len(b_alphas)  # shape after zeta0 max
            self._ballooning_residuals = np.full(b_n_expected, np.nan)
            if getattr(args, 'w_ballooning', 0.0) > 0:
                try:
                    from desc.vmec import VMECIO
                    from desc.objectives import BallooningStability
                    # Use the VMEC object's own wout — reliable, no glob needed.
                    wout_temp = getattr(vmec, "output_file", None)
                    if wout_temp is None or not os.path.exists(wout_temp):
                        wout_temp = os.path.join(
                            os.getcwd(),
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    eq_b = VMECIO.load(wout_temp, L=6, M=6, N=6)
                    lam_residuals = []
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
                        lam_residuals.extend(
                            np.maximum(lam_raw - lam_target, 0.0).tolist())
                    self._ballooning_residuals = np.array(lam_residuals, dtype=float)
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Ballooning FAILED: {exc}")
                    self._ballooning_residuals = np.full(len(b_rhos) * len(b_alphas), np.sqrt(INVALID_OBJECTIVE))

            # ── DESC force-balance residual (no solve, just FB eval on fit) ──
            self._fb_residuals = np.zeros(1)
            if getattr(args, 'w_force_balance', 0.0) > 0:
                try:
                    from desc.vmec import VMECIO as _VMECIO_fb
                    from desc.objectives import ForceBalance as _ForceBalance
                    _wout_fb = getattr(vmec, "output_file", None)
                    if _wout_fb is None or not os.path.exists(_wout_fb):
                        _wout_fb = os.path.join(
                            os.getcwd(),
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    _eq_fb = _VMECIO_fb.load(_wout_fb, L=6, M=6, N=6)
                    _fb_obj = _ForceBalance(eq=_eq_fb)
                    _fb_obj.build(use_jit=False, verbose=0)
                    _fb_vals = np.asarray(_fb_obj.compute(_eq_fb.params_dict), dtype=float)
                    _fb_rms = float(np.sqrt(np.mean(_fb_vals**2)))
                    # Normalize: residual = sqrt(FB_RMS / FB_RMS0 - 1), ~0 at start
                    if not hasattr(self, '_fb_rms0'):
                        self._fb_rms0 = max(_fb_rms, 1.0)
                    self._fb_residuals = np.array([np.sqrt(max(_fb_rms / self._fb_rms0 - 1.0, 0.0))])
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Force-balance FAILED: {exc}")
                    self._fb_residuals = np.array([np.sqrt(INVALID_OBJECTIVE)])

            # ── Quasi-single-stage coil-realizability proxy ──
            if (
                getattr(args, 'w_coil_proxy_bn', 0.0) > 0
                or getattr(args, 'w_coil_proxy_k', 0.0) > 0
                or getattr(args, 'w_coil_proxy_phi', 0.0) > 0
            ):
                try:
                    _wout_cp = getattr(vmec, "output_file", None)
                    if _wout_cp is None or not os.path.exists(_wout_cp):
                        _wout_cp = os.path.join(
                            os.getcwd(),
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
                except Exception as exc:
                    print(f"    [SQuID #{self._n}] Coil proxy FAILED: {exc}")
                    bad = np.array([np.sqrt(INVALID_OBJECTIVE)])
                    self._coil_proxy_residual_bn = bad
                    self._coil_proxy_residual_k = bad
                    self._coil_proxy_residual_phi = bad
                    self._coil_proxy_metrics = {}

            # ── Boundary curvature/twist proxy for coil non-planarity margin ──
            if getattr(args, "w_boundary_curvature", 0.0) > 0:
                try:
                    from squid.diagnostics.boundary_geometry import boundary_geometry_metrics

                    _wout_bg = getattr(vmec, "output_file", None)
                    if _wout_bg is None or not os.path.exists(_wout_bg):
                        _wout_bg = os.path.join(
                            os.getcwd(),
                            f"wout_{os.path.basename(vmec.input_file).replace('input.', '')}.nc")
                    bg = boundary_geometry_metrics(
                        _wout_bg,
                        ntheta=int(getattr(args, "boundary_curvature_ntheta", 48)),
                        nphi=int(getattr(args, "boundary_curvature_nphi", 48)),
                        torus_range="full torus",
                    )
                    pdrot_aw = float(bg["pdrot_1_per_m"].get("mean_area_weighted", np.nan))
                    pdrot_max = float(bg["pdrot_1_per_m"].get("max", np.nan))
                    pdrot_p99 = float(bg["pdrot_1_per_m"].get("p99", np.nan))
                    pdrot_cvar1 = float(bg["pdrot_1_per_m"].get("cvar_top1", np.nan))
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
                dmerc_min=self._dmerc_min,
                dmerc_negative_count=self._dmerc_negative_count,
                balloon_n_pos=int(np.sum(self._ballooning_residuals > 0)),
                balloon_lam_max=float(np.max(self._ballooning_residuals)),
                coil_Bn_max=self._coil_proxy_metrics.get("Bn_max_abs_unitless", ""),
                coil_Bn_rms=self._coil_proxy_metrics.get("Bn_rms_unitless", ""),
                coil_K_rms=self._coil_proxy_metrics.get("K_rms", ""),
                coil_K_max=self._coil_proxy_metrics.get("K_max", ""),
                coil_phi_high=self._coil_proxy_metrics.get("phi_high_mode_fraction", ""),
                coil_r_bn=float(self._coil_proxy_residual_bn[0]),
                coil_r_k=float(self._coil_proxy_residual_k[0]),
                coil_r_phi=float(self._coil_proxy_residual_phi[0]),
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
                pdrot_max=self._pdrot_metrics.get("pdrot_max", ""),
                pdrot_p95=self._pdrot_metrics.get("pdrot_p95", ""),
                pdrot_p99=self._pdrot_metrics.get("pdrot_p99", ""),
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
            if getattr(args, 'w_beta', 0.0) > 0:
                parts.append(f"beta={self._beta:.4f}")
            if getattr(args, 'w_grad_s', 0.0) > 0:
                parts.append(f"f_nabla_s={self._fgs:.3e}")
            if getattr(args, 'w_well', 0.0) > 0:
                well_pct = self._well_depth * 100
                well_tag = "well" if self._well_depth > 0 else "HILL"
                parts.append(f"well={well_pct:+.2f}%({well_tag})")
            if getattr(args, 'w_mercier', 0.0) > 0:
                parts.append(f"DMerc_min={self._dmerc_min:.3e}")
                parts.append(f"DMerc_neg={self._dmerc_negative_count}")
            if getattr(args, 'w_ballooning', 0.0) > 0:
                bpos = int(np.sum(self._ballooning_residuals > 0))
                bmax = float(np.max(self._ballooning_residuals))
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
            parts.append(f"({dt:.1f}s)")
            print(f"    [SQuID #{self._n}]  {'  '.join(parts)}")

        def maxJ_residuals(self):
            self._compute()
            return self._maxj_residuals

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
        (getattr(args, 'w_grad_s', 0.0), squid.grad_s_penalty, "w_grad_s"),
        (getattr(args, 'w_well', 0.0), squid.well_penalty, "w_well"),
        (getattr(args, 'w_mercier', 0.0), squid.mercier_penalty, "w_mercier"),
        (getattr(args, 'w_mercier_margin', 0.0), squid.mercier_margin_penalty,
         "w_mercier_margin"),
        (getattr(args, 'w_ballooning', 0.0), squid.ballooning_penalty, "w_ballooning"),
        (getattr(args, 'w_force_balance', 0.0), squid.force_balance_penalty, "w_force_balance"),
        (getattr(args, 'w_coil_proxy_bn', 0.0), squid.coil_proxy_bn_penalty, "w_coil_proxy_bn"),
        (getattr(args, 'w_coil_proxy_k', 0.0), squid.coil_proxy_k_penalty, "w_coil_proxy_k"),
        (getattr(args, 'w_coil_proxy_phi', 0.0), squid.coil_proxy_phi_penalty, "w_coil_proxy_phi"),
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
            least_squares_serial_solve(prob, max_nfev=max_nfev, grad=True,
                                       abs_step=abs_step, rel_step=rel_step,
                                       **solve_kwargs)
        except EvaluationLimitReached as exc:
            print(f"\n  Stopped by evaluation cap: {exc}")
        t_total = time.time() - t_start
        print(f"\n  Finished in {t_total / 60:.1f} min  ({squid._n} evals)")
    else:
        print("\n  maxiter <= 0: initial evaluation only; optimisation skipped.")

    try:
        obj_f = prob.objective()
    except EvaluationLimitReached:
        obj_f = np.nan
        print("  Final objective skipped: evaluation cap reached at a new trial point.")
    print(f"\n  Initial objective = {obj0:.4e}")
    if np.isfinite(obj_f):
        print(f"  Final objective   = {obj_f:.4e}")
    else:
        print("  Final objective   = unavailable after evaluation-cap stop")

    out_path = os.path.join(run_dir, "input.squid_optimized")
    try:
        vmec.write_input(out_path)
        print(f"\n  Saved: {out_path}")
    except Exception as e:
        print(f"\n  Could not save: {e}")

    for dat in glob.glob("simsopt_*.dat"):
        try:
            os.remove(dat)
        except OSError:
            pass
