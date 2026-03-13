"""
VMEC backend for SQuID optimisation.

Uses simsopt + VMEC2000 Fortran extension to run the equilibrium solver
and the LeastSquaresProblem interface.
"""

import os
import time
import glob
import numpy as np
import netCDF4

from simsopt.mhd import Vmec
from simsopt._core import Optimizable
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

from ..objectives.maxj_residual import _evaluate_squid
from ..objectives.itg_residual import ITGResidual
from ..objectives.penalties import hinge_loss


def wout_to_input(wout_path, input_path, ns=31,
                  prescribed_iota_ax=None, prescribed_iota_edge=None):
    """Convert wout_*.nc to VMEC input file.

    If prescribed_iota_ax / prescribed_iota_edge are given, the AI
    coefficients are set to a simple linear profile instead of fitting
    the iotaf from the wout file.
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
    if prescribed_iota_ax is not None and prescribed_iota_edge is not None:
        ai = np.array([prescribed_iota_ax,
                       prescribed_iota_edge - prescribed_iota_ax])
    else:
        ai = np.polynomial.polynomial.polyfit(s_full, iotaf, min(10, ns_w - 1))

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
        f.write("  NSTEP = 200\n  FTOL_ARRAY = 1.0E-12\n")
        f.write(f"  PHIEDGE = {phiedge:.15e}\n")
        f.write("  GAMMA = 0.0\n  LFREEB = F\n  NCURR = 0\n")
        f.write("  PIOTA_TYPE = 'power_series'\n")
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


def _compute_mirror_penalty(mirror_ratio, mirror_target, mirror_max=None):
    if mirror_max is not None:
        pen_lo = hinge_loss(mirror_target - mirror_ratio, 0.0)
        pen_hi = hinge_loss(mirror_ratio - mirror_max, 0.0)
        return pen_lo + pen_hi
    return (mirror_ratio - mirror_target) ** 2


def _compute_iota_penalty(iota_axis, iota_edge, iota_ax_target, iota_edge_target):
    return (iota_axis - iota_ax_target) ** 2 + (iota_edge - iota_edge_target) ** 2


def _compute_f_grad_s(vmec_ro, s_targets, nu=80, nv=80):
    """Vacuum-proxy ITG target (dB/ds < 0)."""
    wout = vmec_ro.wout
    ns = int(wout.ns)
    nfp = int(wout.nfp)
    xm = np.ravel(np.array(wout.xm, dtype=int))
    xn = np.ravel(np.array(wout.xn, dtype=int))
    xm_nyq = np.ravel(np.array(wout.xm_nyq, dtype=int))
    xn_nyq = np.ravel(np.array(wout.xn_nyq, dtype=int))
    rmnc = np.array(wout.rmnc)
    zmns = np.array(wout.zmns)
    bmnc = np.array(wout.bmnc)
    gmnc = np.array(wout.gmnc)
    a_min = float(wout.Aminor_p)

    theta = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / nfp, nv, endpoint=False)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    m_geom = xm[:, None, None]
    n_geom = xn[:, None, None]
    ang_geom = m_geom * th[None, :, :] - n_geom * ze[None, :, :]
    cos_geom = np.cos(ang_geom)
    sin_geom = np.sin(ang_geom)

    m_nyq = xm_nyq[:, None, None]
    n_nyq = xn_nyq[:, None, None]
    ang_nyq = m_nyq * th[None, :, :] - n_nyq * ze[None, :, :]
    cos_nyq = np.cos(ang_nyq)

    s_grid = np.linspace(0, 1, ns)

    def _rg(fmnc, k):
        return np.sum(fmnc[:, k][:, None, None] * cos_geom, axis=0)

    def _rn(fmnc, k):
        return np.sum(fmnc[:, k][:, None, None] * cos_nyq, axis=0)

    f_total = 0.0
    for s_t in s_targets:
        js = int(np.argmin(np.abs(s_grid - s_t)))
        js = max(1, min(js, ns - 2))
        ds = s_grid[js + 1] - s_grid[js - 1]
        R = _rg(rmnc, js)
        R_u = np.sum(-m_geom * rmnc[:, js][:, None, None] * sin_geom, axis=0)
        R_v = np.sum(n_geom * rmnc[:, js][:, None, None] * sin_geom, axis=0)
        Z_u = np.sum(m_geom * zmns[:, js][:, None, None] * cos_geom, axis=0)
        Z_v = np.sum(-n_geom * zmns[:, js][:, None, None] * cos_geom, axis=0)
        sqrtg = _rn(gmnc, js)
        sqrtg = np.where(np.abs(sqrtg) < 1e-30, 1e-30, sqrtg)
        cross_sq = R ** 2 * (R_u ** 2 + Z_u ** 2) + (Z_u * R_v - R_u * Z_v) ** 2
        grad_s = np.sqrt(np.maximum(cross_sq / sqrtg ** 2, 1e-30))
        dBds = (_rn(bmnc, js + 1) - _rn(bmnc, js - 1)) / ds
        bad_mask = np.where(dBds < 0, 1.0, 0.0)
        xi = (a_min * bad_mask * grad_s) ** 2
        xi_pos = xi[xi > 0]
        if len(xi_pos) == 0:
            continue
        xi_95 = float(np.percentile(xi_pos, 95))
        integrand = xi * np.maximum(xi_95 - xi, 0.0)
        dth = 2 * np.pi / nu
        dze = 2 * np.pi / (nfp * nv)
        f_total += float(np.sum(integrand)) * dth * dze
    return f_total


def run_vmec(args):
    """Optimisation using SIMSOPT + VMEC (LeastSquaresProblem)."""
    print("\n  Backend: VMEC (SIMSOPT + VMEC2000)")

    input_path = os.path.join(
        os.path.dirname(os.path.abspath(args.nc_file)), "input.squid_init"
    )
    print(f"  Converting wout -> {input_path} ...")
    wout_to_input(args.nc_file, input_path, ns=args.ns_vmec,
                  prescribed_iota_ax=args.iota_ax,
                  prescribed_iota_edge=args.iota_edge)

    vmec = Vmec(input_path)
    vmec.run()

    A0 = vmec.aspect()
    if args.aspect_target is None:
        args.aspect_target = round(A0, 1)
    print(f"  Aspect: {A0:.2f}  (target {args.aspect_target})")

    # --- DoFs ---
    surf = vmec.boundary
    surf.fix_all()
    freed = []
    for m in [1, 0, 2, 3]:
        for n in [0, 1, -1, 2, -2]:
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
            self._fgs = 0.0
            self._mirror = 0.0
            self._iota_ax = 0.0
            self._iota_ed = 0.0
            self._n = 0
            super().__init__(depends_on=[vmec])

        def _compute(self):
            try:
                cx = tuple(vmec.x)
            except Exception:
                cx = None
            if cx is not None and cx == self._cache_x:
                return
            self._n += 1
            t0 = time.time()
            try:
                info = _evaluate_squid(
                    vmec, s_vals, alphas,
                    args.num_pitch, T_J=-0.06, mboz=8, nboz=8,
                )
            except Exception as exc:
                print(f"    [SQuID #{self._n}] FAILED: {exc}")
                info = dict(f_maxJ=1e6, f_QI=1e6, mirror_ratio=0.0,
                            iota_axis=0.0, iota_edge=0.0,
                            B_min=0.0, B_max=0.0)
            self._fmaxj = info["f_maxJ"]
            self._fqi = info["f_QI"]
            self._mirror = info["mirror_ratio"]
            self._iota_ax = info["iota_axis"]
            self._iota_ed = info["iota_edge"]

            try:
                self._fgs = _compute_f_grad_s(vmec, s_grad)
            except Exception:
                self._fgs = 0.0

            self._cache_x = cx
            dt = time.time() - t0
            print(f"    [SQuID #{self._n}]  f_maxJ={self._fmaxj:.3e}  "
                  f"f_QI={self._fqi:.3e}  f_nabla_s={self._fgs:.3e}  "
                  f"delta={self._mirror:.4f}  "
                  f"iota=[{self._iota_ax:.3f},{self._iota_ed:.3f}]  ({dt:.1f}s)")

        def f_maxJ(self):
            self._compute()
            return np.sqrt(max(self._fmaxj, 0.0))

        def f_QI(self):
            self._compute()
            return np.sqrt(max(self._fqi, 0.0))

        def mirror_penalty(self):
            self._compute()
            return np.sqrt(_compute_mirror_penalty(
                self._mirror, args.mirror_target,
                getattr(args, 'mirror_max', None)))

        def iota_penalty(self):
            self._compute()
            return np.sqrt(_compute_iota_penalty(
                self._iota_ax, self._iota_ed,
                args.iota_ax, args.iota_edge))

        def grad_s_penalty(self):
            self._compute()
            return np.sqrt(max(self._fgs, 0.0))

        def reg_penalty(self):
            x_cur = np.array(vmec.x, dtype=float)
            return np.sqrt(float(np.sum((x_cur - x0_vmec) ** 2)))

    squid = SQuIDObjective()

    prob = LeastSquaresProblem.from_tuples([
        (vmec.aspect, args.aspect_target, args.w_ar),
        (squid.f_maxJ, 0.0, args.w_maxj),
        (squid.f_QI, 0.0, args.w_qi),
        (squid.mirror_penalty, 0.0, args.w_mirror),
        (squid.iota_penalty, 0.0, args.w_iota),
        (squid.grad_s_penalty, 0.0, args.w_grad_s),
        (squid.reg_penalty, 0.0, args.w_reg),
    ])

    print(f"\n  Weights: w_AR={args.w_ar}, w_maxJ={args.w_maxj}, "
          f"w_QI={args.w_qi}, w_mirror={args.w_mirror}, "
          f"w_iota={args.w_iota}, w_grad_s={args.w_grad_s}, w_reg={args.w_reg}")

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
    abs_step = getattr(args, 'abs_step', 1e-4)
    rel_step = getattr(args, 'rel_step', 0.0)
    print(f"\n  Starting optimisation (max_nfev={max_nfev}, "
          f"abs_step={abs_step:.0e}) ...")
    t_start = time.time()
    least_squares_serial_solve(prob, max_nfev=max_nfev, grad=True,
                               abs_step=abs_step, rel_step=rel_step)
    t_total = time.time() - t_start

    print(f"\n  Finished in {t_total / 60:.1f} min  ({squid._n} evals)")

    obj_f = prob.objective()
    print(f"\n  Initial objective = {obj0:.4e}")
    print(f"  Final objective   = {obj_f:.4e}")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(args.nc_file)),
        "input.squid_optimized",
    )
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
