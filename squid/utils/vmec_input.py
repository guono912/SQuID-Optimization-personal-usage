"""VMEC input reconstruction shared by optimization and geometry transforms."""

import netCDF4
import numpy as np


def wout_to_input(wout_path, input_path, ns=31,
                  prescribed_iota_ax=None, prescribed_iota_edge=None,
                  free_iota=True, zero_current=False, *,
                  target_rmajor=None, nfp_out=None):
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
    with netCDF4.Dataset(wout_path, "r") as ds:
        nfp = int(ds.variables["nfp"][:])
        mpol = int(ds.variables["mpol"][:])
        ntor = int(ds.variables["ntor"][:])
        ns_w = int(ds.variables["ns"][:])

        phi_arr = np.array(ds.variables["phi"][:])
        phiedge = float(phi_arr[-1])
        iotaf = np.array(ds.variables["iotaf"][:])
        if target_rmajor is not None and "presf" not in ds.variables:
            presf = np.zeros(ns_w)
        else:
            presf = np.array(ds.variables["presf"][:])
        scale = 1.0
        if target_rmajor is not None:
            rmajor = float(ds.variables["Rmajor_p"][:])
            if not (np.isfinite(rmajor) and rmajor > 0 and np.isfinite(target_rmajor) and target_rmajor > 0):
                raise ValueError("source and target major radii must be positive and finite")
            scale = float(target_rmajor) / rmajor
            phiedge *= scale ** 2
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
    if target_rmajor is None:
        presf = np.maximum(presf, 0.0)
    p_max = presf.max()
    pn = presf / p_max if p_max > 0 else presf
    am = np.polynomial.polynomial.polyfit(s_full, pn, min(10, ns_w - 1))

    # VMEC stores the full toroidal mode numbers in the source convention.
    # Change the field period only after recovering the reduced mode indices.
    if rmnc.shape[-1] != len(xm):
        rmnc = rmnc.T
    if zmns.shape[-1] != len(xm):
        zmns = zmns.T
    xn_in = xn_full // nfp
    rbc, zbs = {}, {}
    for i in range(len(xm)):
        m, n = int(xm[i]), int(xn_in[i])
        if m >= mpol or abs(n) > ntor or xn_full[i] % nfp:
            continue
        rc, zs = float(rmnc[-1, i]) * scale, float(zmns[-1, i]) * scale
        if abs(rc) > 1e-16 or abs(zs) > 1e-16:
            rbc[(n, m)] = rc
            zbs[(n, m)] = zs

    if target_rmajor is not None:
        # The NFP4 rescaling workflow historically drops sub-1e-14 modes.
        rbc = {key: val for key, val in rbc.items() if abs(val) > 1e-14}
        zbs = {key: val for key, val in zbs.items() if abs(val) > 1e-14}

    with open(input_path, "w") as f:
        f.write("&INDATA\n")
        f.write("  DELT = 0.9\n  TCON0 = 1.0\n")
        f.write(f"  NFP = {nfp if nfp_out is None else nfp_out}\n  MPOL = {mpol}\n  NTOR = {ntor}\n")
        f.write(f"  NS_ARRAY = {ns}\n  NITER_ARRAY = 5000\n")
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
        for (n, m) in sorted(rbc.keys() | zbs.keys()):
            if (n, m) in rbc:
                f.write(f"  RBC({n:d},{m:d}) = {rbc[(n, m)]:.15e}\n")
            zs = zbs.get((n, m), 0.0)
            if abs(zs) > 1e-16:
                f.write(f"  ZBS({n:d},{m:d}) = {zs:.15e}\n")
        f.write("/\n")
