"""
Convert booz_xform netCDF (boozmn_*.nc) to NEO-RT in_file ASCII format.

NEO-RT's boozer_read() (do_magfie_standalone.f90) expects:
  - 5 header lines (skipped)
  - Data line: m0b  n0b  nsurf  nfp  flux[Tm²]  a[m]  R0[m]
  - Per surface (nsurf times):
      - 2 lines skipped   (read(18,'(/)'))
      - Params: s  iota  Jpol/nper[A]  Itor[A]  pprime[Pa]  sqrt_g(0,0)
      - 1 line skipped     (read(18,*))
      - nmode lines: m  n  rmnc  rmns  zmnc  zmns  vmnc  vmns  bmnc  bmns
  where nmode = (m0b + 1) * (n0b + 1), inp_swi = 9 (ncol2 = 8).

Variable mapping (this booz_xform version):
  ixm_b     → m          (poloidal mode numbers)
  ixn_b/nfp → n          (toroidal; booz_xform stores n*nfp)
  rmnc_b    → rmnc       (pack_rad, mn_modes)
  zmns_b    → zmns
  pmns_b    → vmns       (Boozer spectral angle)
  bmnc_b    → bmnc
  bvco_b    → G = mu0*Jpol/(2pi)   on full VMEC radial grid
  buco_b    → I = mu0*Itor/(2pi)   on full VMEC radial grid
  jlist     → 1-based VMEC surface indices for the computed surfaces
  ns_b      → VMEC radial grid size (NOT number of Boozer surfaces)
  For stellarator symmetry: rmns = zmnc = vmnc = bmns = 0.
"""

import numpy as np
from scipy.io import netcdf_file

MU0 = 4.0e-7 * np.pi


def convert_boozmn_to_neort(boozmn_path, output_path="in_file",
                            s_values=None, ns_vmec=None,
                            flux_override=None, a_override=None):
    """
    Convert booz_xform netCDF to NEO-RT ASCII in_file.

    Parameters
    ----------
    boozmn_path : str
        Path to boozmn_*.nc from booz_xform / simsopt Boozer.
    output_path : str
        Output file (default ``in_file`` — the name NEO-RT expects).
    s_values : array-like or None
        Normalised flux values for each Boozer surface.
        If None, computed from ``jlist`` and ``ns_b``.
    ns_vmec : int or None
        Override for VMEC radial grid size when computing s from jlist.
    flux_override : float or None
        Total toroidal flux [Tm²] from VMEC (``wout.phi[-1]``).
        Required when ``phi_b`` in the netCDF is all zeros.
    a_override : float or None
        Minor radius [m] from VMEC (``wout.Aminor_p``).
        Used when ``aspect_b`` is unavailable.

    Returns
    -------
    str or None
        Output path on success, None on failure.
    """
    try:
        f = netcdf_file(boozmn_path, 'r', mmap=False)
    except Exception as e:
        print(f"      [nc_to_neort] Cannot open {boozmn_path}: {e}")
        return None

    try:
        nfp       = int(f.variables['nfp_b'][()])
        ns_vmec_f = int(f.variables['ns_b'][()])
        mboz      = int(f.variables['mboz_b'][()])
        nboz      = int(f.variables['nboz_b'][()])

        # Mode numbers — this booz_xform version uses ixm_b / ixn_b
        xm = np.asarray(f.variables['ixm_b'][:], dtype=int)
        xn = np.asarray(f.variables['ixn_b'][:], dtype=int)   # already ×nfp

        # Harmonic data — shape (pack_rad, mn_modes), one row per surface
        bmnc_2d = np.asarray(f.variables['bmnc_b'][:])
        rmnc_2d = np.asarray(f.variables['rmnc_b'][:])
        zmns_2d = np.asarray(f.variables['zmns_b'][:])
        pmns_2d = np.asarray(f.variables['pmns_b'][:])         # vmns

        n_surfs = bmnc_2d.shape[0]   # actual number of computed surfaces

        # Full-radial-grid arrays (length = ns_b = VMEC grid size)
        iota_full = np.asarray(f.variables['iota_b'][:])
        bvco_full = np.asarray(f.variables['bvco_b'][:])
        buco_full = np.asarray(f.variables['buco_b'][:])

        pres_full = (np.asarray(f.variables['pres_b'][:])
                     if 'pres_b' in f.variables else np.zeros(ns_vmec_f))
        phi_full  = (np.asarray(f.variables['phi_b'][:])
                     if 'phi_b'  in f.variables else np.zeros(ns_vmec_f))

        # Stellarator-symmetric partners (zero if absent)
        has_bmns = 'bmns_b' in f.variables
        bmns_2d = np.asarray(f.variables['bmns_b'][:]) if has_bmns else np.zeros_like(bmnc_2d)
        has_rmns = 'rmns_b' in f.variables
        rmns_2d = np.asarray(f.variables['rmns_b'][:]) if has_rmns else np.zeros_like(rmnc_2d)
        has_zmnc = 'zmnc_b' in f.variables
        zmnc_2d = np.asarray(f.variables['zmnc_b'][:]) if has_zmnc else np.zeros_like(zmns_2d)
        has_pmnc = 'pmnc_b' in f.variables
        pmnc_2d = np.asarray(f.variables['pmnc_b'][:]) if has_pmnc else np.zeros_like(pmns_2d)

        jlist = (np.asarray(f.variables['jlist'][:], dtype=int)
                 if 'jlist' in f.variables else None)

        # Aspect ratio for minor-radius estimate
        aspect = (float(f.variables['aspect_b'][()]) if 'aspect_b' in f.variables else 0.0)
    except KeyError as e:
        print(f"      [nc_to_neort] Missing variable in {boozmn_path}: {e}")
        f.close()
        return None
    finally:
        f.close()

    # ── Normalised flux s per computed surface ───────────────────
    ns_grid = ns_vmec if ns_vmec else ns_vmec_f
    if s_values is not None:
        s_arr = np.asarray(s_values, dtype=float)
        if len(s_arr) != n_surfs:
            print(f"      [nc_to_neort] Warning: len(s_values)={len(s_arr)}"
                  f" != computed surfaces={n_surfs}; truncating/padding.")
            if len(s_arr) > n_surfs:
                s_arr = s_arr[:n_surfs]
            else:
                s_arr = np.pad(s_arr, (0, n_surfs - len(s_arr)),
                               constant_values=s_arr[-1])
    elif jlist is not None and ns_grid > 1:
        s_arr = (jlist - 1.0) / (ns_grid - 1.0)
    else:
        s_arr = np.linspace(0.05, 0.95, n_surfs)
        print("      [nc_to_neort] Warning: s unknown; using linspace.")

    # ── Extract per-surface quantities from full radial arrays ───
    # jlist stores 1-based VMEC surface indices; convert to 0-based
    if jlist is not None:
        jidx = np.clip(jlist - 1, 0, len(iota_full) - 1)
    else:
        jidx = np.linspace(0, len(iota_full) - 1, n_surfs, dtype=int)

    iota_arr = iota_full[jidx]
    bvco_arr = bvco_full[jidx]
    buco_arr = buco_full[jidx]
    pres_arr = pres_full[jidx]
    phi_arr  = phi_full[jidx]

    # ── Derived quantities (SI) ──────────────────────────────────
    Jpol_arr = bvco_arr * (2.0 * np.pi / MU0)
    Itor_arr = buco_arr * (2.0 * np.pi / MU0)

    if n_surfs > 1:
        pprime_arr = np.gradient(pres_arr, s_arr)
    else:
        pprime_arr = np.zeros(n_surfs)

    # sqrt_g(0,0) via Boozer identity:
    #   <sqrt(g)> * <B²> = G*iota + I  (where G = bvco, I = buco in SI)
    sqrtg_arr = np.zeros(n_surfs)
    idx00 = np.where((xm == 0) & (xn == 0))[0]
    for i in range(n_surfs):
        b = bmnc_2d[i, :]
        B00_sq = b[idx00[0]]**2 if len(idx00) > 0 else 0.0
        B2_avg = B00_sq + 0.5 * (np.sum(b**2) - B00_sq)
        if B2_avg > 0:
            sqrtg_arr[i] = (bvco_arr[i] * iota_arr[i] + buco_arr[i]) / B2_avg

    # ── Global header quantities ─────────────────────────────────
    m0b = int(mboz)
    n0b = int(nboz)

    R0 = float(np.abs(rmnc_2d[0, idx00[0]])) if len(idx00) > 0 else 1.0

    if flux_override is not None:
        flux_total = float(flux_override)
    elif np.any(phi_arr != 0):
        flux_total = float(phi_arr[-1])
    else:
        flux_total = 0.0
        print("      [nc_to_neort] WARNING: flux = 0 (phi_b not implemented)."
              " Pass flux_override=vmec.wout.phi[-1] for correct results.")

    if a_override is not None:
        a_minor = float(a_override)
    elif aspect > 1:
        a_minor = R0 / aspect
    else:
        a_minor = R0 / 10.0
        print("      [nc_to_neort] WARNING: aspect_b unavailable."
              " Pass a_override=vmec.wout.Aminor_p for correct results.")

    # ── Build mode tables ───────────────────────────────────────
    # NEO-RT equilibrium (in_file) requires n=0 modes ONLY with
    # m = 0, 1, ..., m0b in ascending order (fast_sin_cos assumes this).
    # Non-axisymmetric harmonics (n≠0) belong in a perturbation file.
    n_bare = xn // nfp if nfp != 0 else xn

    n0_mask = (xn == 0)
    n0_idx = np.where(n0_mask)[0]
    n0_idx = n0_idx[np.argsort(xm[n0_idx])]  # ascending m

    n0b_out = 0           # equilibrium file is axisymmetric
    target_nmode = m0b + 1
    nmode_out = min(len(n0_idx), target_nmode)

    # Find dominant non-axisymmetric harmonic for perturbation estimate
    mid_surf = n_surfs // 2
    B00 = float(bmnc_2d[mid_surf, idx00[0]]) if len(idx00) > 0 else 1.0
    nz_mask = (xn != 0)
    dominant_epsmn = 0.0
    dominant_mph = nfp
    dominant_m0 = 1
    if np.any(nz_mask):
        nz_idx = np.where(nz_mask)[0]
        nz_amps = np.abs(bmnc_2d[mid_surf, nz_idx])
        best = np.argmax(nz_amps)
        gi = nz_idx[best]
        dominant_epsmn = float(nz_amps[best] / abs(B00)) if B00 != 0 else 0.0
        dominant_mph = abs(int(xn[gi]))   # physical toroidal mode number (already ×nfp)
        dominant_m0 = int(xm[gi])
        print(f"      [nc_to_neort] Dominant non-axi harmonic:"
              f" m={dominant_m0}, n={int(n_bare[gi])},"
              f" |bmnc|/B00={dominant_epsmn:.4e}")

    # ── Write in_file (n=0 equilibrium only) ─────────────────────
    with open(output_path, 'w') as out:
        out.write("CC Boozer-coordinate data file\n")
        out.write("CC Converted by SQuID nc_to_neort.py (n=0 equilibrium)\n")
        out.write(f"CC Source: {boozmn_path}\n")
        out.write("CC\n")
        out.write(" m0b   n0b  nsurf  nper    flux [Tm^2]"
                  "        a [m]          R [m]\n")
        out.write(f"  {m0b:4d} {n0b_out:4d} {n_surfs:5d} {nfp:4d}"
                  f"  {flux_total:16.8e}   {a_minor:.8f}   {R0:.8f}\n")

        for i in range(n_surfs):
            out.write("        s               iota"
                      "           Jpol/nper          Itor"
                      "            pprime         sqrt g(0,0)\n")
            out.write("                              "
                      "            [A]           [A]"
                      "             [Pa]         (dV/ds)/nper\n")
            out.write(f"   {s_arr[i]:.8e}   {iota_arr[i]:.8e}"
                      f"   {Jpol_arr[i]/nfp:.8e}   {Itor_arr[i]:.8e}"
                      f"   {pprime_arr[i]:.8e}   {sqrtg_arr[i]:.8e}\n")
            out.write("    m    n      rmnc [m]         rmns [m]"
                      "         zmnc [m]         zmns [m]"
                      "         vmnc [ ]         vmns [ ]"
                      "         bmnc [T]         bmns [T]\n")
            for k in range(nmode_out):
                gi = n0_idx[k]
                out.write(
                    f"  {xm[gi]:3d}  {0:3d}"
                    f"   {rmnc_2d[i, gi]:.8e}"
                    f"   {rmns_2d[i, gi]:.8e}"
                    f"   {zmnc_2d[i, gi]:.8e}"
                    f"   {zmns_2d[i, gi]:.8e}"
                    f"   {pmnc_2d[i, gi]:.8e}"
                    f"   {pmns_2d[i, gi]:.8e}"
                    f"   {bmnc_2d[i, gi]:.8e}"
                    f"   {bmns_2d[i, gi]:.8e}\n")
            for k in range(nmode_out, target_nmode):
                out.write(f"  {k:3d}  {0:3d}"
                          + "   0.00000000e+00" * 8 + "\n")

    print(f"      [nc_to_neort] Wrote {output_path}"
          f"  ({n_surfs} surfaces, {target_nmode} modes, n0b=0)")
    return output_path, dominant_epsmn, dominant_m0, dominant_mph
