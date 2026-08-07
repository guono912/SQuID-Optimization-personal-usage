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

    Writes the FULL Boozer spectrum (all m and n>=0 modes) so that
    NEO-RT sees the complete 3D magnetic field.

    Returns
    -------
    tuple (output_path, epsmn, m0, mph) on success, None on failure.
        epsmn, m0, mph describe the dominant non-axisymmetric harmonic
        for use in the NEO-RT namelist.
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

        xm = np.asarray(f.variables['ixm_b'][:], dtype=int)
        xn = np.asarray(f.variables['ixn_b'][:], dtype=int)   # already ×nfp

        bmnc_2d = np.asarray(f.variables['bmnc_b'][:])
        rmnc_2d = np.asarray(f.variables['rmnc_b'][:])
        zmns_2d = np.asarray(f.variables['zmns_b'][:])
        pmns_2d = np.asarray(f.variables['pmns_b'][:])

        n_surfs = bmnc_2d.shape[0]

        iota_full = np.asarray(f.variables['iota_b'][:])
        bvco_full = np.asarray(f.variables['bvco_b'][:])
        buco_full = np.asarray(f.variables['buco_b'][:])

        pres_full = (np.asarray(f.variables['pres_b'][:])
                     if 'pres_b' in f.variables else np.zeros(ns_vmec_f))
        phi_full  = (np.asarray(f.variables['phi_b'][:])
                     if 'phi_b'  in f.variables else np.zeros(ns_vmec_f))

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
            if len(s_arr) > n_surfs:
                s_arr = s_arr[:n_surfs]
            else:
                s_arr = np.pad(s_arr, (0, n_surfs - len(s_arr)),
                               constant_values=s_arr[-1])
    elif jlist is not None and ns_grid > 1:
        s_arr = (jlist - 1.0) / (ns_grid - 1.0)
    else:
        s_arr = np.linspace(0.05, 0.95, n_surfs)

    # ── Extract per-surface quantities from full radial arrays ───
    if jlist is not None:
        jidx = np.clip(jlist - 1, 0, len(iota_full) - 1)
    else:
        jidx = np.linspace(0, len(iota_full) - 1, n_surfs, dtype=int)

    iota_arr = iota_full[jidx]
    bvco_arr = bvco_full[jidx]
    buco_arr = buco_full[jidx]
    pres_arr = pres_full[jidx]

    # ── Derived quantities (SI) ──────────────────────────────────
    Jpol_arr = bvco_arr * (2.0 * np.pi / MU0)
    Itor_arr = buco_arr * (2.0 * np.pi / MU0)

    if n_surfs > 1:
        pprime_arr = np.gradient(pres_arr, s_arr)
    else:
        pprime_arr = np.zeros(n_surfs)

    idx00 = np.where((xm == 0) & (xn == 0))[0]
    sqrtg_arr = np.zeros(n_surfs)
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
    elif 'phi_b' in f.variables and np.any(phi_full != 0):
        flux_total = float(phi_full[-1])
    else:
        flux_total = 0.0
        print("      [nc_to_neort] WARNING: flux = 0."
              " Pass flux_override=vmec.wout.phi[-1].")

    if a_override is not None:
        a_minor = float(a_override)
    elif aspect > 1:
        a_minor = R0 / aspect
    else:
        a_minor = R0 / 10.0

    # ── Build lookup table: (m, n/nfp) → index in booz_xform arrays
    n_bare = xn // nfp if nfp != 0 else xn
    mode_idx = {}
    for k in range(len(xm)):
        key = (int(xm[k]), int(n_bare[k]))
        mode_idx[key] = k

    # Target modes: m = 0..m0b, n = 0..n0b  (n≥0 for stellarator symmetry)
    nmode_out = (m0b + 1) * (n0b + 1)

    # Find dominant non-axisymmetric harmonic for perturbation info
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
        dominant_mph = abs(int(xn[gi]))
        dominant_m0 = int(xm[gi])
        print(f"      [nc_to_neort] Dominant non-axi harmonic:"
              f" m={dominant_m0}, n={int(n_bare[gi])},"
              f" |bmnc|/B00={dominant_epsmn:.4e}")

    # ── Write in_file with FULL Boozer spectrum ──────────────────
    with open(output_path, 'w') as out:
        out.write("CC Boozer-coordinate data file\n")
        out.write("CC Converted by SQuID nc_to_neort.py (full 3D spectrum)\n")
        out.write(f"CC Source: {boozmn_path}\n")
        out.write("CC\n")
        out.write(" m0b   n0b  nsurf  nper    flux [Tm^2]"
                  "        a [m]          R [m]\n")
        out.write(f"  {m0b:4d} {n0b:4d} {n_surfs:5d} {nfp:4d}"
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

            for n_idx in range(n0b + 1):
                for m_idx in range(m0b + 1):
                    key = (m_idx, n_idx)
                    if key in mode_idx:
                        k = mode_idx[key]
                        out.write(
                            f"  {m_idx:3d}  {n_idx:3d}"
                            f"   {rmnc_2d[i, k]:.8e}"
                            f"   {rmns_2d[i, k]:.8e}"
                            f"   {zmnc_2d[i, k]:.8e}"
                            f"   {zmns_2d[i, k]:.8e}"
                            f"   {pmnc_2d[i, k]:.8e}"
                            f"   {pmns_2d[i, k]:.8e}"
                            f"   {bmnc_2d[i, k]:.8e}"
                            f"   {bmns_2d[i, k]:.8e}\n")
                    else:
                        out.write(
                            f"  {m_idx:3d}  {n_idx:3d}"
                            + "   0.00000000e+00" * 8 + "\n")

    print(f"      [nc_to_neort] Wrote {output_path}"
          f"  ({n_surfs} surfaces, {nmode_out} modes,"
          f" m0b={m0b}, n0b={n0b})")
    return output_path, dominant_epsmn, dominant_m0, dominant_mph
