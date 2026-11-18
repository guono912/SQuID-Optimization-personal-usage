# Skill 09 - Mercier Normalization Protocol

SQuID adopted edge-flux normalization on 2026-07-21. The radial-grid
correction below also applies when revisiting results produced since then.

## Canonical quantity

Use the paper-facing, flux-normalized Mercier criterion

```text
dmerc_flux_normalized(s) = Phi_edge^2 * dmerc_vmec_raw(s)
```

where `Phi_edge = abs(wout.phi[-1])` is VMEC's physical edge toroidal flux in
webers. Read `phi[-1]` directly from the wout. Do not reconstruct it from B0,
major radius, minor radius, or an input-file `PHIEDGE` convention.

The implementation is centralized in
`squid/diagnostics/mercier_normalization.py` and has
protocol ID `squid-flux-normalized-v1`. Mercier components `DShear`, `DWell`,
`DCurr`, and `DGeod` use the same `Phi_edge^2` multiplier.

## Radial grid and affected releases

Native VMEC2000 computes DMerc and its four components at interior full-grid
points. For a length-`ns` array with zero-based index `j`, use
`s = j / (ns - 1)` for `1 <= j <= ns - 2`. Its Mercier loop omits the
axis and the last surface. DESC 0.16 also exports Mercier on its full grid,
sets the axis entry to zero, and computes the edge from its own equilibrium.
For a body comparison, use the same interior mask in both files. Treat the
edge according to the writer when it is explicitly under study.
[VMEC2000 radial coordinate and Mercier loop](https://github.com/hiddenSymmetries/VMEC2000/blob/728af8bd6c796b36a0aa85fe298e507791e57c6e/Sources/Input_Output/mercier.f#L108-L188)
and [DESC 0.16 wout writer](https://github.com/PlasmaControl/DESC/blob/v0.16.0/desc/vmec.py#L762-L781).

At SQuID revision `e3d03f1`, `vmec_half_grid_profile()` drops index zero
and places index `j` at `(j - 0.5) / (ns - 1)`. Diagnosis also uses
`vmec.s_half_grid` for DMerc. Each such label is half a radial step inward.
The stored pointwise values, their signs, and the multiplier `Phi_edge^2`
stay the same. An interval can gain or lose a sample, changing its minimum,
negative count and the radius assigned to the minimum. The mapping can even
include the native VMEC edge slot near the outer edge. This finding is
specific to Mercier and its components. Other wout variables have their own
grid conventions.

Affected consumers include both optimizer backends, QI refinement, diagnostic
summaries, component plots and the Mercier part of the MHD gate. A gate
threshold or candidate ranking can change when the newly included or
excluded point is decisive.

Recheck interval statistics directly from the original wout before using
them for thresholds, rankings or promotion. For a finite interior interval:

```python
import numpy as np
from netCDF4 import Dataset

with Dataset(wout_path) as ds:
    raw = np.ma.filled(ds.variables["DMerc"][:], np.nan)
    phi_edge = float(np.ma.filled(ds.variables["phi"][-1], np.nan))
if not np.isfinite(phi_edge) or phi_edge == 0:
    raise ValueError("invalid edge toroidal flux")
j = np.arange(1, len(raw) - 1)
s = j / (len(raw) - 1)
keep = j[(s >= s_min) & (s <= s_max)]
if not keep.size or not np.isfinite(raw[keep]).all():
    raise ValueError("incomplete Mercier interval")
values = phi_edge**2 * raw[keep]
j_min = keep[np.argmin(values)]
```

Record `keep`, `j_min / (len(raw) - 1)`, `values.min()` and the number of
negative values. A historical summary alone cannot recover the extra sample.
The protocol ID `squid-flux-normalized-v1` identifies the flux scaling and
does not establish that the correct radii were used.

## Numerical check on a four-period equilibrium

The 2026-11-18 backend audit used a parent with 31 radial points and two
accepted candidates. We compared the repository helper at `e3d03f1` with
the physical coordinate above on the same stored arrays. For both body
intervals, the helper selected indices 4 through 29 inclusive. The full-grid
mask selected 3 through 28 for `[0.1, 0.95]` and 3 through 29 for
`[0.1, 0.97]`.

| Wout (SHA256 prefix) | Minimum with v1 coordinates | Minimum on full grid | Negative points, v1 / full at 0.95 | Negative points, v1 / full at 0.97 |
| --- | ---: | ---: | ---: | ---: |
| Parent (`583ff424ecfe`) | -0.236762 | -0.452630 | 26 / 26 | 26 / 27 |
| Rebuilt VMEC candidate (`81f3c0629ea4`) | -0.235828 | -0.451070 | 26 / 26 | 26 / 27 |
| Exported DESC candidate (`2ed6f3d072ae`) | -82.634240 | -125.502089 | 26 / 26 | 26 / 27 |

All minima are flux-normalized. For each file the v1 minimum was index 4,
labeled `s = 0.116667` (physical `s = 0.133333`). The full-grid minimum
was index 3 at `s = 0.1`. All three still fail a zero-negative-point rule.
Their optimizer runs used the core preset with Mercier weights and the hard
MHD gate disabled. The termination checks therefore remain applicable.
A run with an active Mercier objective can follow a different search path;
recomputing a saved candidate measures its score, not that trajectory.

The flux scaling matches the dimensionless quantity plotted as
`Phi_edge^2 D_Merc` in the Stellaris design paper and the edge-toroidal-flux
normalization stated for CIEMAT-QI4X:

- https://publikationen.bibliothek.kit.edu/1000179851/172386752
- https://doi.org/10.1088/1741-4326/ae54ad

## Required names

New machine-readable outputs must use explicit fields:

- `dmerc_vmec_raw_min`
- `dmerc_flux_normalized_min`
- `dmerc_edge_toroidal_flux_wb`
- `dmerc_negative_count`
- `dmerc_min_s`, `dmerc_s_min`, and `dmerc_s_max`

Do not create new fields named `DMerc_min`, `scaled_DMerc`, `dmerc_scaled`, or
`dmerc_min` without a normalization qualifier. Plot axes and prose must state
`Phi_edge^2 D_Merc`, not merely `DMerc`, when quoting a magnitude.

Active SQuID pipelines must not emit a B0-squared Mercier column, even when it
is labeled `legacy`. Reproduce that convention only in isolated archival
analysis outside production optimization, gate, and reporting paths.

## Gates and margins

A positive flux multiplier preserves the pointwise stability sign. The
promotion criterion below requires the interval and resolution to be recorded.
The optimizer applies its hard MHD checks only when `--hard_gate_mhd` is enabled.

```text
dmerc_flux_normalized(s) >= 0 on the declared radial interval
dmerc_negative_count = 0
```

A positive numerical margin is protocol-specific. Declare and calibrate it
against a reference set at fixed radial mask and resolution before using it.
Never transplant the historical raw or B0-squared thresholds `0.08` or `0.1`
into the flux-normalized column.

Use the optimizer keys:

- `mercier_flux_normalized_margin_target`
- `hard_dmerc_flux_normalized_min`

Legacy keys such as `mercier_margin_target`, `hard_dmerc_min`, and
`dmerc_b0_ref` are rejected so an old value cannot be silently reinterpreted.
Changing from raw residuals to flux-normalized residuals also changes the
effective optimization weight; recalibrate `w_mercier` and
`w_mercier_margin` in a short run.

## Comparison protocol

Every comparison must record the normalization protocol, exact radial mask,
VMEC radial resolution, convergence status, beta/profile/current state, B0,
and `Phi_edge`. Compare flux-normalized magnitudes only when those facts are
declared. The sign may be compared more broadly, but still requires a
converged equilibrium and a declared radial interval.

Near-axis points are reported separately. The standard body masks currently
used by different campaigns, `[0.1, 0.95]` and `[0.1, 0.97]`, remain distinct
protocols and must not be merged into one ranking column.

## Near-axis interpretation

For a regular equilibrium with a nonzero leading Mercier coefficient,
`D_Mercier` scales as `1/rho^2`, with `rho = sqrt(s)`. Examine
`s * D_Mercier` and all four components at fixed physical radii. Compare
several radial resolutions and report force residuals. Check a higher angular
and spectral resolution or an independent force-balanced equilibrium when
classifying a marginal crossing. A crossing that settles at a finite radius
has different implications from one that moves toward the axis with radial
refinement. If the scaled coefficient or the sign at fixed radius fails to
converge, report the region as unresolved. State separately what the chosen
body interval excludes.
[Landreman and Jorge (2020)](https://doi.org/10.1017/S002237782000121X)
and [Kim, Jorge and Dorland (2021)](https://doi.org/10.1017/S0022377821000465).

## Historical data

Existing reports produced before 2026-07-21 are not rewritten in place.
Interpret:

- `DMerc` or `dmerc_raw` as VMEC raw only when the producing script confirms it;
- `scaled_DMerc`, `dmerc_scaled_1T`, or `DMerc_1T` as legacy B0-based scaling;
- an unlabeled `DMerc_min` as ambiguous and unsuitable for cross-scale ranking.

Recompute from the original wout with the full-grid mask above before a
historical candidate is promoted or compared across B0 or size.

The following command checks the multiplication identity on its selected
samples. At `e3d03f1` it uses the v1 mask, so a passing identity check
still requires the independent coordinate check above.

```bash
python scripts/diag/check_mercier_normalization.py <wout.nc> --s-min 0.1 --s-max 0.95
```
