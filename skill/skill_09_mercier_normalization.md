# Skill 09 - Mercier Normalization Protocol

This protocol is mandatory for every new SQuID optimization, diagnostic,
gate, table, and plot after 2026-07-21.

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

## VMEC radial grid

Mercier profiles are VMEC half-grid quantities. For an array stored with
length `ns`, element zero is a non-physical placeholder. Use

```text
values = DMerc[1:]
s_half[j] = (j + 1/2) / (ns - 1),  j = 0, ..., ns - 2
```

through `vmec_half_grid_profile()` in the canonical module. Never map the
stored array with `linspace(0, 1, ns)`. That error can move an axis-near point
across the `s=0.1` gate when `ns` changes and can therefore create a false
resolution-dependent sign flip. The same half-grid mapping applies to
`DShear`, `DWell`, `DCurr`, and `DGeod`.

This convention matches the dimensionless quantity plotted as
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

The stability sign is invariant under this normalization. The default hard
gate is:

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

## Historical data

Existing reports produced before 2026-07-21 are not rewritten in place.
Interpret:

- `DMerc` or `dmerc_raw` as VMEC raw only when the producing script confirms it;
- `scaled_DMerc`, `dmerc_scaled_1T`, or `DMerc_1T` as legacy B0-based scaling;
- an unlabeled `DMerc_min` as ambiguous and unsuitable for cross-scale ranking.

Recompute from the original wout with
`squid/diagnostics/mercier_normalization.py` before a
historical candidate is promoted or compared across B0 or size.

Audit any wout without rerunning equilibrium physics:

```bash
python scripts/diag/check_mercier_normalization.py <wout.nc> --s-min 0.1 --s-max 0.95
```
