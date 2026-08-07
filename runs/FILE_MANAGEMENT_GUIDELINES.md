# SQuID File Management Guidelines

This document defines where generated products live, how they are named, and
which files are retained. It applies to both YF_0 and YF_1 campaigns.

## Non-negotiable rules

1. The repository root is source-only. VMEC, DESC, REGCOIL, and diagnostic
   products must never be written there.
2. Every calculation gets one explicit run directory. A run directory contains
   its inputs, outputs, logs, diagnostics, and manifest together.
3. A filename is not the identity of an equilibrium. The manifest is the
   authority for parent, state, physics point, resolution, code revision, and
   checksums.
4. Never overwrite a promoted equilibrium. Create a new run or a new state.
5. Do not delete a promoted node or a failed run before its manifest records
   why it was rejected and where its parent came from.

## Canonical layout

Use the canonical run root `/home/guozx/runs` on WSL:

```text
runs/
  YF_0/<campaign>/<run_id>/
  YF_1/<campaign>/<run_id>/
  archive/<date_or_event>/<run_id>/
```

Each retained run should have:

```text
<run_id>/
  equilibrium/       # wout_*.nc and the exact VMEC input
  diagnostics/       # CSV/JSON/plots, including resolution studies
  logs/              # stdout, stderr, command line, environment summary
  manifest.json      # required metadata and SHA256 values
  README.md          # short scientific conclusion and promotion status
```

Scratch products may be numerous, but they belong inside the active run
directory. When a run is closed, retain only the important nodes and the
evidence needed to reproduce their promotion or rejection.

## Run identifiers and filenames

Use lower-case, ASCII, stable tokens separated by underscores:

```text
<project>_<lineage>_<state>_<physics>_<resolution>_<shortid>
```

Examples:

```text
yf0_0396_topology_clean_vacuum_B0p200_I_m4920_ns192
yf0_0396_coil_return_vacuum_ns192
yf1_w7x_stage3_lowbeta_B0p200_ns128
```

Recommended tokens include `vacuum`, `finite_beta`, `coil_return`,
`fixed_boundary`, `free_boundary`, `repaired`, `promoted`, `rejected`,
`B0p200`, `beta0p200`, `Im4920`, `ns192`, `m12n14`, and `nfp3`.
Do not use spaces, dates as the only identity, or ambiguous names such as
`wout.nc`, `final.nc`, or `new2.nc` in a promoted directory.

## Manifest minimum fields

Every promoted node must record:

```text
asset_id, project, campaign, parent_id, status, source_kind,
path, sha256, nfp, R_major, minor_radius, aspect, B0, beta,
current, ns, mpol, ntor, pressure_profile, iota_summary,
mercier_convention, dmerc_flux_normalized_min, dmerc_vmec_raw_min,
ballooning_lambda_max, epsilon_eff_peak, itg_proxy,
pdrot_p99, pdrot_cvar1, pdrot_max, code_commit, command, created_at
```

The Mercier convention must be explicit. The project default is the
flux-normalized quantity `Phi_edge^2 * D_Merc`; raw VMEC `D_Merc` is retained
only as a secondary diagnostic.

## Retention and promotion

- `P0`: promoted candidate, its parent, and all resolution/operating-window
  evidence. Keep indefinitely.
- `P1`: serious candidate or comparison baseline with complete diagnostics.
- `P2`: short-run screen or intermediate checkpoint; keep until campaign close.
- `P3`: failed scratch output; keep only the manifest, failure reason, and any
  evidence needed to explain a decision.

Promotion is a copy-with-manifest operation, not a rename in place. A promoted
node should be immutable. Archive old nodes under `runs/archive/` and preserve
the source and destination checksums when moving legacy products.

## Tool behavior requirements

- Scripts must accept `--run-dir`, `--output-dir`, or an equivalent explicit
  output path; hard-coded absolute campaign paths are forbidden.
- VMEC calls must run with their working directory set to that run directory,
  because legacy VMEC/SIMSOPT output names are relative to the process cwd.
- Output discovery must search the requested run directory, never `os.getcwd()`
  or the repository root.
- A script that creates a new equilibrium must write a command line and a
  manifest or a machine-readable history file before promotion.
- Source utilities used for real calculations are tracked code, not scratch
  files. Scratch experiments belong in a run directory or an explicitly named
  `scratch/` directory.

## Campaign close checklist

1. Confirm no calculation process is running.
2. Verify the run directory contains the input, output, log, and diagnostics.
3. Recompute SHA256 for every retained `.nc` file.
4. Record the promotion/rejection decision in `README.md` and `manifest.json`.
5. Move legacy root products only into a dated archive with a CSV manifest.
6. Confirm the repository root has no generated VMEC products.
