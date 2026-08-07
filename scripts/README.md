# SQuID command-line entry points

`scripts/` contains thin user-facing wrappers. Workflow orchestration lives in
`squid/cli/`; reusable physics and numerical logic lives elsewhere under
`squid/`.

Use grouped paths in new documentation. Legacy flat paths remain compatible.

## Optimization

```bash
python scripts/opt/optimize.py --input_parameter <config.json>
python scripts/opt/refine_qi.py --nc-file <wout.nc> --run-dir <run>
```

- `optimize.py`: VMEC/DESC optimization, resolved config, history, checkpoints.
- `refine_qi.py`: specialized QI refinement; use only after baseline gates.

## Diagnostics

```bash
python scripts/diag/diagnose.py \
  --nc_file <wout.nc> --output_dir <dir> [--plot] [--extended] [--ae]

python scripts/diag/physical_diagnostics_report.py \
  --wout <wout.nc> --output_dir <dir> [--diagnostics_json <json>]

python scripts/diag/check_mercier_normalization.py \
  <wout.nc> --s-min 0.1 --s-max 0.97
```

- `diagnose.py`: QI/max-J/Bmin, iota/rationals, Mercier/well, ripple, ITG/AE.
- `physical_diagnostics_report.py`: profile-focused MHD/transport report.
- `check_mercier_normalization.py`: audit the flux-normalized Mercier identity.

## Independent gates

```bash
python scripts/gate/mhd_gate.py --wout <wout.nc> --output_dir <dir>
python scripts/gate/gate_checkpoints.py --run_dir <stage> [--watch]
python scripts/gate/coil_feasibility_gate.py --wout <wout.nc> --output_dir <dir>
python scripts/gate/coil_gate_checkpoints.py --run_dir <stage> [--watch]
python scripts/gate/boundary_geometry_gate_checkpoints.py --run_dir <stage> ...
```

Run `--help` before using a gate. Store the exact protocol with every result.

## Seeds and transformations

```bash
python scripts/seed/generate_seed.py --help
python scripts/seed/recompose_vmec_input.py --help
python scripts/transform/rescale_and_gate.py --help
```

Seed/NFP/scale transformations require provenance and reconstruction checks;
see Skills 02 and 08.

## Visualization

```bash
python scripts/viz/viz_report.py \
  --nc_file <wout.nc> --output_dir <dir> [--nfp <NFP>] [--gate_json <json>]
```

Expected outputs include equilibrium overview, MHD stability, neoclassical
transport, ITG proxy, J contours, Boozer surface, scalar JSON, and boundary
HTML. A viz report is not a substitute for an external gate.

## Utilities

```bash
python scripts/util/coil_contour_metrics.py --help
python scripts/util/compare_desc_profiles.py --help
python scripts/util/nc_to_neort.py --help
```

## Output rules

- Always pass an explicit run/output path under `/home/guozx/runs`.
- Optimization refuses a non-empty `run_dir` by default. Use a new directory;
  `--overwrite_run_dir` is an explicit recovery-only override.
- An evaluation-cap stop exits nonzero and writes
  `input.squid_unaccepted_cap` plus `run_status.json`; it never creates a false
  `input.squid_optimized` candidate.
- Never generate `wout`, VMEC input, `threed1`, `simsopt_*.dat`, plots, or
  solver scratch in the source root.
- Preserve `input_parameter.resolved.json`, history, command, source hash, and
  code revision with durable runs.
- Ripple/full-report workflows default to CPU in the current DESC/JAX stack.

## Adding a command

1. Put reusable logic in `squid/diagnostics`, `evaluation`, `objectives`,
   `backends`, or `utils`.
2. Put orchestration/argparse in `squid/cli`.
3. Add a thin grouped wrapper under `scripts/<group>/`.
4. Keep an old flat wrapper if that path already existed.
5. Add import, `--help`, output-isolation, and numerical tests.
