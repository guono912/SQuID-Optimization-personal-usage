# Skill 01 — Environment & Repository Layout

## Machines and paths

- Linux host (WSL Ubuntu-24.04 / remote R7625): project root `/home/guozx/SQuID`.
- Python env: `/home/guozx/fusion_env` (python3.12; DESC, simsopt, netCDF4,
  VMEC2000 extension importable as `vmec`). Activate or call
  `/home/guozx/fusion_env/bin/python` directly.
- Standalone VMEC binary: `xvmec <input.name>` (run in the directory holding
  the input file; produces `wout_<name>.nc`, `threed1.<name>`, `mercier.<name>`).
- DESC W7-X example: `/home/guozx/fusion_env/lib/python3.12/site-packages/desc/examples/W7-X_output.h5`.
- Legacy W7-X VMEC beta scan: `/mnt/d/qifiles/Files/plots/plot_available_energy/wout_W7-X_beta_*.nc`.
- NFP=4 seed conversion repo: `/home/guozx/w7x_squid_nfp4` (`scripts/wout_to_nfp4_input.py`).

## Repository layout

```
SQuID/
  squid/                  # package: backends (vmec_backend, desc_backend), objectives, diagnostics
  scripts/                # all operational entry points (see below)
  configs/                # reusable optimizer JSON configs
  runs/<campaign>/        # one folder per campaign; all outputs live here
  artifacts/              # historical equilibria, do not modify
  skill/                  # this skill collection
  AGENT_OPTIMIZATION_GUIDE.md   # long-form methodology handoff
```

Active campaign: `runs/w7x_r2p35_goodman_chain/` with
`STATUS.md` (live dashboard), `GUIDELINES.md` (playbook),
`EXPERIMENT_LOG.md` (chronology), `SEED_SOURCE_AUDIT.md`,
`SEED_INVESTIGATION_20260611.md` (read its CORRECTIONS section first).

## Script inventory (scripts/)

| Script | Purpose |
| --- | --- |
| `optimize.py` | Main optimizer entry point (`--input_parameter <config.json>`) |
| `diagnose.py` | Baseline physics diagnosis of a wout (QI, maxJ, Bmin, ripple, Mercier...) |
| `mhd_gate.py` | EXTERNAL MHD gate: DESC force balance + ballooning + eps_eff |
| `gate_checkpoints.py` | Batch/watch MHD gate over a run's `checkpoints/` |
| `coil_feasibility_gate.py` | REGCOIL-like current-potential scan (Bn, K, Phi spectrum) |
| `coil_contour_metrics.py` | TRUE coil contours cut from current potential + geometry metrics |
| `coil_gate_checkpoints.py` | Batch coil gates over checkpoints |
| `boundary_geometry_gate_checkpoints.py` | pdrot/k2/H boundary geometry gates |
| `viz_report.py` | Full HTML report (Boozer maps, MHD, ITG proxy, Newcomb) |
| `physical_diagnostics_report.py` | Scalar physics diagnostics report |
| `recompose_vmec_input.py` | Low-mode filter / rescale of a VMEC input boundary |
| `generate_seed.py` | Seed generation helper |
| `rescale_and_gate.py` | Rescale a config and re-gate |

## Run-folder conventions

- One stage = one subfolder: `stageNN_<short_purpose>/` containing
  `config.json`, `input_parameter.resolved.json`, `history.csv`,
  `checkpoints/input.squid_eval_NNNNNN`, final `input.squid_optimized`.
- Gate outputs go to sibling folders, e.g. `mhd_<stage>_<eval>/`,
  `coil_<stage>_<eval>/`, `coil_contour_<stage>_<eval>/`.
- Checkpoints store VMEC inputs; rebuild wout with `xvmec` if missing
  (`gate_checkpoints.py` does this automatically).
- Seeds derived from a checkpoint live in `seeds/<name>_from_<stage>_<eval>/`.

## VMEC input conventions (critical)

- In `&INDATA`, boundary coefficients are `RBC(n,m)` / `ZBS(n,m)` where the
  FIRST index is the PER-PERIOD toroidal index n (boundary is
  `sum RBC cos(m*theta - n*NFP*phi)`). With NFP=4, `n=1` means physical
  toroidal mode 4.
- In a `wout_*.nc` file, `xn` is the PHYSICAL toroidal mode number
  (already multiplied by NFP). Do not confuse the two; this exact confusion
  produced the invalid Section-3 table in `SEED_INVESTIGATION_20260611.md`
  and the mis-indexed `input.desc_beta1p5`.
- Beta is set by `PRES_SCALE` x `AM(...)` polynomial; iota free with
  `NCURR=1, AC(0)=0`. `PHIEDGE` scales as `R^2` under geometric scaling.
