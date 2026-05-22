# SQuID Stellarator Optimisation Skill

Use this note as the working playbook for CLI agents helping with SQuID
stellarator configuration optimisation.

## Baseline Workflow

1. Keep the repository root clean. Write diagnostics and optimisation outputs
   under `runs/<run-name>/`; historical files belong in `artifacts/`.
2. Run a baseline diagnosis before changing optimisation targets:

   ```bash
   python scripts/diagnose.py \
     --nc_file <wout.nc> \
     --output_dir runs/diagnose_<case> \
     --plot
   ```

3. Compare JSON reports, not screenshots alone. Important fields are:
   `core.f_QI`, `core.f_maxJ`, `core.f_Bmin`,
   `core.qi_surface_rms`, `core.maxj_interval_pass_ratio`,
   `diagnostic_facts`, `stability.mercier_summary`,
   `stability.well_summary`, `geometry.iota_scan`, `ripple`, and
   `geometry.axis`.
4. Only enable expensive transport diagnostics after the core geometry is
   promising:

   ```bash
   python scripts/diagnose.py --nc_file <wout.nc> \
     --output_dir runs/diagnose_<case> \
     --extended --ae
   ```

## Available Seed Sources

- Local historical equilibria are under `artifacts/legacy_root/`.
- Near-axis/QSC-generated seed material is under `/home/guozx/pyqsc`, including
  exported VMEC inputs in `/home/guozx/pyqsc/seed_vmec_exports/`.
- Additional QSC-derived reference files are available at
  `/home/guozx/constellaration/nc_files/wout_squid_optimized_qsc.nc` and the
  matching HTML report.

## Reading Diagnostics

The Python diagnostics should compute facts and cheap proxies. The agent, not
the diagnostic script, decides how to adjust the next optimisation run.

- High `f_QI` or high `core.qi_surface_rms` means the field strength is not
  sufficiently omnigenous on the checked surfaces. Check `qi_surface_rms` to
  identify the radial location before changing global weights.
- Low max-J pass ratio means the second adiabatic invariant is not decreasing
  outward enough. Check `maxj_interval_pass_ratio` and
  `maxj_interval_worst_lambda`; shallow trapped particles often fail first.
- Positive `f_Bmin` means `B_min(s)` is not growing outward at the target rate.
  Increase `--w_bmin` or reduce shape changes that flatten the magnetic well.
- A negative magnetic well depth is an engineering red flag even if QI/max-J
  scores improve. Treat it as a stability constraint, not cosmetic output.
- Ripple results can come from DESC, NEO-RT, or a Boozer proxy. Do not compare
  values across sources as if they were identical physics metrics.
- Check `geometry.iota_scan.nearest_low_order`,
  `geometry.iota_scan.near_low_order`, and
  `geometry.iota_scan.crossings` before changing iota-related weights. These
  are facts relative to the configured denominator/distance thresholds, not an
  automatic accept/reject decision.
- Use `stability.mercier_summary.min`,
  `stability.mercier_summary.negative_count`,
  `stability.well_summary.edge`, and `stability.well_summary.min` as the cheap
  VMEC stability facts available in the default diagnostic pass. Ballooning,
  island-width, bootstrap-current, and high-fidelity turbulence conclusions
  require heavier tools and should not be inferred from these proxies alone.

## Parameter Adjustment Heuristics

- Prefer `configs/*.json` over long command lines for optimisation runs:

  ```bash
  python scripts/optimize.py --input_parameter configs/core_r2_assist.json
  ```

  Command-line arguments are still allowed for one-off overrides. Every run
  writes the final effective argument set to
  `runs/<run-name>/input_parameter.resolved.json`; use that file when comparing
  runs or preparing a commit.
- Use `--mode core` for baseline fixed-boundary VMEC optimisation with simple
  QI, max-J, B_min, aspect ratio, and regularisation.
- Use `--mode core_r2_assist` when the simple QI score is already reasonable
  and R2 should gently steer field-line structure. Keep R2 compressed
  (`qi_r2_arr_out=false`) for optimisation.
- Use `--mode maxj_repair` when diagnostics show acceptable QI but poor
  `maxj_interval_pass_ratio` or weak outward `B_min` growth. This mode raises
  max-J/B_min weights while keeping QI active enough to prevent backsliding.
- Treat `--mode engineering` as a late-stage preset. Do not mix too many weakly
  calibrated engineering or transport penalties into early geometry searches.
- If QI improves but max-J worsens, do not simply raise all weights. First
  inspect the failing radial interval and trapped-depth map, then raise
  `--w_maxj` or `--w_bmin` selectively.
- If aspect ratio rises during optimisation, keep `--w_ar` active and avoid
  freeing many high-order boundary modes before the low-order shape is stable.
- If iota drifts too far, use `--w_iota` with explicit `--iota_ax` and
  `--iota_edge` in free-iota mode. Use `--prescribe_iota` only for controlled
  experiments, since it can hide whether the boundary shape naturally supports
  the desired transform.
- If `geometry.iota_scan` shows a close approach to a low-order rational over
  the checked radial range, consider steering the profile away with iota
  targets or reducing the boundary changes that caused the profile shift.
- If finite-difference optimisation is noisy, lower the diagnostic resolution
  for early iterations and increase it only for final polishing.
- Keep `--w_grad_s` off for early core optimisation unless the base geometry
  is already acceptable; ITG targets are more model-dependent than QI/max-J.

## Backend And Equilibrium Notes

- The main optimisation path is fixed-boundary VMEC through SIMSOPT: boundary
  Fourier coefficients are the design variables, and VMEC enforces ideal-MHD
  force balance for each trial boundary.
- The DESC backend is currently a diagnostic/fallback shell, not the primary
  production optimiser. Use it only for controlled experiments until its
  objective assembly and output handling match the VMEC path.
- Free-boundary or coil-coupled optimisation should be a separate late-stage
  workflow after a fixed-boundary target is credible. Do not combine coil/free
  boundary degrees of freedom with early QI/max-J repair unless the run has a
  specific engineering purpose.

## Target Implementation Status

The active optimiser now sends resolution-normalised residual vectors to
SIMSOPT for simple QI, max-J, B_min, aspect, and regularisation targets.
The logged `f_QI`, `f_maxJ`, and `f_Bmin` values are squared residual norms,
so they are closer to mean-square errors than grid-size-dependent sums.

R2 squash-stretch-shuffle QI is available as an optional target:

```bash
python scripts/optimize.py \
  --nc_file <wout.nc> \
  --w_qi_r2 0.1 \
  --qi_r2_nphi 301 --qi_r2_nalpha 24 --qi_r2_nbj 301
```

Keep R2 off or lightly weighted during early optimisation. Before making it
the default:

1. Compare simple QI and R2 QI on existing cases in `artifacts/legacy_root/`.
2. Calibrate `--w_qi_r2` at low resolution.
3. Increase R2 resolution only for final polishing.
4. Confirm the resulting configurations still pass max-J, B_min, ripple, well,
   and iota diagnostics.

## R2 Calibration Notes

Initial low-resolution scans used:

```bash
python scripts/diagnose.py --nc_file <wout.nc> \
  --skip_ripple --qi_r2 \
  --qi_r2_nphi 101 --qi_r2_nalpha 8 --qi_r2_nbj 101 \
  --qi_r2_mpol 8 --qi_r2_ntor 8 \
  --output_dir runs/r2_scan_<case>
```

Observed behaviour:

- R2 separates the known good, intermediate, and poor QI cases consistently
  with the simple QI diagnostic.
- The compressed `(surface, alpha)` R2 residual (`--qi_r2_arr_out` off) and
  full `(surface, alpha, phi)` residual (`--qi_r2_arr_out` on) produce the same
  `f_QI_R2` after normalisation; use compressed mode for optimisation.
- A higher diagnostic resolution
  `nphi=201, nalpha=16, nBj=201, mpol=12, ntor=12` changed `f_QI_R2` by only a
  few percent on tested cases, so the lower setting is adequate for ranking.
- Current representative values at `s=[0.4, 0.5, 0.6]`:
  good cases are around `f_QI_R2 ~ 5.7e-5`, intermediate cases around
  `1.4e-4`, and poor cases around `1.8e-3`.

Practical recommendation: start with `--w_qi_r2 0.05` to `0.2` alongside the
simple QI target. Do not make R2 the sole QI objective until a short optimisation
scan shows it improves R2 without degrading max-J pass ratio, B_min growth,
well depth, or ripple.

## Git Hygiene

- Commit code and small documentation only.
- Keep `artifacts/legacy_root/`, `artifacts/old_archive/`, and `runs/*`
  ignored unless a small summary file is intentionally added.
- Before pushing, avoid storing GitHub tokens in `.git/config`; use a clean
  remote URL and credential storage instead.
