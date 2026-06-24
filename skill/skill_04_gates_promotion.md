# Skill 04 — External Gates and Promotion Rules

## MHD gate (the promotion gate)

```bash
python scripts/mhd_gate.py --wout <wout.nc> --output_dir <gate_dir> [--L 6 --M 6 --N 6]
python scripts/gate_checkpoints.py --run_dir <stage_dir> [--watch]
```

What it computes: VMEC sanity (ier, beta, iota, DMerc over s in
[0.1, 0.97], well depth), DESC force-balance re-solve (L=M=N=6), DESC
ballooning lambda on rho = {0.1,...,0.95} x 8 alphas (nturns=2,
nzetaperturn=80, 5 zeta0), Nemov effective ripple on 9 surfaces.

### Interpreting the verdict — IMPORTANT

`verdict: PASS` only means: no negative DMerc points, `n_unstable <= 0`
ballooning issues flagged, eps_eff <= 0.05, well >= 3%. It does NOT check:

- `DMerc_min >= 0.08` (campaign promotion threshold) — check the JSON field
  `vmec.DMerc_min_gated` yourself. Exploratory points may proceed with
  `>= 0` but must be labeled exploratory.
- Edge ballooning beyond rho = 0.95. Route1's limiting mode lived at the
  rho=0.95 envelope; before promoting a marginal candidate, rerun the gate
  with an extended rho list including 0.975 and 0.99 and >= 12 alphas.
- Margin quality. Treat `lambda_max in (-2e-5, 0]` as MARGINAL, not stable;
  pressure noise of 0.001% beta flips such points (route1 1.411 vs 1.412).

### Promotion checklist (all must hold)

1. `ier_flag = 0`, beta within 0.02% absolute of branch target (no silent drift).
2. `DMerc_min_gated >= 0.08`, `DMerc_negative_count = 0`.
3. Ballooning `n_unstable = 0` AND `lambda_max <= -2e-5` (margin, not zero),
   confirmed on the extended-edge rho grid.
4. `ripple_peak <= 0.025` (prefer <= 0.015 on the desc_seed branch where it
   is cheaply available).
5. Boozer high-B topology visually acceptable (`viz_report.py`), max-J not
   edge/high-field concentrated.
6. Fixed-protocol coil review acceptable (Skill 05).
7. Gates rerun after any beta/profile/resolution/scale change.

## Beta-ceiling scan (pressure-only)

To find the ballooning ceiling of a FIXED boundary: copy the input, scale
`PRES_SCALE` to a beta grid (e.g. 1.40/1.43/1.50/1.60), `xvmec` each, gate
each. Record the first WARN. Remember the result is a property of that
boundary; it is NOT the branch ceiling — co-evolving the boundary while
ramping pressure can move it (route1 lesson: pressure-only ceiling 1.411%,
and small low-order probes did not move it; a different repair
parameterization is required).

## Diagnosis (cheaper, not a gate)

```bash
python scripts/diagnose.py --nc_file <wout.nc> --output_dir runs/diagnose_<case> --plot
python scripts/viz_report.py ...   # HTML, Boozer maps, Newcomb, ITG proxy
```

Use `diagnose.py` JSON fields (`core.f_QI`, `maxj_interval_pass_ratio`,
`stability.mercier_summary`, `geometry.iota_scan`, `ripple`) for steering;
raw viz ballooning is a diagnostic and is overridden by the external gate.

## Bookkeeping

Gate outputs: `<gate_dir>/<stem>_<timestamp>_gate.json` + DESC `.h5`.
Batch summary: `<stage>/gate_results/gate_summary.csv`. Copy the decisive
numbers (beta, DMerc_min, lambda_max, n_unstable, ripple_peak, verdict)
into STATUS.md when a decision is made.
