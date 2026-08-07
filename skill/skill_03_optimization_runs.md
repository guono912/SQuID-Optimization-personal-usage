# Skill 03 - Optimization Runs

## Launch from a reviewed config

```bash
cd /home/guozx/SQuID
PY=/home/guozx/fusion_env/bin/python
$PY scripts/opt/optimize.py --input_parameter <campaign>/configs/<stage>.json
```

The resolved configuration is written to
`<run_dir>/input_parameter.resolved.json`. This file, not the template, is the
authority for what ran.

Current presets are `core`, `core_r2_assist`, `maxj_repair`, `engineering`,
`edge_bal_repair`, and `edge_bal_direct`. Presets are starting points, not
validated universal recipes.

## Minimum config content

Declare:

- parent `nc_file` or `vmec_input_file` and `run_dir`;
- backend and preset;
- numerical resolution, evaluation cap, checkpoint cadence, finite-difference
  step, free mode limits, and shape bounds;
- active weights and their physical purpose;
- target values, hard guards, and radial protocols;
- pressure/current/iota policy;
- a one-sentence hypothesis for the run.

## Choose degrees of freedom deliberately

`max_dofs` is only a count. Inspect the printed free-DoF list and confirm the
intended `(m,n)` families are present. Use `free_m_max/free_n_max`, shape
anchors, and `dof_bound_frac` to define the local search.

Use low-order modes first for a local repair. Open higher modes only when a
measured blocker requires them and geometry tails are guarded.

## Calibrate weights instead of copying them

Objective scales depend on sampling, normalization, residual-vector length,
and code version. Historical weights are evidence about one run, not defaults.

For a new basin:

1. evaluate the parent objective decomposition;
2. perturb one control or mode family;
3. measure target and guard responses;
4. choose weights that make the intended response visible without overwhelming
   hard guards;
5. run a capped short probe before a long stage.

Keep one primary objective idea per probe. MHD sign, topology, beta/current
identity, and numerical convergence should be hard guards where supported.

## Monitoring

`history.csv` is a steering log. Track:

- whether the named blocker moves;
- beta, current, iota profile/shear, and rational crossings;
- flux-normalized Mercier and components on the declared half-grid mask;
- QI/max-J/Bmin/ripple or engineering proxy relevant to the hypothesis;
- tail statistics, not only means or maxima;
- VMEC failures and elapsed time.

Checkpoint every few evaluations in a new basin. A final solver state after an
evaluation-cap exception is not necessarily an accepted optimizer step.

## Stop rules

Stop and record the response when:

- the blocker is flat across the evaluation budget;
- improvement requires an MHD/topology/numerical failure;
- geometry develops a non-convergent local tail;
- the same failure survives a guarded and a wider parameterization;
- the run drifts to another beta/current/profile policy.

After two unproductive parameterizations, switch profile, control family, or
seed instead of retuning weights indefinitely.

## Postcheck

Never rank directly from low-resolution history. Rebuild selected checkpoints
at high radial and spectral resolution, then run Skills 04, 05, 08, and 09 as
applicable. Record before/after values with identical protocols.
