# Skill 03 - Optimization Runs

## Launch from a reviewed config

```bash
PY=python  # From the checkout with the solver environment active
$PY scripts/opt/optimize.py --input_parameter <campaign>/configs/<stage>.json
```

The resolved configuration is written to
`<run_dir>/input_parameter.resolved.json`. It records the settings at launch. With `backend=auto`, confirm the selected backend
in the log. Values derived from the equilibrium can be set after this file is written.

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
intended modes are present. VMEC uses `free_m_max/free_n_max` to restrict
the mode search and can apply hard coefficient bounds. DESC uses its own mode
order and treats `dof_bound_frac` as a soft penalty.

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
- flux-normalized Mercier and components, with an independent radial-mask
  check as described in [Skill 09](skill_09_mercier_normalization.md);
- QI/max-J/Bmin/ripple or engineering proxy relevant to the hypothesis;
- tail statistics, not only means or maxima;
- VMEC failures and elapsed time.

The history columns differ by backend. Use independent diagnostics for missing
quantities such as ripple. Preserve checkpoints for promising trials.

## Read the termination record

| State in `run_status.json` | Action |
| --- | --- |
| `converged` and `accepted: true` | Rebuild VMEC's `input.squid_optimized` or read DESC's `wout_desc_optimized.nc`. For DESC, `candidate_evaluation` identifies the exported point. |
| `initial_only` | Inspect the baseline. A successful `--maxiter 0` run exits 0 with `accepted: false`. |
| `evaluation_cap_recovered_start` / `evaluation_cap` | The total evaluation budget stopped VMEC / DESC. VMEC saves its pre-solve boundary as `input.squid_unaccepted_cap`. |
| `solver_evaluation_limit` / `iteration_limit` | The solver exhausted its own allowance. Review the history before allocating more work. |
| `candidate_rejected` / `solver_failed` | Inspect the solver message, available rejection reasons and diagnostic output. |

Budget stops and rejected candidates exit 2. An interrupted process can leave
no status file. VMEC's solver allowance is `maxiter * (n_free + 1)`; its total
evaluation count also includes finite-difference probes. DESC counts the
initial point toward its total allowance and limits simplex iterations with
`maxiter`. The final history row can be a discarded trial.

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


## Backend selection

VMEC uses SIMSOPT residuals with SciPy nonlinear least squares. DESC solves force balance with DESC and
reads its exported wout with SIMSOPT for the core SQuID terms. DESC needs a
wout as input. Auto selection prefers VMEC when its extension is available.
The CLI rejects active objectives and hard gates exclusive to the other
backend. For example, high-B topology uses DESC and coil-proxy weights use VMEC.
