# Skill 03 — Configuring and Running Optimization Stages

## Launching a stage

```bash
cd /home/guozx/SQuID
python scripts/optimize.py --input_parameter runs/<campaign>/configs/<stage>.json
```

Config JSON groups: `name, mode, nc_file, backend, run_dir, note`,
`numerics{}`, `weights{}`, `targets{}`, `coil_proxy{}`, optional `gates{}`.
The resolved argument set is always written to
`<run_dir>/input_parameter.resolved.json` — use that for comparisons.

Modes (presets, overridable): `core`, `core_r2_assist`, `maxj_repair`,
`engineering`, `edge_bal_repair`, `edge_bal_direct`.

## Numerics defaults that worked in this campaign

```json
"numerics": {
  "maxiter": 4, "max_total_evals": 30-40, "max_dofs": 12-14,
  "num_alpha": 6, "num_pitch": 24, "num_surfaces": 4, "ns_vmec": 64,
  "abs_step": 5e-5, "checkpoint_every": 3, "dof_bound_frac": 0.012
}
```

- `checkpoint_every` 2-3 for ANY new basin (a previous stage lost its best
  point at an uncheckpointed eval).
- `dof_bound_frac`: 0.003-0.006 guarded repair, 0.006-0.012 normal step,
  larger only for deliberate basin hopping.
- `max_dofs <= 24`; low-order modes move first.

## Weight guidance (per branch type)

Physics-first: `w_qi 0.8-1.0`, `w_maxj ~2`, `w_bmin ~1.2`, `w_well ~20`
(target_well 0.04), `w_mercier_margin 20-35` (margin target 0.04-0.08),
`w_iota` per goal, weak coil proxy (`w_coil_proxy_bn <= 200`).

Coil repair: raise `w_coil_proxy_bn` to 300-400 with
`coil_proxy_bn_max_target = 0.0034`, keep MHD terms on, keep
`w_shape_anchor` strong to protect physics.

Known traps:

- `w_grad_s` strong (>= 2e-3) destroys QI/ballooning; safe 3e-4 - 8e-4.
- `w_ballooning` (internal DESC ballooning penalty) is on a DIFFERENT
  numerical scale than the external gate (stage07: internal residual
  ~0.078 while external lambda ~1e-6). Do not expect internal ballooning
  optimization to move the external gate; prefer gate-in-the-loop
  checkpoint selection.
- An iota target only acts outside `iota_tolerance`; combined with a strong
  `w_shape_anchor` and small `dof_bound_frac` it can be a no-op
  (stage02 desc_seed: iota_edge moved < 0.001 toward a 0.04 push).
  To actually move iota: tolerance <= half the desired change, anchor
  <= ~300, and enough evals/DoFs (or use `--prescribe_iota` for a
  controlled experiment only).

## Monitoring a run

- `history.csv` columns include per-eval `f_QI, f_maxJ, beta, iota_ax/edge,
  well, dmerc proxy, coil proxy bn/k, aspect, wall time`. Penalties are
  steering logs ONLY — never acceptance criteria.
- Run the external gate watcher in parallel:

  ```bash
  python scripts/gate_checkpoints.py --run_dir runs/<campaign>/<stage> --watch
  ```

- Stop a stage when 20-30 evals produce no new gate-passing improvement,
  or when only low-iota/high-ripple points improve engineering metrics.

## After a run

1. Gate ALL durable checkpoints (not just the final point).
2. Select by external gates + fixed-protocol coil contour review
   (Skill 04/05). If `pdrot` and true contour curvature conflict, true
   contour curvature wins.
3. Write a STATUS.md entry: config, seed, best checkpoints, gate numbers,
   decision, blocker.
