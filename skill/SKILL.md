# SQuID Agent Workflow

This is the canonical operating procedure for an Agent evaluating or
optimizing a stellarator equilibrium with SQuID.

The objective is not to minimize every scalar. The objective is to identify
the current blocker, choose a physically meaningful control, measure the
response, and retain only candidates that survive independent checks.

## 1. Establish provenance before computing

For every input equilibrium, record:

- absolute path and SHA256;
- source type: published target, fixed-boundary result, free-boundary result,
  coil-return equilibrium, vacuum point, or finite-beta operating point;
- NFP, `ns/mpol/ntor`, major/minor radius, aspect ratio, B0 proxy, beta;
- `NCURR`, `CURTOR`, current profile, pressure profile, and `PHIEDGE`;
- parent equilibrium and transformation used to create this file.

Do not treat two files with similar names as the same configuration. See
Skill 01 for paths and the run manifest rules.

## 2. Create an isolated run

Use `/home/guozx/runs/YF_0` for the compact device and
`/home/guozx/runs/YF_1` for the larger device.

```text
<project>/<campaign>/<run_id>/
  equilibrium/
  diagnostics/
  viz/
  logs/
  manifest.json
  README.md
```

All solver output must stay in this tree. Never calculate from the repository
root and later search for whichever `wout.nc` appeared.

## 3. Run the baseline diagnosis

```bash
cd /home/guozx/SQuID
PY=/home/guozx/fusion_env/bin/python

$PY scripts/diag/diagnose.py \
  --nc_file <wout.nc> \
  --output_dir <run>/diagnostics \
  --plot
```

Add `--extended` when ITG diagnostics are needed. Add `--ae` only after the
equilibrium passes cheaper sanity and MHD checks. Use `--skip_ripple` only for
explicitly labeled fast screening.

Generate the standard visual report for serious candidates:

```bash
$PY scripts/viz/viz_report.py \
  --nc_file <wout.nc> \
  --output_dir <run>/viz \
  --nfp <NFP>
```

The visual report is diagnostic. Force-balanced ballooning and promotion
decisions come from Skill 04.

## 4. Read metrics in this order

### A. Numerical existence and geometry

Reject or repair first when VMEC did not converge, force residuals are not
small, flux surfaces are broken, the boundary has cusps/self-intersections, or
results change sign under modest resolution refinement.

Fixed-boundary VMEC is a real ideal-MHD force-balance solve for the prescribed
boundary and profiles. It is not proof that a practical coil set reproduces
that boundary. A coil-return/free-boundary equilibrium is the later test of
that realization.

### B. Macroscopic stability

Inspect:

- `Phi_edge^2 * DMerc` minimum, negative count, radial location, and all four
  components on the declared half-grid interval;
- magnetic well profile;
- force-balanced ballooning `lambda_max` and unstable-point count;
- Newcomb/related ideal-MHD diagnostics when available.

Mercier sign is a hard fact; a positive margin is protocol-specific. Never
reuse raw or historical B0-scaled thresholds. Follow Skill 09.

### C. Transform and resonance exposure

Inspect the complete `iota(s)` profile, shear, sign changes, and low-order
rational crossings. Keep nominal NFP-compatible resonances separate from the
broader set that can be driven by symmetry-breaking coil or assembly errors.
A reduced denominator or `q<=N` crossing count is not a resonance-risk score.
The deciding follow-up is resonant normal-field response, local shear,
island-width/overlap analysis, and Poincare topology using the actual
coil/mgrid field. Follow Skill 10.

### D. Confinement and transport

Inspect effective ripple by radial surface, Boozer `|B|` topology, QI/max-J
diagnostics, and ITG/AE proxies. Do not compare ripple values produced by
different implementations or surface grids without labeling the protocol.

For a low-field pure-D device, alpha-particle confinement is not a design
gate. Thermal-particle confinement, finite-orbit-width risk, bootstrap/current
behavior, and transport-compatible profiles remain relevant to long pulses.

### E. Engineering exposure

Boundary pdrot and curvature identify difficult regions but do not establish
coil feasibility. Use current-potential/REGCOIL-like metrics and filamentary
coil-return results as higher-fidelity evidence. Keep protocols fixed when
comparing candidates. See Skill 05.

## 5. Diagnose the blocker before choosing an optimizer

| Observed problem | First response |
| --- | --- |
| Reconstructed iota/current/beta differs from parent | Fix input conversion or profile preservation; do not optimize yet |
| Mercier/well fails only at low beta | Scan pressure level and profile shape at fixed boundary |
| Ballooning limits the high-beta end | Pressure/profile ladder, then guarded boundary continuation |
| iota crosses dangerous rationals or shear collapses | Scan current/profile response; protect the whole iota profile |
| Ripple/QI is poor while MHD is sound | Core/QI repair with MHD and topology guards |
| max-J/Bmin is the main defect | `maxj_repair` response probe |
| Physics is acceptable but coil proxy is poor | Late engineering or coil-aware branch with physics hard guards |
| Several unrelated metrics are bad and short probes are flat | Switch seed family; do not keep retuning one basin |
| Coil-return physics collapses | Optimize robustness around the target or repair the returned boundary; compare both against the same coil perturbation ensemble |

Change one physical idea per short response probe. A large weighted sum hides
which control produced the apparent improvement.

## 6. Configure a short optimization

Prefer a reviewed JSON file over a long command line:

```bash
$PY scripts/opt/optimize.py --input_parameter <campaign>/configs/<stage>.json
```

Choose the closest preset, then override it deliberately:

- `core`: QI/max-J/Bmin and basic shape control;
- `core_r2_assist`: core optimization with a light R2 QI diagnostic;
- `maxj_repair`: targeted max-J/Bmin response;
- `edge_bal_repair` or `edge_bal_direct`: edge/ballooning-oriented work;
- `engineering`: late engineering proxy work, never an early universal mode.

For every run:

- inspect the printed free-DoF list;
- use explicit `run_dir`;
- bound the shape step and keep a parent-profile/iota anchor;
- checkpoint often enough to retain the best response;
- cap evaluations for a probe;
- record the fully resolved parameters.

Do not copy historical weights blindly. Objective magnitudes depend on grids,
normalization, and residual-vector length. Calibrate weights from the parent
objective decomposition and one-variable response probes.

## 7. Monitor and stop intelligently

Use `history.csv` to answer whether the targeted blocker moves and which
guardrail pays for it. Stop when:

- the target changes only at numerical-noise scale;
- the improvement is purchased by a hard MHD/topology failure;
- the optimizer reaches an evaluation cap without accepting a step;
- two parameterizations show the same harmful local response.

An evaluation-cap state is not an accepted optimum. It may be postchecked as a
response sample, but it must not be promoted as the optimizer result.

## 8. Rebuild and independently evaluate survivors

For each survivor:

1. rebuild at `ns >= 128` (often 192 for marginal Mercier cases);
2. verify VMEC convergence and force residuals;
3. rerun the same diagnostic protocol as the parent;
4. run the external MHD gate and corrected viz report;
5. check pressure/current/profile perturbations around the operating point;
6. test at a second spectral/radial resolution;
7. run the fixed-protocol engineering/coil proxy comparison.

Low-resolution optimization values are for direction finding, not promotion.

## 9. Decide: promote, retain as control, or reject

Promote only when all hard gates pass and the candidate improves the intended
blocker without an unpriced new risk. Keep useful trade-off points as controls.
Reject failed probes with a short reason so they are not repeated.

Every decision record must include parent hash, config, code revision,
resolution, metric protocols, key before/after values, and the next action.

## 10. Required specialist reading

- Any Mercier number: [Skill 09](skill_09_mercier_normalization.md).
- Any rational-crossing, iota-window, or island-risk claim:
  [Skill 10](skill_10_iota_resonance_policy.md).
- Any B/beta/pressure/current/iota change:
  [Skill 08](skill_08_finite_beta_lowB_continuation.md).
- Any seed construction or NFP reinterpretation:
  [Skill 02](skill_02_seed_generation.md) and
  [Skill 07](skill_07_seed_portfolio.md).
- Any promotion claim: [Skill 04](skill_04_gates_promotion.md).
- Any coil-feasibility claim: [Skill 05](skill_05_coil_workflow.md).
- Before trusting surprising output: [Skill 06](skill_06_pitfalls.md).
