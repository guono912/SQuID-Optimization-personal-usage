# Agent Optimization Guide

This file is the handoff guide for future agents working on this SQuID
stellarator optimization campaign. It summarizes the active objectives,
accepted tradeoffs, useful branches, gates, and recommended next experiments.

The detailed chronological log for the coil-aware W7X branch is:

- `runs/w7x_modified_coil_optimized/STATUS.md`

## Current Top-Level Decision

The next major plan is to rerun the full design chain:

1. start from W7X-derived geometry,
2. truncate/recompose the VMEC boundary into the working nfp=4 family,
3. optimize at several finite-beta targets,
4. compare physical performance across beta layers,
5. keep enough engineering constraint to show coil manufacturability is not
   obviously impossible.

Keep the main physics archive unchanged:

- `runs/stage2_eval130_R2p5`
- `runs/stage2_eval130_R2p5_ns128/wout_squid_optimized_ns128.nc`

This configuration is still the main finite-beta physics reference. Do not
overwrite it or promote engineering-only descendants over it without explicit
user approval.

Current main-reference approximate VMEC values:

- `Rmajor_p ~= 2.50 m`
- `Aminor_p ~= 0.233 m`
- `aspect ~= 10.73`
- `betatotal ~= 0.01883` or `1.88%`
- `b0 ~= 2.60 T`
- VMEC `bmnc max ~= 2.81 T`

The latest low-curvature branch is useful for coil engineering tests, not as a
QI/max-J replacement:

- `runs/w7x_modified_coil_optimized/stage2_mhd_hard_coil_push`
- best engineering checkpoint:
  `checkpoints/wout_squid_eval_000032.nc`
- conservative engineering checkpoint:
  `checkpoints/wout_squid_eval_000024.nc`

## Active Scientific/Engineering Priorities

The user currently wants a practical finite-beta configuration that can be made
coil-realizable. The priority stack depends on branch type.

For the next W7X truncation/recomposition campaign, the primary comparison is
physical performance as a function of beta. Engineering constraints should be
kept as guardrails, not allowed to dominate the physics search. A point with
excellent coil metrics but poor QI/max-J/topology is only an engineering test
point.

For physics candidates:

1. Finite-beta MHD stability.
2. QI topology and high-field `|B|` contour quality.
3. max-J quality.
4. Acceptable effective ripple and iota.
5. One representative coil-manufacturability metric must be acceptable.

For coil-forward engineering test points:

1. Finite-beta MHD stability.
2. Coil contour feasibility and boundary curvature.
3. `Bmax <= 3 T` and coil length `< 5 m`.
4. QI/max-J/iota/ripple are diagnostics, not hard gates.

Do not confuse these two branch types. `stage2_mhd_hard_coil_push/eval032` is
excellent as a low-curvature engineering test point, but it is not a physics
candidate because QI/max-J and raw viz ballooning are poor.

## Metric Priority Order

Use this ordering when comparing candidate configurations. First reject points
that fail mandatory gates, then rank the survivors.

Overall order:

1. numerical and geometry sanity
2. MHD stability hard gate
3. QI topology
4. max-J behavior
5. effective ripple / neoclassical proxy
6. iota and shear reasonableness
7. representative coil manufacturability proxy
8. ITG bad-curvature flux-compression proxy
9. boundary geometry proxy

Details:

- Numerical and geometry sanity: VMEC/DESC convergence, normal flux surfaces,
  no obvious spikes, cusps, broken surfaces, or resolution-dependent failures.
- MHD stability: `DMerc`, force-balanced ballooning, Newcomb, and branch beta
  target. This is pass/fail and cannot be traded away.
- QI topology: Boozer `|B|` contour structure, especially high-field
  `B/Bmax = 0.75-0.98` topology. For QI, constant-`|B|` contours should close
  poloidally, not toroidally, and `Bmax` contours should be straight at the
  field-period boundaries. Wrong high-B topology is usually hard to repair
  later.
- max-J: reject high-field or edge-concentrated max-J violations. Some
  low-impact degradation is acceptable only if QI topology and MHD remain good.
- Ripple: use effective ripple/ripple peak and radial structure as the main
  neoclassical transport proxy.
- iota/shear: avoid the known low-iota coil-friendly basin when choosing a
  physics candidate.
- Coil proxy: for physics candidates, use true coil contour `curvature_max` as
  the representative engineering metric; record length, `Bmax`, `Bn_max/|B|`,
  `curvature_rms`, and clearances.
- ITG proxy: use as a tie-breaker or warning, not as a reason to sacrifice
  MHD/QI/max-J.
- Boundary geometry: use `pdrot`, principal curvature, and curvature maps as
  engineering risk diagnostics and local repair guidance.

## Definition of "Good Enough to Use"

MHD stability is mandatory and cannot be traded away. For other metrics, the
goal is not mathematical perfection; stop when the configuration is good enough
to support the next design step.

Do not use raw optimizer penalty values as universal accept/reject criteria.
Values such as `f_QI`, `f_maxJ`, high-B topology penalties, and curvature
penalty terms depend on resolution, grids, normalization, Boozer fitting
choices, and implementation details. They are optimizer logs, not physical
acceptance metrics.

Use physical diagnostics and independent proxies for decisions:

- physical diagnostics: beta, `Bmax`, iota profile, Mercier/Newcomb/ballooning,
  effective ripple, magnetic well, Boozer `|B|` contour topology
- transport proxy: ITG bad-curvature flux-compression profile and map
- coil proxy: current-potential `Bn_max/|B|`, true contour curvature, coil
  length, coil-plasma distance, coil-coil distance
- geometry proxy: boundary `pdrot`, principal curvature statistics, and
  curvature maps

Optimizer penalties may be used only to steer a run or compare two checkpoints
computed with the exact same code, grid, and normalization.

Physics-use criteria, after all MHD gates pass:

- beta: at or near the branch target; for the high-beta branch, `1.8-2.0%`.
- QI: Boozer `|B|` contours should show the intended quasi-isodynamic topology.
  Use contour plots and high-B topology proxies, not only scalar QI penalties.
- max-J: maximum-J behavior should be visually and radially consistent with the
  design goal. Reject points where violations are concentrated in high-field
  regions or near the plasma edge.
- high-B topology: `B/Bmax = 0.75-0.98` contours should close poloidally on
  the Boozer map; reject points with dominant toroidal closure, fragmented
  islands, or broken high-field contours. Also check that the maximum-`|B|`
  contours are approximately straight at the field-period boundaries.
- effective ripple: prefer `ripple_peak <= 0.025`; usable up to `0.03` if QI
  and MHD are strong.
- iota: keep within a sensible branch range; large drops into the known
  low-iota coil-friendly basin should not be accepted for physics candidates.
- ITG proxy: record and prefer lower values, but do not reject a point solely
  on ITG proxy unless it is much worse than the archive at the same beta.
- convergence check: if a proxy is used in a decision, rerun or compare it at a
  second resolution/grid and make sure the conclusion is not a grid artifact.

Engineering proxy gate for physics candidates:

- Use true coil contour `curvature_max` as the single representative
  manufacturability metric.
- Compare `curvature_max` only at fixed contour-generation settings:
  offset `0.35a`, regularization `lambda=1e-8`, and the same coil extraction
  resolution.
- For a physics candidate, the curvature metric should be no worse than the
  `stage2_eval130_R2p5_ns128` reference under this identical protocol. A modest
  improvement is enough; do not optimize this metric at the expense of physics.
- If a point is physically superior but slightly worse in curvature, keep it as
  a physics candidate but record the engineering warning and run a guarded
  curvature repair.

Rationale: `Bn_max/|B|` and current-potential proxies are useful, but previous
runs showed they can improve while actual coil contours remain twisted. True
contour curvature is the best single scalar guardrail for coil manufacturability
available in this workflow, provided the comparison protocol is fixed.

## Hard Gates

Use these as default finite-beta MHD gates unless a branch explicitly changes
them:

- VMEC must converge.
- `DMerc_min >= 0.08` for production-like comparisons.
- `DMerc_negative_count = 0`.
- force-balanced ballooning `n_unstable = 0`.
- force-balanced `ballooning_lambda_max <= 0`.
- Newcomb metric should be positive where available.
- beta should not silently drift away from the branch target.

For early exploratory screening, `DMerc_min >= 0` can be tolerated, but do not
promote such points.

Important caveat: raw `viz_report.py` ballooning values can differ from the
force-balanced outer gate. Prefer the force-balanced gate for accept/reject
decisions, and keep raw viz ballooning as a diagnostic.

## When to Use Force Balance

Use force balance at three different levels, with different costs.

Inner optimizer:

- Use VMEC fixed/free-boundary equilibrium evaluation for most objective calls.
- Keep explicit `w_force_balance` weak or off unless a branch is DESC-centered.
- Do not run expensive force-balanced DESC solves on every exploratory
  checkpoint unless the run is already near acceptance.

Checkpoint gate:

- Run force-balanced MHD gates every `3-5` checkpoints in serious repair runs,
  and every `10-20` evals in broad scans.
- Use the force-balanced result for pass/fail decisions on ballooning and
  Newcomb. This should override raw viz ballooning if they disagree.
- Required before calling a checkpoint "MHD-stable":
  `DMerc_negative_count = 0`, positive DMerc margin, force-balanced
  `ballooning_n_unstable = 0`, force-balanced `lambda_max <= 0`, and nonnegative
  Newcomb where available.

Final validation:

- Always run force balance on the best checkpoint from each beta layer before
  comparing branches.
- Always rerun force balance after changing beta, pressure/current profile,
  Fourier resolution, radial resolution, or global scale.
- If a point only passes MHD before force balance, label it as "raw VMEC/viz
  candidate", not as an accepted configuration.

## Engineering Gates

Current engineering constraints and proxies:

- coil length `< 5 m`
- `Bmax <= 3 T`
- improve true coil contour curvature, not only current-potential proxy
- preserve enough coil-plasma and coil-coil clearance

For physics-first branches, do not optimize all engineering metrics at once.
Use true coil contour `curvature_max` as the main engineering accept/reject
metric, while recording length, `Bmax`, `Bn_max/|B|`, `curvature_rms`, and
clearances.

Current contour-proxy settings used in recent comparisons:

- winding surface offset: `0.35a`
- regularization: `lambda = 1e-8`
- recent low-curvature reference values around the current branch:
  - `Bn_max/|B| <= 4.28e-3`
  - `curvature_max < 8.75 1/m`
  - `curvature_rms <= 4.04 1/m`

These numbers are not universal physical thresholds. They are useful for
same-protocol comparisons among the existing stage2-derived branches.

`stage2_mhd_hard_coil_push/eval032` achieved approximately:

- `Bn_max/|B| = 4.113e-3`
- `curvature_max = 8.723 1/m`
- `curvature_rms = 3.987 1/m`
- `min coil-plasma distance ~= 0.0574 m`
- `min coil-coil distance ~= 0.868 m`
- proxy contour length mean `~= 2.30 m`

If a collaborator's real optimized coils exceed 5 m, treat that as a mismatch
between the current contour proxy and the real coil workflow. Calibrate against
their actual coil output before changing plasma scale.

## Radius Scaling Warning

Do not shrink the `R=2.5 m` configuration just because a length constraint is
mentioned. Under uniform geometric scaling by factor `s`:

- lengths scale as `s`
- coil-plasma and coil-coil distances scale as `s`
- mean/principal curvature and `pdrot` scale as `1/s`
- Gaussian curvature scales as `1/s^2`

So shrinking from `2.5 m` to `2.0 m` (`s=0.8`) shortens coils but worsens
curvature and clearance by about `25%`. Since the current proxy contour length
is already under `5 m`, shrinking is not the first move. Prefer a scale sweep
only after real coil outputs require it:

- `s = 1.00, 0.95, 0.90, 0.85, 0.80`

For each scale, recompute MHD, coil contour metrics, `Bmax`, length, curvature,
clearances, and `Bn_max/|B|`.

## When to Optimize at One Size and Rescale Back

Using a temporary scale, such as optimizing near `R=1.8 m` and later rescaling
back to `R=2.5 m`, can be useful, but it is a numerical/search tactic rather
than a physical shortcut. Use it only when it helps the optimizer escape a bad
local basin or changes the relative strength of engineering proxies in a useful
way.

Appropriate uses:

- continuation experiment: test whether a boundary shape family survives across
  scale, beta, and field targets
- search tactic: temporarily make curvature/coil proxies more visible to the
  optimizer, then rescale back and re-equilibrate
- engineering sweep: determine whether coil length, clearance, curvature, and
  `Bmax` constraints are scale-limited

Not appropriate:

- do not use scale tricks to claim MHD stability without rerunning finite-beta
  equilibrium and force-balanced gates at the final target size
- do not accept a point whose coil metric is good only at the temporary scale
- do not assume QI/max-J/high-B topology is unchanged after finite-beta
  re-equilibration at the final size

Mandatory protocol for scale tricks:

1. Save the pre-scale seed and record the scale factor.
2. Optimize at the temporary size with the intended beta/field convention
   explicitly recorded.
3. Rescale to the final target size.
4. Rerun VMEC at the final size and target beta/pressure profile.
5. Rerun force-balanced MHD gates, Boozer/QI topology, max-J, ripple, ITG proxy,
   coil contour metrics, `Bmax`, length, and clearances.
6. Compare the final-size result to the original same-size baseline, not to the
   temporary-size intermediate.

For the current campaign, scale tricks are secondary. The preferred first
campaign is the W7X truncation/recomposition beta-layer study at the target
engineering scale. Use scale continuation only if the target-size branches
plateau or if real coil outputs show the `5 m` length or `3 T` field limits
cannot be met otherwise.

## Objective Terms and How to Use Them

Available optimization terms in `scripts/optimize.py` include:

- QI: `w_qi`
- R2 QI assist: `w_qi_r2`
- max-J: `w_maxj`
- B-min / mirror / aspect: `w_bmin`, `w_mirror`, `w_ar`
- beta and iota anchors: `w_beta`, `w_iota`
- ITG grad-s proxy: `w_grad_s`
- Mercier margin: `w_mercier`, `w_mercier_margin`
- ballooning penalty: `w_ballooning`
- force balance: `w_force_balance`
- high-B QI topology: `w_highB_topology`
- QSS-style coil proxy: `w_coil_proxy_bn`, `w_coil_proxy_k`,
  `w_coil_proxy_phi`
- shape anchor: `w_shape_anchor`
- LCFS boundary curvature proxy: `w_boundary_curvature`

Use `w_grad_s` only as a diagnostic or weak term unless the branch explicitly
prioritizes transport proxy optimization. Past runs showed that strong
`w_grad_s ~= 0.002-0.003` pulls the search into a basin that sacrifices QI and
ballooning. Safer range:

- `w_grad_s = 3e-4` to `8e-4`

Use hard MHD gates when `w_grad_s` is active:

- `ballooning_n_unstable = 0`
- `ballooning_lambda_max <= 0`
- `DMerc_min >= 0.08`

For coil proxy terms:

- `w_coil_proxy_bn` can reduce current-potential `Bn`.
- `w_coil_proxy_k` can lower current-density proxy.
- These do not guarantee real modular coils are simple.
- Always follow with contour/coil geometry gates.

For boundary curvature:

- use it as a weak source-term regularizer
- target `-k2`, `H`, `pdrot_aw`, and `pdrot_max`
- do not expect `pdrot` to be fixed quickly
- it helps true contour curvature, but high `pdrot_max` can remain

Recent curvature-penalty settings that produced useful local moves:

- `boundary_pdrot_aw_target ~= 1.45`
- `boundary_pdrot_max_target ~= 16`
- `boundary_k2_abs_max_target ~= 98-100`
- `boundary_H_abs_max_target ~= 49-50`

Treat these as optimizer tuning parameters, not physical acceptance criteria.
Check any accepted point with independent contour geometry and MHD gates.

## Literature-Backed Methods Worth Using

The working plan should use the following methods from the QI/SQuID and
coil-proxy literature. Keep them separated by cost and reliability.

Use in the inner or checkpoint loop:

- QI topology screening from Boozer `|B|` contours. Goodman-style QI design is
  built around three geometric conditions: constant-`|B|` contours close
  poloidally, `Bmax` contours are straight at field-period boundaries, and
  bounce distance is approximately field-line independent. In this project,
  high-field topology is the most important part to screen visually/proxy-wise.
- Maximum-J proxy. The maximum-J property is strongly tied to favorable trapped
  particle curvature, fast-particle confinement, MHD behavior, and trapped
  particle mode suppression. Use it as a physical ranking metric after QI
  topology and MHD pass.
- Magnetic-well / Bmin radial-growth diagnostics. These are useful low-cost
  support metrics for maximum-J and MHD trends, but not substitutes for the
  full max-J and MHD gates.
- ITG bad-curvature / flux-compression proxy. PRX Energy 2024 and critical
  gradient work both emphasize the role of bad curvature, local shear,
  connection length, and flux-surface expansion. In this project, use ITG as a
  proxy plot and weak/tie-breaker objective, not as a hard inner objective.
- Quasi-single-stage coil proxy. Use `Bn_max/|B|` and current-potential
  quantities to bias away from coil-impossible plasma shapes, but always verify
  with true contour curvature, distances, and eventually the collaborator's real
  coil workflow.
- Boundary surface geometry proxy. The 2026 coil non-planarity study supports
  using principal-direction rotation rate (`pdrot`) and principal-curvature
  statistics as real coil-hardness proxies. Use them for repair guidance and
  risk screening, not as dominant physics objectives.

Use as outer-loop or final validation:

- Force-balanced ballooning and Newcomb metric. Prefer force-balanced gates for
  pass/fail; raw fit ballooning from visualization is a warning diagnostic.
- Effective ripple and bootstrap-current related neoclassical diagnostics.
  Goodman-style QI targets should ultimately be judged by physical consequences
  such as effective ripple and low bootstrap tendency, not only raw QI residuals.
- Fast alpha orbit loss. This is too expensive for every checkpoint, but should
  be used on final beta-layer winners.
- Available energy / trapped-electron-mode proxy. The Goodman QI paper and
  related work evaluate trapped-electron turbulence tendencies. Use this as a
  final physics discriminator if the tooling is available.
- Real coil optimization or collaborator coil workflow. Any final engineering
  claim must survive a real coil run, because current-potential and contour
  proxies can miss torsion/twist details.

Use for seed discovery, not direct promotion:

- Near-axis QI database screening. The NAE database is useful for finding seeds
  with good effective ripple, low beta sensitivity/Shafranov-shift sensitivity,
  maximum-J prevalence, and reasonable shaping. Do not promote NAE points until
  finite-aspect reconstruction and finite-beta continuation pass the same gates
  as W7X-derived candidates.
- `solve_geo=True`-style near-axis reconstruction. Do not replace it with naive
  Frenet inversion of scalar functions; the previous local attempt showed the
  inverse problem is ill-conditioned. If this route is resumed, use coupled axis
  coefficient + NAE-parameter optimization.

## High-B QI Topology

The high-field `|B|` contour topology is a known weak point in several branches.
For QI repair branches, include or screen:

- high-B percentiles around `B/Bmax = 0.75` to `0.98`
- Boozer-map poloidal closure of constant-`|B|` contours
- straightness of maximum-`|B|` contours at field-period boundaries
- penalties for toroidal closure, broken contours, and multi-island behavior

Do not try to repair bad high-B topology only by pushing scalar QI. If topology
is wrong, prefer finding a better seed or using a dedicated topology penalty.

## Newcomb and ITG Visualization

`viz_report.py` now includes:

- Newcomb ballooning metric profile
- ITG bad-curvature flux-compression proxy
- MHD stability plot with Mercier, ballooning, and Newcomb

Use these as screening/proxy plots, not as gyrokinetic replacements. Do not
run full GK every optimization cycle.

ITG proxy is best plotted as:

- radial profile of bad-curvature flux-compression proxy
- heatmap on `(theta, zeta)` or Boozer-like coordinates
- overlay/compare with bad-curvature coverage and `grad_s`

Keep ITG proxy weak in optimization unless MHD gates are hard.

## Branch Map

Main archive:

- `runs/stage2_eval130_R2p5`
- `runs/stage2_eval130_R2p5_ns128`
- Role: main physics archive, beta about `1.88%`.

Coil-aware W7X redo:

- `runs/w7x_modified_coil_optimized/phase01_seed`
- `runs/w7x_modified_coil_optimized/phase02_qi_mhd`
- Role: showed current-potential proxy improves, but true contour geometry can
  worsen. Do not keep pushing as main path.

Stage2 paired repair:

- `runs/w7x_modified_coil_optimized/stage2_paired_repair`
- best checkpoint: `eval015`
- Role: small local coil improvement near stage2, not a major engineering win.

Curvature margin repair:

- `runs/w7x_modified_coil_optimized/stage2_curvature_margin_repair`
- useful checkpoint: `eval020`
- Role: proved boundary curvature source terms can be reduced without
  immediately breaking MHD.

Coil contour guard:

- `runs/w7x_modified_coil_optimized/stage2_eval020_coil_contour_guard`
- useful checkpoints: `eval008`, `eval012`
- Role: true contour curvature improved; still not a physics candidate.

MHD-hard / coil-forward:

- `runs/w7x_modified_coil_optimized/stage2_mhd_hard_coil_push`
- best engineering checkpoint: `eval032`
- conservative engineering checkpoint: `eval024`
- Role: current strongest low-curvature finite-beta MHD-stable engineering
  test branch. Not a QI/max-J candidate.

NAE database branch:

- `runs/nae`
- Role: near-axis seed screening. `dataframe.pkl` should be pre-screened before
  finite-aspect finite-beta continuation. Avoid all-candidate short runs.

## Recommended Next Experiment: Beta-Layer Pareto Study

The most informative next campaign is to rerun the W7X
truncation/recomposition -> finite-beta optimization chain across beta layers,
not to keep pushing a single beta `2%` point. Create beta-layer branches and
optimize each until no useful progress remains.

Suggested beta targets:

- low beta: `0.8%` to `1.0%`
- mid beta: `1.3%` to `1.5%`
- high beta: `1.8%` to `2.0%`
- stress beta: `2.2%` to `2.5%` only as exploration

For a clean comparison, derive all branches from the same seed family:

- preferred seed family: the W7X truncation/recomposition seed chain
- archive comparison seed: `stage2_eval130_R2p5_ns128`
- engineering seed: `stage2_mhd_hard_coil_push/eval032` or `eval024`

Do not mix unrelated seed families in the first Pareto comparison unless the
goal is explicitly seed discovery.

For each beta layer:

1. Build the finite-beta W7X-derived seed at the target beta.
2. Run physics-first optimization: MHD + QI + max-J + high-B topology.
3. Keep `curvature_max` as the engineering guardrail.
4. If physical metrics are good but curvature is poor, run a small guarded
   curvature repair.
5. Stop when 20-30 evals produce no new gate-passing improvement.

Record for each accepted checkpoint:

- beta
- `Bmax`
- coil length
- `Bn_max/|B|`
- contour `curvature_max`, `curvature_rms`
- min coil-plasma distance
- min coil-coil distance
- `DMerc_min`, negative count
- ballooning `n_unstable`, `lambda_max`
- Newcomb min
- QI, max-J, ripple, iota
- high-B topology metric
- boundary `pdrot_aw`, `pdrot_max`, `k2_min`, `H_min`

Expected outcome:

- beta `~1%`: easiest for coil and curvature, weaker physics payoff
- beta `~1.5%`: likely best engineering/physics compromise
- beta `~1.9%`: close to current stage2, harder coil problem
- beta `>2.2%`: likely too hard without a new seed

Primary deliverable for this study:

- a table comparing the best usable checkpoint at each beta layer
- one selected "recommended physics point"
- one selected "engineering demonstration point" if different
- clear reason why higher beta was or was not usable

## Stop/Promote Criteria

Stop a local branch if any of these happen:

- no new gate-passing checkpoint after `20-30` evals
- curvature improvement below `~1%` while MHD margin degrades
- only low-iota/high-ripple points improve engineering metrics
- raw high-B topology becomes clearly wrong
- real coil workflow shows no improvement despite proxy gains

Promote a point only if its branch purpose is satisfied:

- physics candidate: finite-beta MHD + QI topology + max-J acceptable + true
  contour `curvature_max` at least warning-level acceptable
- engineering test point: finite-beta MHD + true coil metrics improved
- final candidate: must satisfy both physics and representative engineering
  gates

## Practical Agent Rules

- Do not overwrite `stage2_eval130_R2p5`.
- Do not promote `eval032` as a physics result.
- Do not trust QSS-style coil proxy alone.
- Do not use strong ITG weighting without hard MHD gates.
- Prefer checkpoint-every `3-5` for exploratory repairs.
- Use small `dof_bound_frac` when close to a good point:
  - `0.003-0.006` for guarded repairs
  - `0.006-0.010` for MHD-hard engineering pushes
- Always run outer gates on checkpoints, not only the final optimizer output.
- Keep a short `STATUS.md` entry for every new branch with config, seed, best
  checkpoints, gates, and decision.
