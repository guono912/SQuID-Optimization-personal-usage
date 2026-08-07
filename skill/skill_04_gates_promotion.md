# Skill 04 - Independent Gates and Promotion

Optimizer penalties steer a run. Promotion uses independently recomputed
physics and engineering evidence.

## Three evaluation levels

### Level 1: cheap diagnosis

```bash
PY=/home/guozx/fusion_env/bin/python
$PY scripts/diag/diagnose.py \
  --nc_file <wout.nc> --output_dir <run>/diagnostics --plot
```

Use this to reject numerical failures and identify likely blockers. It is not
the final force-balanced ballooning or coil-realization gate.

### Level 2: physical report and visualization

```bash
$PY scripts/diag/physical_diagnostics_report.py \
  --wout <wout.nc> --output_dir <run>/physical_report \
  --diagnostics_json <run>/diagnostics/<report.json>

$PY scripts/viz/viz_report.py \
  --nc_file <wout.nc> --output_dir <run>/viz --nfp <NFP>
```

Use these for profiles, Mercier components, iota/rationals, Boozer structure,
ripple, ITG, ballooning protocol checks, and reviewable figures.

### Level 3: external MHD gate

```bash
$PY scripts/gate/mhd_gate.py \
  --wout <wout.nc> --output_dir <run>/mhd_gate --L 6 --M 6 --N 6
```

This performs VMEC sanity checks, flux-normalized Mercier/well assessment,
DESC force-balance fitting, ballooning, and effective ripple. Review CLI
`--help` for the exact active radial/alpha protocol and any extended-edge
option. A gate JSON is only comparable to another gate run with the same
protocol.

## Hard requirements

Unless a campaign explicitly tightens them, promotion requires:

- VMEC converged with finite, acceptably small force residuals;
- no broken/self-intersecting boundary or unresolved geometry singularity;
- `Phi_edge^2 * DMerc >= 0` and zero negative points on the declared body
  interval, reproduced at final radial resolution;
- no unstable ballooning points in the declared rho/alpha protocol;
- finite effective-ripple calculation with an acceptable campaign threshold;
- intended beta/current/pressure/iota operating point reproduced;
- no unexplained transform sign flip or vanishing shear;
- no unresolved high-risk rational response;
- engineering evidence at the fidelity required by the campaign.

There is no universal positive Mercier margin, ripple threshold, ballooning
margin, or weight-independent QI score. Campaign targets must name the code,
grid, radial mask, resolution, and reference calibration.

## Resolution and robustness

For a serious candidate:

1. compare screening and final `ns` (normally at least 128; use 192 when
   Mercier is marginal);
2. repeat with higher `mpol/ntor` when spectral truncation is plausible;
3. perturb beta, pressure-profile coefficients, and current around nominal;
4. verify sign/margin conclusions do not depend on one sample crossing the
   radial gate boundary;
5. use the physical VMEC half-grid defined in Skill 09.

## Rational surfaces

The iota scan reports exposure, not island width. Maintain two lists: nominal
NFP-compatible resonances ranked by their lowest symmetry-compatible
`(m_sym,n_sym)`, and low-denominator symmetry-breaking modes evaluated under a
declared coil-error ensemble. The broad `q<=N` count is not a hard gate.

If a relevant surface is crossed, promotion requires a response study using
the actual or perturbed coil field: complex resonant `B_mn`/`a_mn`, local
shear, Poincare/island analysis, estimated island width, and overlap where
possible. Merely deleting a crossing from a scalar table is not a substitute
for response robustness. Use the definitions and examples in Skill 10.

## Promotion record

Write to the run README/manifest:

- parent and candidate hashes;
- exact equilibrium state and operating point;
- VMEC and diagnostic resolutions;
- Mercier convention, radial interval, components, and minimum location;
- ballooning/ripple protocol and result;
- iota profile, shear, and rational response status;
- confinement/ITG results used in the decision;
- engineering protocol and result;
- decision: promoted, retained control, or rejected, with reason.

Never overwrite the parent or promote an evaluation-cap trial state without a
separate converged rebuild and postcheck.
