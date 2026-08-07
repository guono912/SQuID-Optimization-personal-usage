# Skill 05 - Coil-Side Proxies and Handoff

SQuID does not replace the coil group's detailed design workflow. Its role is
to avoid handing over obviously hostile plasma boundaries and to measure how
coil realization changes the plasma physics.

## Evidence hierarchy

### 1. Boundary geometry diagnostics

pdrot, principal curvatures, curvature tails, and singular-area fractions
locate difficult surface regions. They are not coil-manufacturability scores.
Use them for local repair guidance and regression monitoring.

### 2. Inner-loop current-potential proxy

`w_coil_proxy_bn`, `w_coil_proxy_k`, and related low-resolution objectives can
steer optimization away from poor winding-surface response. Their absolute
numbers are optimizer-protocol dependent and must not be compared with a
separate REGCOIL or filamentary calculation.

### 3. Fixed-protocol current-potential evaluation

```bash
PY=/home/guozx/fusion_env/bin/python
$PY scripts/gate/coil_feasibility_gate.py \
  --wout <wout.nc> --output_dir <run>/coil_proxy \
  --offsets <fixed_values> --lambdas <fixed_values>
```

Record normalized normal-field error, current-density norms/tails, spectral
content, winding-surface definition, regularization, and all resolutions.
Compare candidates only within the same named protocol.

### 4. Filamentary coils and coil-return equilibrium

This is the decisive physical realization test. Compare target and coil-return
equilibria using the same diagnostic protocols. Normal-field error alone is
not sufficient: a spatially concentrated resonant error can damage iota,
Mercier/DGeod, ripple, or islands despite a good global RMS.

## Physics-side requirements to give the coil group

Provide:

- target equilibrium hash and exact operating point;
- allowed normal-field error protocol and hotspot maps;
- protected iota profile/shear and two rational lists from Skill 10: nominal
  NFP-compatible modes and symmetry-breaking error-field exposure modes;
- acceptable degradation bands for Mercier, ballooning, ripple, and ITG;
- plasma/coil clearance envelope and any forbidden boundary changes;
- required coil-return/free-boundary reconstruction cases;
- perturbation cases for manufacturing and assembly tolerance studies.

Do not prescribe the coil group's conductor, support, stress, cooling,
power-supply, or assembly solution unless the task explicitly includes those
engineering models.

## Robustness loop

For each serious target:

1. generate a nominal coil solution;
2. compute its coil-return equilibrium;
3. evaluate the same physics vector as the target;
4. perturb coil positions, currents, and allowed manufacturing errors;
5. extract complex resonant harmonics, including non-NFP modes introduced by
   the perturbations, and identify which components drive physics loss;
6. feed a robustness objective or sensitivity map back to boundary design;
7. repeat until the target is not dependent on an unrealistically exact coil.

## Decision rule

Use boundary metrics as warnings, current-potential metrics as intermediate
evidence, and coil-return/perturbation physics as the deciding evidence. A
smaller global `Bn` that worsens resonant response or coil-return stability is
not an improvement.
