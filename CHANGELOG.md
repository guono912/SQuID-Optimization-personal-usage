# Changelog

## Unreleased

### Documentation

- Defined the canonical two-list rational-resonance policy: nominal
  NFP-compatible harmonics versus symmetry-breaking error-field exposure.
- Added the minimum symmetry-compatible harmonic-order formula and prohibited
  using a broad `q<=N` crossing count as an island or promotion score.
- Linked rational-surface promotion and coil-return guidance to the new
  resonance policy.

## 1.1.0 - 2026-08-07

This release establishes the maintained SQuID baseline used for YF_0 and YF_1
equilibrium work.

### Correctness

- Standardized Mercier reporting as `Phi_edge^2 * VMEC.DMerc` on the physical
  VMEC half-grid while retaining explicitly named raw values.
- Preserved pressure and current data during `NCURR=1` wout reconstruction.
- Added reconstruction gates for complete iota, pressure, and current-density
  profiles as well as endpoint iota, total current, and beta.
- Corrected signed rational scans, including iota sign changes and zero
  crossings.
- Unified the reported ISS04 field scale with the canonical `|bvco|/Rmajor`
  convention and retained the flux/area estimate only as a named fallback.
- Replaced the erroneous doubled torus-area proxy with resolved LCFS
  quadrature and a correct `4*pi^2*R*a` fallback.
- Isolated legacy VMEC products in explicit run directories.
- Refused accidental reuse of non-empty run directories and prevented
  evaluation-cap trial points from being mislabeled as optimized outputs.

### Structure

- Moved reusable CLI implementations into `squid/cli/` and retained compatible
  legacy wrappers.
- Grouped user-facing scripts by function.
- Added install metadata and console entry points.
- Established one Agent-facing documentation hierarchy and external run-file
  policy.

### Verification

- Unit and VMEC-backed integration tests.
- Legacy/grouped CLI parity tests.
- Real v8 diagnosis, optimization, postcheck, and visualization workflow.
- Root-artifact, documentation-link, and command smoke checks.

### Known constraints

- VMEC and the validated SIMSOPT binary stack remain external dependencies.
- DESC 0.16 effective ripple remains CPU-only in the validated environment
  because its `nufft2` path is not CUDA-lowered.
- Fixed-boundary force balance does not establish coil realizability; promoted
  candidates still require coil-return and robustness checks.
