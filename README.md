# SQuID

SQuID is a stellarator equilibrium optimization and diagnostic toolkit built
around VMEC/SIMSOPT, DESC, Boozer-space diagnostics, and engineering proxies.

This repository contains code and operating documentation. Generated
equilibria and campaign results live under `/home/guozx/runs`, not in the
source tree.

## Start here

Agents should read documents in this order:

1. [`skill/SKILL.md`](skill/SKILL.md): end-to-end decision workflow.
2. [`skill/SKILLS_INDEX.md`](skill/SKILLS_INDEX.md): task-specific references.
3. [`scripts/README.md`](scripts/README.md): exact CLI inventory.
4. [`runs/FILE_MANAGEMENT_GUIDELINES.md`](runs/FILE_MANAGEMENT_GUIDELINES.md):
   run layout, naming, retention, and promotion.

Do not begin from an old handoff, a campaign `STATUS.md`, or a historical
config unless the current task explicitly targets that campaign.

## Quick start

```bash
cd /home/guozx/SQuID
PY=/home/guozx/fusion_env/bin/python

# Cheap baseline diagnosis
$PY scripts/diag/diagnose.py \
  --nc_file <wout.nc> \
  --output_dir /home/guozx/runs/<project>/<campaign>/<run>/diagnostics \
  --plot

# Full visual report
$PY scripts/viz/viz_report.py \
  --nc_file <wout.nc> \
  --output_dir /home/guozx/runs/<project>/<campaign>/<run>/viz

# Optimization from a reviewed JSON config
$PY scripts/opt/optimize.py \
  --input_parameter /home/guozx/runs/<project>/<campaign>/configs/<stage>.json
```

Before promoting a result, follow the independent gate and resolution checks
in [`skill/skill_04_gates_promotion.md`](skill/skill_04_gates_promotion.md).

## Environment

The validated local environment is `/home/guozx/fusion_env`. For development,
install this source tree without replacing its validated solver stack:

```bash
/home/guozx/fusion_env/bin/python -m pip install -e . \
  --no-deps --no-build-isolation
```

The package metadata lists lightweight Python requirements and optional DESC,
VMEC/SIMSOPT, and test extras. A successful Python package installation alone
does not provide the external VMEC executable or validate binary compatibility;
use the environment checks in `skill/skill_01_environment.md`.

## Repository layout

```text
squid/       reusable physics, objectives, diagnostics, backends, and CLI logic
scripts/     thin user-facing entry points; no new physics formulas belong here
configs/     examples and historical templates; copy and review before use
tests/       unit, compatibility, and integration tests
skill/       current agent operating documentation
runs/        documentation pointer only; generated runs live outside the repo
artifacts/   historical source material, not active campaign output
```

## Non-negotiable rules

- Use an explicit `--run_dir` or `--output_dir` for every calculation.
- Diagnose the parent equilibrium before changing it.
- Treat optimizer penalties as steering signals, not acceptance metrics.
- Use `Phi_edge^2 * VMEC.DMerc` on the physical VMEC half-grid.
- Preserve current and pressure profiles when reconstructing `NCURR=1` inputs.
- Separate nominal NFP-compatible resonances from symmetry-breaking rational
  exposure; never promote or reject from a `q<=N` crossing count alone.
- Rebuild selected candidates at high resolution and rerun independent gates.
- Never overwrite a promoted equilibrium or infer identity from its filename.
- Keep the source root free of `wout_*.nc`, `input.*`, `threed1.*`,
  `simsopt_*.dat`, plots, and solver logs.

## Tests

```bash
cd /home/guozx/SQuID
PY=/home/guozx/fusion_env/bin/python

# Fast suite
$PY -m pytest -q

# Include wout-backed integration tests
SQUID_TEST_WOUT=<trusted_wout.nc> JAX_PLATFORMS=cpu $PY -m pytest -q
```

DESC 0.16 effective-ripple calculations use a `nufft2` path that is not
CUDA-lowered in the current environment. Keep ripple and full report workflows
on CPU unless the dependency stack is revalidated.
