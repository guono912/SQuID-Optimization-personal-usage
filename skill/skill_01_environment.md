# Skill 01 - Environment and Repository Boundaries

## Supported environments

Use the checkout and solver Python for the host on which the job runs.
The two maintained installations have different paths.

| Installation | Checkout | Solver Python |
| --- | --- | --- |
| WSL | `~/SQuID` | `~/fusion_env/bin/python` |
| R7625 | `~/Projects/SQuID-Optimization-personal-usage` | `~/miniconda3/envs/fusion/bin/python` |

Verify the environment before a long run:

```bash
PY=python  # From the checkout with the solver environment active
$PY -c "import netCDF4, simsopt, desc, jax; print(jax.__version__)"
$PY -m pytest -q
```

Use `SQUID_TEST_WOUT=<trusted.nc>` to enable wout-backed integration tests.

## Source versus generated data

The repository contains reusable code and documentation. `/home/guozx/runs`
contains calculations. Never write solver products to the source root.

Forbidden at the root:

```text
wout_*.nc  input.*  threed1.*  simsopt_*.dat  boozmn_*.nc
plots      solver logs         temporary VMEC/DESC files
```

Every command must receive an explicit `--run_dir`, `--output_dir`, or output
path. VMEC construction and execution must occur while the process cwd is the
run directory because legacy Fortran units bind relative paths early.

## Code ownership

```text
squid/backends/      VMEC and DESC execution
squid/objectives/    optimizer residuals and penalties
squid/diagnostics/   reusable physical diagnostics and conventions
squid/evaluation/    independent gates and engineering evaluation
squid/cli/           argument parsing and workflow orchestration
scripts/<group>/     thin user entry points
tests/               unit, compatibility, and integration tests
skill/               current Agent instructions
```

Do not add physics formulas to `scripts/`. Put reusable logic under `squid/`
and retain legacy flat script paths as thin compatibility wrappers.

## CPU and GPU policy

JAX can detect the installed RTX 5060 Ti, but DESC 0.16 EffectiveRipple uses
`nufft2`, which lacks a CUDA lowering in the current environment. Use
`JAX_PLATFORMS=cpu` for ripple, full reports, and production comparisons.
Revalidate the complete workflow before changing this policy; a successful
standalone JAX array operation is not sufficient.

## Remote synchronization

GitHub `master` records the shared revision. WSL and R7625 have separate
working trees. Compare both HEADs and any uncommitted changes before syncing.

Before a controlled sync:

1. check which checkout active jobs use and defer updates to that checkout until they finish;
2. run tests locally;
3. inspect `git status` on both machines;
4. sync source only, never `/home/guozx/runs` by accident;
5. record the code commit in new run manifests.

## File identity

Use SHA256 plus manifest metadata. A filename such as `final.nc` or `wout.nc`
is not an identity. Follow `runs/FILE_MANAGEMENT_GUIDELINES.md` for naming,
promotion, and retention.
