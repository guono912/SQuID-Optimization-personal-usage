# Skill 01 - Environment and Repository Boundaries

## Supported environments

Primary paths on WSL and the remote Linux host:

```text
source repository: /home/guozx/SQuID
Python:            /home/guozx/fusion_env/bin/python
campaign results:  /home/guozx/runs
YF_0 results:      /home/guozx/runs/YF_0
YF_1 results:      /home/guozx/runs/YF_1
```

Verify the environment before a long run:

```bash
cd /home/guozx/SQuID
PY=/home/guozx/fusion_env/bin/python
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

The local WSL repository is the code-maintenance authority. The remote host is
for compute. Do not synchronize while either worktree has unreviewed changes.

Before a controlled sync:

1. finish or stop calculations;
2. run tests locally;
3. inspect `git status` on both machines;
4. sync source only, never `/home/guozx/runs` by accident;
5. record the code commit in new run manifests.

## File identity

Use SHA256 plus manifest metadata. A filename such as `final.nc` or `wout.nc`
is not an identity. Follow `runs/FILE_MANAGEMENT_GUIDELINES.md` for naming,
promotion, and retention.
