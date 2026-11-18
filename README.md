# SQuID

SQuID is a Python toolkit for optimizing and assessing quasi-isodynamic stellarator equilibria. Starting from a VMEC equilibrium, it varies Fourier coefficients of the plasma boundary and evaluates the magnetic field on a set of flux surfaces. The core objective combines departures from quasi-isodynamicity, violations of the maximum-J condition and radial variation in the minimum field strength. Additional terms control aspect ratio, boundary displacement and other quantities chosen for a particular design.

The package also evaluates an existing equilibrium without optimizing it. Those diagnostics are useful when a change improves one objective at the expense of another. For example, a lower quasi-isodynamic residual can accompany a negative Mercier criterion over part of the plasma.

## Choose a backend

| | VMEC | DESC |
|---|---|---|
| Starting point | An original VMEC input or a `wout` from which the code reconstructs an input | A VMEC `wout` imported into DESC at the requested spectral resolution |
| Boundary search | SIMSOPT residuals and finite differences with SciPy nonlinear least squares | Nelder-Mead search over boundary coefficients, with a DESC force-balance solve at each trial |
| Evaluated equilibrium | VMEC solution at the trial boundary | DESC solution exported as a `wout` and read through SIMSOPT for the core SQuID terms |
| Accepted output | `input.squid_optimized` | `wout_desc_optimized.nc` |

Use an explicit `--backend vmec` or `--backend desc` when comparing runs. The backends have different equilibrium representations and objective implementations. Compare changes within each run, then evaluate the two candidate equilibria with the same diagnostic settings. For VMEC runs with finite pressure or toroidal current, the original input preserves profile specifications that may be lost in a `wout`. Reconstruction from a `wout` includes a check for drift in the operating point.

## Install in a solver environment

Use Python 3.10 or later. VMEC optimization requires SIMSOPT and its working VMEC extension. The DESC route also needs DESC and the SIMSOPT interface used to read its exported equilibria. From the checkout, install the Python package into an environment that already contains the solver stack:

```bash
python -m pip install -e . --no-deps --no-build-isolation
python scripts/opt/optimize.py --help
```

Package dependencies and optional extras are listed in [pyproject.toml](pyproject.toml). The [environment guide](skill/skill_01_environment.md) covers solver imports and compatibility checks. CPU is the tested setting for effective-ripple calculations and full reports with DESC 0.16.

## Run a first case

Supply an existing equilibrium and a new directory outside the checkout. These commands use the baseline diagnostic, the core VMEC preset and a finite evaluation budget. Replace the example path with your input file.

```bash
export JAX_PLATFORMS=cpu
WOUT=/absolute/path/to/wout_parent.nc
RUN="$HOME/runs/squid-first-case"

python scripts/diag/diagnose.py --nc_file "$WOUT" \
    --output_dir "$RUN/parent" --skip_ripple --plot

python scripts/opt/optimize.py --backend vmec --mode core \
    --nc_file "$WOUT" --run_dir "$RUN/vmec" --maxiter 5 --max_total_evals 30
```

For DESC, use the same `--nc_file`, select `--backend desc` and give it a separate run directory. For a VMEC input deck, supply `--vmec_input_file` in place of `--nc_file`. The input equilibrium and campaign outputs are kept outside the source tree.

For repeated experiments, start with a [configuration template](configs/README.md) and pass `--input_parameter path/to/stage.json`. The optimizer applies the mode preset, the JSON settings and then the command-line overrides. It records the resulting parameters in `input_parameter.resolved.json`. Unknown option names and active objectives exclusive to the other backend produce an error before the run directory is created.

## Read the result

`history.csv` records the evaluated trials. Read `run_status.json` for the termination reason, evaluation count and objective values. A trial, including the last row of the history, may be a finite-difference probe or a point that the optimizer discarded.

| State | Interpretation |
|---|---|
| `converged`, `accepted: true` | The optimizer met its stopping criterion and exported a checked candidate. For DESC, `candidate_evaluation` identifies the evaluated equilibrium saved to disk. |
| `evaluation_cap_recovered_start` or `evaluation_cap` | The total evaluation budget stopped the search. VMEC writes its pre-solve boundary to `input.squid_unaccepted_cap` for recovery. |
| `solver_evaluation_limit` or `iteration_limit` | The solver used its own step or evaluation allowance. The best objective reported so far can be useful for a follow-up run. |
| `initial_only` | A successful `--maxiter 0` run evaluated the starting equilibrium and wrote an initial-state file. |
| `candidate_rejected` | The selected point failed final checks. Inspect the status and diagnostic output before choosing a restart. |

A successful evaluation-only run and an accepted candidate exit with code 0. A completed search with no accepted candidate exits with code 2. On VMEC, `maxiter` sets the least-squares allowance to `maxiter * (number of free coefficients + 1)`; finite-difference probes also count toward `max_total_evals`. On DESC, `maxiter` limits simplex iterations, and `max_total_evals` includes the initial point. A run interrupted before finalization can lack a status file.

## Assess a candidate

Rebuild a selected VMEC input, or use the exported DESC equilibrium, and evaluate it at the intended operating point and resolution. The repository includes a physical diagnostics report, a visualization report, MHD and coil-feasibility checks, and tools for screening saved checkpoints. For example, with a rebuilt or exported candidate:

```bash
CANDIDATE=/absolute/path/to/wout_candidate.nc
python scripts/viz/viz_report.py --nc_file "$CANDIDATE" --output_dir "$RUN/viz"
python scripts/gate/mhd_gate.py --wout "$CANDIDATE" --output_dir "$RUN/mhd"
```

For Mercier stability, report `Phi_edge^2 * VMEC.DMerc` on the physical VMEC half-grid and record the radial interval. When interpreting rotational-transform profiles, distinguish field-period-compatible resonances from exposure to symmetry-breaking perturbations. A count of rational crossings alone does not give an island width. The [promotion protocol](skill/skill_04_gates_promotion.md) specifies the independent checks and resolution studies for a design claim.

## Where to work

| Task | Source |
|---|---|
| Change an objective or residual | [Objective terms](squid/objectives), [Boozer and particle-orbit calculations](squid/core), [backend assembly](squid/backends) |
| Change config parsing, backend selection or output | [Optimizer entry point](squid/cli/optimize.py), [path helpers](squid/utils/run_paths.py) |
| Diagnose a configuration or assess a checkpoint | [Diagnostics](squid/diagnostics), [evaluation routines](squid/evaluation), [command inventory](scripts/README.md) |
| Plan a campaign and preserve its evidence | [Workflow](skill/SKILL.md), [topic index](skill/SKILLS_INDEX.md), [run layout and retention](runs/FILE_MANAGEMENT_GUIDELINES.md) |

Historical stage configurations and [archived material](artifacts/README.md) record earlier campaigns. Review the parent, profiles, weights, conventions and grids before reusing one. Preserve the original equilibrium, the code revision and the input hash with each run.

## Tests

Run the fast suite from the checkout. Set `SQUID_TEST_WOUT` to an equilibrium compatible with the installed VMEC stack to include the equilibrium-backed tests.

```bash
python -m pytest -q
SQUID_TEST_WOUT="$WOUT" JAX_PLATFORMS=cpu python -m pytest -q
```
