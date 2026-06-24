# SQuID Agent Skill Collection — Index

Operational skill documents for CLI/IDE agents working on the SQuID
stellarator optimization campaigns. Read this index first, then the skill
that matches your task. The legacy general playbook is `SKILL.md`.

| Skill | File | Use when |
| --- | --- | --- |
| Environment & layout | `skill_01_environment.md` | First contact with the repo, running anything |
| Seed generation & conversion | `skill_02_seed_generation.md` | Creating/recomposing NFP=4 seeds from W7-X or other sources |
| Optimization runs | `skill_03_optimization_runs.md` | Configuring and launching `optimize.py` stages |
| External gates & promotion | `skill_04_gates_promotion.md` | Deciding PASS/WARN, promoting checkpoints |
| Coil feasibility workflow | `skill_05_coil_workflow.md` | Coil proxy, REGCOIL-like scans, true contour metrics |
| Known pitfalls & failure modes | `skill_06_pitfalls.md` | Before trusting any number or repeating an old experiment |
| Campaign design method | `../runs/w7x_r2p35_goodman_chain/DESIGN_METHOD_R2p35_NFP4.md` | Planning the R=2.35 m, NFP=4 campaign |

## Non-negotiable working rules (apply to every skill)

1. Promotion decisions use EXTERNAL gates (`scripts/mhd_gate.py`,
   fixed-protocol coil contour review), never internal optimizer penalties.
2. A "PASS" from `mhd_gate.py` is necessary but NOT sufficient for promotion:
   the gate does not check `DMerc_min >= 0.08`; check it yourself.
3. Coil metrics are comparable only at the fixed protocol
   (offset `0.35a`, `lambda = 1e-8`, same M/N resolutions).
4. Every new branch gets: a `config.json`, checkpoints every 2-3 evals,
   a gate summary, and a short entry in the campaign `STATUS.md`.
5. Rerun all gates after ANY change of beta, profiles, resolution, or scale.
6. Keep physics-first and engineering-only branches labeled and separate.
