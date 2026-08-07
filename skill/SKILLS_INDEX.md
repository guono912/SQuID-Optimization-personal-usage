# SQuID Agent Documentation Index

Read `SKILL.md` first. Open a topic below only when the workflow reaches it.

| Task | Document | Authority |
| --- | --- | --- |
| End-to-end diagnosis and optimization | [SKILL.md](SKILL.md) | Primary workflow |
| Environment, paths, and repository boundaries | [Skill 01](skill_01_environment.md) | Operational |
| Importing, reconstructing, or generating seeds | [Skill 02](skill_02_seed_generation.md) | Operational |
| Configuring and running optimization | [Skill 03](skill_03_optimization_runs.md) | Operational |
| Independent MHD/physics gates and promotion | [Skill 04](skill_04_gates_promotion.md) | Operational |
| Coil-side proxies and handoff boundary | [Skill 05](skill_05_coil_workflow.md) | Operational |
| Known numerical and physical failure modes | [Skill 06](skill_06_pitfalls.md) | Mandatory review |
| Screening a portfolio and deciding repair vs switch | [Skill 07](skill_07_seed_portfolio.md) | Strategy |
| Low-B, finite-beta, pressure, current, and iota scans | [Skill 08](skill_08_finite_beta_lowB_continuation.md) | Strategy |
| Mercier normalization and radial-grid convention | [Skill 09](skill_09_mercier_normalization.md) | Mandatory protocol |
| Historical evidence behind older rules | [Evidence map](KNOWLEDGE_SOURCES_20260710.md) | Evidence only |

## Conflict resolution

Use this precedence order when documents disagree:

1. Current code and tests.
2. `SKILL.md` and Skills 01-09.
3. The active campaign's `GUIDELINES.md` and `STATUS.md`.
4. Historical handoffs, archived guides, and old configs.

A campaign may tighten a gate, but it must not silently change metric
definitions. Record protocol, code revision, radial grid, and resolution.

## Documentation ownership

- General reusable instructions belong in `skill/`.
- Exact command options belong in `scripts/README.md` and CLI `--help`.
- Project targets belong in `/home/guozx/runs/YF_0/...` or `YF_1/...`.
- Chronology and failed experiments belong in campaign logs.
- Historical conclusions must not be copied into general skills as universal
  thresholds without a named calibration protocol.
