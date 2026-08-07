# Run storage

Generated results are stored outside the source repository at
`/home/guozx/runs`.

```text
/home/guozx/runs/
  YF_0/     compact-device campaigns, normally R about 1.0-1.2 m
  YF_1/     larger-device campaigns, normally R above 2 m
  archive/  closed or migrated campaigns
```

Use one campaign directory per scientific objective and one run directory per
calculation. Keep equilibrium, exact input, diagnostics, logs, manifest, and a
short decision README together.

The current asset index, important-node list, SHA256 inventory, and migration
map live directly under `/home/guozx/runs`. Do not duplicate those changing
tables in the source repository.

See `FILE_MANAGEMENT_GUIDELINES.md` for naming, retention tiers, immutable
promotion, and cleanup rules.
