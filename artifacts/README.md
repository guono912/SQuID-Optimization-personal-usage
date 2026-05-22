Artifact layout for local SQuID work.

- `legacy_root/`: historical VMEC/DESC/diagnostic files moved out of the
  repository root. These files are intentionally ignored by git.
- `old_archive/`: old standalone scripts and reference experiments kept for
  manual lookup only. Active code should live under `squid/` or `scripts/`.

New optimisation or diagnosis outputs should go under `runs/<run-name>/`.
