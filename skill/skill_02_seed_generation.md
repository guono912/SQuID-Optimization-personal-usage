# Skill 02 - Seed Import, Reconstruction, and Generation

Use this skill whenever a new lineage enters SQuID.

## Classify the source

Record one of:

- published or database VMEC wout;
- VMEC input with known profiles;
- DESC equilibrium/checkpoint;
- near-axis/pyQSC construction;
- ConStellaration boundary record;
- fixed-boundary optimized target;
- coil-return/free-boundary equilibrium.

Preserve the original file unchanged and record its SHA256.

## Prefer direct VMEC inputs when available

If the exact parent input exists, pass it with `--vmec_input_file`. This best
preserves pressure, current, auxiliary profile knots, tolerances, and spectral
settings.

When only a wout exists, SQuID can reconstruct an input. In `NCURR=1` mode it
must preserve `AC/AC_AUX/CURTOR`; it must not silently replace the source
current with zero. The reconstruction gate compares endpoint iota, current,
and beta against the source before optimization starts.

Use `--zero_current` only for an explicitly labeled vacuum/current-removal
experiment.

## Baseline reconstruction test

Before freeing boundary coefficients:

1. reconstruct at moderate resolution;
2. run VMEC without optimization;
3. compare R/a/aspect, beta, current, `PHIEDGE`, pressure, iota endpoints and
   profile, Mercier sign/components, and well;
4. repeat at the intended final resolution if the point is marginal.

Do not optimize away a converter error.

## NFP conversion and Fourier indexing

VMEC input coefficients use per-period `RBC(n,m)` and `ZBS(n,m)`. A wout's
`xn` is the physical toroidal mode number and already includes NFP. Convert
between them explicitly.

Changing NFP is not geometric scaling. Distinguish:

- filtering modes compatible with a new periodicity;
- reindexing a helical family so its fundamental is reinterpreted at new NFP;
- directly optimizing a native seed at the desired NFP.

These operations answer different physical questions. Record the method and
verify `RBC(0,0)` equals the intended major radius. Overlay physical-space
cross-sections and report boundary reconstruction RMS before using a converted
seed.

For fixed-NFP low-mode filtering/rescaling:

```bash
PY=/home/guozx/fusion_env/bin/python
$PY scripts/seed/recompose_vmec_input.py \
  --input <input.src> \
  --output <run>/equilibrium/input.recomposed \
  --nfp <NFP> --R0 <major_radius> --mmax <m> --nmax <n> \
  --preserve-profiles
```

Review `--help`; do not infer flags from historical campaign scripts.

## Analytic and near-axis seeds

Generated/near-axis seeds are screening starts, not promoted equilibria.
Require:

- finite-aspect-ratio VMEC convergence;
- no sign-flipped or vanishing transform unless intended;
- vacuum and low finite-beta response;
- boundary and coil-proxy screening;
- high-resolution reconstruction before serious optimization.

## Seed acceptance record

Store source hash, NFP conversion, scale transformation, profiles, current
policy, reconstruction errors, baseline diagnostics, and rejection/promotion
decision. Use Skill 07 to decide whether to repair this basin or switch.
