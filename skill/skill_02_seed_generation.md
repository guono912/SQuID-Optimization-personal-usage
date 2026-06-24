# Skill 02 — Seed Generation & NFP/Scale Conversion

## Canonical NFP=5 → NFP=4 reindex pipeline (the campaign standard)

Use when deriving an NFP=4 seed from any W7-X-like NFP=5 equilibrium.

1. Obtain an NFP=5 VMEC wout. For the DESC h5 source:

   ```python
   from desc.examples import get
   from desc.vmec import VMECIO
   eq = get("W7-X")[-1]            # final EquilibriaFamily member
   VMECIO.save(eq, "/tmp/desc_w7x_nfp5_wout.nc")
   ```

2. Reindex + truncate + rescale with the standard tool:

   ```bash
   python /home/guozx/w7x_squid_nfp4/scripts/wout_to_nfp4_input.py \
     --wout <nfp5_wout.nc> --nfp-out 4 --mpol 5 --ntor 5 \
     --target-rmajor 2.35 --free-iota
   ```

   This keeps modes with `xn % 5 == 0` (all of them for a proper NFP=5
   wout), maps reduced `n_new = xn/5`, so the W7-X FUNDAMENTAL helical
   harmonic (physical n=5) becomes the NFP=4 fundamental (physical n=4).
   Helical shaping is preserved by reinterpretation, not by physical
   periodicity filtering.

3. Set the beta target: edit `PRES_SCALE` with `AM = [1, -2, 1]`
   (p ~ (1-s)^2) or copy the source AM polynomial. Iterate `PRES_SCALE`
   until `betatotal` in the wout matches the target (it is not linear in
   PRES_SCALE at fixed boundary, but nearly so for small steps).

4. Run VMEC: `xvmec <input.name>`; check `ier_flag=0`, then run the
   external gates (Skill 04) before optimizing.

## Recomposition / low-mode filtering at fixed NFP

```bash
python scripts/recompose_vmec_input.py \
  --input input.src --output input.dst \
  --nfp 4 --R0 2.35 --mmax 4 --nmax 4 [--taper] [--preserve-profiles]
```

- Rescales all coefficients by `R0/RBC(0,0)` and `PHIEDGE` by the square.
- `RBC(0,0)`-only edits ("shiftR") change R without rescaling shaping —
  this produced the route2 seed and is a legitimate nonuniform move, but
  always re-gate after it.

## What NOT to do

- Do NOT build NFP=4 seeds by 2D-FFT periodicity filtering of an exactly
  NFP=5-periodic boundary: all non-multiple-of-5 harmonics are identically
  zero, so the result is the axisymmetrized W7-X plus FFT leakage noise.
  (`desc_seed/convert_desc_to_nfp4.py` did this AND had a normalization
  bug; its products `wout_desc_w7x_nfp4*.nc` are deprecated.)
- Do NOT hand-write VMEC inputs with helical content at per-period
  `n = NFP`: per-period `n=1` is the fundamental. (`input.desc_beta1p5`
  failure mode.)
- Do NOT trust spectral comparison tables unless the (0,0) coefficient
  reproduces the known major radius exactly — that is the cheapest
  normalization check.

## Correct way to compare two seeds' spectra

1. Convert both through the SAME pipeline to the same NFP, mpol/ntor, R.
2. Compare per-period reduced-index `RBC/ZBS` tables directly
   (see `SEED_INVESTIGATION_20260611.md` Section 10 for the format).
3. Separate the QUALITATIVE question (are the sources distinct?) from the
   QUANTITATIVE one (how much do amplitudes differ?). Verify with a
   physical-space check: overlay cross-sections at phi = 0, pi/(2*NFP),
   pi/NFP and report RMS boundary distance.

## Available seed sources (audited 2026-06-11)

| Source | Type | Notes |
| --- | --- | --- |
| `wout_W7-X_beta_*.nc` (16 files) | VMEC NFP=5, mpol=10/ntor=50 | current campaign ancestor (vacuum member) |
| DESC `W7-X_output.h5` final member | DESC NFP=5, L=M=N=12, beta~2% | desc_seed ancestor; finite-beta profiles built in |
| simsopt `get_w7x_data()` | coils+axis only | for coil-forward / free-boundary work only |
| `runs/stage2_eval130_R2p5_ns128` | SQuID-optimized R2.5 archive | route2 ancestor; do not overwrite |
| NAE/pyqsc exports `/home/guozx/pyqsc/seed_vmec_exports/` | near-axis seeds | screening only, never direct promotion |
