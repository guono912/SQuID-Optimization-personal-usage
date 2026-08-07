# Skill 08 - Finite-Beta, Low-B, and Iota Continuation

Use this skill whenever changing magnetic field, pressure, beta, current, or
iota. These are independent controls and must not be collapsed into a single
"scale the equilibrium" operation.

## Declare the scaling policy

Always state which quantity is held fixed:

- **fixed beta, vary B:** scale pressure approximately with `B^2` while
  changing `PHIEDGE`;
- **fixed absolute pressure, vary B:** beta rises approximately as `1/B^2`
  and VMEC may fail at low B;
- **pressure-only:** keep `PHIEDGE` and boundary fixed, change `PRES_SCALE`;
- **geometry scaling:** scale boundary coefficients and `PHIEDGE` with the
  correct length powers, then re-gate at final size.

Record `PHIEDGE`, `PRES_SCALE`, all `AM(...)`, achieved beta, B0/volavgB,
`NCURR`, `CURTOR`, and `AC/AI` in every comparison row.

## What the 2026-07 low-B scans established

- Lowering B while preserving beta can converge down to about `B=0.2 T`, but
  raw Mercier numbers grow strongly with field normalization. Compare sign,
  negative-count, radial location, and `Phi_edge^2*VMEC.DMerc`; do not
  compare raw DMerc magnitudes across B as if they were dimensionless margins.
  Historical B0-squared columns are legacy-only. Follow Skill 09.
- Lowering B at fixed pressure is much more severe. The direct `B=0.2 T`,
  fixed-pressure case failed VMEC in the three-axis scan.
- Ramping beta down together with B can cross into Mercier failure even when
  a fixed-beta path stays positive. Pressure level and profile shape matter,
  not just beta as one scalar.
- A soft-edge pressure profile changed the blend-0.50 topology-first case from
  strongly negative Mercier to positive DMerc with zero gated negative points.
  Pressure-shape scans are therefore a first-class branch, not bookkeeping.

Source tables:

- `/home/guozx/runs/mini_steady_lowB_20260708/bfield_continuation_preserve_iota/`
- `/home/guozx/runs/mini_steady_lowB_20260708/topology_first_pressure_scan/`
- `/home/guozx/runs/mini_steady_lowB_20260709/three_axis_scaling_key_ns128/`

## Iota and current continuation

- `NCURR=0`/`AI` prescribed-iota runs answer whether an equilibrium can exist
  with that profile; they do not show that the boundary naturally supplies it.
- `NCURR=1` with `AC/CURTOR` is the production test for self-consistent iota.
- Scan current and pressure together when iota moves through rational surfaces.
  The 2026-07-09 current scan moved iota substantially, but the high-current
  end introduced a low-order rational crossing.
- Always report sign conventions explicitly. Some imported families use
  negative iota; compare absolute rational locations only after confirming
  orientation conventions.

## Resolution is part of the result

Low-B Mercier behavior was highly resolution-sensitive. One topology-first
repair looked positive at `ns=64` but was strongly negative at `ns=128`.
Therefore:

1. use low resolution only for direction finding;
2. rebuild the selected input at `ns>=128` and a tight VMEC tolerance;
3. recompute `dmerc_flux_normalized` on the declared radial interval and
   record `Phi_edge`;
4. run the external MHD gate before promotion.

Never average or merge ns64 and ns128 values in one ranking column.

## Finite beta can stabilize or destabilize

The Goodman fixed-boundary ladders show that vacuum Mercier sign alone is not
a reliable seed filter:

- several iota `0.3-0.5` lines changed from negative vacuum DMerc to positive
  DMerc as beta increased toward `~1%`;
- the high-iota `1.2` GH23 line became more negative with pressure;
- positive Mercier does not imply ballooning stability; these ladders lacked
  a complete external promotion gate and remain response evidence.

Recommended sequence for a new seed family:

1. vacuum smoke;
2. fixed-boundary beta/profile ladder;
3. external MHD gate on the best response points;
4. only then start shape optimization or coil work.

Evidence: `/home/guozx/runs/goodman_finite_beta_lift_20260703/`.
