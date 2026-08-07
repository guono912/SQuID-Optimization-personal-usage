# Skill 10 - Iota Rational Resonance Policy

Use this policy whenever reporting rational surfaces, selecting iota targets,
setting rational penalties, reviewing coil-return topology, or comparing
configurations with different NFP.

## The rule that must survive handoffs

A rational crossing is a location where a perturbation can resonate. It is not
an island, and the reduced denominator alone is not a resonance-risk score.
Risk depends on the perturbation spectrum, NFP symmetry, local shear, island
width, overlap with neighboring islands, and symmetry-breaking errors.

Assume the straight-field-line harmonic convention

```text
cos(m*theta - n*phi),       resonance when m*iota - n = 0.
```

For a reduced rational `iota = p/q`, an exactly NFP-periodic field contains
only physical toroidal mode numbers `n` that are multiples of NFP. The lowest
symmetry-compatible representation is therefore

```text
k_sym = NFP / gcd(NFP, abs(p))
m_sym = k_sym * q
n_sym = k_sym * p
```

Use `abs(m_sym)` and `abs(n_sym)` for order ranking; retain the sign of `n_sym`
when the iota orientation is negative. Always verify the code's Fourier
convention before comparing stored VMEC/SIMSOPT indices with physical `n`.

Examples:

| NFP | reduced iota | lowest symmetric `(m,n)` | interpretation |
| ---: | ---: | ---: | --- |
| 4 | `8/9` | `(9,8)` | low-order natural resonance |
| 4 | `9/10` | `(40,36)` | high-order in the symmetric spectrum |
| 3 | `9/10` | `(10,9)` | low-order natural resonance |
| 2 | `1/3` | `(6,2)` | natural resonance despite reduced numerator 1 |
| 2 | `2/7` | `(7,2)` | low-order natural resonance |

Thus a `9/10` crossing is not generically harmless or dangerous. It is much
less exposed in an exact NFP=4 spectrum than in an NFP=3 spectrum. A
symmetry-breaking coil or assembly error can still drive the low-order
`(m,n)=(10,9)` harmonic, so it remains in the error-field exposure list.

## Keep two resonance lists

Every serious report must distinguish:

1. **Symmetry-compatible natural resonances.** Rank by the lowest physical
   `(m_sym,n_sym)` allowed by the nominal NFP symmetry. These govern the
   nominal ideal boundary and exactly periodic coil model.
2. **Symmetry-breaking exposure resonances.** Include ordinary low-denominator
   fractions that can be driven by coil placement, current, manufacturing, or
   assembly errors. These are evaluated against an explicit perturbation
   ensemble, not treated as present at nominal amplitude.

The broad `q<=N` scan is useful for the second list and for locating possible
surfaces. It must not be labeled a nominal island count or used as a universal
hard gate.

## What determines risk

For every crossed surface that matters, record:

- rational value, radial location, and local `d iota / ds`;
- NFP and lowest symmetry-compatible `(m_sym,n_sym)`;
- complex resonant normal-field or field-line Hamiltonian coefficient,
  including phase;
- nominal symmetric-coil value and values under the declared error ensemble;
- estimated island width in the documented coordinate convention;
- neighboring-island spacing and Chirikov overlap;
- Poincare topology, stochastic-layer width, and last closed surface;
- beta, pressure/current profile, and coil-current dependence.

Island width scales schematically as

```text
delta_s proportional to sqrt(abs(a_mn) / (m * abs(d iota / ds))).
```

The exact coefficient is convention dependent. Do not compare widths from
different radial coordinates or harmonic normalizations without conversion.
Near-zero shear can make a small resonant field important. Conversely, a
crossing with negligible resonant amplitude need not produce a measurable
island.

## Optimization and promotion policy

- Never optimize a scalar count of all low-denominator crossings by default.
- Never define `rational_hard_targets` from denominator alone. Hard targets
  must be justified by low symmetric order, a measured/returned resonant
  harmonic, or a campaign-specific error-field study.
- Protect the complete iota profile, finite shear, and sign. An iota zero
  crossing or sign flip remains a topology failure independent of this
  resonance ranking.
- Moving iota is only one repair control. Prefer reducing the responsible
  resonant harmonic or improving shear locally when that preserves the rest of
  the physics and coil response.
- Promotion is decided by resonant response, island width/overlap, Poincare,
  and coil-return robustness. A crossing table alone cannot promote or reject.
- If an island chain is intentional, such as an edge island divertor, verify
  its phase, size, beta/current robustness, and separation from internal
  islands rather than penalizing it as a generic crossing.

## Current SQuID output semantics

- `diagnose.py` field `low_order_iota_crossing_count` and the associated
  `q<=...` table are broad reduced-fraction exposure scans. They are not NFP
  risk rankings.
- `viz_report.py` field `nfp_dangerous_rationals_crossed` lists nominal
  NFP-coupled harmonics within its declared `m/k` limits.
- `physical_diagnostics_report.py` intentionally includes non-NFP
  low-denominator modes as assembly-error exposure markers.

Until a single CLI report emits both lists with `k_sym/m_sym/n_sym`, preserve
these names and state explicitly which list is being discussed. Do not merge
their counts into one score.

## Published precedent

CIEMAT-QI4X does not avoid every rational between iota about 0.8 and 1. It
specifically controls the NFP=4-compatible `8/9` resonance using rotational
transform, local shear, resonant coil-field optimization, and HINT-3D island
verification. Higher natural resonances such as `12/13` are checked, while an
edge `4/4` chain is intentionally retained for the island divertor.

Stellaris similarly uses an NFP=4 profile near `0.86-0.98`, verifies the
coil-produced free-boundary equilibrium, and assesses resonant topology rather
than rejecting every reduced rational crossing. These examples motivate the
two-list policy; they do not define a universal preferred iota range.

Primary references:

- CIEMAT-QI4X: https://doi.org/10.1088/1741-4326/ae54ad
- Stellaris: https://doi.org/10.1016/j.fusengdes.2025.114868
- Magnetic-gradient coil-distance proxy:
  https://doi.org/10.1088/1361-6587/ad1a3e
