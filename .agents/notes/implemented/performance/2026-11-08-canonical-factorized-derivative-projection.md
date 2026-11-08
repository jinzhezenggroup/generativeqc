# Decision: project correlated factors before the first AO pullback

Status: implemented
Date: 2026-11-08

## Problem and measured boundary

The first complete pinned H2IO iodine-ECP CBS energy/force replay passed on
the stationary-density-product candidate, library SHA-256
`944f546b8d63c3407f45c201c2ea14e480c68a1ab182c847cc2657f8b3a4254b`,
Slurm job 4130 on n2's RTX PRO 6000. Native complete endpoint time was
2863.197875082027 seconds; maximum component/composite directional-force error
over both independent FD steps was `7.65392111539212e-9 Eh/Bohr`.

Its 97-AO correlated MP2 derivative still submitted 1,500,625 ordered native
shell quartets. Completed derivative telemetry attributed 1961.590311688
seconds to shell consumers and only 7.735879896 seconds to host pullback/scatter.
The same frame's stationary derivative submitted 198,765 canonical quartets.
The finite conventional source work, not another convergence failure or missing
device storage, was now the dominant complete-endpoint barrier. These are
source-matched measurements of the earlier binary, not a new speedup claim.

## Decision

Project the sparse Fock and occupied/virtual correlation MO cotangent entries
onto all eight exact ERI permutations before transforming the first AO slot.
The common coefficient matrix acts on all four slots, so projection commutes
with the complete pullback. Keep repeated MO-index contributions in the average;
multiply the final local AO cotangent by its distinct shell orbit, using the
existing canonical consumer. No symmetry of the supplied Fock/correlation
weights is assumed.

Use at most the existing admitted `max_shell_width*N^3` first-stage capacity.
Reserve it once before submitting any quartet, and reuse it across first shells.
If this optional reservation fails for a resource reason, retain the existing
compact ordered factorization without mixing partially accumulated canonical
and ordered gradients. Explicit compact and full staged layouts remain
independent validation paths. All remaining buffers retain their admitted
shapes; no molecular AO-rank-four buffer or new CUDA arithmetic is introduced.

The exact stationary-density-product specialization remains first choice for
its exact unit occupied-diagonal Fock and zero correlation cotangent. The new
projection handles genuinely correlated weights, never a cutoff approximation.

## Invariants and qualification

- Preserve exact conventional/ECP inputs, FP64, coefficient ownership, physical
  response gates, convergence limits and the unchanged 43-GiB endpoint budget.
- Resource failure must select a complete bounded fallback before the first
  two-electron callback. Numerical/backend failures must not become fallbacks.
- Native comparisons cover nonsymmetric Fock/correlation weights, dense expanded
  cotangents, ordered compact/full layouts, repeated shells/atoms and multiple
  occupied orbitals. The common-translation-unit probe additionally checks
  independent molecular ERI center derivatives through Cartesian d and spherical
  d/f shells and the production numeric staging bound.
- New default resource tests enforce ownership, cleanup and admitted capacity,
  not exact new allocation call counts or a production-source fingerprint.
- Complete pinned component/CBS energies, zero-net-force and both independent
  directional FD steps must pass on the actual new binary before claiming
  complete issue qualification or endpoint performance improvements.

## Rejected alternatives and revisit conditions

Increasing only the Slurm allocation hides symmetry-related source work.
An unconditional AO-rank-four cotangent would discard bounded staging. Retaining
eight ordered derivative evaluations to save a modest host staging buffer is
not the default when that same full buffer is already admitted; it remains the
explicit resource fallback. Revisit when complete profiles show projected host
staging dominates, retaining these independent gates and bounded fallback.

## References

- #3034 and PR #3046.
- [First-stage compact fallback](2026-10-11-compact-factorized-mp2-first-stage.md).
- [Second-stage compact fallback](2026-11-06-compact-factorized-mp2-second-stage.md).
- [Stationary specialization](2026-11-08-stationary-rhf-density-cotangent.md).
- `tests/native/mp2_derivative_pullback_probe.cpp`.
- `tests/python/test_conventional_ecp_cbs_regression.py`.
