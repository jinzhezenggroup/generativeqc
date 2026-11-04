# Decision: shared exact CPU ERI shell-orbit scheduling

Status: implemented
Date: 2026-11-04

## Problem

The first factorized CPU derivative consumer removed per-component compilation
as the dominant Ni barrier, but a matched-state 600-second trial still returned
no HVP. Its retained profile contained 122,273 raw first-component contractions,
116,726 weighted-IR constructions and 2,419,977 shell-signature conversions.
J and full-range K visited the same ordered shell quartets separately. Bounded
numeric storage did not bound or reduce that scientific work.

## Decision

Move the existing exact AO orbit helpers into the compiler's integral scheduling
owner and migrate both runtime and tests directly. Do not retain runtime import
aliases. Extend that owner with distinct shell orientations and arbitrary
ordered-cotangent folding. For a canonical shell quartet, transpose each distinct
ordered weight block back to canonical component order and add it exactly once.
Repeated shells require their full component blocks, not a blind factor of eight.

CPU first-integral sources publish bounded vectorized component batches. The
directional Fock source evaluates each canonical raw quartet once for each
distinct radial operator, reusing a full-range jet for both J and K. Physical
density, exchange coefficient, spin factors and atom scattering stay with the
runtime consumer. Immutable mathematical requests and Cartesian normalizations
are reused per angular/operator class; no integral tensor or state-specific
numeric cache is retained. S/T/V keep their existing mathematics and tiling.

Full-range CPU weighted gradients and second sources fold arbitrary ordered
weights before evaluating a canonical quartet. Second-order folding is optional
resident storage: preflight the output plus two shell-weight arrays and retain
the ordered schedule if they cannot fit. Release each borrowed weight before
requesting another callback to preserve that bound. The existing factorized
lease/component-lowering fallback remains separate and unchanged. CUDA weighted
execution and range-separated weighted-source schedules are not promoted.

## Scientific and resource invariants

- Validate every ordered callback, even for exact physical translations.
- Preserve public spherical cotangents and existing normalization exactly once.
- Sum raw contributions before any presentation symmetrization or projection.
- Admit no numerical screening or chemical/symmetry assumptions about weights.
- Keep independent-slot primitive derivatives intact at repeated centers.
- Record primitive/component work, ordered shell-weight blocks and reduced shell
  quartets; do not confuse a bounded allocation or fewer compilations with a
  qualified endpoint performance result.

## Validation

Executable orbit coverage includes one, two, four and 23 shells; the latter is
the real Ni/H def2-SVP shell count. Dense independently symmetrized tensors test
unequal shell extents and arbitrary nonsymmetric weights, including diagonals.
Noncanonical shape, nonfinite and complex weights reject rather than disappear
behind a canonical representative.

A physical real-spherical mixed s/p/d/f fixture freezes arbitrary public weights,
pulls them into the existing Cartesian derivative workspace and compares native
first derivatives and HVPs with independent libcint nuclear gradients and two
central-gradient steps. Fixed absolute gates are 3e-10 for gradients and 2e-8
for extrapolated HVPs. This is a source-level scientific test, not a complete
Ni SCF/Hessian or installed public endpoint qualification.

Initial retained receipts under `/tmp/gqc-2843-2849/` are
`eri-symmetry.{log,xml}` (9 passed, 74.76 seconds),
`symmetry-targeted.{log,xml}` (4 passed, 268.81 seconds, two JUnit warnings),
and `symmetry-first-guards.{log,xml}` (13 passed, 19.58 seconds). The last
includes the tight-budget one-center folding fallback and existing compiled
first-derivative contracts. All applicable hooks pass, and the compiler
structure checker reports 586 modules with no dependency errors. The larger
regression and finite Ni endpoint trial are recorded separately; these initial
receipts do not assert their success.

## Rejected alternatives and remaining gates

Do not apply AO diagonal multiplicities to shell blocks, multiply one weight by
eight, assume density symmetries for arbitrary cotangents, or materialize the
coordinate-by-AO-rank-four derivative tensor in production. Do not cache numeric
integrals across geometry/lifetime changes to avoid source work.

The old 600-second failure, the new finite Ni trials and full endpoint gates
remain distinct evidence. No speedup, public B3LYP support, issue completion or
default-grid/installed qualification follows from this schedule alone.

## References

- Issues #2224 and #2843; draft PR #2977.
- `2026-11-04-factorized-cpu-eri-derivative-consumer.md` under architecture notes.
- `python/generativeqc_compiler/integral/eri_symmetry.py`.
- `tests/python/test_eri_symmetry.py`.
- Local retained profile: `/tmp/gqc-2843-2849/ni-factorized-profile.txt`.
