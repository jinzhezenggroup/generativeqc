# Decision: compare kinetic branch boundaries in physical coordinates

Status: implemented
Date: 2026-10-11

## Problem

The imported TPSS correlation `min(tau_W/tau, 1)` predicate reconstructs the
ratio through fractional spin-density powers. At the exact dyadic polarized
point `rho=(1,1)`, Cartesian gradients `((1/4,0,0),(1/4,0,0))`, and
`tau=(1/128,1/128)`, the physical ratio is exactly one but the reconstructed
binary64 ratio is `0x1.0000000000001p+0`. The resulting constant branch gives
zero half-vtau instead of the pinned Libxc value `-3.3455150091360384`.
Energy and occupied-space tangents can still agree, concealing a wrong full
potential matrix.

This is a reachable physical density, not an undefined-input limitation. Take
AO values `(1,0)`, x-derivatives `(1/8,1)`, and per-spin `D=diag(1,0)`.
The unoccupied Fock entry is half-vtau. The one-sided PSD path
`D_alpha(h)=diag(1,h)` has energy derivative `-3.3455150091360384`, while the
fixed-electron PSD path `D_alpha(h)=diag(1-h,h)` also disagrees with the original
Fock contraction. Both paths remain valid for sufficiently small positive h.
The exact same rounded feature inputs are supplied to both native source and
independently installed PySCF 2.14.0 / Libxc 7.0.0.

## Decision

The existing physical-coordinate canonicalizer rewrites only matching
`select_le(normalized_ratio, 1, true_value, false_value)` predicates before
symbolic differentiation. It compares `sigma_total <= 8*(rho_total*tau_total)`
and retains the original branch payloads. It matches source structure rather
than functional names; unrelated bounds and other comparisons stay unchanged.
Restricted embedding uses the same literal-zero polarization and spin splitting
as the importer. Repeated canonicalization is idempotent, including the guarded
fallback's retained original node.

Multiply density and tau first, so a large density does not overflow at
`8*density` when the final product is representable. When the product is at or
below the binary64 minimum normal value, retain the original predicate rather
than introducing an underflow-induced branch change. This range guard is a
machine-representation condition, not a scientific tolerance. Automatic work
floors place ordinary admitted work points far above that threshold.

## Rejected alternatives

Replacing the entire normalized ratio, rather than only its predicate, fixed
the exact rank-one example but changed raw empty/floored-spin derivatives.
Three existing polarized TPSS boundary rows then failed the unchanged
absolute `2e-11` / relative `2e-9` E/vxc gate, with half-vtau errors up to
`3.205e-8`; the original and predicate-only versions pass those rows. Do not
ship that whole-invariant rewrite or relabel its failed measurements as evidence.

A documentation-only limitation, post-hoc vxc clipping, tolerance widening,
functional blacklist, or hiding unoccupied matrix entries does not correct the
valid-direction derivative defect. No worker floor is perturbed.

## Evidence and boundaries

The checked four-row fixture records the exact dyadic case and the three
empty/floored-spin controls, with independently recomputed pinned oracle
outputs. All eleven native energy/physical-feature coefficients use the
existing E/vxc acceptance gate. The old source fails the rank-one fixture and
both inward PSD derivative tests while passing all three spin controls.

The predicate-only native CPU source passes all 37 RKS and 37 UKS existing TPSS
work-boundary points, with maximum absolute errors `6.7502e-14` and `1.1935e-13`.
The source/test suite also covers adjacent branch sides, exact rational branch
oracles at large/tiny positive work coordinates, subnormal/zero product
fallback, non-target comparisons, unchanged payloads, and idempotence.
CPU/CUDA expression identities match; no local GPU execution is claimed.

Canonical regeneration updates affected catalog node counts, the source/product
manifest, and the complete mathematical CPU-shard baseline. Compiler source
identities deliberately change. Earlier CUDA/SCF measurements belong to their
recorded source closure and must not be relabeled as measurements of this fix.
The previously reported aggregate curated KS driver exit 1 and its matching
native-consumer control remain historical failures; this repair does not claim
that aggregate driver passed. AOT defaults, public energy-only scope, and force
restrictions are unchanged.

## Revisit when

A more general physical-coordinate simplifier can demonstrate unchanged raw
Libxc work-domain derivatives and finite-arithmetic behavior across spin floors,
all branch sides, and independent valid-density directions. Any additional
backend/runtime qualification should be recorded against its actual source.

## References

- PR #2276: the same-input rank-one diagnosis supersedes the provisional review.
- `tests/python/test_libxc_physical_kinetic_predicate.py`.
- `tests/data/xc/physical-kinetic-predicate/tpss.json`.
- `tests/python/test_libxc_automatic_point_native.py`.
- [Shared automatic point and spin-screening decision](2026-10-11-shared-automatic-point-spin-screening.md).
