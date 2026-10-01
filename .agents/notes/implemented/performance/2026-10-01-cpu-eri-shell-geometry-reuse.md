# Decision: reuse compiler-owned primitive ERI geometry across shell components

Status: implemented
Date: 2026-10-01

## Problem

The value-only CPU ERI lowering introduced by #1662 removed the dynamic scalar
recurrence from s/p/d production, but retained an AO-first traversal. Every
Cartesian component independently reconstructed the same primitive product
centers, pair decay, prefactor and Boys vector. Final endpoint profiles still
placed about 72% of water/def2-SVP and 94% of formaldehyde/def2-SVP time in integral
preparation.

## Decision

Expose component-independent expression roots from the existing shell-class
scalar DAG. Cut each CPU component DAG at those roots, then apply the existing
scalar factorization/lowering. Emit one common geometry constructor and retain
the existing 313 center/axis symmetry representatives. The shared shell-class
value and derivative equations remain the scientific owner; no Gaussian-product
or recurrence formulas are copied into the CPU lowering or native scheduler.

Traverse canonical shell quartets natively. Prepare lossless component records
once per shell quartet, evaluate common geometry/Boys once per primitive quartet,
and accumulate its requested Cartesian components before advancing the primitive
tuple. Keep normalization, contraction coefficients, FP64 accumulation,
eightfold tensor scatter, and spherical transformation in their existing native
owners. The component buffer is bounded by 6^4 records, independent of molecule
size. It is invocation-local and cannot outlive or be reused across geometry
changes.

The fast route requires value-only work and an all-s/p/d system. Derivative and
f+ systems retain the former AO traversal and explicit primitive fallback.
Range-separated ERIs and the independent RawSource oracle are unchanged.

## Rejected alternatives

- Handwritten CPU pair/product-center or recurrence mathematics would add a
  second scientific owner and increase CPU/CUDA divergence
- Emitting all 10,000 Cartesian scalar functions or all shell-wide unrolled
  functions would sacrifice the bounded code inventory and raise offline
  compilation cost
- A molecule-wide primitive-quartet cache would add memory/lifetime identity
  requirements; immediate component consumption provides reuse with fixed scratch

## Invariants

Shell-pair canonical order is allowed to differ from global AO-pair order. Each
physical eightfold orbit is evaluated exactly once, and all tensor permutations
are populated. Same-shell component pairs and identical shell-pair quartets need
their own triangular bounds. Tests cover contracted s/p/d values, spherical
projection, derivative/f+ fallback, and four distinct d shells in both storage
orders (the complete 1,296-component buffer).

Common geometry may be center/axis-permuted only according to the existing ERI
symmetry map: the two inverse pair exponents and four shifts follow center order,
the product-center difference flips sign on bra/ket exchange, and the prefactor,
rho and Boys vector are invariant. The requested Coulomb order remains bounded
by the value-only IntegralIR.

## Evidence

The exact source-derived census preserves 326,255 primitive component evaluations
for water/def2-SVP while reducing common geometry/Boys evaluations to 38,111.
Formaldehyde/def2-SVP preserves 2,261,946 component evaluations while reducing
geometry/Boys evaluations to 199,362. These are work counts, not timing claims.
H2/STO-3G has no geometry reuse and must be retained as an overhead control.

Endpoint qualification compares incrementally against #1662's qualified binary,
with independent PySCF/libcint numerical checks, identical spherical BSE inputs,
core guess, DIIS history 8, energy tolerance 1e-10 and density RMS tolerance 1e-8.
Cold, fresh-object warm and changed-geometry measurements remain separate.

## Revisit when

Remaining profiles identify pair construction, component dispatch, contraction,
or spherical transformation as dominant. Subsequent optimizations should remain
compiler-owned and be measured against this slice rather than repeating #1662's
speedup claim.

## References

- #1662: compiler-owned CPU value-only ERIs
- #762 / #926: compiler-first shared CPU/CUDA scientific ownership
- `docs/maintainer/performance_engineering.md`
- `benchmarks/results/cpu-eri-20261001/README.md`: prior endpoint qualification
