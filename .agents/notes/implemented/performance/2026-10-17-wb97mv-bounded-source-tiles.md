# Decision: bounded canonical WB97M-V Hessian source tiles

Status: implemented
Date: 2026-10-17

## Problem

The response-reuse qualification only established tiny-system performance.
Source-matched larger probes rejected a 12-atom/28-AO STO-3G water tetramer and
a 6-atom/48-AO def2-SVP water dimer. Neither produced a Hessian. The first
failure was the moving-grid planner's `20*points*AO*all_primitives*3` work bound,
not a GPU out-of-memory condition. Independently, the exact nuclear source
required all `2*shells^4` ordered group descriptors and rejected both group
counts at 100,000. Simply raising these guards would retain excessive storage
and duplicate primitive work.

## Decision

Use authenticated normalized shell topology for the AO primitive/polynomial
incidence upper bound. Check that bound against actual device records before
evaluating any AO jet; an understated bound must fail closed. No quadrature,
primitive, derivative or floating-point precision changes are involved.

Retain the original ordered/resident schedule when admitted. Otherwise enumerate
canonical shell-pair pairs and partition their complete distinct permutation
orbits into bounded windows. Reuse the existing Cartesian and derivative-slot
permutations, linked AD closure, primitive chunking, public projection and
ordered incidence reductions. Subdivide a window when its source/batch arena
does not fit. Do not repeatedly rebuild every preceding window while shrinking
a later high-angular window. Global source, primitive, projection and reduction
guards remain global. Host metadata is bounded by the source-group guard;
only one window's numerical and descriptor storage is resident on CUDA.

Reserve the shared status/publication panels before sizing tile arenas. A
diagnostic candidate exposed a real off-by-resource error here: a tile could
consume the full 64 MiB integral budget before the final common panels were
added. Keep that negative result rather than increasing the integral budget.
Project high-angular groups with the existing strided CTA; a 128-public-quartet
batching heuristic is not a scientific or capacity prerequisite.

For full Hessians, retain frozen exact `F_R(D)` panels and frozen HVP columns,
under a separate optional 8 MiB cap inside the original aggregate budget.
Quadratic exact J/K energy implies `d(gradient)/dD : dD = F_R(D) : dD` for
signed RKS/UKS densities. The compiler-AD source still supplies every first and
second nuclear derivative. Contract these AO panels after the physical CPKS
solve instead of regenerating four-index jets for the moving density. This is
O(3N*spin*AO^2 + (3N)^2) storage. Cache-disabled and allocation/budget fallbacks
retain the complete original source action. A standalone HVP never builds the
full-Hessian cache. DF retains its fitted primary and uses this only for LR.

## Invariants and qualification

- All numerical work remains linked CUDA; host scheduling metadata is not a
  Python, CPU numerical or reference-oracle fallback.
- No screening, precision, SCF, CPKS, quadrature, endpoint-time or GPU-memory
  acceptance control is relaxed.
- Canonical permutations include derivative slots, not scalar symmetry factors.
- A partially built source/frozen cache cannot authorize publication.
- Host admission tests exercise 663,552 ordered groups inside 64 MiB, global
  source/work refusal, disabled fallback and malformed linked closure.
- Component gates compare tiled/scalar schedules with independent libcint ERIs
  and derivatives, plus independent AO/Becke moving-grid values.
- Complete RKS/UKS direct and DF/ECP/spherical matrices must agree with the
  directional fallback at 3e-12 absolute/relative; independent full-matrix
  comparison retains the 3e-6 absolute gate.

## Consequences and remaining limit

This fixes storage/work-admission cliffs, not arbitrary molecular scaling.
The optional 128 KiB dense CPKS representation is unchanged; larger rotation
spaces still execute physical matrix-free GMRES. Recycled/block response and
the remaining VV10/source work need separate endpoint-qualified optimization.
Coarse 8/4/8-grid timings do not establish chemical grid convergence or general
superiority over GPU4PySCF. Retain complete endpoints, work counts, failures and
sampled device memory with the measurement source identity.

## References

- Supersedes the capacity boundary, not the historical measurements, in
  `2026-10-17-wb97mv-response-reuse-and-grid-reductions.md`.
- `docs/developer/dft_hessian.md`
- PR #2594
