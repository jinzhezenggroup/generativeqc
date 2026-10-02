# Decision: retain one fresh-SCC GFN2 runtime per calculator

Status: implemented
Date: 2026-10-03

## Problem

The public Python `singlepoint()` created and destroyed a context, system and
prepared calculation for every call. Each prepared GFN2 calculation also owned
an exclusive bridge. Consequently the native CPU/CUDA caches could not survive
between public calls, even though they already implement topology/control
identity checks and numerical refresh. Repeating a molecule rebuilt allocations,
solver handles and the SCC graph, including disposable setup validation work.
xTBloom's public calculator retains its corresponding native context.

## Decision

Retain one native context in each Python GFN2 calculator. Serialize the complete
prepare/execute/result-read transaction, including explicit cleanup, with a
per-owner reentrant lock. `clear_cache()` releases resident storage and permits
later reconstruction. Collection also releases the owner. Backend/device changes
retire the context before creating a replacement; failed creation leaves it empty.

A method-neutral polymorphic workspace slot in `core::ContextState` owns the
GFN2 bridge. Prepared calculations keep strong references to that bridge and
independent immutable copies of their atoms and scientific controls. The API
context mutex protects acquisition/replacement. The existing bridge mutex still
covers execution and orbital snapshot extraction. A replaced workspace can
therefore outlive the slot while an outstanding prepared calculation uses it.

The bridge submits the full request on every call with `SCC_START_FRESH`. Native
cache identity and coordinate admission remain authoritative. There is one
committed native cache; transactional replacement may temporarily allocate one
candidate beside it. Existing SCC graph fallback and failure settlement remain
unchanged. No global cache, result cache, warm density or new scientific kernel
is introduced.

## Rejected alternatives

- Reuse only the Python context: the exclusive bridge in each prepared
  calculation would still discard the expensive native cache.
- Cache the prepared calculation/result by coordinates: this complicates the
  immutable preparation contract and risks stale controls, properties or SCC.
- Retain an unbounded molecule-keyed/global map: unnecessary memory growth and
  lifetime ambiguity. One owner per calculator is sufficient for this endpoint.
- Remove CUDA seed/setup validation while changing lifetime: this is a separate
  numerical-admission decision. Retention removes its repeated cost without
  weakening those checks.

## Invariants

Every result comes from fresh SCC at the supplied geometry. Energy/force output
selection, element ordering, topology, charge, spin and SCC controls must be
validated before reuse. Failure must not poison the next valid calculation.
Separate calculators must not share mutable runtime state. Cleanup and result
reads cannot race execution. A bounded entry count does not bound the storage
needed for a large molecule.

## Evidence

`test_gfn2_runtime_retention.py` exercises cold-versus-retained energies, forces
and SCC iterations across geometry, property, topology and control changes,
invalid coordinates/spin and SCC nonconvergence recovery, concurrent calls and
cleanup, exactly-once destruction and rebuild, and backend/device replacement.
A counted non-scientific bridge compiles the real method adapter to verify
ownership, distinct contexts, prepared-request isolation and replacement lifetime.
The failure/recovery gate caught stale context detail after a normal nonconverged
return. The execution ABI now replaces that earlier failure with the current
nonconvergence diagnostic while preserving borrowed strings on success.
Independent tblite goldens and directional finite differences remain separate
scientific gates. `test_gfn2_orbital_snapshot_transaction.py` protects the existing
execute/snapshot transaction.

Qualification on n2 RTX PRO 6000 Blackwell, Slurm job 2051, used Release
sm_120/CUDA 12.9.1, the same LP64 scipy-openblas32 provider, one BLAS thread,
300 K, modified Broyden history 8/damping 0.4, maximum 300 iterations, and
energy/charge tolerances 1e-10/1e-8. The ten-case cohort contains seven tblite
fixtures and 24/96/192-atom water clusters; each has cold 1 + repeated 5 +
changed-geometry 5 samples, all fresh SCC and host-visible energy/force calls.
The baseline uses both the previous native binary and its previous Python
package. The comparator checks the geometry of every sample before any timing
ratio is accepted.

| Endpoint median | Before | Retained | xTBloom |
| --- | ---: | ---: | ---: |
| H2 repeated | 9.082 ms | 2.771 ms | 2.688 ms |
| 24 atoms repeated | 54.224 ms | 43.522 ms | 41.657 ms |
| 96 atoms repeated | 170.628 ms | 140.182 ms | 129.619 ms |
| 96 atoms changed | 170.703 ms | 140.159 ms | 129.606 ms |
| 192 atoms repeated | 441.966 ms | 352.091 ms | 314.493 ms |
| 192 atoms changed | 442.329 ms | 352.375 ms | 314.703 ms |

These endpoints still trail xTBloom. All 110 SCC iteration counts match both
comparators. Versus the previous build, maximum energy/force differences are
0 Eh/4.17e-17 Eh/bohr; versus xTBloom, 5.69e-14 Eh/5.17e-15 Eh/bohr.
Retention reduces runtime construction from one per call to one per compatible
calculator lifetime; fresh SCC iteration work and per-request geometry refresh
are unchanged. Optional topology/control replacements still rebuild.

The CPU/CUDA endpoint, lifecycle, tblite and finite-difference cohort passed
41 tests. Compute Sanitizer memcheck passed the three CUDA retention scenarios
with zero errors. Host ownership/orbital tests passed 15 tests (three CUDA
opt-in skips), and the native RHF API test passed, including borrowed diagnostic
string stability. Pre-commit ownership/structure checks passed.

Local ignored receipts are under `.artifacts/n2/retention/`, with runner and
validation logs under `.artifacts/`. Measured binary SHA256 identities:

- Previous PR: `7ebd9ab642c624616bdff2ad0c50c8b3513082b9e8e36ad12554a325cb956d05`.
- Retained: `efbc3d0a4b0b374544c2780d48aab9df1b9fb4cb21df295a9276ccd8eb71afed`.
- xTBloom main `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`:
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Consequences

GFN2 calculators now keep CPU/GPU resources after a completed singlepoint until
cleanup or collection. Applications with many long-lived calculators should call
`clear_cache()` when resident work is no longer useful. Shared-calculator calls
are serialized, while separate calculators remain independent. Other methods'
singlepoint context lifetime is unchanged.

## Revisit when

The public API gains explicit reusable calculation owners, resource-budgeted
cache admission, or independently qualified warm-SCC requests. Preserve fresh-SCC
behavior unless a new public contract explicitly opts into a different start.
