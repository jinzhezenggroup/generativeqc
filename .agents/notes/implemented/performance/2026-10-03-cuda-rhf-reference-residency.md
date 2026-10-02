# Decision: bounded optional ERI residency for CUDA RHF reference export

Status: implemented
Date: 2026-10-03

## Problem

CCSD and CCSD(T) request a physical CUDA RHF reference. The shared owner
previously disabled ordinary small-system ERI residency for reference export
and recomputed unscreened integrals inside every matrix-direct Fock build.
After correcting CC value-source admission, the 56-AO energy endpoint still
spent 775/136/866 seconds in cold/warm/changed-geometry RHF on node2. The MO
provider and correlation solve together took only about 37 seconds.

## Decision and invariants

Reuse the existing dense ERI producer and Fock consumer within the shared RHF
owner for eligible s/p physical-reference exports. Do not add integral formulas,
a CC-specific RHF solver, CPU oracle work or another physical reference contract.
Keep the matrix-direct arena. After querying the actual device and host solver
workspaces, admit an optional device ERI allocation only if the complete
reference budget can hold it. Cap this quartic cache at 256 MiB independently of
the caller's budget. Higher angular momentum remains on the direct evaluator.
The normal HF small-system selection and generated force ownership are unchanged.

The optional allocation has its own bucket resource owner and device-capacity
accounting. Release it in stream order. A budget shortfall or an allocation
failure selects the existing bounded fallback; other CUDA errors propagate.
Recompute geometry-dependent values after coordinate changes. Recheck solver
and cache bytes on every cached execution, not only on first setup. Exported
physical P/F/C/epsilon still pass the same stationarity and canonical gates.

The existing optional progress journal reports completed reference work after
stream synchronization: AO count, SCF iterations, physical Fock builds, resident
ERI bytes/values constructed and complete reference capacity. These counts are
not host graph-construction calls. A zero resident-value count on the direct
route does not mean zero integral work; that route recomputes during Fock builds.

## Evidence

Initial complete endpoints pass the exact-basis PySCF 2.14.0 gates and all 40
public CCSD/CCSD(T) CPU/CUDA tests on node1. The 28-AO energy endpoint takes
4.77 seconds cold, 4.24 seconds warm and 4.26 seconds after changing geometry;
28-AO forces take 18.50/17.89/17.90 seconds. Node2 completes all four 56-AO
energy calls in 67.28/66.79/66.80/66.84 seconds. These preliminary measurements
use a source-identical numerical candidate before adding the diagnostic journal
and extracting the admission predicate; final-binary evidence is retained with
qualification before promoting this PR.

The native resource probe compares physical energies, density, Fock matrices
and orbital energies against an independent CPU RHF solve. It executes the
exact selected and fallback budgets, rejects minimum-minus-one, and injects
optional device allocation pressure through the real numeric ledger instead
of exhausting a shared GPU. The initial 7-AO probe observes 19,208 resident
bytes, an exact 8,435,684-byte fallback peak, one optional rejection and zero
live ledger bytes after release. Host tests cover overflow, the 256 MiB ceiling,
force exclusion, higher angular momentum and exact admission boundaries.

## Rejected alternatives and revisit conditions

Removing the old reference guard from ordinary persistent ERI selection would
only cover <=16 AOs and could turn optional storage into a mandatory rejection.
Raising that ordinary threshold globally would change unrelated HF workloads.
Neither preserves reference-specific queried workspace admission. Enabling a
second CC-owned integral/Fock implementation would duplicate scientific owners.

This cache is rebuilt for each standalone RHF call and is not borrowed by the
following MO provider. Revisit cross-phase or prepared-endpoint reuse only with
explicit geometry/source identity, charged overlapping lifetimes, independent
reference gates and full endpoint measurements. Revisit the storage ceiling or
angular domain only with complete workload and resource evidence. The public
CCSD(T) force boundary remains 28 AOs; the separate 56-AO degeneracy failure is
not addressed by this scheduling change.
