# Decision: shared resident force inventory after a bounded census

Status: implemented
Date: 2026-10-06

## Problem

HF constructed psss resident tasks while packing each system. The Direct J/K
matrix-value packer explicitly skipped those tasks, and its historical Build
mode produced ket indices without bra tasks. Reusing that field alone in DFT
would leave an empty lease and never select resident force execution.

## Decision

`make_direct_force_resident_bra_schedule` owns the host inventory for every
force consumer. It first counts each system's p-s bras and s-s kets, task chunks
and maximum bra primitive domain. Checked arithmetic and a metadata byte budget
precede allocation. The second pass emits the same cache-oriented ket indices
and 128-ket bra chunks as the qualified HF packer.

HF packing calls the builder after complete shell-pair offsets are available.
A later stationary consumer can call it on the matrix-value packer's immutable
pair metadata without rebuilding or changing basis/primitive data. The legacy
ket-only matrix packing behavior remains explicit with `include_tasks=false`.

## Invariants

- Every p-s/s-s pair product belongs to exactly one resident task in its system.
- The builder does not screen work or introduce J/K, spin or hybrid policy.
- Budget rejection publishes empty views. CUDA capacity admission and bounded
  fallback remain execution concerns, independent of this host inventory.
- Compiler-owned launch chunk size and existing `PsssResidentTask` ABI are used.

## Evidence

The node1 host inventory probe independently enumerates mixed s/p/d pair products
over batched, empty and multi-chunk systems. Exact ownership, no duplicates,
maximum primitive capacity, exact-budget admission, one-byte-short rejection,
ket-only compatibility and malformed-offset rejection pass. The combined host
qualification campaign passes 34 tests, including independent force controls,
capacity admission and asynchronous optional-allocation rollback.

## Rejected alternatives

Turning on Build in the matrix packer would still omit bra tasks and disable its
pair-vector reserve preflight. A DFT-only reconstruction would duplicate the HF
inventory and allow chunk/ownership policy to drift. The shared builder avoids
both problems while retaining the existing matrix packing contract.

## Revisit when

The compiler changes resident chunk policy or a new class needs a different
primitive identity. Extend the shared inventory with explicit class ownership
and repeat independent batched domain qualification.
