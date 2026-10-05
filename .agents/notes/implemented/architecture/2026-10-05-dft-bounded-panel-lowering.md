# Decision: bind real DFT grid projections through canonical tensor lowering

Status: implemented
Date: 2026-10-05

## Problem

#1886 requires DFT consumers to use the same request/candidate/prepared boundary
as post-HF. The grid owner called a thin GEMM wrapper directly for density and
orbital projections. Active AO counts, point tails, orbital tails and requested
jet subsets vary after preparation. Enumerating all shapes would grow host
storage with the capacity product; preparing vendor descriptors per tile would
move provider planning into the scientific replay loop.

## Decision

Represent both projections by the existing TensorIR einsum `jpu,uv->jpv`.
Fold the contiguous jet/point axes into one free matrix axis, without packing
or replicating the right operand. DFT may depend on TensorIR as CC and MP2 do;
TensorIR retains its independent `common`-only dependency direction.

Add a shared bounded packed-domain binding. The compiler emits canonical
request/candidate identities and symbolic descriptors. Preparation validates
the maximum descriptor and selects a shape-polymorphic implementation once.
Replay validates the exact descriptor against immutable mode/extent/precision/
layout/update bounds using stack storage, then uses the shared matrix executor.
There is no per-shape plan, allocation or mutable shape cache.

The bound admits strict-FP64 library and generated execution. Providers requiring
exact-shape plans retain explicit rejection evidence. Their implementation is
not approximated by unbounded plan caching. The old provider allowance remains;
the new context replaces the old grid BLAS handle, so handles are not duplicated.
The numeric arena retains its existing 4 MiB reserved workspace for ABI/layout
compatibility; the new shared provider uses its qualified zero-workspace policy.
One 16 KiB host reservation covers the binding and simultaneous stack descriptor.

## Invariants

- The source owner still canonicalizes density and weighted orbital factors.
- AO maps, empty tiles, feature-mask offsets, tau-only jets, orbital tails and
  identical restricted-spin provenance retain their existing semantics.
- Error audits remain on the owner stream and use its sticky error word.
- Optional preparation failure may choose only the admitted generated FP64
  fallback. Execution failures propagate without resubmission.
- Capture is rejected by the shared boundary until replay work accounting is
  available. Existing ordinary grid leases are the migrated consumer; native
  XC's captured density/Vxc kernels require their own subsequent migration.
- Binding compatibility includes the immutable runtime domain as well as the
  compiler template; diagnostic template hashes alone do not license reuse
  with a different capacity or stream.

## Rejected alternatives

Becke/AO recurrence tuning does not implement this architectural goal. The
ordinary KS square products are a low-work consumer in the retained large-force
profile. Native XC's fused symmetric/mixed density kernels cannot be replaced
by an ordinary GEMM without retaining their canonicalization, precision,
resource and capture contracts. They are deliberately outside this first PR.

## Evidence

Qualification uses node1's RTX 5090 through finite Slurm allocations. Independent
fixture/CPU feature comparisons exercise full/local/empty maps, all feature masks,
spin reuse and density/orbital routes. Additional tests force generated fallback
and verify exact semantic work counts and a single preparation across tails.
The shared fixed-table CUDA tests protect the common executor extraction.
Complete consumer timings and final test results are recorded with the PR;
this architectural migration does not promote a new optional provider or claim
a full SCF/force speedup.

## Revisit when

A finite tile shape inventory or a qualified genuinely shape-polymorphic
provider can admit optional library/AOT plans with complete retained-resource
and endpoint evidence. Connect capture only with physical replay accounting.

## References

- #1886 canonical lowering architecture; #1890 production migration.
- `docs/developer/lowering_providers.md`.
