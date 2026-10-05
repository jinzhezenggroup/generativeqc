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
One 32 KiB host reservation covers the binding, descriptor copies, and bounded
preparation selection scratch as well as the replay stack descriptor.

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
Host/compiler/publication gates: 582 passed, 13 opt-in skips. Slurm suites:
140 grid/shared-executor tests and 178 density/spatial/fallback tests passed.
After extending the host reservation to include preparation scratch, the
focused resource/consumer suite passed 32 tests. Memcheck and initcheck each
passed four tau-only density/orbital/provider cases with zero errors.
The CMake-generated native grid translation unit also compiles with the
explicit CXX/CUDA ccache launchers; command and cumulative cache statistics
are retained in `.artifacts/1890-grid/`.

Complete fixed-density PBE energy/Vxc endpoints include the GPU grid and the
existing native CPU XC consumer. Five warm samples follow the first call;
every sample is checked against the independently generated retained fixtures.
They are diagnostic endpoint measurements, not isolated GEMM timing:

| AO / points | Route | Baseline warm median (s) | Migrated warm median (s) | Calls / summands per endpoint |
| --- | --- | --- | --- | --- |
| 96 / 12288 | density | 0.079394 | 0.078215 | 96 / 226492416 |
| 96 / 12288 | orbitals | 0.087210 | 0.087616 | 192 / 188743680 |
| 192 / 24576 | density | 0.246586 | 0.241936 | 192 / 1811939328 |
| 192 / 24576 | orbitals | 0.279571 | 0.278702 | 576 / 1509949440 |

Work is unchanged; the migrated counters are checked against those equations.
Maximum energy/matrix errors over all samples are below 2.2e-14 / 3.2e-15.
First density endpoint calls were 0.177689/0.251849 s baseline and
0.161311/0.243279 s migrated at 96/192 AO. Migrated binding preparation was
0.015807/0.001689 s and occurs once, outside repeated endpoint execution.
These small warm differences do not establish a performance improvement.

Baseline source is `6a0e4eede`; the baseline generated grid binary SHA256 is
`8a2dfaa032fb14c3fcb9363e1a50a7ef3061d54e25163833a0dca1d6a2456226`.
The final migrated grid binary is
`86c0134dba3b3a22b7880ca4a8ecae541e6f3f630f81eeb1345b7e75575e362a`.
Both runs explicitly compose their generated grid with the same pinned native
AO-normalization library, revision `9ba032c783addfeede96890c894b7cc9447cde95`,
SHA256 `429a609e109352a47c01b4e421cbc1c17b4c744d9ea2ffac2f58ebcf785df46a`.
This is not a current-head full native-library or SCF/force benchmark.
`endpoint.py`, `endpoint-baseline.json`, `endpoint-final.json`, initial failures,
all compiler/test logs and earlier resource-reservation measurements remain in
the ignored artifact directory. No optional provider/default promotion follows
from this architectural migration.

## Revisit when

A finite tile shape inventory or a qualified genuinely shape-polymorphic
provider can admit optional library/AOT plans with complete retained-resource
and endpoint evidence. Connect capture only with physical replay accounting.

## References

- #1886 canonical lowering architecture; #1890 production migration.
- `docs/developer/lowering_providers.md`.
