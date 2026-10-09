# Decision: shared host ragged-topology validation

Status: implemented
Date: 2026-10-08

## Problem

The native xTB common schema owned reusable validation of immutable ragged
atom/shell/AO storage alongside GFN spin, geometry-generation and pair-cutoff
policy. Its location obscured the distinction between host storage contracts
and scientific method ownership.

## Decision

`src/runtime/ragged_topology.*` owns topology validation, fail-closed binding,
exact borrowed projections and ordered element identity. The native common
implementation adapts the same descriptors into this owner. It contains no
replacement topology or projection algorithms and the shared owner cannot
depend on the native xTB subtree.

Existing native POD declarations remain byte-identical. Field-wise adapters
preserve their pointer/count/token values without casting unrelated objects.
The generic topology view describes explicit pairs as a borrowed byte image
with the asserted two-int64 layout; host inspection uses `memcpy` for each
pair. Structural admission never dereferences this image. This preserves the
native pair type used throughout CUDA without strict-aliasing violations.

The extracted owner reuses checked byte multiplication from
`runtime/bounded_workspace.hpp`. It neither allocates nor retains storage.
Production consumers are CUDA host setup and SCC admission. Compiling this
owner in a CPU build does not establish a CPU scientific consumer.

## Invariants

- Preserve all fourteen alias ranges, check order and first-error diagnostics
- Preserve enum numbers, zero-count/null rules, complete square matrix slices,
  atom/shell/orbital ownership, optional pair maps and AO bucket permutations
- Preserve exact projection pointers and the element fingerprint's seed,
  version, field order and exclusion of memory addresses
- Keep first-endpoint-major explicit topology pairs distinct from the native
  sparse pair-list traversal contract
- Preserve output clearing and existing self-binding behavior
- Keep spin packing, generation validation, physical cutoffs, topology cache
  replacement, numerical policies and result publication in their native owners
- Do not change CUDA adapters, kernels, launches, capture, provider calls or
  generated source artifacts to perform this host-only ownership transfer

## Rejected alternatives

Relocating the entire schema would move scientific spin and cutoff policies.
Aliasing all native descriptor types to newly named generic types would alter
CUDA-visible type identities. Replacing CUDA resources or graph owners would
also change status, device-selection or synchronization contracts and belongs
to a separate review.

## Evidence

The baseline is exact master commit
`882f5c5e0bc060b5777d5060c9cb89714bb4429b`, independently confirmed before freezing
the old schema. Qualification compares the unchanged old implementation with
the real new owner and compatibility adapters, including diagnostics, borrowed
identity, ABI, inaccessible device metadata and allocation behavior. A separate
host harness traces the actual unchanged CUDA bridge definitions with controlled
CUDA boundary failures. These host tests do not substitute for GPU execution.

The focused gates are `tests/python/test_ragged_topology_preservation.py` and
`tests/python/test_ragged_topology_cuda_bridge.py`: 30 tests pass, including
737 identical old/new host bridge traces. The aggregate host suite passes
102 tests with four explicit CUDA endpoint skips. A cached full CPU build and
all 78 CTests pass. The native allocation/link probe requires Linux; the CUDA
host trace probe requires POSIX protected mappings. Static provenance checks
remain available on other platforms. Real GPU execution was not run.

## Consequences

The native common file remains because it still owns method policy. This is
implementation ownership retirement, not whole-file retirement or a reduction
claim for the handwritten-CUDA inventory. The common .cpp/.hpp were not direct
CUDA provenance manifest entries; all existing hashed CUDA source and manifest
bytes remain unchanged.

## Revisit when

Another production consumer needs the same exact ragged-storage contract, or a
separately qualified change intentionally broadens that contract. Do not invent
an additional consumer or move generation/spin policy to justify this owner.
