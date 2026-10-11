# Decision: stage masked library products in charged caller storage

Status: proposed; local correctness candidate, device qualification pending
Date: 2026-10-10

## Problem and source binding

The shared matrix adapter promises mask preservation, but its cuBLAS branches
do not use `active`. Native kernels return before an inactive write. Both
production RHF/UHF and KS callers reach this adapter with resident masks.

The adapter's Git blob is identically
`fdeb1cde14b1715c23b51d20d718f887741f9790` at historical measured
`f6f9279f6d06909563abeef0b7b0ae767bd7fbc1` and master
`b9f344b1d84a419c59fb2833fcaac76919311a65`. PR #2213 reports 1,080 failing
masked-library cases among 2,880 retained H100 cases, including Graph replay.
This is primitive evidence; it does not prove an incorrect public KS singleton
endpoint. The historical negative receipt must not be relabeled as a candidate
pass.

## Proposed repair and proof

Borrow a disjoint precharged output-sized scratch span from the existing numeric
owner. Masked library GEMMs preserve their existing arithmetic, submissions and
strides but write only scratch. The existing selected-copy kernel then publishes
one system's output iff its device mask is nonzero. An inactive output therefore
receives no write at all, preserving signed zero and NaN payloads as well as
ordinary sentinels. Every step is ordered on the same borrowed stream. The
scratch span is reused only after the preceding copy and remains alive with the
arena across captured replay. No new CUDA arithmetic or mask host read is added.

The adapter checks representable positive shapes, signed CUDA grid bounds,
scratch capacity, and scratch overlap with all operands/output/mask before any
enqueue. Library submission failure does not enqueue the publication copy; a
later-spin failure may leave scratch modified but cannot publish partial masked
output. Asynchronous execution errors remain the caller's existing stream error
responsibility. This does not promise atomic failure semantics for unmasked GEMM.

RHF/UHF reserves the span in its checked arena when its existing resolved policy
chooses cuBLAS. KS uses its unchanged shape-only provider admission in its shared
allocation/resource-query partition. The numeric bytes are not hidden inside
the opaque library-retention allowance. No default/provider feature is disabled.

## Rejected alternatives

- A non-null mask is not permission to silently switch a forced library request
  to native. An absent scratch span is a rejected request instead
- Copying the mask to the host or synchronizing at each call breaks async and
  Graph semantics
- Device alpha=0/beta=1 is not a proof of preserving inactive bytes or avoiding
  NaN/signed-zero transformations
- Allocating inside the adapter breaks prepared ownership, budget and capture
  contracts; aliasing an unrelated live solver buffer requires a separate
  complete liveness proof

## Evidence and remaining gates

Host tests compile the complete adapter translation unit and real RHF arena,
execute the actual selected-copy kernel body under host launch-index emulation,
and independently compare active results with NumPy. They exercise changing
masks and input values on the same queued operations, inactive NaN inputs and
bit-pattern sentinels, every physical/spin broadcast, transpose/scale, guarded
capacity/overlap and provider errors. KS dry/materialized storage queries and
RHF native/library capacity deltas are checked directly. An unchanged-master
host control overwrites all 18 inactive elements of a three-system spin case.

No fresh GPU execution, full build, or complete SCF scientific qualification was
performed for this repair. The borrowed-span harness adaptation is separate from
#2213's protocol fix and must retain historical receipts unchanged. Final
exact-source review and required enforced CI govern the normal protected merge
queue. Source-matched CUDA/Graph and complete-endpoint qualification remain
necessary before the corresponding runtime, scientific or performance claims;
the disclosed absence of supplemental GPU evidence is not an additional merge
gate.

## Consequences and revisit condition

Masked library calls retain full-batch arithmetic plus one masked copy, and each
admitted owner retains one extra output-sized numeric span. This is a correctness
tradeoff, not a speedup. Revisit only with a qualified vendor masking primitive or
an independently proved charged workspace-reuse design that preserves these
same output, stream, Graph, provider, and resource contracts.

References: #1873, #2213; `docs/developer/cuda_matrix_products.md`.
