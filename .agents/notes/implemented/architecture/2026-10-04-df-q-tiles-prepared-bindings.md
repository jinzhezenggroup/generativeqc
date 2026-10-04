# Decision: retain Q tiles through prepared contraction integration

Status: implemented
Date: 2026-10-04

## Problem

The auxiliary-Q tile work in #1914 used borrowed GEMM callbacks. Master acquired
prepared native contraction ownership in #1891, commit
`ff2c16a3aabd6d5a8505ca9be3227a938d6175b9`, before the tile branch landed. A merge
that retained callbacks would bypass the new descriptor lifetime and precision
contract; retaining only master's three tables would discard tile execution.

## Decision

Keep the one-Q prepare, auxiliary and iteration tables. Add a separate auxiliary
batch table with at most two variants, the admitted full tile and a nonunit tail.
One-Q tails use the original auxiliary table. Bind all shapes after optional
provider/arena admission, before any iterative work. Charge the full retained
host descriptor storage and its conservative construction copy in each candidate
budget, including both variants when present.

The owner retains the same optional tile-to-one-Q-to-scalar allocation retry.
Independent replay still reserves and executes its original one-Q action. Shared
native descriptors own provider dispatch, precise semantic summands, runtime
shape checks and sticky output auditing. The compiler-generated ordered consumer
still adds every Q contribution to the retained accumulator in ascending order;
no tile subtotal or new cast is introduced. Lambda reuses the same consumer with
its existing stage-specific descriptor guard and unchanged generated arithmetic.

## Rejected alternatives

Restoring legacy callbacks loses the prepared native contract. Preparing a tail
inside an iteration adds hidden work and uncharged storage. Combining a one-Q
variant with full/tail variants in one table violates the bounded two-variant
contract. A dedicated batched table avoids all three problems.

## Evidence and limits

- Host projection validates the new batched recipes against native semantic
  descriptor checks at nonrepresentative runtime shapes.
- The live generated bind functions exercise all one-Q/full/tail combinations
  and compare descriptor charges with the retained tables.
- Constructor fault injection and provider lifetime tests cover the actual owner
  retry/unwind paths, preserving the already-landed DIIS ring.
- Integration source emulation executes the generated primal prepare, auxiliary
  and ordered-consumer code together with the live prepared context/descriptor
  implementation at a host-emulated CUDA/cuBLAS boundary. This is source and
  semantic validation, not real-device qualification or endpoint timing.

The frozen endpoint receipts from #1914 retain their original source identity.
They are not rewritten or relabeled as evidence for this integrated tree. New
real-device or performance claims require their own exact-source qualification.

## References

- #1914, #1909, #1891
- [Prepared native descriptors](2026-10-05-native-contraction-bindings.md)
- [Auxiliary Q tiles](../performance/2026-10-05-df-cc-auxiliary-tiles.md)
