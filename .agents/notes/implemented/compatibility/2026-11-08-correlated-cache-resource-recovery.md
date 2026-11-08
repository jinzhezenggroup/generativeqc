# Decision: Recover optional correlated caches before resource failure

Status: implemented
Date: 2026-11-08

## Problem

Review of #3046 found that projected MP2 derivatives reserved only the first
stage before canonical shell callbacks, so later allocations could fail after
partial accumulation. Energy-cache admission ignored energy-only diagnostic
controls and retained optional storage while retrying a resource-refused energy
phase. Persistent RHF iteration graphs could replay a previous cold/warm startup
policy.

## Decision and invariants

Reserve all four coexisting projected staging capacities within the existing
full staging bound, before shell callbacks. Reuse those owners throughout the
projection; release every reservation on resource refusal before taking the
compact ordered route. Do not replay after partially accumulated shell work.

Honor energy-only cache/tiling disables, including the one-block fallback when a
force cover cannot fit. On borrowed-energy length/allocation failure, retire the
optional cache, consume only the expected CUDA allocation status, and retry the
streamed route. Numerical and unrelated backend failures still propagate.

Recapture a persistent RHF iteration graph when cold/warm startup policy changes.
Transient resident/screened captures must not overwrite the persistent graph's
policy metadata. Normal physical stopping gates and arithmetic remain unchanged.

## Evidence and boundaries

Host regressions inject refusal at each projected staging reservation, compare
with independent dense/compact pullbacks, exercise cache controls and typed
resource/numerical failures, and replay cold/warm/transient graph transitions.
Existing #3046 GPU CBS receipts keep their original source and binary identities.
This repair does not claim a new GPU endpoint, performance or peak-memory campaign.

## Rejected alternatives

Catching allocation failures after shell callbacks would risk double-counting a
partial gradient. Retaining the optional MO cache during streamed retry defeats
the intended resource fallback. One plan-level policy updated by every transient
graph capture can falsely license reuse of an older persistent graph.
