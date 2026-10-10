# Decision: source-owned bounded CUDA graph inspection

Status: implemented
Date: 2026-10-10

## Problem

The actual CUPTI pending-event control in Slurm job 6903 contains a graph wait
that blocks twice, but records zero graph synchronization waits. Therefore
activity counters alone cannot qualify the graph layer of #1629. The earlier
[observed activity decision](2026-10-10-cupti-observed-residency-activity.md)
remains valid; this additional source observation does not retroactively change
that qualification or its source/library identities.

## Decision

A private method-neutral, thread-local source observer ABI is always available
without linking CUPTI into production. RHF/UHF's retained graph owner emits its
actual definition, host submission and destruction boundaries. A separately
compiled optional CUPTI/CUDA sidecar inspects the graph synchronously at those
boundaries, outside CUDA/CUPTI API callbacks. Node storage and total traversal
are bounded before execution; child graphs are recursively enumerated with
independent depth limits and cycle detection. Conditional nodes are retained but
not treated as a statically executed branch.

Source lifetime generations are assigned even when observation is disabled.
Attaching late cannot fabricate a previously observed definition. Borrowed CUDA
handles never escape the callback; stored identities are source generations and
CUPTI graph/exec/node IDs. A distinct CUSTOM1 correlation stack connects actual
host graph-launch APIs to source submissions while CUSTOM0 continues to own the
public lifecycle phase. Source callbacks cannot change graph ownership, CUDA
return statuses, equations, capture exception cleanup or retry behavior.

## Rejected alternatives

- CUDA graph inspection inside CUPTI API callbacks risks unsafe runtime reentry.
- Pointer identity conflates separately owned lifetimes after address reuse.
- A static wait node multiplied by host launch count is not a wait execution
  count, especially with device tails or conditionals.
- Adding CUPTI as a production dependency changes the execution contract.
- A thread-local observer cannot claim worker-thread or other-owner coverage.

## Invariants

- Only the matching callback/context can detach; replacement during dispatch is
  rejected. Exceptions and recursion are observation errors, not solver errors.
- Each successful instantiation has a lifetime generation; destruction is
  observed before handles are invalidated. Rebuilds acquire a new generation.
- Bounded record overflow, traversal/query failure, unknown node types/flags,
  unmatched launch correlation and incomplete lifetimes prevent intact receipts.
- Failed host submissions are retained, not counted as accepted graph work.
- Successful collection remains INCOMPLETE. Scientific payload dependencies,
  graph execution multiplicity and implicit blocking remain unqualified.

## Evidence

CPU lifecycle tests execute the real HF graph owner with deterministic CUDA
doubles: exception abandonment/retry, recapture, thread isolation, device flags,
callback exceptions/recursion and generation exhaustion preserve behavior.
Native sidecar probes use actual CUPTI 28/CUDA headers but link only CPU doubles.
They cover nested event/semaphore nodes, conditionals, unknown types, capacity,
depth/node bounds, CUDA/CUPTI query failures, pending/failed submissions and
no-write short reads. Protocol tests ensure zero synchronization activity cannot
hide declared graph waits, and never promote static inventory to execution
coverage or a residency PASS. These tests are not GPU qualification.

### Scheduled qualification

Slurm job 6915 on node1/RTX 5090 exercised the separately compiled actual retained
HF graph owner in an independent pending-event control. One declared external event
wait node and two accepted host submissions were observed. Independent event
queries returned not-ready before both submissions, and the consumer stream
blocked about 0.25 seconds twice. CUPTI still reported zero graph event waits.
The source inventory exposed the wait node but correctly left execution counts
and complete wait coverage unset. This is not endpoint timing or speedup evidence.

The same job captured real public prepared RHF/UHF endpoints with matched ordinary
history. Four observed graph lifetimes contained 16/16/21/25 nodes (RHF) and
17/17/22/26 nodes (UHF), with 2/2/3/3 accepted host submissions. Actual device-launch
flags were set on all four, so dynamic execution remains explicitly unqualified
even though no static event/semaphore nodes were found in these particular graphs.
The 2762/2822 activity records and 106/110 inventory records have zero losses,
query/dispatch errors, outstanding submissions or unassigned activity. Matched
energy error is below `5e-14` Eh and force error below `4.4e-15` Eh/bohr. Iteration
histories remain `[2,8,2]`, `[1,1,1]`, `[2,2,2]`, `[1,1,1]`, `[2,2,2]`. These gates
check instrumentation non-regression, not a newly qualified scientific method.

Slurm job 6916 explicitly opted into the two ragged direct-HF resource regressions
(both passed) and reproduced both RHF/UHF allocation receipts as PASS. Warm owned
device allocation counts/requested bytes remain zero; intercepted host allocations
remain honestly nonzero. The earlier unopted resource invocation in job 6915
skipped all 18 cases and is not counted as regression evidence.

Both scheduled jobs use source tree
`6e5ee96388dd59c8efb87e2e5a5c97bc661bb03e`, rebuilt native library SHA-256
`3045cc226da7d443e54d6206adb0aa000290ffb6cab2fe22acc3942c15f54792`,
and collector SHA-256
`89d6ea902e7eb34753ce939d5a1a1b8b4790c46cec6ffd162140ac9f815bb220`.
The native build reuses verified ccache 4.5.1 and retains its existing Release
sm_120/AOT-disabled qualification configuration; no defaults are promoted.
Raw records, independent contracts, numerical gates, native input/configuration
hashes, logs, actual binaries and the frozen source archive are retained locally
in ignored `.artifacts/issue1629-graph-owners/`. This evidence append is later than
the frozen qualification tree and does not rewrite it. Broader CPU validation
passed 185 tests and 57 subtests; formatter/lint and whitespace checks passed.

## Consequences and remaining scope

The disabled observer adds no CUDA operation or request allocation. Optional
inspection changes profiling work, not scientific work; no latency claim is
made. Only RHF/UHF's retained graphs are observed. Shared general graph owners,
graph mutation, device-launched execution counts, opaque callbacks, implicit
waits and payload/host-transform dependencies remain required for full #1629.

## Revisit when

A source owner can bind payload publication/iteration roles and dependencies,
another graph owner is explicitly covered, or actual device/conditional graph
execution can be independently calibrated. Current usage is documented in
`docs/maintainer/residency_receipts.md`.
