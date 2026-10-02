# Decision: stream large Becke pair rows through a bounded cooperative block

Status: implemented, opt-in pending NVIDIA qualification
Date: 2026-10-02

## Problem

The stationary owner can admit 128 atoms after #1703, but its only cooperative
Becke schedule retains all 64-byte pair states at once and is limited to 32 atoms.
At 96 atoms the full table would require 291,840 dynamic shared bytes per block.
Merely raising the atom cutoff cannot satisfy the existing device/resource
contract. The generic large-system route still executes both pair passes serially
inside each point worker. Earlier A/B work already distributes point workers and
retains center geometry; repeating those changes would not address this cliff.

## Decision

Keep the existing full-pair retained schedule through 32 atoms. For 33–128 atoms,
when cooperative execution is explicitly requested, stream four canonical
triangular rows per block through a bounded shared strip. All blocks still own
independent point workers. A forward strip generates factor/log state and gathers
only incident edges into the per-atom log products. After global-in-point
normalization, a reverse strip regenerates only the needed pair state and consumes
its adjoint before reusing storage. There is no floating-point atomic reduction.

Both schedules call the same existing scalar norm, ratio, polynomial and log
Graphs, plus the shared pair construction, pair-adjoint, normalization and
point-motion functions. The reverse strip deliberately calls `point_pair<false>`
and `pair_adjoint<false>`, preserving lazy reverse logarithms and the saturated
slope exit. Tiling changes scheduling, not the Becke partition or its derivative.

The per-atom gather visits its lower neighbors only when its own row is in the
strip, followed by later strip rows. This preserves each atom's original
triangular subsequence and avoids scanning every possible neighbor per strip.
A team integer validity vote publishes producer state and propagates failure
collectively; it does not replicate a whole strip scan in every thread. The vote
uses the already charged control padding as a separate integer field. It must
not modify the AO-admission predicate: a fast lane could otherwise invalidate
that predicate while a slower lane is still deciding whether to enter the team.
The vote also has an exit barrier after reading its result, preventing a faster
lane from publishing a later failure before another lane reads the prior vote. Final owner motion stays ordered.

## Resources, work and lifetime

- Largest strip capacity is `r*(2*natom-r-1)/2` pair states, `r=min(4,natom-1)`
- Shared bytes are `16 + 64*capacity`, including the existing static control
- At 96 atoms: 374 states / 23,952 total shared bytes per block
- At 128 atoms: 502 states / 32,144 total shared bytes per block
- Persistent point lanes, the 8 MiB lane-scratch cap, arena bytes, optional center
  cache and admission budgets are unchanged
- Native configuration independently rechecks actual device shared/launch caps;
  a mismatch retains the admitted generic route
- Every unordered pair is evaluated exactly once in each of the two passes.
  Large tiled cooperation does not claim C's small-domain 2-to-1 pair-state reuse
- Logical pair visits and pair-state metrics retain their full counts. Direct
  center norms and generated log calls do not gain extra passes
- No pair screening, grid-point dropping, approximate arithmetic or change to
  scientific tolerances is introduced
- State dies at each point/tile/reset. Stream ownership, sticky upstream failures,
  transactional output and bounded submission windows retain their current rules

## Evidence and qualification

Host-thread tests execute the actual emitted helper with independent barriers,
1/7/32 workers, 1/4/7-row strips, partition iterations 1/3/5, atom counts through
128, cached/direct centers, changed geometry, canaries and the existing exact-zero,
clipping, nonfinite/collision and empty-tail cases. Instrumentation requires exact
pair, norm and log work-count parity with the generic helper. Independent
60-digit Decimal directional finite differences, translation and permutation
cover both a small case and a multi-strip 11-atom case.

The emitted geometry-kernel host harness covers the native-to-generated seam at
12/33/96/128 atoms, implicit/explicit owners, external seeds, tails and sticky
failures. Native admission tests compile and execute actual configuration code
against explicit device-cap stubs. Capacity-audit fingerprints are refreshed from
the changed source and the mutation-rejection suite remains required.

The opt-in device suite has dedicated 67-point PBE0 geometry probes at 96 and 128
atoms, comparing generic/tiled schedules, cached/direct centers, tight/full point
capacity, tails and changed geometry. This is a targeted diagnostic, not a full
96-atom benchmark campaign. Complete force oracles retain the existing opt-in
qualification switch and strict gates.

GPU_NOT_RUN. Host threads and work counts do not establish device correctness,
occupancy, absence of device races, or endpoint speedup. Before automatic promotion,
run the exact head's targeted CUDA probes and Compute Sanitizer, then complete
cold/warm/changed-geometry endpoint non-regression under the existing policy.
The production default remains generic. The AO/XC contraction inside a cooperative
point remains serial and is explicitly not claimed to be accelerated here.

## Rejected alternatives

- Raising the old cutoff retains an unbounded quadratic shared-memory demand
- Retaining every pair state in global memory would multiply scratch by all
  simultaneous point workers and trade an execution cliff for a memory cliff
- Reducing point workers to fit pair-state storage changes independent concurrency
  and the user's existing budget allocation without performance evidence
- Exact or approximate locality pruning is a different scientific/numerical policy;
  it is not needed to distribute the current complete pair traversal

References: #1672, #1684, #1685, #1703; previous
`2026-10-01-cooperative-becke-point-state.md` note; `test_becke_tiled_cooperative.py`.
