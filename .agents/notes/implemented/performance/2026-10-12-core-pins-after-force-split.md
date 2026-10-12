# Decision: retain core hotspot owners after the complete-force split

Status: implemented
Date: 2026-10-12

## Problem

PR #2368 originally moved six complete test modules to core-c using measurements
from before the complete UCCSD gradient modules acquired their own force shard.
After #2367 landed, that assumption was obsolete. Run 38268849163, at source head
`d004fa5922c5097b9e82f4361c4469ef3d5b319b`, finished core-a/core-b pytest in
1055.64/1105.91 seconds, but core-c hit the unchanged 30-minute step budget with
77% reported progress. Its workers collected 12,622 items before execution.

The same run's completed A/B JUnit reports sum to 2960.783/3217.280 worker-seconds.
The core-c debug artifact's monotonic start-to-next-start intervals attribute at
least 1027.729 worker-seconds to the six transferred modules; exchange-queue work
was still unfinished. These intervals include setup, teardown, and inter-test
overhead and are not bare scientific-call timings. The corrected-Lambda module
alone occupied 2062.358 worker-seconds, versus 972.102 seconds in the older JUnit
baseline. Its repeated setup is addressed separately by #2371. Core-c also spent
about 300 seconds before its first worker test started.

## Decision

Keep explicit whole-file pins, but put CUDA XC point contracts, the UCCSD solver,
and CC triples tiles on core-a; put DF Lambda mixed precision, the direct exchange
queue, and DF occupied triples on core-b. These are their ordinary owners in the
landed 1,664-file master inventory. Pins prevent future filename insertions from
silently shifting the expensive modules onto core-c. Every pinned file continues
to count in the modulo-three inventory before its owner override is applied.

## Invariants

- Preserve every scientific test, assertion, reference, event-specific exclusion,
  worker count, distribution mode, time budget, and cancellation policy
- Preserve each unrelated file's modulo slot, including when a pinned file is
  tracked but absent from disk; missing tracked files must fail closed in pytest
- Never select absent/untracked hotspot files merely because a pin exists
- Preserve dedicated force/compiler precedence and complete nightly selection

## Evidence and limits

The routing/budget suite exercises the real shell selector. Additional synthetic
inventories cover all five events, absent/untracked/tracked/missing pins, and
interleaved filename insertions. The actual inventory remains exactly once, and
the repaired current-master distribution is unchanged. Only the six intended
owners differ from the failing PR head.

Do not interpret workload sums as predicted elapsed-time savings. The source head
also has expensive collection and corrected-Lambda setup, and native cache state
differs from the older run. This ownership repair alone does not prove that every
timeout is cured; exact-head Actions remain the performance and merge gate.

## Rejected alternatives and revisit conditions

Keeping all six transfers used an obsolete capacity assumption. Removing the pins
entirely would restore current owners but resume filename-driven drift. Increasing
budgets or deleting scientific coverage would not repair the load regression.
Revisit the explicit pins when fresh complete-run data shows sustained imbalance,
especially after fixture and collection improvements land.

## References

- [Failing current run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38268849163)
- [Core-c diagnostic artifact](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38268849163/artifacts/11759295302)
- [Earlier baseline](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38251932879)
- [Dedicated force routing](https://github.com/jinzhezenggroup/generativeqc/pull/2367)
- [Independent fixture repair](https://github.com/jinzhezenggroup/generativeqc/pull/2371)
