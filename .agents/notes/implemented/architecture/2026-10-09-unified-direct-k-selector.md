# Decision: one prepared selector for Rys-task and work-scheduled direct K

Status: implemented
Date: 2026-10-09

## Problem

PR #2133 promotes a qualified quartet-parallel Rys-task producer, while #2135
promotes primitive-work scheduling of incumbent generated K producers. Their
launcher overloads both take four scalar arguments with different meanings:
the third is either a task mask or a class, and the fourth a class or a boolean.
Combining them by overload resolution can silently select the wrong producer.
Separate prepared schedule and mask fields also allow topology and dispatch
policy to drift during integration.

## Decision

`DirectExchangeSelection` freezes all three alternative masks and the queue
schedule in one value. Preparation parses the K lowering once, intersects each
alternative with reachable value classes, and uses the registry's qualified
preference only for unset/empty lowering. Launch takes the typed selection,
class and spin contract, without environment reads.

Precedence is Rys-task, restricted-only block, component-Rys, work, incumbent.
Only one alternative family is enabled by preparation; precedence remains
deterministic for overlapping supplied masks. The work registry retains its
existing unsupported-class fallback. Rys-task's existing multiwarp queues use
cross-chunk filling for the Work enum, without a new task work-kernel variant.
J/HF keep their two-argument adapter and zero-initialized incumbent scheduling.

The two environment controls remain independent. Lowering `incumbent` rolls
back task preference, not work scheduling. Task `fill` rolls back work buckets,
not task preference. Set both for the defaults preceding the two promotions;
task `incumbent` additionally disables cross-chunk filling.

## Rejected alternatives

- Adding another scalar/boolean overload retains silently convertible argument
  ambiguity and distributes policy across call sites.
- Giving work priority over Rys-task discards the target/class-qualified producer.
- Reimplementing task recurrence inside work kernels changes scientific/source
  identity unnecessarily; preserve both independently qualified leaf producers.
- Treating either environment control as a global rollback conflates independent
  execution choices and changes the explicit experimental contract.

## Invariants

Screening, recurrence, normalization, FP64 accumulation, symmetry/spin semantics
and force equations are unchanged. Prepared topology and launcher use the same
frozen schedule. Failed launches propagate without retry into partial output.
Capability and automatic preference remain distinct; no target/class promotion
is added beyond #2133. Optional bounded storage fallbacks remain intact.

## Evidence

Integration starts at master `6b439bb003d50b6dd854e81400977195bb4bf460` and combines
#2133 `76ade1377531cbfe953d8462578c29272dd0c175` with
#2135 `6e97c0d2379dc68b411aaa010c9088a2535e0578` without rewriting their receipts.
Host probes exercise both controls, unset/empty values, reachable masks, both
spins, frozen environment mutations, invalid values, dispatch precedence and
the real no-AOT stub. Existing emitted concurrent-worker admission censuses
remain applicable to both leaf families. Host production compilation uses
verified ccache and CUDA 12.9.1 headers.

Independent emission from each frozen PR verifies byte identity for 21 incumbent
streams, their work companions, all twelve task-class shards and the old
component-Rys/K-block shards. All six pre-companion retained-source hashes stay
pinned. The production bundle fixture changes deliberately only in the four
sm_120 shards and registry source; validation-only companion omission does not
alter production emission or registry expectations. The concrete host provider
and generated registry both compile through ccache.

The selector probe covers 2,592 combinations of lowering, scheduling, reachable
coverage, class and spin. The focused host suite passes 211 tests with 18 PySCF
dependency skips; the extended suite also exercises compiler/module ownership,
source identities, queue admission, ABI and bounded allocation contracts.
The standard-library `functools.partial` adapter introduced by the work emitter
is admitted explicitly without relaxing its filesystem or ownership bans.
The final combined host run passes 1,516 tests with 19 skips: 18 lack the
optional PySCF reference dependency and one optional resident-force CUDA compile
gate has no configured `GENERATIVEQC_NVCC`. Compiler, shared-SCF,
electronic-structure and codegen-test ownership checks report zero errors;
the default inventory registers all 85 discovered controls. Current local Slurm
exposed only drained `node3` at the initial offline integration stage, so that
snapshot did not claim combined GPU endpoint or sanitizer qualification.

Cold v2 records retain effective task scheduling and reject mixed schedules.
Legacy v1 receipts remain schedule-unspecified. The separate historical GPU
qualification does not establish a combined cold-speedup claim; new joint
endpoint/oracle/sanitizer qualification must use finite Slurm allocations.

## Consequences and revisit conditions

There is one prepared K dispatch contract and no additional device allocation.
Revisit precedence or coupling only with target/class-specific independent
matrix gates, complete endpoint work/timing evidence and explicit rollbacks.

## Subsequent qualification and queue ownership

The integration was reconciled onto master
`6acb3e70e3031b4a7ef44ef77df551b354e34e59`, retaining the merged #2135 reviewed
head `e903d467a62b8aa4ace2ab6513912033e571f436`'s source-retention checks and
historical fixture correction. A same-binary exclusive-node campaign of the
original union passed scientific/sanitizer gates but failed both 96-atom Cold
median comparisons; its 48-atom gains did not justify publication.

The initial decision to interpret task `work` as Fill is superseded by
[warp-private task work buckets](../performance/2026-10-09-rys-task-work-buckets.md).
The typed selector, independent rollback axes, incumbent/work byte snapshots
and original task arithmetic remain intact. Consult that note for current
qualification rather than treating the offline or separate-PR receipts as
qualification of the combined policy.
