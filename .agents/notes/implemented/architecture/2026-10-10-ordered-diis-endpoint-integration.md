# Decision: integrate ordered DIIS through explicit endpoint selection

Status: implemented
Date: 2026-10-10

## Context and decision

Draft #2190 at `ca51f2e53f62fa21548b4c83296b8631c8e431e8` preserves an
ordered-FP64 incremental Gram implementation against `f87d51ab`. Its 19-file
delta has eleven paths changed by upstream, including merged #2159 (`828b8c1b0`).
The original unconditional endpoint and arena replacements cannot be applied
without overriding the qualified serial-default / opt-in pending-row contract.

The reconciled implementation retains `GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM`:
unset/0 is serial and 1 admits the existing separately charged raw Gram cache.
A subordinate `GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION` selects the
reducer. Unset/cooperative preserves #2159's existing pending-row path; ordered
uses the generated scalar-order dot and in-kernel lazy refresh. The subordinate
value is consulted only when incremental DIIS is admitted and history is at
least two. Invalid active values fail shared resource-query/execution admission.
No second cache or unconditional numeric allocation is introduced.

HF and both KS paths call the new ordered entry for the explicit ordered choice;
that branch does not submit the cooperative pending-row grid. HF remains
unnormalized, KS remains normalized, and KS final closure still bypasses DIIS.
This is a reachable endpoint feature, not an unused internal primitive.

## Plan and numerical ownership

The reducer is captured in each plan. HF's actual delete/recreate gate compares
both mode flags, with a second fail-closed check in the driver. Merely adding the
driver comparison would reject rather than rebuild a cached plan. A mode change
therefore discards prior graph/history ownership before execution. KS retains
its constructor-selected mode for the plan's entire lifetime, including captured
and ordinary execution. Existing beginning/refinement resets set count=head=0.

Keep fixed dimensions, immutable live residuals, ordered stream operations and
disjoint history/cache/solve buffers for a live cache. Reset count=head=0 before
changing routes. Ordered insertion one intentionally leaves its norm unwritten;
a subsequent cooperative insertion would not repair that norm. Switching from
serial or cooperative state can instead retain missing or differently rounded
old-old entries. Dimensions and count/head alone are not a numerical cache key.

The ordered route preserves the incumbent scalar FP64 contraction order. It
performs zero dots on insertion one, computes the missing prior diagonal on
insertion two, and thereafter refreshes the live new row only. Normalized
retirement narrows the valid submatrix without corrupting raw cache entries.
Optional caller-owned work counters measure actual vector-dot work, not endpoint
performance or public work diagnostics. The existing cooperative reducer keeps
its separate qualified reduction identity.

## Correctness repairs

The original ordered entry lacked master's invalid count/head guard. An invalid
head could write outside a system's history, and invalid counts could admit
unestablished cache entries. Both incremental instantiations now share the
pre-write reset-to-zero/copy-current-Fock behavior. Histories zero/one still admit
null history/cache pointers and perform only the original Fock copy.

Reject exact cache/solve aliasing before launch because solve clearing otherwise
destroys the cache. The ordered entry also validates positive dimensions,
spin/history bounds, pointers, one-dimensional grid, complete warp, sufficient
grid coverage and division-based storage extents. Arbitrary partially overlapping
allocations remain a documented caller-precondition violation, as for the
incumbent pending-cache interface.

## Rejected approaches

- Replacing #2159 or enabling ordered by default would silently change its policy
  and conflate numerical work removal with default/crossover qualification.
- Reinterpreting incremental=1 alone would change the existing reducer identity.
- Landing only an unused internal entry would not deliver the requested usable
  endpoint integration. That earlier local sketch was superseded.
- Switching reducers on a live ring would retain incompatible numerical state.

## Evidence and limits

The source is reconciled against exact master
`985caaf01688bd6df3388b2c7985af3176efbf9d` (9,388 verified blobs). An independent
non-implementing source reviewer found no correctness/default regression in the
reachable integration. Arena files, compact DF code, pending-row/DF partial
launchers and existing raw-cache launcher ABI remain preserved.

A compiled host probe executes the real generated dot/ring adapter through a
lane shim for 378 arithmetic/lifetime cases, including capacities 2 through 64,
bitwise scalar order, independent long-double dots, wrap, retirement, inactive
neighbors, uncleared resets and actual-work counters. Another compiles the real
admission body with a launch recorder. Removing alias rejection or lazy prior
initialization causes the corresponding check to fail.

Selector and actual HF/two-KS dispatch fragments are compiled with launch
recorders over a 96-case mode/history matrix. Public KS resource and fleet-budget
probes exercise both reducers through the production query/selector code and
confirm equal cache charges, inactive malformed-value handling and active
rejection. HF native resource regressions are extended for equal reducer charges and
invalid active admission, but are not executed without the full native build. The final focused host suite reports 116 passed and one skip. Source-structure,
ownership, formatting and exact-tree patch checks also pass; their receipts are
retained with the source-matched review artifacts.

These host tests are not CUDA compilation, synchronization, graph replay,
device endpoint or performance measurements. CUDA tooling/device and a full
native library are absent here. Historical #2190 combined-tree GPU evidence does
not become evidence for this reconciled source. Source review and host checks
are the current evidence; no endpoint speedup, default promotion, fresh GPU qualification or issue closure
is claimed.

The original implementation's measured identities remain preserved without
reinterpretation in [the historical ordered-Gram note](2026-10-10-incremental-ordered-diis-gram.md).

## References

- https://github.com/jinzhezenggroup/generativeqc/pull/2190
- https://github.com/jinzhezenggroup/generativeqc/pull/2159
