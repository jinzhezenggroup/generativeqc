# Decision: expose graph setup and resource lifetime fences under their real owners

Status: implemented
Date: 2026-10-10

## Problem

The explicit HF input extension qualifies all 147 observed H2D records in its
public RHF/UHF histories, but leaves 18 stream synchronization records without
source ownership. API names and matching counts identify candidates, not the
actual owners. A cleanup fence can occur during an endpoint, so classifying all
resource teardown as harmless final publication would hide operational costs.

## Decision

Declare graph capture/upload drains at their existing call sites under
`hf_graph_setup` with a preparation role. Declare bucket and ordinary eigensolver
release fences under their respective resource owners and a new `lifetime` role.
The added scalar scopes preserve conditional calls, streams, device restoration,
CUDA statuses, ignored destructor returns, exception unwinding, resource release
order and scientific launch order. They add no CUDA operation, allocation,
completion fence or production CPU/reference path.

Lifetime work is conservatively higher risk, including during public close;
the source tag alone never authorizes a legal hot-path exemption. Preparation
declarations apply to these exact graph-management calls, not entire files or
any arbitrary synchronization under a public execute phase.

Expose `by_owner` alongside `by_role` as separate projections of the same joined
source/API/activity records. Neither projection is an extra physical work count.
Require exact runtime CUDA return-code agreement: two different nonzero errors
are not matching observations. Failed source calls and observed synchronization
activity continue to be separate quantities.

The graph-only callback ignores the recognized independent source layout 2.
Combined capture validates both stores; unknown layouts remain failures. Without
this demultiplexing, legal boundary records delivered by the shared submitting-
thread callback would incorrectly invalidate a graph-only inventory. This is
compatibility between observation domains, not a scientific fallback.

## Invariants

- Layout 1 graph records and layout 2, 14-word boundary records remain independent.
- Existing owner/role/site numbers remain stable; new declarations append tags.
- Native artifacts remain the authority for tag names; no shadow Python registry.
- Source storage stays preallocated and bounded; loss and correlation faults fail closed.
- Nested owner invocations do not inherit the enclosing bucket's publication role.
- CUDA APIs still run exactly as before, even if observation fails or is disabled.
- Owned fields and helper invocations are not scientific producer/transform/consumer links.
- Full dependency, implicit-wait and graph-execution coverage remain incomplete;
  matching all observed explicit transfer/sync records cannot authorize PASS.

## Evidence

CPU tests compile the real SDK/collector with CUDA/CUPTI doubles, not real GPUs.
They cover nested bucket/graph/resource scopes, independent owner projections,
conservative lifetime risk, exact failed-status matching, graph-only foreign-
layout handling and unknown-layout rejection. The real graph lifecycle test
retains original exception, failure, retry and launch behavior and now uses the
verified native compiler-cache fixture. Scheduled qualification is separate;
no existing source tree is retroactively relabeled for this extension.

## Scheduled qualification (appended after source freeze)

Slurm job **6931**, `main`, `gpu:5090:1`, `node1`, completed with exit 0
on 2026-10-10 under a finite ten-minute allocation. The pre-evidence source tree
is `141757363af3f114d1fbbe861cde59fd06853992` (9,194 manifest entries), native
library SHA-256 `89316e9d4be2125633d7260854232146c670ac06c5fb236b7adb47ceae48c6d3`,
and combined collector SHA-256
`72d1fef064e9910c50f5dd45df5b6139c72cdb00a02ca768df7cec373913499b`.
The Release sm_120 build reuses verified ccache 4.5.1; AOT remains disabled.
Appending this evidence does not retroactively change the frozen tree.

Both RHF and UHF public prepared histories contain 363 matched source operations,
8 graph-preparation fences and 4 bucket-resource release fences. Each retains
**six unannotated production synchronization records**, down from 18 in the
previous boundary slice. The ordinary eigensolver resource owner is absent from
these measured histories; instrumentation presence is not execution coverage.
All activity/source/graph stores are intact, with no reported loss, parse,
dispatch or outstanding-operation errors. `by_owner` and `by_role` are alternate
projections and cannot be added together. The four lifetime fences remain higher
risk. The graph-only observer also keeps intact source graph inventory in a
public CUDA H2 first/warm run; its independent CPU energy error is
`4.440892098500626e-16`.

RHF/UHF H2D totals remain 6,649/8,041 bytes and D2H totals 3,306/5,586 bytes.
Maximum matched ordinary-CUDA energy error is `2.842170943040401e-14` for each method;
force maxima are `4.884981308350689e-15` and `7.216449660063518e-15`.
All endpoint iteration counts match, within the 1e-10 energy / 1e-9 force gates.
The scheduled resource regression suite passes 39 tests and source-matched
allocation receipts pass for both methods. Local CPU suites pass 78 focused
tests and 274 broader tests (57 subtests); these CPU suites are not GPU evidence.

Ignored local `.artifacts/issue1629-hf-lifetimes/` retains the source archive,
manifest, exact binaries/build configuration, raw CUPTI and numerical histories,
Memray allocation receipts, Slurm controller record and logs. `aggregate.py`
uses the archived source to reverify its tree and recompute saved diagnostics;
`qualification-summary.json` retains those results. Archive and binary checksums
were reverified before this append. All residency results remain `INCOMPLETE`,
with no zero-round-trip or endpoint timing claim.

Numerical-reference clarification: the RHF/UHF history errors above compare
observed execution with ordinary CUDA execution outside capture, not with CPU.
Only the separately named graph-only H2 check uses a CPU reference in this
qualification. The earlier CPU label for the history error was incorrect;
frozen raw data and scientific code are unchanged. The later observed-work
ratchet qualification adds separately labeled native-CPU backend comparisons.

The subsequent [device-ledger lifetime owner slice](2026-10-10-device-ledger-lifetime-fence-owner.md)
qualifies the six remaining explicit fences under a different frozen tree; it
does not retroactively relabel this graph/resource-only evidence or complete
scientific dependency and implicit/graph-wait coverage.

## Remaining scope

Other scientific owners (including the explicit host-staged DF consumer), real
cross-owner scientific dependency links, dynamic/device-tail graph execution and
implicit blocking still require independent qualification. No endpoint timing
or repository-wide zero-unexpected-round-trip claim follows from this slice.

## References

- [Source-owned boundary declarations](2026-10-10-source-owned-residency-boundaries.md)
- [Source-owned graph inventory](2026-10-10-source-owned-cuda-graph-inventory.md)
- Current contract: `docs/maintainer/residency_receipts.md`
