# Decision: bounded shared producer reuse in cooperative reductions

Status: implemented; bounded numerical qualification, production incomplete
Date: 2026-10-13

## Problem

Issue #783 retains repeated virtual work after #1680. The #1866 groups and
subsequent coordinate-proof repair describe overlapping candidates rather than
a safe execution partition. Equal shapes alone do not authorize scalar reuse,
and moving an output earlier can overwrite another allocation's live storage.

## Decision

Consume the existing coordinate-proved facts only under the default-off
`shared_streamed_reductions` schedule. Select deterministic disjoint subgroups
of two to four siblings, with at most 32 shared producers. Execute at the first
sibling only after proving all materialized leaves are ready and no early
output range intersects storage live before its original execution. Preserve
the existing allocation plan; reject groups rather than silently extending
lifetimes in an already allocated arena. Respect target shared-memory limits.
Reject recursively expanded scalar/read cost above 512 or inline depth above
64, including diamond DAGs whose source spelling grows exponentially.

Cache proved scalars per logical output/reduction coordinate. Reuse the
existing scalar emitter with explicit read substitution through only the
flat-index-preserving paths covered by the planner's proof. Opaque producers
keep their existing checked evaluator. Keep each sibling's accumulator dtype,
lane order, shuffle tree, coefficient scaling and publication checks.

Ordinary and resident launch emission share the same immutable partition.
Prepared provider recipes retain their separately qualified unfused execution;
the scientific consumer receives no provider-valued schedule axis. The schedule
field participates in executable identity; no equation changes or reuse of old
compiled artifacts is authorized. Cached plan facts are rebuilt on replacement.
The tuner execution key also includes the actual selected partition. Merely
requesting the flag with no admitted group does not split identical work.

## Rejected alternatives

- Launch all overlapping candidates: a sibling could execute multiple times.
- Fuse every RCCSD(T) sibling at once: its 36-output candidate creates an
  unbounded increase in register/shared-storage demand and compilation size.
- Move launches without storage proof: live input or intermediate slots can
  alias a future sibling's output.
- Memoize only the direct operand: expensive recursively shared producers
  remain independently evaluated through their pointwise descendants.
- Switch provider names in scientific code: bypasses semantic lowering owners.

## Evidence and limits

The unchanged upstream host baseline passed 163 tests with 12 device skips;
Linux's expanded baseline passed 237 with 12 device skips. Latest integrated
host validation passed 250 with 20 device skips. Tests cover
empty/tail domains, unsupported mappings, bounded/disjoint selection, storage
overlap rejection and target shared-memory rejection. Static work estimates
replace the selected groups' repeated scalar demand with cached demand; actual
hardware instruction counts and dynamic allocator peaks remain separate facts.

H200 / driver 570.124.06 / NVCC 12.9.86 retained two distinct source snapshots:
54 device/host tests for the bounded candidate and separate instrumented scalar
counts, then 21 latest-source tests including mixed accumulation, CUB fallback,
JVP/VJP, F-order intermediate storage, transpose aliases, CUDA graph replay and
negative-control recovery. Neither result is relabeled as the other's source.

The latest full (T) lifecycle includes control/static feed preparation, owner
creation, upload, all runtime pages, result download and cleanup after one
reusable specialization. Median default / unfused / shared times are:

| Synthetic shape | default | unfused | shared |
| --- | ---: | ---: | ---: |
| 4 occupied, 8 virtual | 52.560 ms | 47.679 ms | 46.793 ms |
| 8 occupied, 16 virtual | 111.687 ms | 200.884 ms | 122.368 ms |

Shared reuse improves the streamed route but the 8x16 full lifecycle is about
9.6% slower than the default. Sequential samples are not an interleaved
promotion guard. Earlier retained-owner timings exclude creation/cleanup and
must not be substituted for the complete lifecycle. The option stays off.

First-page instrumented virtual scalar-body entries drop from 18,419,712 to
7,507,968 (4x8), and 246,448,128 to 102,531,072 (8x16). These are separately
compiled diagnostic artifacts, not SM instructions or production timings.
CUPTI records successful launch APIs and scoped allocator peaks; PTXAS confirms
the production resource gates. Neither establishes total driver/context memory
or dynamic hardware register-live peaks. Verified sccache wraps the actual NVCC
driver, but combined `--shared` calls are reported as non-compilation calls with
zero hits; compiler-cache use must not be reported as a cache-hit benefit.

Raw records retain `NUMERICAL_PASS_ENDPOINT_RECORDED_QUALIFICATION_INCOMPLETE`.
The benchmark consumes synthetic supplied tensors and excludes the preceding
SCF/CCSD solve and complete molecular forces. No full RCCSD(T), production
acceptance or parent #783 closure is claimed. Reconstruction patches, checksums,
per-source numeric rows and the negative default comparison are retained under
`benchmarks/results/shared-streamed-reductions-20261013/`.

The exact-head review then exposed an emitter bookkeeping gap: an unrelated
1,200-node virtual chain is legal outside the paths actually inlined around
caches, but recursive reachability overflowed Python's stack. An unrelated
diamond also made the root-access discovery repeat paths exponentially.
The tested repair provided by `njzjz-bot` computes reachability topologically
and visits only reachable, previously unseen root-access nodes. Preserve this
implementation contribution when selecting a genuinely independent reviewer.
Host verification passed 252/20 skips; 15 before/after ordinary/resident
fixture pairs retain source bytes, partitions and identities. A separate new
H200 source-bound suite passed 23 tests and repeated representative (T) oracle,
replay/recovery, PTXAS and lifecycle checks. These results belong to the repair
snapshot in `reach-repair-records.json.gz`, not the earlier recorded source.

## Revisit when

Source-matched device measurements justify different subgroup bounds or a
planner-owned lifetime reorder, and complete endpoint evidence justifies
promotion. Provider-bound fused recipes require their own common contract.
