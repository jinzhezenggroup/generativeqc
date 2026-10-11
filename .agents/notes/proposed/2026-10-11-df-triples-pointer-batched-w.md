# Proposal: pointer-batched pairs of FP64 occupied W seeds

Status: proposed
Date: 2026-10-11

## Problem

After #2232/#2235, the bounded energy owner still dispatches two ordinary
contractions for each of o^3 distinct W seeds. Two seeds sharing their first
occupied index borrow the same integral panel and publish to independent moment
cubes. Their other occupied indices are exchanged. Batching those two existing
products changes provider dispatch, not the mathematical W inventory.

Do not reopen unconditional two-sided ladder dressing: its retained degree-five
cancellation counterexample is already decisive. Do not revive the cuBLASLt
prototype's unresolved resource/concurrency boundary for this experiment.

## Isolated evidence

Ignored artifacts are under
`.artifacts/df-triples-w-batched-triage-20261011/` and
`n2:/data/jzzeng/qc-df-triples-w-batched-triage-20261011/`.
The experiment consumes the actual emitted W recipe at source head
31c3d6b353fd9f64e460c505bcbffab7ed98f516. No production source, endpoint library,
public ABI or provider default changes.

Finite Slurm jobs 2881, 2882 and 2883 run on node2/main/gpu:pro6000:1 with
five-minute limits and assigned device visibility intact. Each is terminal
with exit zero. The provider reports cuBLAS 120901, CUDA runtime 12090 and
driver 13020 on an RTX PRO 6000 Blackwell Workstation Edition. Both arms use
pedantic FP64, atomics disabled and the same four-MiB workspace.

| Isolated complete two-seed action | Ordinary median ms | Candidate median ms | Ratio |
| --- | ---: | ---: | ---: |
| Raw first-product batching, 2881 | 6.981408119201660 | 6.628091812133789 | 1.053306x |
| Checked first-product batching, 2882 | 7.056676149368286 | 6.741908073425293 | 1.046688x |
| Checked both-product batching, 2883 | 7.055717945098877 | 6.599434137344360 | 1.069140x |

Each campaign retains all 32 measurements in ordinary/candidate/candidate/
ordinary order, eight repetitions per measurement. Do not pool campaigns or
present these repeated component samples as fresh-process endpoint statistics.
Host stream-wall medians agree with the device-action signal; these are still
component measurements, not complete CCSD(T) timing.

The raw probe groups the two first products before both second products and
omits production finite-audit kernels. Its signal only motivated the checked
probe. The checked probes restore the ordinary control's actual sequence:
first W1, audit, first W2, audit, second W1, audit, second W2, audit. Candidate
products are grouped only across independent outputs; their W1 audits still
precede W2 accumulation. The receipt field named
`matches_original_seed_execution_order` describes this **control** sequence,
not a claim that batching preserves the candidate's original enqueue order.

All probes compare every one of 21,587,722 output values against the ordinary
arm: maximum difference is zero. Sixty-four separately accumulated long-double
samples check both contractions and both occupied views; maximum error is
2.4379370050509053e-15 under the micro-only 1e-11 gate. This is not independent
molecular energy qualification.

Both checked arms audit 43,175,444 output values per complete two-seed action.
The both-product candidate changes four GEMM dispatches/four audit kernels to
two dispatches/two audit kernels plus two pointer-binding kernels. Every
binding launch is inside the measured action. The same 48-byte device pointer
table is reused in stream order; no panel copy, zero-stride vendor assumption,
new numeric cube or CPU scientific work is introduced. Sticky finite state is
checked after qualification and after the complete measurement sequence.

For o9, there are 324 two-seed groups and 81 singletons. Scientific W work
remains 729 seeds / 1458 products. Both-product batching would reduce W driver
calls from 1458 to 810 and add 648 pointer-binding launches; panel builds and
contraction summands must remain unchanged. A projection from these isolated
medians is not a verified endpoint gain.

The raw combined compile/link invocation passes through ccache but is
uncacheable. Checked jobs split cached object compilation from linking and
retain per-command cache logs proving supported cache misses. No existing
compiler cache is cleared or disabled.

`verify.py` checks terminal status, all 32 measurements, emitted/source hashes,
exact dispatch/binding/audit counts, work and numerical receipts without GPU
execution. An initial local metadata assertion incorrectly expected four raw
candidate dispatches; it is repaired without repeating any GPU action. The
both-product input metadata initially copied first-product counts. Its corrected
metadata retains the original as `initial-first-product-provenance.json`;
`qualified-probe-receipt.json` binds the accurate counts and unchanged original
numerical/timing output bytes.

## Required next gates

Only a useful matched complete endpoint warrants a production implementation
and PR. First build an isolated actual-owner candidate against the exact
qualified #2235 library, replacing the minimum immutable-checked objects. Retain
the #2171 HF / #2221/#2232/#2235 CC consumer boundary instead of claiming a full
latest-master library or latest-master HF/force qualification.

Before promotion:

- Bind grouping to the compiler-owned original two W descriptors and provider
  capabilities; do not add a native shape-only vendor override or hidden hook.
- Admit and journal the pointer storage/host binding explicitly. Refusal/OOM
  must retain existing panel/precision choices and original W execution rather
  than shrink an earlier resource choice merely to admit this optimization.
- Preserve all singleton/alias cases, complete per-product finite checks and
  sticky errors, provider-absent/generated execution, stream ordering and exact
  driver-versus-semantic work counters. Keep force/response unchanged.
- Qualify independently changing occupied/amplitude views, disjoint outputs,
  tails, unsafe arithmetic, budget refusal and sanitizer behavior only in the
  newly changed domains. Reuse immutable passing baseline evidence.
- Require original independent total-energy 1e-8 Eh, separate-(T) 1e-10 Eh and
  physical R1/R2 1e-10 gates, complete endpoint wall, trajectory/work and exact
  admitted-capacity receipts. A component gain is insufficient for a PR.

## Actual-owner feasibility pair

The isolated actual owner now lives under
`.artifacts/df-triples-w-batched-owner-20261011/` and the corresponding
`n2:/data/jzzeng/qc-df-triples-w-batched-owner-20261011/` directory. It retains
the incumbent #2235 library exactly:
8072bb71d8ce1cd3e9d5285e4007914b0dd8db6c6aca4f25cd97be6f3a7f3d88.
Candidate library is
f07b30b5d8f0547237b1a7878f0d61b9d12eb5c545d73b5e0c0e72cf12ab12f9.

Only the native owner and its generated CUDA leaf object rebuild, using verified
ccache; borrowed link inputs remain checksum-verified immutable. The emitted
CUDA .cuh/.cu leaf bytes themselves remain identical. The generated inline
header intentionally changes its dispatch/code identity. The leaf object is
rebuilt to keep that consumed header closure consistent, not because its
scientific energy kernel changes. Existing result/Inputs/public ABI types remain
unchanged. This is not a clean full latest-master build.

The new device pointer table charges 48 bytes, adding 256 bytes to the aligned
triples arena. Optional admission occurs after the original panel/provider
choices. Exact old three-panel budget retains those choices and ordinary W
dispatch; typed numeric OOM during context preparation drops the optional table
before any input/scientific work. Execution failures cannot cause retries.
The prototype's temporary heap-owned Context and read-only descriptor accessor
are qualification scaffolding, not a production ownership recommendation.

Actual emitted map/group gates cover occupied sizes 1..32 in both strict and
unqualified-precision modes (64 outputs). The latter includes all-equal groups
of six seeds: a two-element collection would overflow and silently damage the
fallback. Preserve the six-element constant-storage collection in production.
The first host compilation omitted the ordinary public include path; retain
the failed log and the successful corrected host gate, without GPU repetition.

Slurm 2884 builds both objects successfully and the first actual-owner fixture
passes independent energy and exact work/dispatch assertions. Its checker then
incorrectly requires the cold retained-provider observation to equal the
incumbent's already-warm observation: 10 MiB versus 8 MiB. This is not an energy
failure, and does not change the already admitted 96-MiB provider allowance.
Keep the original failed checker log. Enforce the unchanged allowance and the
exact arena + provider allowance + host-binding ledger instead of asserting
that unrelated cold/warm observations are identical.

Slurm 2885 reuses the exact successfully built library, with no recompilation.
Its four targeted actions pass: o4/v3/Q5, o3/v7/Q5 (>256-point tail), exact old
three-panel budget refusing the optional table, and nonfinite arithmetic with
unpublished sentinels. The same job then completes one separately retained
candidate-then-control complete endpoint pair:

| Metric | Control | Candidate |
| --- | ---: | ---: |
| Complete wall seconds | 48.190556841902435 | 47.84401593520306 |
| Triples seconds | 3.757218845 | 3.483849015 |
| CCSD seconds | 24.921561608 | 24.915016147 |
| Triples driver GEMMs | 1556 | 908 |
| Triples contraction summands | 2326012282334 | 2326012282334 |

Complete wall is 0.719105% shorter (1.007243x); triples is 1.078468x. Do not pool
this pair with the micro campaigns, #2235 or earlier endpoint observations, and
do not claim statistical/global performance promotion. Both arms retain the
same CC trajectory and all non-timing CC work. W still evaluates 729 seeds /
1458 products; 324 pairs/81 singletons reduce W dispatches to 810, with 98 panel
dispatches unchanged. There are 648 added pointer bindings and 648 fewer W audit
dispatches; every original W audit value remains. Vendor-internal GPU kernel
counts are not inferred from driver API counts.

Both original independent gates pass without relaxation: total energy 1e-8 Eh,
separate-(T) 1e-10 Eh and physical R1/R2 1e-10. Errors are respectively
2.4300561562995426e-12 Eh, 9.269841838577264e-14 Eh,
2.9439020665655846e-12 and 1.3072413702734653e-12. Energy bits agree in the pair.
CC device numeric capacity remains 8,583,749,632 bytes and reported CC capacity
is unchanged. Whole-process peak RSS is 1,589,248 bytes higher for the candidate;
that is a pair-specific observation, not a universal resource bound.

Both GPU jobs are terminal; the follow-up exits zero. No new production patch,
commit, PR, merge or Release is submitted for this prototype. The useful endpoint
signal now warrants canonical prepared-provider integration. Move pointer
binding/dispatch out of compiler glue into a typed backend pair executor, replace
the prototype-only descriptor accessor, preserve in-place Context ownership,
and independently gate scratch/output/input aliasing, stale context/capture,
provider/precision fallback, budget/OOM, unsafe arithmetic and sanitizers. The
canonical implementation needs its own source-matched endpoint receipt because
its host validation/dispatch changes; do not transfer this prototype timing by
claiming byte identity or repeat unrelated baseline suites for master activity.

## Master impact

Observed master advances through 8dd792c33 (#2233). The earlier #2229/#2230
changes do not alter the W generator, triples owner or tensor contraction
consumer. #2233 changes matrix qualification driver tests for caller-owned
scratch, not the production W producer or this standalone probe. No passing
GPU matrix, endpoint, force or unrelated suite is repeated
solely because master advances. Latest-master reference qualification remains
separate from this frozen W experiment.

After the complete owner pair, master advances to 82bdc5c18 through #2234,
#2231 and #2220. #2234 adds a standalone compiler UCCSD amplitude-layout module,
not an RCCSD/W generator or runtime change. #2231 changes DFT batch-force routing.
#2220 changes KS resident final validation and adds a square-padded panel-product
helper used by that KS path; it does not alter the consumed W contraction/runtime
headers. None of these commits changes this frozen RCCSD/(T) consumer closure.
No GPU/endpoint/force/UCCSD suite is repeated because these commits land. The
#2235 head remains 31c3d6b35, open, with no unfinished or failed checks in the
post-pair authoritative snapshot; this is not merge authorization.

## References

Canonical integration and its separately measured evidence are now recorded in
`../implemented/performance/2026-10-11-df-triples-independent-pair.md`. The
prototype/micro results in this proposal remain historical and are not
transferred to that implementation by identity or sample pooling.

- `../implemented/performance/2026-10-11-df-triples-panel-traversal.md`.
- `../rejected/2026-10-10-unconditional-two-sided-df-ladder.md`.
- `2026-10-10-df-cc-lt-ladder-selection.md`.
