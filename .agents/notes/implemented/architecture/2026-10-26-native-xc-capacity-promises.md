# Decision: reserve future owners before optional XC admission

Status: implemented; bounded KS promises and optional XC admission qualified
Date: 2026-10-26

## Problem

The historical native resource cutover in closed PR #2586 disables local AO maps
and point batching whenever a public device ledger is bound. The diagnosis in
PR #2798 reproduced two large endpoint regressions by disabling those same
owners in the baseline. Removing the guards would instead allow optional panels
to consume capacity needed by later fleet members, retained force consumers or
geometry rebuilds. Today's unallocated bytes are not a future-owner bound.

## Accounting primitive

`DeviceCapacityReservation` reserves a supplied native endpoint bound before
its mandatory owners are constructed. `DeviceCapacityScope` lets those owners
convert the promise into actual numeric allocations or retained provider
allowances, without charging the same bytes twice. Unscoped allocations can use
only capacity left after all promises. Exceeding an owner's promise fails even
if another owner has unused capacity.

Retiring an allocation restores its promise while the owner is alive. Thus a
geometry rebuild can reuse the released capacity without exposing it to optional
storage. Closing the reservation releases only unused capacity; actual numeric
and provider charges remain until their separate owners die. Shared allocation
metadata has no locking destructor, so generation replacement and registry
erasure cannot recursively lock the ledger. Allocation failure and host-registry
failure restore credit under the same mutex as ordinary charge rollback.

The public descriptor now explicitly distinguishes selected inventory (default,
also used by the private ABI and Python planner) from a raw numeric cap (CLI).
Identical one-row inputs retain different authority; neither implies exclusive
ownership. Public diagnostics expose unused promises separately from numeric
and provider charges. This resolves the metadata distinction, not bound coverage.

The primitive alone does not calculate an endpoint bound, prove an external
inventory or admit XC. The integration below supplies bounded native authority;
external inventories retain their dense/single-tile path. The restored cutover
still needs full endpoint qualification before submission as a recovered fast path.

## Integration requirements

- Obtain native bounds for the complete mandatory fleet, serialized rebuild
  excess, every lazy retained force owner and DF response reservation. Compose
  allocator/compiler-owned leaves; do not duplicate compiler schedule equations.
- Distinguish a raw numeric cap from a caller-selected external inventory. The
  latter can reserve compatibility consumers invisible to the native method.
  Array length or currently observed live bytes is not a safe discriminator.
- Preserve shared-handle semantics: independently prepared owners must retain
  their own future promises. Do not silently grant an exclusive ledger to KS.
- Admit optional AO maps, spatial grid copies and point-batch panels only from
  explicitly available capacity after those promises. Preserve dense/single-tile
  fallbacks and the incumbent resource failure behavior.
- Expose unused promises distinctly in diagnostics before enabling them in
  production. Numeric requested/live/peak counts and allocation journals must
  continue to describe actual allocations, not reservation metadata.

## Rejected shortcuts

Do not reserve only the observed preparation peak: it omits lazy force/response
owners and can depend on geometry. Do not spend the numeric-cap-only SDK/CLI
handle as if its bytes were a complete native plan. Do not add optional Python
rows alone and claim recovery of the installed SDK/CLI routes.

Evictable arenas would require a different lifecycle: graph invalidation,
stream fences, callback-safe reclamation and layout fallback. They are not an
implicit alternative to correct future-capacity accounting.

## Evidence and acceptance

The host CUDA-double test executes the actual ledger allocator/provider code for
exact/below-limit promises, multiple future owners, numeric/provider conversion,
nested and foreign scopes, driver/registry failure, address reuse, rebuild credit
and cross-thread frees after reservation close. It does not establish CUDA
science, actual allocation bounds or endpoint performance.

Before the cutover can be proposed for acceptance, retain the historical full
20 SDK/30 CLI cases in both orders and the unchanged 1.05 timing gate, alongside
native-versus-planner bounds, exact/below-limit behavior, all lifecycle/oracle
gates and full endpoint allocation receipts. The historical 51/100 timing failures
remain failures; this accounting implementation does not relabel them.

## References

- #2227 R2; #933 prepared resources
- #2586 historical cutover (`a97112c4e` to `d474d0b2`)
- #2798 historical diagnosis and retained raw evidence
- `src/runtime/resource_ledger.hpp`, `src/runtime/resource_cuda.cuh`
- `tests/python/test_device_ledger_requested_bytes.py`

## Native Fock composition follow-up

The additive Fock capacity query now composes Direct bounds, the shared DF
policy and resident-promotion leaves, and the earlier source metric phase.
Mixed Direct/DF promotion acts on the remaining DF subbudget; its ceiling must
add the already retained Direct provider back. Initial source construction can
exceed a tiny value cap before the later plan fails, so its metric phase must
not be clipped to that cap. The KS fleet now accepts this independent Fock
setup peak with serialized-rebuild semantics and an unchanged v1 wire format.

The current local tests cover executable policy/subbudget/promotion sweeps,
source and sum overflow, topology queries without CUDA, and retained fleet
neighbors during setup. Three native tests and 30 Python tests passed (11
real-CUDA resource tests skipped); the Python set includes allocation-event
source-factory doubles and the existing policy/KS planner tests. This is a
numeric sizing increment, submitted as PR #2805 on top of #2803. No promise or
optional-XC guard is activated by it. Opaque provider accounting, initial guess
coverage and every independently callable prepared-owner port still need
integration before complete endpoint qualification.

## Native KS integration under qualification

The local integration reserves the counted native fleet before constructing
any item. Its initial domain is explicit numeric-cap admission with strict
FP64 all-electron s/p PBE/PBE0 Direct or homogeneous full-range DF. Preliminary
guesses, independent range/nonlocal/D4 corrections, automatic/generated-split
programs, explicit incremental J/K, application precision budgets and the
experimental PBE point-precision controls keep the incumbent admission path.
A capacity miss also retains that path; host allocation errors still propagate.

The promise composes native state, dense XC and resident grid, Fock retained
response and initial setup, quadrature setup, declared matrix-provider allowance
and Direct stationary capacity. Lazy one-electron export must coexist with the
already retained fleet, so its deferred setup is separate from the excess needed
while rebuilding a retired item. A paired gradient needs both coordinate-vector
outputs even for atoms without orbital shells. The old one-electron envelope
alone underbounds that sparse topology; the paired form includes the additional
output. An extra DIIS history-square buffer conservatively covers diagnostic
changes before rebuilding the geometry.

The batch owns the promise before its item vector, ensuring all item allocations
retire before it closes. Construction, replay, warm restore and independently
callable final-state/resident/derivative ports bind the same promise. Scoping only
batch execution misses single-calculation forwarding through `single_owner()`.
A null scope withholds an unrelated owner's active promise rather than borrowing
its credit. Multiple prepared owners may still share the same ledger.

`MatrixLibraryOwner` converts its declared provider allowance only under a native
promise, before creating the provider; it releases that allowance after provider
destruction. External selected inventories already withhold this allowance and
must not be charged again. These receipts do not measure opaque CUDA context,
cuBLAS or cuSOLVER internal allocation peaks.

Six focused native tests and 112 Python tests passed before the final paired-
gradient bound correction. The corrected source subsequently passed all 121
native CPU tests and pre-commit, and a cached CUDA Release build passed its three
resource tests. Slurm job 14466 passed 40 installed SDK probes (the original 20
models with caller metadata retained or closed, each cold/warm/moved). Job 14475
passed larger/shared-owner and ragged-batch lifecycle probes, including a 21-AO
matrix-provider allowance, independent single/batch derivative ports, stale
tokens, exact shared capacity, failed preparation, retry and moved rebuilds.

## Optional XC admission under qualification

Only a native complete promise permits optional XC under a public ledger. The
entire mapped AO arena is charged as unpromised storage, including its dense
scratch; the mandatory dense credit stays available for allocation fallback.
Spatial execution copies and point-batch panels also allocate under unpromised
scopes. Their bounds are clipped to currently unpromised capacity, and the
compiler's existing point-panel planner and one-tile fallback remain authoritative.
The active promise is restored before later mandatory owners are constructed.
Selected external inventories and configurations outside the bounded native
domain retain the incumbent dense/single-tile behavior.

Slurm job 14476 incrementally built this version and repeated all 40 SDK probes
and the larger/shared/ragged lifecycle probe successfully. The earlier frozen
source archive/patch, installed SDK and receipts were retained; only its completed
build directory was reused, with a separate installation prefix for this version.
Both source variants passed the original SDK energy/force oracle gates against
the authenticated PySCF 2.14.0 reference (120 phase observations each). The optional
version's largest errors were approximately 1.07e-9 Hartree and 1.34e-10 Hartree/Bohr,
below the unchanged 1e-8 and 1e-6 gates. These probes do not establish timing recovery.

The grid implementation has changed since the historical campaign to support
newer defaults. Its old source checksum therefore cannot be carried forward as
current evidence. For the SDK's explicit version-1 prescription, regeneration
matched every saved point, weight, raw measure, owner and position array checksum
for both geometries. This preserves the original independent oracle inputs.
The current CLI default grid needs its own reference qualification; a historical
default-grid oracle must not silently stand in for it. Keep the complete original
case/property inventory, both timing orders and the 1.05 ratio gate when refreshing
that qualification. The old 51/100 timing failures remain historical failures.

## Current-source qualification and DF response headroom

The refreshed source starts at `1ec57de7eefd523f1a079351faf142fce6b74135`
with candidate patch SHA-256
`ebd5d9a7dd1004ddcf6656a26b377262a56ad509592ed290c56512d1419dcbd0`.
Slurm job 14485 built independent baseline/candidate CUDA Release installations;
four candidate resource/lifecycle CTests and 20 baseline/40 candidate SDK probes
passed. Job 14509 passed the complete current CLI/SDK preflight: 60 variant/model
rows, 216 actual commands and 60 SDK records, including explicit DF-force refusals.
Fresh independent PySCF 2.14.0 references cover the current version-4 PBE CLI
default grid, both spins, Direct/DF and both geometries. Their analytical gradients
passed reconverged finite differences at two steps. Local CPU validation passed
121 CTests and 406 Python tests plus 25 subtests, with 13 skips.

The adjacent-cap campaign did **not** pass. Job 14510 hit its 30-minute limit
after 32 completed models; its raw shell trap misleadingly wrote zero although
the scheduler session returned 143 and no completion marker existed. Separate
continuation 14511 preserved that evidence and completed nine more models before
the UHF DF CLI search found a successful endpoint with one rejected allocation
at 559,744 bytes. The frozen zero-rejection gate rejected it. Do not convert that
search result into an admitted cap, discard the model or relax the gate.

A debugger return probe on the actual counted allocator identified a 448-byte
request from `execute_cuda_df_hf_gradient`'s response arena. The incumbent panel
planner considered its local response allowance and physical free memory but
omitted the active native ledger's remaining capacity. Its existing OOM recovery
then selected a smaller panel and completed the same physical calculation. The
first repair clipped the initial policy input to `device_resource_available(device)`.
It passed the new UHF regression and all four resource/lifecycle CTests in job
14525, but fresh cap job 14527 exposed a lost RHF fallback: its minimum increased
from 558,761 to 558,825 bytes. The reduced policy input also changes the
single-fitted-tensor selection, so it is not equivalent to the incumbent
allocation retry. That first repair is retained as an intermediate result,
not a final accepted capacity policy.

The revised repair preserves the existing algebra and retained-tensor selection.
Immediately before allocating its response workspace, it checks the active
ledger/promise's actual remaining bytes and applies the same panel halving as
the existing physical-OOM recovery. Non-shrinking mandatory storage still refuses.
The physical-allocation retry remains necessary for fragmentation or concurrent
consumers; reported rejection counters must not be cleared to disguise it.

The added real-GPU regression uses public DF-RHF/UHF energy/force endpoints at both
geometries. It obtains the unconstrained complete peak, then repeats at that
peak for RHF and one byte below it for UHF, requiring independent PySCF energy/force agreement, zero rejected
allocations and final cleanup. This tests bounded endpoint behavior rather than
an exact number of internal allocations. The repaired source still requires
fresh build, cap, numerical and full timing qualification. All original 50
models, both orders, 21 samples, three warmups and the 1.05 gate remain unchanged.

Evidence is retained under `.artifacts/xc-current/` in this worktree and the
corresponding `qc-2227-xc-current-*` directories on n1. n5 cannot execute these
frozen n1 products because its glibc/libstdc++ is older. Failed profiling setup
attempts there are retained as failures, with no CUDA science or timing claim;
the matching n1 runtime is required for these installed-product measurements.

The small private DF late-port probe passed all eight storage/spin models
(auto/dense/packed/packed-single, RKS/UKS), including independent single ports
and cold/warm/moved batch ports on the first repaired installation. A separate
21-AO three-water probe passed automatic RKS but failed UKS final-state physical
validation before its late integral consumer. This is retained as an unresolved
scientific fixture/control result, not a resource pass or authorization to relax
the final-state validator. Larger DF and unsupported public DF-force domains
must not inherit the small-model admission evidence.

The same 21-AO UKS final-state failure was reproduced without any public resource
ledger or native promise, isolating it from this admission change. Its fixture
remains unqualified. The first expanded RHF regression in job 14531 also requested
one byte below its already minimal measured endpoint peak; that invalid test
assumption failed with typed OOM. RHF must instead retain admission at the measured
peak, while UHF exercises its smaller-panel fallback below the unconstrained peak.
The failed job is retained, and the corrected test requires fresh execution.

## Final panel repair qualification

The corrected final source is the same measured base `1ec57de7e` plus patch
`591cd8b728e89abd06cc63c25d0ddb6e95c3db9547de6274886cad5eddf11512`.
Job 14533 passed its cached CUDA Release build/install, four resource/lifecycle
CTests, 40 SDK metadata-lifetime probes and all eight small DF late-port models.
The expanded RHF regression fails against the retained first-clamp installation,
providing a negative control for the capacity regression. Job 14534 passed the
complete 60-row SDK/CLI preflight (216 commands and 60 saved SDK records).

Job 14536 independently qualified all 50 final-source SDK/CLI cap boundaries:
full cap and exact cap succeed with unchanged published iteration work and
independent science; one byte below and one byte total return typed OOM. The
RHF DF auto/dense minimum is restored to 558,761 bytes; UHF auto admits at
559,709 bytes without a rejected allocation. Earlier cap values seeded offline
search only. Job 14535's harness failed immediately with a Python TypeError in
an expected-refusal branch; its corrected successor uses separately exercised
control flow. Neither that harness failure nor the earlier source failures is
relabeled successful.

Job 14537 completed 160 separate Nsight executions covering both installed
variants and every SDK/CLI route, with independent science checks on every
trace. A read-only SQLite audit verified trace identities and found no recorded
CUDA runtime errors or inconsistent tracked allocation lifetimes. Kernel,
transfer, launch, graph and requested-allocation peaks are retained separately
from clean timing. Nsight 2025.1.3 warns that the CUDA 13 driver is traced with
12.9 libraries; this limits completeness claims. These are observed events,
not complete physical RSS, opaque provider storage, Fock counts or effective
precision telemetry.

Both historically diagnosed SDK endpoints now have matching baseline/candidate
kernel counts and tracked device peaks: PBE RKS Direct has 6,185 kernels and
16,857,506 requested device bytes; DF has 3,450 kernels and 80,107,322 bytes.
This is diagnostic recovery of execution structure, not timing acceptance.
The unchanged full 100-comparison timing campaign remains the acceptance gate.

The measured base tree is identical to published master
`e1e244197411b58da42aa8a2cbce49014d7ed3ff` (the squash of #2805). Differences in
Git ancestry therefore introduce no additional source changes into this
comparison. Later documentation/evidence edits must stay distinguishable from
the frozen measured patch.

## Sampling cost and future diagnostic screening

The fixed CLI matrix launches 11,520 fresh processes: 30 routes, eight blocks,
24 observations including warmups, and two geometries. The first three completed
CLI blocks cost 28.49–29.00 minutes each; 98.48–98.76% of that wall time is inside
the subprocess invocation. This includes process-wait detection and does not
separate CUDA initialization from calculation. Optimizing JSON bookkeeping alone
cannot materially shorten this campaign. SDK blocks cost 1.85–2.16 minutes.

Estimate campaign cost before scheduling future candidates. Retain the complete
numerical/resource preflight, then use a short diagnostic screen to find gross
regressions before committing to final qualification. A proposed screen spanning
both Direct spins, DF storage modes and the RHF/UHF response-panel paths uses six
SDK/eight CLI routes, ABBA, one warmup and three measured samples per block. The
current observations forecast roughly six minutes for that screen, versus 246
minutes for the full timed blocks; preflight, queueing, compilation and auditing
are additional. This estimate is not a measured short-run result or statistical
acceptance. The original complete matrix, both orders, sample counts and 1.05
gate remain required for this candidate. Do not mix a future screening population
with the frozen qualification population or accept early from partial results.

The read-only cost calculation and inputs are retained as
`.artifacts/xc-current/plan_sampling_cost.py`, `sampling-cost-plan.json` and
`timing-overhead-audit.json`. Current job 14539 is not restarted or modified.

## Completed final endpoint qualification

Job 14539 completed with tool and worker exit status zero and both final input
and product integrity checks zero. All original 100 science/work/timing
comparisons pass: SDK maximum ratios are 1.0044809944829631 (ABBA) and
1.0051072050620615 (BAAB); CLI maxima are 1.0249240201055871 and
1.0472140739522102. All 50 routes, 21 samples and three warmups per block,
both orders and the 1.05 gate were retained. No failed comparison was removed.

The [reviewed evidence bundle](../../../../benchmarks/results/native-prepared-xc-promises-20261026/README.md)
retains the exact measured patch, original checkers and inputs, full-precision
observations, boundary controls and profiling summaries. Its offline replay
recomputes every timing/work decision and independent science gate. Shared
significant-speedup/default-promotion qualification remains explicitly not run;
passing the original nonregression protocol does not claim a speedup. Historical
51/100 failures and all rejected intermediate repairs retain their original
dispositions. Subsequent documentation/inventory/evidence edits do not change
the production or test source qualified by the frozen patch.
