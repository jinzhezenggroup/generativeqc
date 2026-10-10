# Decision: Capture actual allocation lifetimes and assert zero by domain

Status: implemented
Date: 2026-10-10

## Problem

The requested-byte extension in
`../compatibility/2026-10-10-device-ledger-requested-bytes.md` preserves real
counter evidence but cannot reconstruct allocations/frees or phase boundaries.
Closing #1630 requires runtime observation rather than invented journals from
live/peak differences. V1's zero assertion applies to every named domain; it
cannot represent a device-zero prepared route alongside nonzero host requests.

## Decision

Attach an optional finite journal to the existing native device ledger. Capture
starts while that owner is inactive, snapshots retained generations, and reserves
event slots before replay. Existing registry insertion/retirement/free paths
append POD records under their existing lock. The capture handle outlives the
ordinary ledger handle so cached-buffer releases are still recorded. Slot loss
invalidates evidence without reallocating observation storage or altering CUDA
allocation, visibility, equations, precision or provider selection.

Add a compatible v2 receipt contract with explicit per-window zero-allocation
domains. Unasserted domains must still provide full named-owner event coverage,
identity, continuous phase state and consistent metrics. V1 retains its stronger
global-assertion meaning and exact field shapes. This is not a blanket ban on
host allocation and does not relax a promised device-zero gate.

Use optional pinned Memray 1.20.0 for actual host malloc-family requests rather
than adding or replacing a scientific allocator. A continuous nonaggregated
capture is normalized with finite parsing bounds, preserved address lifetimes
and explicit pre-tracker/mapped-memory/Python-suballocator exclusions. It does
not represent all process heap/RSS or authenticate source/build/execution by
itself. Caller-controlled phase markers and separately pinned contracts remain
necessary for coordinated prepared receipts.

## Negative evidence and rejected alternatives

- A real direct `ctypes.CDLL(None).malloc/free` experiment produced **zero**
  profiler events, despite actual requests. Direct function pointers bypass
  intercepted PLT call sites: treating that result as zero heap work is wrong.
- The bounded compiled marker's PLT calls produced the expected two allocations
  of 31/47 bytes, four owned lifetime events, requested/peak 78, and one excluded
  pre-tracker free. Require visible unique markers; never guess from a reused
  address or replace them with direct libc function pointers.
- Reconstructing events from aggregate counters hides release/reuse and cannot
  validate ownership continuity. Actual native generations are authoritative.
- Growing a journal during replay defeats the allocation ratchet. Drop events
  explicitly and reject complete coverage instead.
- Applying a device-zero guarantee globally to host instrumentation/publication
  either makes honest receipts impossible or encourages omitting host evidence.
  Explicit admitted domains solve that representational gap without weakening
  the numerical or lifecycle gates.

## Evidence and remaining work

The host-double tests compile the production C API and allocation wrapper and
exercise initial owners, asynchronous allocation/free, delayed generation reuse,
capture bounds/loss, handle-independent lifetime, short-buffer ABI protection
and legacy counters. Adversarial parser tests protect v1 and scoped v2 contracts,
history/owner continuity, marker ambiguity, unsupported encodings and event limits.

Actual host profiling verifies the production dense-response-oracle `to_dense`
input-buffer hoist: 1024 per-column 8192-byte calloc requests become one request,
while both paths produce the independently specified `2 I` matrix exactly. This
is an allocation-count gate for that oracle consumer, not an HF/DFT endpoint
timing or speedup claim.

The optional dependencies live in the isolated ignored profiling environment;
no production/compiler dependency was added. Source-matched GPU journal and
coordinated host/device phase evidence must be retained under ignored artifacts
or the repository's accepted benchmark evidence owner before broad closure.

Real prepared **device-domain** v2 receipts passed for RHF and UHF in Slurm
job 6873, node1/main, one RTX 5090, finite 10-minute limit, preserving assigned
visibility. The independently pinned pre-execution contracts cover preparation,
first/warm energy, first/warm forces, changed geometry and closing publication.
Both journals contain 12 real allocation/release events; warm energy/force
requests are zero and closing releases leave zero live owned-device bytes.
The two existing prepared GPU regressions also passed. Maximum observed energy
and force differences against matched ordinary prepared execution are below
`1.5e-14` Eh and `2.3e-15` Eh/bohr respectively. These are instrumentation
non-regression gates, not independent qualification of a new scientific method.

The capture's working-tree object is
`5388e8676e5d5b6fff097db69ec0b0ba11389572` over baseline
`f87d51ab616cc9aa774bd344a15e268cef7eaf05`. Its native library SHA-256 is
`4898b3aa36f92f1d4e8ad1815ae3e4c0a295f292588dc3b106097ac1f188c858`.
Original expected/receipt/verification/numerical files are retained in ignored
`.artifacts/issue1630-journal/` locally and in the node1 qualification copy.
Later parser-history hardening and documentation updates are not retroactively
claimed to be that frozen tree. The current strict verifier also passes those
retained receipts. These remain device-only receipts: the real host adapter and
its allocation-reduction gate do **not** by themselves qualify a combined
host/device prepared lifecycle. The reusable coordinated producer and joint
runtime proof remain the next required work before closing #1630.

## Joint producer qualification (2026-10-10)

The outstanding combined-producer gap described above is now implemented in
`tools/capture_prepared_allocations.py`. It freezes source through a separate
temporary Git index, exports Git modes/blob identities plus SHA-256 digests,
reconstructs the tree from actual copied bytes, and refuses extra production
source inputs. Pinning precedes execution, includes installed library/marker
hashes and the actual Slurm-assigned GPU UUID, and cannot overwrite contracts.
Capture verifies these inputs again after execution. A different imported
Python checkout or native library is rejected.

The producer attaches a bounded journal to the real prepared native ledger
before its preparation binding, while one continuous optional Memray tracker
covers host malloc-family requests. It uses public prepared endpoints rather
than an alternate SCF implementation. Explicit CUDA fences establish completion.
Compiled markers remain live until tail cleanup to avoid address reuse.
Endpoint boundaries are before their marker, so marker/snapshot payloads belong
to publication; setup/publication boundaries include their ending marker. Every
event is consumed through finalized host EOF. Ordinary matched prepared history,
including changed geometry, is compared outside the monitored lifecycle.

Slurm job **6883**, node1/main, one RTX 5090, finite 10-minute limit, produced
passing joint RHF and UHF receipts for tree
`5ac7f3b928161a72c2e7ebefcfc57bf731551186` over the same baseline
`f87d51ab616cc9aa774bd344a15e268cef7eaf05`. Native inputs were unchanged from the
earlier qualified journal build: library SHA-256 remains
`4898b3aa36f92f1d4e8ad1815ae3e4c0a295f292588dc3b106097ac1f188c858`.
The compiled marker SHA-256 is
`f34fbdf9a0d7d7ac10c7b4a0b2a9f7432153da082a5d82f6598dd0ba37682d3c`.
Build configuration and verified ccache provenance accompany the proof. The
shell/stationary-force AOT qualification flags are retained as build identity,
not promoted as production defaults.

Each method has 13 lifecycle windows and 12 actual device lifetime events;
warm energy/force device request count and requested bytes are zero, and close
leaves zero owned-device bytes. Host raw event counts are 8648/8692 (RHF/UHF).
RHF warm energy/force have **513/527** observed malloc-family requests and
**164403/165051** requested bytes; UHF has **515/529** and **165795/166443**.
These honest nonzero host observations are not hidden by the device-only zero
assertion. Requested-byte accumulation, peak ownership and final live ownership
remain separate; retained host result/observer objects need not be freed at EOF.

Maximum matched energy error is below `1.5e-14` Eh, and maximum force error below
`5.2e-15` Eh/bohr. Actual and ordinary iteration counts agree: `[2,8,2]` for first
energy, `[1,1,1]` for warm energy, `[2,2,2]` for first forces, `[1,1,1]` for warm
forces, and `[2,2,2]` after the geometry change. This is an instrumentation
non-regression comparison, not a new independent scientific method qualification
or an endpoint timing claim.

Both receipts and independent current-verifier rechecks are retained under
ignored `.artifacts/issue1630-joint-final/`, with contracts, raw `.bin` files,
source manifest, hashes, diagnostics and logs. The frozen Git tree/archive owns
that evidence; later editorial/documentation changes do not retroactively change
its identity. An earlier joint qualification, job 6881, is retained separately.
Job 6883's capture commands passed; its later pytest command could not start
because the separate optional venv lacked pytest. After installing test-only
dependencies there (not in the shared runtime), job **6886** passed 29 host/hoist
tests with GPU opt-in cases explicitly skipped; job **6888** then explicitly
enabled and passed the two RHF/UHF prepared CUDA regressions. Missing test
dependencies and skipped GPU tests were not counted as GPU qualification.

Local #1630 acceptance now has concrete gates for synthetic hot allocation
detection, legal setup classification, bounded-vector growth handling, actual
host/device counts/bytes/peak ownership, scoped warm CUDA zero requests, and the
independent numerical `2 I` response-buffer hoist count reduction `1024 -> 1`.
This completes the missing runtime integration locally, not a merge or GitHub
issue closure. Other pathways retain their own unqualified ownership scopes.

## Revisit when

The supported host interception boundary changes, a new profiler/version is
qualified, native numeric buffers bypass the existing registry, or a consumer
requires mapped-memory/Python-suballocator ownership. Do not silently widen
current counters to those layers or add their bytes as physical-memory totals.
