# Generated resident ABI and span host contracts

Status: implemented, host-only test seam
Date: 2026-10-20

## Decision

Replace only the resident ABI declaration source assertion in
`test_tensor_cuda_resident.py` with compiled execution of the complete
`resident_source` result. The test injects the existing outer ordinary-emission
boundary with a host owner/CUDA dependency fixture. Actual int64 passthrough
control plans include an empty floating literal required by normal CUDA precision
admission. They need no floating-input validation or computation launches; their
normal production emitters run without patching private helpers. The test does not
extract or reconstruct resident function bodies or span rows. The generated
exports and control-span tables remain the subject under test, together with an
unchanged copy of production `src/tensor/cuda_resident.cuh` and `metrics.hpp`.

The dependency fixture supplies host storage, CUDA API responses and an owner.
It queues copies until synchronization, so a missing synchronization cannot pass
merely because a stub copied immediately. It never computes a TensorIR result.
This is neither production Context lifecycle qualification nor CUDA/kernel,
arithmetic/symmetry, ordinary-emitter or scientific numerical qualification.

## Executable oracles

- Resolve and call all five resident exports; compare plan identity and ABI
  version, use the production Metrics layout, and reject null run arguments and
  uninitialized static state.
- Upload distinct raw bytes and compare the entire arena against an independent
  Python bytearray, including untouched gaps. Independently seed the entire arena
  before testing downloads, avoiding a matching upload/download bug cancelling
  itself out. Test multiple unequal-length int64 control slots and empty spans.
  These cases do not qualify FP32/FP64 spans or computed-output storage.
- Exercise invalid slots, incorrect byte lengths, null owners/host buffers,
  allocation offset/extent bounds, injected copy/synchronization/device errors,
  nonzero statuses, diagnostics and stream draining. Record all CUDA copy kinds
  per operation and require H2D uploads/D2H downloads without fixing call counts;
  require successful runs to drain their
  queued readback as well as failed runs. Preserve the input pin gate.
- Opt-in `VIBEQC_RESIDENT_HOST_CONTROLS=1` accepts formatting and private table
  renaming while rejecting ABI-version, missing diagnostics, offset, length and
  status corruption using the same behavioral oracle. Separate controls corrupt
  the copied header's copy kind and omit or move the successful-run drain early.
  Normal gates use the unchanged production header; only opt-in damage controls
  transform a temporary copy. These controls edit generated/test-copy text;
  they are not a required-CI private spelling/hash gate.

## Boundaries retained

The ordinary ABI and doubled-pair computed-output span source assertions are
deliberately retained: replacing ordinary emission with an owner seam cannot
prove ordinary exports, and int64 passthrough plans cannot replace the original
floating/computed-output span guard. All arithmetic, symmetry, post-run,
lifetime, dependency-integrity and GPU numerical tests remain.
No production generator/runtime, mathematics, resource default or selector changes
are part of this slice. Parent issue #2296 remains open.

## Validation

Run `tests/python/test_tensor_cuda_resident.py` with a real Linux C++ compiler and
the shared `native_cxx` fixture, setting `CCACHE` to a verified sccache (preferred)
or ccache executable. Preserve compiler-cache statistics, finite compile/link
timeouts and exact source provenance. Windows prerequisite skips do not establish
Linux host acceptance. Run the opt-in controls separately from ordinary CI.

The initial `b94da4079` implementation passed 24 resident cases and a related
154-pass/19-CUDA-opt-in-skip Linux gate with Python 3.11, G++ 12.3 and sccache
0.16.0 (10 compilations/misses, no failures). However, review P1
`PRRT_kwDOUFEw6s6uNDqj` correctly identified that patching private validation and
launch helper names would still reject a harmless rename. Those results do not
qualify the subsequent narrower head. The final slice removes both private
patches and restores the original doubled-pair span assertion; it requires its
own source-matched Linux gate and latest-head review.

The first pure-int64 revision failed before compilation (5 failed, 6 setup
errors): `program_precision` requires a floating value for plan identity. The
replacement uses a legitimate zero-element floating constant through the public
TensorIR API. It does not bypass precision admission, initialize floating data or
qualify any nonempty floating span.

The `149b11c9e` narrower sources passed the Linux gate with 156 cases and 19 explicit
CUDA-device opt-in skips, including all 26 resident cases (five opt-in controls).
The cache recorded 11 compilations/misses and zero compilation failures.
The three qualified test/helper/fixture LF hashes were checked against the
committed blobs. The compiler/runtime boundary remains host-only; this result
does not qualify ordinary emission, floating validation, kernels or science.

Review P2 `PRRT_kwDOUFEw6s6uNLy_` identified a stale diagnostic in the static-state
failure oracle. The opt-in diagnostic-suppression control reproduced that gap:
the old ABI oracle accepted a nonzero, synchronized return without a fresh
diagnostic, so the control failed with `DID NOT RAISE AssertionError`. Clear the
buffer before the static-state call and require the initialization diagnostic;
the same control must then reject the damaged wrapper. This is test-oracle repair,
not a production error-path change.

After this diagnostic-oracle repair, the `bdb33adb2` Linux gate passed
157 cases with 19 CUDA-device opt-in skips, including all 27 resident cases and
six opt-in controls. The diagnostic-suppression control now rejects the damaged
wrapper; no production source was changed. Ruff/format and whitespace checks
passed, and final test/helper/fixture LF blobs matched the qualified sources.

Reviews `PRRT_kwDOUFEw6s6uNU3K` and `PRRT_kwDOUFEw6s6uNU3M` identified missing
copy-kind observation and successful-run synchronization checks. The host CUDA
dependency fixture now records supplied copy kinds; its getter compares all to
the public H2D/D2H constants without depending on a particular numeric encoding.
The run oracle checks synchronization and an empty pending-copy queue immediately
after the successful call. An omitted or prematurely moved drain is rejected
before any subsequent error path can
mask it; negative-fixture destruction discards fake pending borrowed pointers
rather than accessing expired stack readback storage. No production code changes
are needed for these test-contract repairs.

The `f57eb109a` source-matched Linux gate passed 160 cases with 19 explicit
CUDA-device opt-in skips, including all 30 resident cases and nine opt-in
controls. Wrong copy kinds, missing drains and early drains are rejected while
the harmless table-renaming/format control passes. The normal-gate production
resident/metrics headers remain unchanged; test/helper/fixture LF blob hashes
match the qualified sources. This remains host evidence only.

Review `PRRT_kwDOUFEw6s6uNf7G` identified the same direction-observation gap for
the generated arithmetic-status readback. The run oracle now clears prior copy
observations and requires every observed copy kind to be D2H. A dedicated opt-in
control changes only that generated readback direction, independently of the
copied-header upload/download direction control. This tests status transport,
not arithmetic correctness or device execution.

With that readback-kind check, the `4aae37118` Linux gate passed 161
cases with 19 explicit CUDA-device opt-in skips, including all 31 resident cases
and ten opt-in controls. Test/helper/fixture committed LF blobs were checked
against the qualified source hashes; the host-only boundary remains unchanged.

Reviews `PRRT_kwDOUFEw6s6uNo4I` and `PRRT_kwDOUFEw6s6uNo4Q` identified two more
dependency-model gaps: zero-valued reversed status-copy endpoints could pass, and
an end event recorded before the readback incorrectly drained later copies. The
fake owner now registers its arena and error storage as logical device ranges;
the CUDA copy dependency rejects direction/endpoint mismatches. Events capture
their stream and a monotonic copy-queue watermark, and event synchronization only
completes that captured prefix. Distinct event identities avoid conflating begin
and end. Stream synchronization still drains all submitted copies. This models
only the required host transport boundary, not a general CUDA runtime or kernels.

Dedicated opt-in controls reverse the generated status-copy pointers and move end
event recording ahead of the readback. The queue semantics follow the
[CUDA 12.9.1 event API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html):
recording captures existing work, later submissions are excluded, and a newly
created unrecorded event has no captured work. Logical endpoint admission follows
the public [copy-direction contract](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__MEMORY.html),
without claiming an exact device error code.

The `49ade5d5f` source-matched Linux gate with this dependency model passed 163
cases with 19 explicit CUDA-device opt-in skips, including all 33 resident cases
and twelve opt-in controls. Against the previous `4aae37118` dependency fixture,
the two new controls fail with `DID NOT RAISE AssertionError`, reproducing both
false positives. Restore the qualified fixture before using the acceptance gate;
these old-model red results are retained negative evidence, not final acceptance.
After restoring the qualified source hashes, both controls passed (2 passed,
31 deselected). The full-gate cache delta was 18 compilations/misses and zero
failures; this establishes cache use rather than cache hits or GPU acceptance.

Review `PRRT_kwDOUFEw6s6uN8uK` found that elapsed-time admission still accepted a
missing begin record. The fixture now distinguishes created/unrecorded events
from recorded events and rejects elapsed-time requests lacking either completed
record, following the linked CUDA event API. It supplies no timing-performance
oracle. A missing-begin control tests the resulting failure boundary. The run
oracle also exercises existing copy, synchronization and device-check failure
injections with fresh diagnostics and drained queues; an ignored-readback-status
control checks that nonzero API responses are not swallowed. These are status
transport checks, not synthetic tensor arithmetic.

The `b7b3647a2` source-matched Linux gate passed 165 cases with 19 explicit CUDA-device
opt-in skips, including all 35 resident cases and fourteen opt-in controls. The
qualified test/helper/fixture LF hashes match the committed blobs. Full-gate
cache evidence records 20 compilations/misses, zero failures; this is not timing,
device or scientific acceptance.

Review `PRRT_kwDOUFEw6s6uOMrb` identified that zero-initialized fake error storage
masked removal of the per-run reset. The owner now starts with a nonzero stale
status in every byte. Copies and fills share one FIFO work queue, so a late reset
cannot retroactively alter an earlier readback; this models only CUDA transport
primitives, not kernels. Missing-reset and late-reset controls require a healthy
run to reject the stale-status behavior. The pending-work observer covers both
operations, without requiring any fixed number of calls.

The existing invalid-slot/length cases also exercise the high bit of public
`size_t` arguments, using the original small buffers. Optional ABI-narrowing
controls verify that a 32-bit slot or byte-count parameter cannot silently
truncate those invalid values into valid transfers.
Short (`size - 1`) and long (`size + 1`) transfers are both rejected, preserving
exact extent equality rather than just an upper bound.

The former affine download seed repeated every 256 bytes, matching planned
alignment and masking an output-only offset shift. Adding a linear block term
still admits some unaligned collisions. Use a fixed-seed non-affine byte stream
and verify unique windows for every nonempty output length over each current
small arena. The output-only `+256` control selects a nonempty output row whose
shifted span remains within capacity and leaves the input table unchanged, so
rejection must come from actual download mapping rather than bounds or upload
damage. Old-oracle comparisons omit the newly added seed-quality assertion so
it cannot masquerade as detection of the download defect.

The final combined source-matched Linux gate passed 170 cases with 19 explicit
CUDA-device opt-in skips, including all 40 resident cases and nineteen controls.
The missing-reset control fails against the prior zero-status fixture; the
late-reset control also fails against its immediate-fill variant with a nonzero
seed. The output-only shift fails against the former periodic download oracle
without the new data-quality guard. All three RED receipts report `DID NOT RAISE`.
The old fixture comparison uses only a statistics-query naming adapter, not
different transfer behavior. Restore the final sources: the five affected
reset/download/ABI-width controls then pass, and all three source hashes match
the full gate. Full-gate cache delta: 25 compilations/misses, zero failures.

The superseded `b7b3647a2` CI run 39380033113 was not green: core-b job
118488566953 failed `test_owned_descendant_is_retired[True]` when reading
`/proc/<pid>/stat` raised `ProcessLookupError` instead of the caught
`FileNotFoundError`. That test's blob was identical in the baseline, this branch
and then-current upstream. The independently owned fix in still-open draft
#2250 was not merged or duplicated here. This negative CI receipt does not
replace latest-head required CI. After adding short-length rejection, the final
source-matched Linux host gate again passed 170 cases with 19 explicit device
opt-in skips; its three qualified LF blobs match the submitted sources.

Review `PRRT_kwDOUFEw6s6uO7I0` found that zero/short status copies could preserve
the initialized local zero while still satisfying direction and completion. Copy
observations now carry byte counts as well as kinds, and the run oracle requires
the complete native `int` status object. Dedicated zero/short-count controls
exercise that transport contract without generating a numerical kernel error.

Zero-length calls still validate owner, slot and exact span length. The existing
cases reject null-owner, out-of-range-slot and nonempty-span requests even when
both host pointer and requested byte count are zero. Legitimate empty spans accept
either null or a nonnull one-byte guard buffer with zero count, preserving the
guard and arena.

Before delivery the branch normally integrates protected upstream `457a47955`
(#2640 materialization-rule contracts, after #2637) without modifying those
owners' code. Its TensorIR import/materialization delta requires fresh host
qualification; earlier counts are not transferred to that integrated tree.
The dependency constants explicitly match public CUDA 12.9.1 encodings
H2D=1, D2H=2 and D2D=3 ([SDK types](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__TYPES.html)).
The symbolic comparisons do not rely on a private enum layout; this correction
does not extend the fixture into other CUDA APIs.
The final integration qualification uses a full archive of the reviewed staged
Git tree in a fresh immutable directory, not an old `HEAD` archive with only test
overlays. Its ignored manifest records the complete tree and selected producer,
runtime and test blob/byte identities. The resulting commit must retain that
exact candidate tree; delivery still requires latest-head CI and formal review.

Reviews `PRRT_kwDOUFEw6s6uPMkr` and `PRRT_kwDOUFEw6s6uPwe1` add two bounded
host contracts. A separate real host thread holds the fixture Context mutex;
run/upload/download must promptly report a fresh busy error without changing
arena, host buffers or admitted/pending work. The holder has finite acquisition,
handshake and automatic release, and every normal/exceptional test path releases
and joins it. Existing serial success oracles run after release to test recovery.
Opt-in guard bypass and blocking controls exercise rejection without leaving a
thread behind. This is owned-host-mutex admission, not general production Context
lifecycle or GPU concurrency qualification.

After real input uploads and before the independent download seed, a real
passthrough `resident_run` must preserve the complete tensor arena. A control
retargets only status reset/readback to the arena and is rejected by this public
upload/run combination; no additional private status-member getter is required.

An exploratory broader run included the unmodified ccache-specific fixture tests
with sccache selected: `--print-stats` was unsupported and a relative-tool child
test timed out. That run was interrupted and is not an acceptance result. A later
stalled compile request was also interrupted; the final successful gate used an
independent sccache server port and the existing persistent cache without clearing
it or stopping the original server. Windows exploratory checks had path-parser
and isolated-import failures and native prerequisite skips; they do not substitute
for the Linux evidence above.

## Revisit when

An ordinary-emission host compilation seam can retain actual ordinary exports
without replacing their bodies, or a broader production-owner ABI gate is needed.
Do not remove further source guards based on this bounded host fixture.
