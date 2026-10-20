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
  nonzero statuses, diagnostics and stream draining. Preserve the input pin gate.
- Opt-in `VIBEQC_RESIDENT_HOST_CONTROLS=1` accepts formatting and private table
  renaming while rejecting ABI-version, generated offset, length and status
  corruption using the same behavioral oracle. These controls edit generated text;
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

The final narrower sources passed the Linux gate with 156 cases and 19 explicit
CUDA-device opt-in skips, including all 26 resident cases (five opt-in controls).
The cache recorded 11 compilations/misses and zero compilation failures.
The three qualified test/helper/fixture LF hashes were checked against the
committed blobs. The compiler/runtime boundary remains host-only; this result
does not qualify ordinary emission, floating validation, kernels or science.

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
