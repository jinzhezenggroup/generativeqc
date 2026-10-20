# Generated resident ABI and span host contracts

Status: implemented, host-only test seam
Date: 2026-10-20

## Decision

Replace two source-spelling assertions in `test_tensor_cuda_resident.py` with
compiled execution of the complete `resident_source` result. During generation,
the test explicitly injects the ordinary emitter, validation-kernel emitter and
computation-launch emitter dependencies. It does not extract or reconstruct
resident function bodies or span rows. The generated exports and tables remain
the subject under test, together with an unchanged copy of production
`src/tensor/cuda_resident.cuh` and `metrics.hpp`.

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
  itself out. Test unequal lengths, FP32/FP64/int64 item sizes and empty spans.
- Exercise invalid slots, incorrect byte lengths, null owners/host buffers,
  allocation offset/extent bounds, injected copy/synchronization/device errors,
  nonzero statuses, diagnostics and stream draining. Preserve the input pin gate.
- Opt-in `VIBEQC_RESIDENT_HOST_CONTROLS=1` accepts formatting and private table
  renaming while rejecting generated offset, length and status corruption using
  the same behavioral oracle. These controls intentionally edit generated text;
  they are not a required-CI private spelling/hash gate.

## Boundaries retained

The ordinary ABI source assertion is deliberately retained: replacing ordinary
emission with an owner seam cannot prove ordinary exports. All other arithmetic,
symmetry, post-run, lifetime, dependency-integrity and GPU numerical tests remain.
No production generator/runtime, mathematics, resource default or selector changes
are part of this slice. Parent issue #2296 remains open.

## Validation

Run `tests/python/test_tensor_cuda_resident.py` with a real Linux C++ compiler and
the shared `native_cxx` fixture, setting `CCACHE` to a verified sccache (preferred)
or ccache executable. Preserve compiler-cache statistics, finite compile/link
timeouts and exact source provenance. Windows prerequisite skips do not establish
Linux host acceptance. Run the opt-in controls separately from ordinary CI.

The source-matched Linux qualification used Python 3.11, G++ 12.3 and sccache
0.16.0: the resident file with controls passed all 24 cases; the final related
resident/planner/FP32/compiler-identity/compiler-structure/GEMM/integer-cost gate
passed 154 cases with 19 explicit CUDA-device opt-in skips. The compiler cache
recorded 10 compilations/misses and no failures, not cache hits. Ruff, formatting
and whitespace checks also passed.

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
