# Decision: test ordered-Gram admission through the production launch ABI

Status: implemented
Date: 2026-10-19

## Problem and decision

Refs #2296. The original host test sliced a CUDA wrapper by signature/namespace
spelling and replaced one kernel launch with a regular expression. This could
reject harmless implementation refactoring without exercising the real public
launcher. The first replacement called the extracted production preflight
normally, but its permanent test still could not observe wrapper misrouting.

The required admission gate now calls `launch_update_diis_cached_gram` linked to
the actual `generativeqc` shared library. Its executable exports a process-local
`cudaLaunchKernel` recorder with ELF dynamic-symbol interposition. This records
calls originating inside the production DSO, rather than using `--wrap` only on
test objects. Every admitted call must reach the recorder exactly once; every
rejected call must throw `std::invalid_argument` with zero submissions. A missing
interposer therefore fails instead of silently accepting a no-device refusal.
The recorder validates the compiled launch payload and public argument routing,
without naming a kernel's private function or reading its implementation body.

The existing internal header retains the unchanged ordered-Gram preflight. The
permanent driver does not reference its private name: implementations may rename,
move or inline it while preserving the production launch ABI. No installed API,
exception semantics, kernel/template dispatch, math, default, allocation or
stream-ordering policy changes.

## Required execution and retained coverage

`generativeqc_ordered_gram_host_admission_tests` is a Linux/NVIDIA shared-library
test registered in `cmake/GenerativeQCTests.cmake`. A standard-library Python
runner supplies independent, unbounded integer byte products around native
storage limits. It checks that the executable consumes every supplied case.
The existing CUDA compile job explicitly builds and runs this target alongside
both existing Fock/proposal tests, retaining `CUDA_VISIBLE_DEVICES=""`, parallel
2, their selections and time limits. There is no skip-77 result for this gate.
If its target is absent or its recorder is missed, required CI fails.
The existing step identifier is retained: an initial title change broke seven
unchanged cache-policy anchor tests. Their reproduced RED is retained; restoring
that identifier keeps both the old policy coverage and the added host gate.

The original 13 rejected and three valid/disabled bindings remain in the public
launcher driver. Coverage expands to 37 independent rejections, every required
pointer, both dimension signs, geometry/history bounds, exact cache/solve alias,
copy-only nullable/aliased storage, NaN/inactive/invalid-count borrowed state,
non-host-readable addresses, public defaults and non-default stream/shared/work
routing. Partial overlap, actual allocation extents, live-ring identity and
stream ordering remain the original caller preconditions.

`tests/python/test_ordered_ring_gram_host.py` retains the original 378 arithmetic
cases unchanged: capacities 2 through 64, bitwise ordered results, independent
long-double oracle, wrap, retirement, uncleared resets, inactive neighbors and
work counters. Its helper-only admission probe migrates to the mandatory native
host gate rather than becoming an optional or skipped test. No production guard
implementation is copied into a test.

## Replaced source assumptions

| Removed assumption | Executable replacement |
| --- | --- |
| Signature and namespace spelling delimit the wrapper body | Link and invoke the existing production launcher ABI |
| One regex match identifies the CUDA launch | Real DSO API interception, zero submissions on rejection, mandatory positive recording and argument-routing checks |
| Extracted guard text compiles with an inserted recorder | The unchanged public driver survives a private-helper rename while wrapper bypass, misrouting and guard removal fail |
| Addressability guard spelling | Independent Python byte products and consumed-input counts; 300 native 64-bit boundary cases |
| Namespace composition requires a private scheduler helper | A test-owned client tag models the HF import; an empty exported CUDA detail namespace reproduces ambiguity |

## Evidence and limitations

Clean baseline upstream: `bc311645f797614e6e4985808696f7d2cc2220db`.
The LF archive SHA-256 is
`2d2e02728c1301aa2e240e78a7e43d7651fde2cf76674c57f76bf1d7c958dc32`.
The original focused baseline passed two tests. After admission migration the
related CPU suite passes 28 tests, including all 378 arithmetic cases, and the
new native host CTest passes the 37/3 cases plus 300 independent byte cases.

Local controls use verified sccache 0.16.0, GCC and NVCC 12.9.86. The exact
production DIIS translation unit is compiled into a shared-library control;
the real CMake helper builds, links and executes the unchanged permanent driver.
Dynamic symbols confirm the launch entry is in that DSO and the recorder is
exported from the executable. This local component DSO is distinct from the
complete production DSO exercised by the required CUDA CI lane.

Renaming the private preflight in its header and production caller, then
recompiling the permanent driver unchanged, passes. Producer-DSO mutations of
25 guard conditions, public-wrapper bypass, admission-shape misrouting and
kernel-argument misrouting all fail. Removing the executable's CUDA API hook
also fails. These 29 negative controls exercise the real public launcher rather
than a copied or extracted guard. An auxiliary real `-m32` guard control passed
260 byte-boundary cases and rejected deletion of the solve-square bound. That
branch is unreachable with admitted 32-bit shape parameters in a 64-bit address
space; the 64-bit run alone does not qualify it.

The initial PR head's full CUDA package build exposed a generic nested `detail`
namespace collision in the HF consumer. The dedicated ordered-Gram namespace
fix preserves that consumer unchanged; retain the original CI RED. A later
formal delta COMMENT (review 5572545147) identified the missing permanent DSO
routing gate. Its body was initially not fully read, and supplementary helper
controls did not resolve that gap. The required public-launcher gate above is
the correction. Discovery supplies explicit EOF; a live attached PTY reproduced
the former blocked child and passed after repair.

The existing device test also retains its added 11 public-launcher pointer
rejections, but its device-gate return 77 is supplementary evidence, not a PASS
for the required host gate or scientific device acceptance. No kernel executes
in the API-recorder process. No GPU synchronization, complete SCF endpoint,
scientific acceptance or performance improvement is inferred. Read required CI,
latest-head review and merge status from the current PR separately.

Opt-in mutation scripts and raw receipts remain ignored artifacts under
`/inspire/hdd/project/chemicalreaction/ordered-host-2296-1019/` on the slice-owned
CPU notebook. They are not permanent source-spelling or fingerprint gates.

## Revisit when

Extend real-device qualification for any kernel, synchronization, math, cache
lifetime or dispatch change. Host recording cannot qualify those changes.
Preserve the reducer rationale in
[the endpoint integration note](2026-10-10-ordered-diis-endpoint-integration.md).
