# Decision: call the ordered-Gram host admission contract directly

Status: implemented
Date: 2026-10-19

## Problem and decision

Refs #2296. The ordered-Gram host test sliced the production CUDA wrapper by
signature/namespace spelling and replaced a particular kernel launch with a
regular-expression recorder. Harmless CUDA source refactoring could break the
test without changing admission behavior.

The existing internal `scf/cuda/scf_diis_kernels.hpp` now holds the inline host
preflight in its private `detail` namespace.
The production wrapper calls it immediately before its unchanged kernel launch.
The host test includes and calls this function normally. The existing native
CUDA test also calls the public launcher with each required binding absent,
before its existing Slurm/device gate, so bypassing the helper is observable.
No public signature, exception category/message, kernel, template instantiation,
stream, shared-memory, normalization, allocation or reduction policy changes.

Pointers are borrowed device addresses. Host preflight checks nullness and exact
cache/solve aliasing without reading or writing their contents. Partial overlap,
actual allocation extents, live-ring identity and stream ordering remain caller
preconditions, as documented by the public interface. Disabled histories zero
and one retain their copy-only/null-history/null-cache exception.

## Behavior replacing the removed source guards

| Removed assumption | Executable replacement |
| --- | --- |
| Wrapper signature/namespace must delimit a source slice | Compile and call the production header; public-launcher negative cases in the existing native test |
| A regex must recognize exactly one launch | Real full-translation-unit NVCC compile and host launch interception; rejected calls submit no work, admitted calls preserve the launch packet |
| Extracted guard text must compile with an inserted recorder | Original 13 rejections and three valid/disabled cases retained; 37 rejections now cover each required pointer, both shape signs, geometry, history and alias errors |
| Addressability guard spelling | Python unbounded-integer byte products determine acceptance around exact storage boundaries, including copy-only histories |
| Host validation may inspect borrowed state | NaN/inactive/invalid-count state remains untouched; non-host-readable sentinel addresses are admitted without dereferencing |

The original 378 arithmetic cases remain intact: capacities 2 through 64,
bitwise ordered results, independent long-double oracle, wrap, retirement,
uncleared resets, inactive neighbors and work counters.

## Source-matched evidence

Baseline upstream: `bc311645f797614e6e4985808696f7d2cc2220db`.
The LF Git archive SHA-256 is
`2d2e02728c1301aa2e240e78a7e43d7651fde2cf76674c57f76bf1d7c958dc32`.
The original focused suite passed two tests before production edits. The new
focused suite passed 29 tests across ordered-ring host, endpoint contract,
ordered-history oracle and host DIIS symmetry tests.

On the slice-owned CPU notebook, GCC and NVCC 12.9.86 compiled through verified
sccache 0.16.0. Full `scf_diis_kernels.cu` compilation succeeded for `sm_90`.
A host linker interception of `cudaLaunchKernel` exercised the actual compiled
wrapper: all 37 invalid cases submitted no work, valid/disabled/poison cases
reached the launch recorder, and stream/shared-memory/normalization/work and
optional kernel arguments retained their values. This is host wiring evidence,
not execution of the kernel.

Opt-in qualification removed each of the 13 shape/geometry/input conditions,
eight enabled-history pointer conditions, exact alias rejection, history extent,
the disabled-history exception and double-byte scaling: all 25 mutations were
rejected by the behavior test. A real `-m32` build passed 260 independent byte
boundary cases. Deleting the solve-square upper bound then failed on
`batch=1048576, nbf=1, spins=1, history=64`: history bytes fit while solve bytes
exceed 32-bit addressability. The solve bound cannot be reached by the admitted
32-bit shape parameters on a 64-bit address space, so the 64-bit run alone does
not qualify that branch.

The existing native test compiled/linked against the real CUDA translation unit,
completed its 11 public-launcher rejection cases, then returned its established
device-gate code 77. No GPU execution, synchronization, complete SCF endpoint,
scientific acceptance or performance improvement is inferred from these checks.
CI/review/merge status must be read from the current PR separately.

Local qualification scripts and raw logs are opt-in ignored artifacts, not
permanent required-CI source-spelling/fingerprint checks. Remote receipts are
under `/inspire/hdd/project/chemicalreaction/ordered-host-2296-1019/` on the
slice-owned `i2296-ordered-host-1019` CPU notebook. Do not use another owner's
resource to rerun them.

## Revisit when

Change the host contract when the public launcher deliberately changes admission
or storage semantics. Extend real-device tests for any kernel, synchronization,
math, cache lifetime or dispatch change; host preflight evidence cannot qualify
those changes. Preserve the historical reducer decision in
[the endpoint integration note](2026-10-10-ordered-diis-endpoint-integration.md).
