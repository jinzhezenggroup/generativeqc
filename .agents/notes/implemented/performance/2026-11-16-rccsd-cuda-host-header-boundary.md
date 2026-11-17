# Decision: keep RCCSD host replay out of CUDA admission headers

Status: implemented
Date: 2026-11-16

## Problem and decision

The UCCSD frontend audit also exposes the same issue in RCCSD: CUDA owners
include large CPU evaluator bodies just to borrow shared types and admission
queries. The conventional RCCSD header alone is about 2 MB and is included by
generated responses and handwritten native owners.

Guard the CPU evaluator block in the main header and each CPU resolvent/moment
evaluator in the triples-Fock header with `#ifndef __CUDACC__`. Keep shape,
capacity, reuse, selected-program identity and bounded-reassociation admission
queries outside those guards. All ordinary C++ evaluators remain unchanged.
Neither numerical programs nor generated CUDA source/output inventories change.

This is deliberately **not** a global change to `_cpu_function`: for example,
the host side of `src/hf/rhf_frame_response.cu` intentionally calls its generated
CPU preconditioner. Backend ownership must be checked at each consumer rather
than inferred merely from a CUDA source extension.

## Complete-object evidence

Verified sccache 0.16.0 and NVCC 12.9.86 compile the complete baseline/candidate
objects with `sm_80`, `-O3`, C++20, relaxed constexpr and `--fmad=false`:

| CUDA owner | Baseline preprocessed bytes | Candidate preprocessed bytes | Reduction |
| --- | ---: | ---: | ---: |
| RCCSD | 9,227,216 | 7,358,464 | 20.25% |
| Triples response | 5,763,899 | 3,895,147 | 32.42% |
| Triples Fock | 5,651,006 | 3,725,496 | 34.07% |

Both complete triples units retain byte-identical PTX and CUBIN. RCCSD PTX
differs only in NVCC's per-translation-unit internal-symbol discriminator on
CUDA/Thrust standard-library constants; normalizing that discriminator gives
identical PTX. All **2,373 allocated ELF sections**, including **1,183 executable
sections**, retain identical types, flags, sizes and payload bytes. Do not call
the entire RCCSD CUBIN byte-identical: its symbol-name metadata differs.

The source-equivalence check allows only the new host guards in the two headers
and verifies that all three generated CUDA sources are byte-identical. These
are one-off migration proofs, not permanent source/body fingerprint tests.

The baseline has zero cache hits and 12 CUDA-stage misses; the candidate has
two CUBIN hits and ten misses. Observed complete-object times are respectively
108/105, 21/17 and 11/7 seconds. These mixed-cache, overlapping observations
do **not** establish cold-build speedups. Actual removed frontend input, rather
than those timings, is the work-reduction evidence.

The handwritten CUDA solver, Lambda response and triples-response owners also
compile completely against the new headers. All twelve of their CUDA-stage
requests miss the cache and finish without compiler failures.

## Executable contracts and scope

Four focused backend-boundary/dependency tests pass in 43.19 seconds. The new
RCCSD probe checks CPU-only replay/resolvent APIs, equal CPU/CUDA-interface
admission for ordinary, extreme-aspect and empty shapes, overflow failures and
CPU insufficient-capacity rejection. Actual CPU energy and both residuals match
the independent unlowered expanded TensorIR program. The CUDA-interface mode
uses host compilation and does not claim physical-device execution.

Another 27 arena-liveness, generator-evidence and scalar-reduction tests pass
in 21.30 seconds. Compiler structure reports 604 modules and zero dependency
errors. The official Libxc native catalog checker still passes without another
snapshot update; no qualification or acceptance projection changes. The new
test is selected by existing tracked-file core-shard discovery.

This follows the
[UCCSD boundary decision](2026-11-16-uccsd-cuda-host-header-boundary.md).
Do not merge unrelated numerical graphs into a larger unity source or remove
scientific/method/architecture coverage to improve compilation statistics.
Full manylinux wheel acceptance and complete Direct endpoint/runtime gates
remain separate qualifications.
