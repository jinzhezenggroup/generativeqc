# Decision: bind rank-k shapes and the fixed GCC host closure

Status: implemented
Date: 2026-10-10

## Problem

The standalone harness executed both `3x5` and `17x9` panels but originally
reused metadata emitted from only the `3x5` TensorIR program. Dimensions are
part of request and scientific identity, so numerically correct `17x9` rows
were mislabeled. Compilation identity also hashed only the selected `g++`
driver, although that driver invokes separately mutable programs and consumes
system headers, startup objects and libraries.

## Decision

Emit independent lowering portfolios for every fixed qualification shape. The
native harness selects request, candidates, target and compilation identity by
`n/k`; generated request identities map to the exact flattened prefix count,
which construction and execution both enforce. This is a finite
qualification matrix, not a shape-polymorphic production contract.

The fixed GCC 11 host closure is a staged manifest with path-independent roles
and SHA-256 identities. It includes the driver, `cc1plus`, assembler,
`collect2`, linker, their dynamic dependencies, every file under the compiler's
reported C++ include search roots, resolved startup/library inputs, effective
GCC specs and the linker's default script. The driver bytes named by the
manifest must match the explicitly selected `-ccbin` executable.

CPU source staging reads that manifest without executing a compiler. GPU
qualification regenerates it from the actual compiler, requires byte equality
with the staged manifest, then regenerates and compares the CUDA metadata before
compilation. Bare GCC child names resolve from the same final PATH used by
compilation. Ambient compiler/include/library overrides remain rejected. The
directly compiled native harness is a separate fixed-recipe identity input.

## Rejected alternatives

- Reusing one request for several runtime sizes mislabels scientific identity.
- Calling the request shape-polymorphic would invent a contract TensorIR does
  not provide.
- Hashing only the `g++` driver leaves independently mutable children and
  sysroot inputs outside the build identity.
- Absolute host paths are provenance, not portable identity; manifest roles
  and content hashes remain relocation independent.

## Invariants

- Every timed row uses metadata emitted from its exact `n/k` TensorIR program.
- Runtime `n/k` and the flattened logical batch/spin prefix exactly match the
  fixed request; no capacity-subset polymorphism is inferred.
- Missing shapes, host roles, compiler bytes or manifest mismatches fail before
  compilation.
- Source generation remains stdlib-only and never probes a compiler or GPU.
- The manifest is a bounded identity for this fixed recipe, not a claim that all
  possible host compilation environments are hermetic.

## Evidence and scope

Mutation/missing-input tests cover CUDA and staged host manifests. Shape tests
require distinct request/scientific identities for `3x5` and `17x9`. qz Job
`i1877-rankk-host-1010y2` generated a 3,666-role GCC closure; Job
`i1877-rankk-h100-1010z10` regenerated the same manifest from the final PATH,
compiled exact commit `68de92cd32a356c1c05e41c3935cc206da77a7ea`, and passed the
complete H100 matrix with distinct identities for every weighted/shape family.
That source also binds the constructor to the request's exact flattened prefix
count and includes the directly compiled native harness in compilation identity
schema v4.

## Revisit when

The compiler owns a general shape-polymorphic request schema or a repository-wide
hermetic host toolchain abstraction that can replace this qualification-local
manifest.

## References

- #1877 and PR #2174
- `tools/generate_rank_k_host_toolchain_manifest.py`
- `tools/generate_symmetric_rank_k_cuda.py`
- `tools/qualify_symmetric_rank_k_cuda.sh`
