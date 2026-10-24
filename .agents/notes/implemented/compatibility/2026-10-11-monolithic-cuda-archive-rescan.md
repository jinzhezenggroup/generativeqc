# Decision: rescan the monolithic CLI's core and GFN2 CUDA archives

Status: implemented
Date: 2026-10-11

## Problem

The first automatic master native CLI distribution build after #2283 compiled
its CUDA and host sources, then failed at the final executable link. The link
line scanned `libgenerativeqc.a` before whole-archiving `libgenerativeqc_gfn2_cuda.a`.
GFN2 CUDA objects introduced references back to separately archived host plan
schema helpers and the Linux lazy driver-import trampolines only after the host
archive had already been searched. Both families were consequently undefined.

The driver imports were already generated and compiled into the core. This was
not a missing CUDA toolkit dependency or missing generated science source.

## Decision

For the monolithic CLI only, when the GFN2 CUDA archive exists and the linker
supports CMake's `RESCAN` feature, group the core and whole-archived CUDA target at
the final executable link. CMake permits a nested `LINK_LIBRARY:WHOLE_ARCHIVE`
inside `LINK_GROUP:RESCAN` and gives group membership precedence over standalone
transitive references. Retain the existing core-to-CUDA dependency and device-link
ownership unchanged. Shared SDK/wheel builds, CPU-only builds and toolchains
without the rescan feature retain their existing link expression.

## Rejected alternatives and invariants

- Do not add `CUDA::cuda_driver` on Linux to paper over the missing driver symbols:
  the existing lazy `libcuda.so.1` imports permit startup without an NVIDIA driver.
- Do not drop CUDA whole-archive consumption: registration/device-link-only
  members must remain linked even when no ordinary host symbol references them.
- Do not whole-archive the entire core or add a cyclic target dependency merely
  to induce repeated archives. Only this final link's actual archive pair needs
  rescanning; unrelated objects and shared target graphs need no change.
- This changes linkage only, not CUDA kernels, mathematical paths, or runtime
  admission. No GPU numerical or performance qualification follows from it.

## Evidence

`tests/python/test_native_cli_link.py` executes the production final CLI CMake
helper with tiny compiled host stand-ins and the actual generated C/assembly lazy
imports. Monolithic CUDA-shaped, ordinary shared and CPU-only links run from the
build and install trees; constructor-only registration remains present, and the
monolithic executable has no GenerativeQC or CUDA-driver shared dependency.
Negative controls reproduce both original missing-symbol families with the old
order, reject omitted imports, and observe missing registration when whole-archive
consumption is removed. Compiler-cache-backed host probes do not substitute for
the automatic full CUDA distribution build and artifact smoke/packaging stages.

References:
- Failed master job: https://github.com/jinzhezenggroup/generativeqc/actions/runs/38151819911/job/114509681942
- CMake group semantics: https://cmake.org/cmake/help/v3.24/manual/cmake-generator-expressions.7.html#genex:LINK_GROUP
