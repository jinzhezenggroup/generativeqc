# Decision: Use the CXX host-link rule for CUDA wheel imports

Status: implemented
Date: 2026-10-06

## Problem

Provider-free wheels discover CUDA imports from a strict failed host link and
retry with generated lazy-import objects. This launcher must run for both the
mixed-language main library and CUDA-only stationary shared libraries.

CMake 4.4.4 does not apply target linker launchers to NVIDIA's CUDA host-link
rule: the rule uses `CMAKE_CUDA_HOST_LINK_LAUNCHER`, rather than a compiler
placeholder or `CMAKE_LINKER`, where CMake injects the launcher. Setting
`CUDA_LINKER_LAUNCHER` therefore silently leaves CUDA-only targets unwrapped.
CMake 4.1 added linker-launcher support for Fortran, not CUDA.

## Decision

Select `LINKER_LANGUAGE CXX` for provider-free wheel targets and attach the
supported `CXX_LINKER_LAUNCHER`. Preserve the target's CUDA sources and all
separable-compilation/device-link properties. Keep shared usage requirements in
the existing `generativeqc_cuda_wheel_imports` interface target.

The CXX launcher is available before the repository's CMake 3.24 minimum. No
wheel-only CMake 4.1 override is needed. Removing that override also avoids a
wheel/editable mismatch: both package build states enable wheel-mode CMake.

Linker diagnostic discovery continues while new provider symbols are found.
A fixed four-link limit is insufficient for lld's default twenty-error limit.
The fixed link inputs have a finite symbol set, and repeated/unknown unresolved
symbols still fail as soon as discovery stops making progress.

## Evidence and limits

- A host-only CMake/Ninja test supplies NVIDIA compiler-identification results
  and inspects generated rules without executing CUDA commands. The prior CUDA
  host rule omits the launcher; the CXX rule includes it. The CUDA device-link
  object remains an input to the final shared library.
- Host tests exercise a real strict CXX link, lazy cudart/cuBLAS loading,
  rank-2k argument forwarding, unrelated unresolved symbols, truncated
  diagnostics spanning more than three batches, and no-progress failure.
- These tests do not establish CUDA kernel or complete wheel correctness.
  The normal manylinux CUDA wheel workflow provides that qualification.

## Revisit when

CMake documents and implements NVIDIA CUDA host-link launcher support and the
actual CUDA-only/device-linked wheel targets pass the same qualification.

## References

- [PR #1979](https://github.com/jinzhezenggroup/generativeqc/pull/1979)
- [CMake linker-launcher documentation](https://cmake.org/cmake/help/latest/prop_tgt/LANG_LINKER_LAUNCHER.html)
- [NVIDIA host-link rule](https://github.com/Kitware/CMake/blob/v4.4.4/Modules/CMakeCUDAInformation.cmake#L89-L92)
- [Launcher expansion](https://github.com/Kitware/CMake/blob/v4.4.4/Source/cmRulePlaceholderExpander.cxx#L347-L353)
- [Independent device-link selection](https://github.com/Kitware/CMake/blob/v4.4.4/Source/cmLinkLineDeviceComputer.cxx)
