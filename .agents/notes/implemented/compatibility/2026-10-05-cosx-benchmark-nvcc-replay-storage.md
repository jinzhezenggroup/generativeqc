# Decision: fixed stack replay storage for the COSX CUDA benchmark

Status: implemented
Date: 2026-10-05

## Problem

The weighted COSX endpoint harness prevented the complete COSX CUDA regression
target from compiling with CUDA 12.9 and the production C++20/sm120 flags. In the
actual translation unit, NVCC's retained host source spelled a nested
`std::array` replay extent as `unsigned long((6))`. G++ rejected that spelling as
a template argument. This was a compilation failure, not numerical evidence or
a failed GPU qualification.

## Decision

Keep the four-route and six-replay namespace constants and use fixed stack
arrays for samples, energies and paired errors. Copy the three checked error
components into each replay slot. Keep the other route arrays, route rotation,
CPU subset oracle, full-grid pairing and semantic-work assertions intact.
The host work probe imports the actual constants while counting six replays
independently; its static assertion protects the benchmark dimensions.

## Rejected alternatives

Moving bounds from local variables to namespace `constexpr std::size_t` did
not repair the full target. Neither unsigned constants nor named nested-array
aliases, including literal replay extents, avoided the retained-host failure.
Small isolated nested-array sources compiled with size_t, unsigned and int
bounds. Those probes did not reproduce the full translation unit and cannot
establish a general restriction on `std::array` or diagnose a vendor root cause.

The first stack-array attempt retained whole-array assignment for paired
errors, which CUDA correctly rejected as a non-modifiable lvalue. Copying all
three components fixes that separate C++ error without changing arithmetic.

## Invariants

Preserve four provider masks, six builds per route, both geometries, exact
full/tail work counts and the original numerical gates. Storage stays on the
host stack with compile-time bounds. No provider admission, production default,
device allocation or timed endpoint work changes.

## Evidence

The actual `generativeqc_cosx_cuda_tests` CUDA 12.9 target compiled and linked on
n1 with explicit ccache use after the stack-array/copy change. The ignored
`.artifacts/1884-cosx-symmetric/` directory on n1 retains
`nvcc-host-emission/` (full command, host emission and small probes),
`failed-raw-array-assignment/`, the successful `array-compile-probe.log`, and
before/after cache statistics. These compilation receipts do not substitute
for final-source complete-endpoint GPU and sanitizer qualification.

## Revisit when

A supported CUDA/toolchain combination compiles the complete test translation
unit with nested replay arrays. Reproduce with the full target before replacing
the bounded stack storage; isolated examples were insufficient here.

## References

Issues #1884 and #1886; PR #1973; `tests/native/cosx_weighted_endpoint_benchmark.cuh`
and `tests/python/test_cosx_weighted_endpoint_work.py`.
