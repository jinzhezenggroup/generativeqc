# Decision: explicit host OBJECT targets with per-component codegen

Status: implemented
Date: 2026-10-13

## Problem

A single `generativeqc` compilation owner obscured CPU/module boundaries.
A first extraction of ten CPU sources demonstrated that CMake OBJECT targets
can preserve the shipped artifact, but expanding it naively exposes generated
header build-order races and differences in CUDA/wheel compile interfaces.

## Decision

Extend the first two CPU OBJECT targets with four component-aligned host targets
for integrals, SCF, DFT and Post-HF/CC numerical implementation, retaining the
public `generativeqc` target and unchanged method/API entry points. Every
source is compiled by exactly one owner. Each new component explicitly
depends on the custom targets producing its generated headers. Inherit
compilation definitions/options/include paths from the parent and explicitly
add toolkit include directories for CUDA-enabled host `.cpp` builds.

Retain CUDA `.cu`, AOT generated source files, device-link, non-qualifying
method adapters and CUDA split-hybrid consumers on their existing owner until
they have a separately named, auditable generation boundary.

## Rejected alternatives

- Per-module STATIC archives: archive extraction can silently discard
  registration/export-only objects and alter monolithic CLI behavior.
- Depending every component on `generativeqc`: creates a target cycle and
  defeats compile isolation.
- Moving all `.cpp` sources automatically: CUDA-gated headers, generated
  inputs and source-specific compilation policies require explicit ownership.

## Invariants and validation

The installed shared library, wheel layout, C ABI, monolithic static CLI and
CUDA device-link remain the same. Configure-time checks reject duplicate
translation-unit ownership and missing named generator targets. The isolated
CMake fixture compiles and links an OBJECT component that consumes a header
created during the build, including inherited parent definitions. Full
CPU, CUDA, wheel and static CLI builds and numerical regression tests remain
the repository CI acceptance gate.

## Revisit when

CI measures a significant cold-build or PCH overhead from object boundaries,
or when generated/CUDA source ownership can be split without violating device
link, export, and source-identity semantics.

## References

- `docs/developer/architecture.md`
- `cmake/GenerativeQCSources.cmake`
- PR #2378 (initial two-target extraction)
