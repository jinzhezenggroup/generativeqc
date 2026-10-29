# Decision: seal stationary CUDA artifacts with their installation RPATH

Status: implemented
Date: 2026-10-29

## Problem

The Linux NVIDIA host-link path introduced by #2856 gives stationary CUDA
plugins an automatic build RPATH for their shared providers. Their generated
manifests bind the post-build ELF hash. CMake subsequently removes or replaces
that RPATH during installation, leaving a correctly rejected manifest/binary
pair (#2866). On source `228a2a68c08d`, all four selected PBE0/r2SCAN SP/SPD plugins
showed this failure. The production loader reported a binary integrity mismatch.

Read-only inspection of those real ELF files found only removal of DT_RUNPATH:
145–155 changed bytes per plugin, with all other bytes unchanged. Independent
mutations of host text and CUDA fatbinary bytes were rejected by the existing
strict ELF derivation verifier. This observation diagnoses packaging; it does
not qualify any molecular endpoint.

## Decision

Set `BUILD_WITH_INSTALL_RPATH` on the Linux NVIDIA public shared-library target.
It therefore uses the caller's final `INSTALL_RPATH` when linked, before the
manifest seals its bytes. Normal installation copies the authenticated artifact
without changing it. Device-link ownership, archive pruning, public exports,
compiler flags and executable CUDA images retain their existing contracts.

## Rejected alternatives

Do not weaken the production loader, ignore binary hashes, or rewrite manifests
during loading. Merely allowing a known RPATH transformation in an external
installation checker does not repair the production manifest's stale hash.
A global `CMAKE_BUILD_WITH_INSTALL_RPATH=ON` can diagnose this behavior, but the
target property fixes the affected default without changing unrelated targets.

## Evidence and limits

The compiled HOST regression uses the actual host-link helper and a shared C++
provider, which forces an automatic build RPATH. Before this change both empty
and `$ORIGIN` installation policies invalidate the sealed hash. With the change,
build/install bytes match and the installed public entry point returns the
independent expected value. The relative-RPATH case loads without an
`LD_LIBRARY_PATH` override. These tests probe no CUDA compiler or GPU.

Real-device registration, numerical accuracy, full SDK coverage and performance
remain separate qualifications. Wheel post-processing must retain its own final
artifact publication and integrity contracts.
