# Decision: explicit optional CUTLASS SDK and compiled qualification identity

Status: implemented
Date: 2026-10-05

## Problem

The native AOT owner and registry offers existed, but the native probe invoked
nvcc directly. A successful standalone probe did not demonstrate that CMake
consumers received the same headers and provider macro. A version label alone
also cannot establish which external header bytes produced a binary.

## Decision

Use an opt-in interface target, `generativeqc_cutlass`, with an explicit external
SDK root and qualified 3.9.2 version. Propagate it publicly to internal library
consumers. Re-read the selected root at configuration, avoiding stale find_path
cache selection when the SDK changes. Ordinary builds do not discover or fetch
the dependency. Keep wheel packaging explicitly unavailable until its artifact
contract is implemented.

Build the existing native qualification probe through this same interface.
Its artifact identity includes external header contents, owned sources, compiler
and assembler versions, generated compilation commands, and linked executable
bytes. Compute the digest after linking and pass it to execution, avoiding a
self-referential digest embedded in the binary. This is a qualification artifact;
it neither installs a production artifact loader nor supplies resource facts.

## Rejected alternatives

- Automatically finding system headers can reuse a cached SDK after ROOT changes.
- Recording hand-written compiler flags misses options added by the build system.
- A version-only SDK identity permits different modified headers under one label.
- Enabling region selection from a build option bypasses resource and endpoint
  qualification, particularly the context-retained module charge.

## Invariants

Scientific request and recipe identities are unchanged. GPU execution stays
inside finite Slurm allocations. Compilation uses ccache. CUTLASS availability
does not qualify a method default, reclaim module storage, or permit capture.

## Evidence

`tests/python/test_cutlass_build.py` exercises actual CMake admission, propagation,
changed-root reconfiguration and default-off behavior without executing a GPU.
`tests/python/test_native_cutlass_binding.py` builds and executes the existing
160-case affine oracle probe using the real external SDK.

## References

- #1886, #1888
- [Native family](2026-10-05-native-cutlass-aot-contraction.md)
- [Registry contract](2026-10-05-cutlass-aot-registry-contract.md)
