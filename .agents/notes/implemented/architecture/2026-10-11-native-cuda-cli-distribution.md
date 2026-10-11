# Decision: Native CUDA CLI release artifacts use a monolithic core

Status: implemented
Date: 2026-10-11

## Problem

The existing C++ CLI is Python-free at runtime but requires a separate
`libgenerativeqc.so`. Existing CUDA CI uses fast compiler flags for gate
latency and does not package a versioned executable. Users need an artifact for
every successful master build and a release path that cannot publish an
accidental tag or overwrite an existing release.

## Decision

Keep the shared ABI-compatible core as the SDK and wheel default. A
distribution-only CMake flag builds that same core as a static archive and links
it into the native CLI. A `cuda-release-cli-sm120` preset retains the reviewed
production AOT path and disables `--Ofast-compile`, with ccache and C++ PCH.
Do not build production releases with the fast PR gate's compiler options.

Integrate an independent release-mode job into the existing CI workflow rather
than broadening required PR jobs: it depends on the existing CUDA compile gate,
uploads immutable per-commit Actions Artifacts, and publishes a GitHub Release
only when an explicitly pushed `vX.Y.Z` tag matches CMake and is on master.
Read/check existing releases first and never overwrite their assets.

Keep external CUDA/system DSOs externally provided. The native release preset
retains core-linked AOT shell kernels and stationary CPU derivative shards but
disables the standalone stationary CUDA .so targets: those C export-based modules
are called from the Python-specific force adapter and are not linked into the
native CLI. Different stationary profiles currently export colliding C names,
so simply combining the libraries would not preserve method dispatch. Fail
packaging if any GenerativeQC-owned .so remains in the distribution installation.
Preserve build provenance, license notices and deterministic archive checksums.

## Rejected alternatives

- Automatically publishing a release on each master push: version identity is
  not changed by every merge, and this would create unapproved release churn.
- Forcing a monolithic shared + static core into ordinary CI/SDK configurations:
  this recompiles expensive generated CUDA units and changes existing API links.
- Claiming all CUDA + driver DSOs are in one fully static ELF: the CUDA driver
  is a host runtime contract and cuBLAS/cuSOLVER providers remain dynamic.
- Linking multiple Python stationary-force wrappers into the CLI without
  namespaced symbols and a new native dispatch/identity contract: the wrappers
  export overlapping C names and the CLI does not call this Python-owned path.
- Changing normal CUDA PR builds to full release code generation: would hurt
  feedback latency and compiler cache behavior.

## Invariants

- No GitHub Release or new tag on ordinary master pushes.
- A tag must equal the CMake version and point to master history.
- Never overwrite an existing Release or its assets.
- PR build gates remain optimized for feedback time.
- The shared C++ SDK and Python wheel remain the default.
- Distribution archives contain one GenerativeQC executable and no local `.so` files.
- Source-generated CUDA code retains the pinned scientific identity and AOT
  manifest; user-visible packages must include notices.

## Evidence

`python3 -m unittest discover -s tests/python -p test_native_cli_package.py`
validates deterministic tarballs, checksums, license inclusion and rejection
of separate stationary CUDA `.so` files without a GPU. CI release-mode jobs additionally check actual ELF
linkage and smoke test native CLI methods and basis catalog. CUDA hardware
execution/performance qualification still requires an appropriate GPU runner;
a successful compile-only run is not evidence of device-level speed.

## Consequences

Master pushes have a separate release-mode build job. The CLI avoids compiling
or shipping Python-only force plug-ins, reducing distribution size and compile
work without removing native CLI functionality. Only sm_120 is initially
distributed. External CUDA/host runtime providers remain necessary.

## Revisit when

Reliable static CUDA toolkit/provider linkage or a validated relocatable CUDA
runtime distribution is available, and when real-device smoke/performance tests
can cover additional GPU architectures.

## References

- `CMakeLists.txt` and `CMakePresets.json`
- `.github/workflows/ci.yml`
- `docs/user/native_cli.md`
- `tools/package_native_cli.py`
