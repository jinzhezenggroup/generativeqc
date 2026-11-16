# Test registration is not a Libxc scientific dependency

Superseded by [the master reconciliation](../compatibility/2026-11-16-r1-master-reconciliation.md). The whole-file exemption below is no longer active.

The retained direct/all-electron FP64 E/F evidence authenticates the full source
census. Qualification invalidation is narrower than that provenance census.

`cmake/GenerativeQCTests.cmake` owns native test/probe targets and CTest properties.
The root CMake file invokes `generativeqc_add_native_tests` only under
`GENERATIVEQC_BUILD_TESTS`; the module links consumers to `generativeqc`, not the
reverse, and does not modify production target sources, definitions, or options.
For example, #3087 only expanded AOT test registration and configured that test's
private definitions/include directory. That should not expire measured E/F.

Exclude this exact module from dependency invalidation, not from the authenticated
inventory. Keep top-level CMake, other/unknown CMake modules, public headers, and
shared native/generator inputs conservative. Do not move production configuration
into this test-only module without revisiting its ownership classification.

Tests replay the immutable campaign with only test registration changed and check
that scientific endpoints remain admitted while provenance reports the change.
They also exercise additions/removals/edits across CPU/CUDA and compiled/E/F,
ensuring the neighboring production CMake and ABI inputs still invalidate reuse.
No native or GPU measurement is made by these tests, and no historical receipt is
rewritten. This exemption alone does not resolve other shared-input changes on
#3090 or qualify later numerical/AOT/integral changes.
