# Decision: validate cached OpenBLAS successes with independent links

Status: implemented
Date: 2026-10-10

## Problem

The capability fingerprint introduced in PR #2176 did not include effective
transitive usage requirements, transitively included headers, configured imported
locations, or library bytes. Reusing a stale positive could enable an unavailable
API in the production CPU linear-algebra translation unit. Hashing previously
observed dependencies also misses newly appearing include-search results.

## Decision

Treat cached successes as hints. When at least two results were successful,
validate those exact probe sources in one fresh CMake project, using separate
executables (or separate archives for the caller's source-only check mode).
CMake's source-signature `try_compile` imports the actual provider target graph.
A deferred project hook adds the other checks with matching compile/link settings
and their own capability-result definitions. Every executable links independently.

The first target's post-build commands build every remaining check. Merely using
`add_dependencies` is insufficient: Unix Makefiles `try_compile` invokes a
`/fast` target that bypasses those dependencies. A fresh completion marker proves
the hook attached all checks, and successful native build proves they succeeded.
The generated build directory is cleared for each validation; compiler-launcher
caches remain enabled and can safely reuse compilation.

Cached failures are always checked again. Failed/incomplete batches invalidate
all successes and run the ordinary individual checks. Custom toolchains, project/rule
hooks, generators other than Unix Makefiles, Ninja, and Ninja Multi-Config,
and semicolon-containing required flags also retain individual checks.
Deferred inclusion observes the final generated directory policy state, avoiding
policy capture before CMake applies the caller's PIE/export settings.

## Rejected alternatives

- Incomplete metadata/content hashes cannot safely skip current validation
- One aggregate executable changes linker semantics: another probe can supply a
  missing definition or extract an earlier archive that rescues a later one
- Silently replacing this with unconditional three-project probing discards the
  opportunity to share CMake project generation on ordinary configurations

## Evidence and consequences

The regression suite covers positive/negative changes in target requirements,
headers, conditional include discovery, header shadowing, configured library
locations, and same-path/same-mtime archives. Archive-order tests protect separate
link semantics. Compiler instrumentation verifies three actual compiles and three
independent links during an all-positive, one-project reconfigure. The 22-case
suite passes with GCC 14 on CMake 3.24.3 / Unix Makefiles, and CMake 4.4.4 /
Unix Makefiles, Ninja, and Ninja Multi-Config. The exact original PR module fails
all twelve added provider-identity transitions. A separate static-library
regression fails if required archiver options are omitted from the extra targets.

This is a three-to-one reduction in generated check projects, not zero probing
or fewer independent links. Two positives plus one negative use two projects;
a failed batch can require four projects before subsequent configurations settle.
No probe executable is run, and no numerical kernel or tolerance changes.

A local warm-cache measurement with the real scipy-openblas32 0.3.34.0.0 CMake
package found two successful capabilities and one unavailable local-thread API.
That case uses two projects. Eight interleaved repeats per implementation gave
median seconds of 0.324 versus 0.352 (Unix Makefiles) and 0.330 versus 0.313
(Ninja), comparing ordinary individual checks with batching. These small mixed
results establish no universal wall-time speedup; the claimed saving is project
generation count, while preserving fresh independent compile/link validation.

## References

- PR #2176: https://github.com/jinzhezenggroup/generativeqc/pull/2176
- `tests/python/test_cmake_cpu_linalg_cache.py`
