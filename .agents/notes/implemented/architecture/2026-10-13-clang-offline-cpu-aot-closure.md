# Decision: opt-in native Clang offline CPU closure

Status: implemented; Linux qualification pending
Date: 2026-10-13

## Problem

The offline CPU collector admitted only native system GCC. Clang also defines
`__GNUC__`, but uses an integrated frontend/assembler and resource directory;
removing the GCC banner guard would misidentify its downstream inputs.

## Decision

Extend the existing collector through explicit `compiler_family="clang"` with
`compiler="clang"`. Preserve the default GCC recipes, reasons and salt policy.
Do not introduce a second cache, store, emitter or production dispatch owner.

Admit only the ELF driver resolved by `/usr/bin/clang` with native Linux
x86_64/aarch64 target. Disable default configuration for every query and returned
compilation. Require one actual `-###` integrated C object job in that driver,
with matching resource directory and target; bind and recheck the complete plan.
Retain the shared finite deadline, regular-file/hash checks and conservative
GNU make depfile grammar.

## Rejected alternatives

- Treating Clang as GCC via `__GNUC__`, standalone cc1/as or dumpspecs queries.
- Guessing manifest completeness from resource files, depfiles or loaded-library
  enumeration. The caller alone asserts an immutable complete snapshot, including
  absent/shadowing/PCH namespaces and opaque assembler inputs.
- Assuming GCC's random-seed option works as Clang cache identity. Use an accepted
  debug-compilation-directory command salt and require a real cached `.incbin`
  invalidation test. No debug flags are admitted; numerical expressions are unchanged.
- Executing the diagnostic job string or silently accepting external assembler,
  plugin/module/cross/sysroot configuration domains.

## Invariants

Incomplete/missing/drifting inputs yield no reusable key. Object compilation
uses the same immutable snapshot, returned flags, driver, environment and cwd
between matching pre/post collections. A store miss is not proof of compiler-cache
freshness. Manifest completeness and unobserved restoration between observations
remain producer responsibilities.

## Evidence

Windows host checks: 123 passed, 23 skips, 20 subtests passed; compiler structure
538 modules, zero dependency errors. Native API contract/boundary checks pass.
The Windows public API docstring checker reports repository-wide path/export
classification diagnostics; no unrelated APIs are changed.

Linux tests retain real GCC controls and add real Clang `-M` versus independently
compiled `-MD`/`-H` dependency observations, grammar/decoy controls, verified
store publication/reuse, execution against Decimal polynomial references,
environment poison/`__has_include` invalidation and opaque `.incbin` cache behavior.
Actual Linux execution is still required; Windows skips are not evidence of it.

## Consequences and revisit conditions

This is a narrow offline C object protocol. It does not establish production
packaging, installed bundles, runtime-cache/downstream completeness, Libxc
scientific promotion, CUDA resources or endpoint performance. Keep #1123 open.
Extend the admitted job/configuration domain only with real source-matched input
observations and independent object/cache qualification.

## References

- `docs/developer/libxc_cpu_aot_closure.md`
- #1123; previous GCC collector #2218
- [Clang toolchain](https://clang.llvm.org/docs/Toolchain.html)
- [Clang default configuration](https://clang.llvm.org/docs/UsersManual.html#configuration-files)
