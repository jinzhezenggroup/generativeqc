# Decision: opt-in native Clang offline CPU closure

Status: implemented
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
Recheck Clang invocation-entry selection around queries and before returning the
key; return the canonical executable explicitly for the object handoff.
Actual `-H` observations guard Clang make output's slash normalization and
unquoted tabs. Reject unsupported path spellings even with real decoys instead
of weakening the preserved GCC parser.

## Rejected alternatives

- Treating Clang as GCC via `__GNUC__`, standalone cc1/as or dumpspecs queries.
- Guessing manifest completeness from resource files, depfiles or loaded-library
  enumeration. The caller alone asserts an immutable complete snapshot, including
  absent/shadowing/PCH namespaces and opaque assembler inputs.
- Assuming GCC's random-seed option works as Clang cache identity. Use an accepted
  debug-compilation-directory command salt. Require fresh `.incbin` object bytes
  plus an ordinary cacheable-C replay/salt-miss control. ccache may safely decline
  `.incbin`; that fallback alone does not prove a cache hit or salt hashing.
  No debug flags are admitted; numerical expressions are unchanged.
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
Actual Linux qualification on Ubuntu 24.04/x86_64 with Clang 18.1.3, GCC 13.3.0
and verified ccache 4.9.1: **147 passed, 22 subtests passed, no SKIP**. Ruff,
compiler structure (538 modules/zero errors), native API contracts/boundaries and
the public API docstring check pass. The numerical fixture links and executes
the retrieved stored object against Decimal references for both Clang and GCC.
It retains input text, recipe, exact compiler/launcher/environment, pre/post
closures, MD dependencies/H trace, object/library hashes and numerical rows in
`qualification.json` alongside the objects. Corruption uses a separate store;
the qualified object remains intact.

Source-matched qualification snapshot: committed base `c330f089f976af9654f65c735de23fe964bc6b2e`
via `git -c core.autocrlf=false archive`, archive SHA-256
`eea4357231506e691ff7d0431cf9b31e13774240d9fc685078b8316fad402d71`, plus patch
`15d3674e1bd3d15f642ea7b313d0c551bb501f4d0220c35ed010e07dee706c0a`.
Qualified collector/test bytes are respectively
`dc7afcff9f4391cfd0671076b0142403a1f29f7a8d1572caa5e71e83455925cb` /
`fee4392d272972d01ce957b1f61697c1d4f7bcaafa0d482003e82fbf9fa21b66`.

The same final snapshot's ordinary C cache probe records one real unchanged
command replay hit and one additional miss after refreshing an opaque caller
configuration manifest/salt. Object SHA-256:
`f1e81e75ec7d86b5d2211cc3056e5465aa50a5f1f67179c00c8c8d63f422dcd5`;
full recipe/closure/counter receipt SHA-256:
`b71b888830fa6498819385346e1bce2785f14e68ea1b86fba85d0444d3ad0006`.
No cache is cleared, weakened or path-normalized to obtain the hit.

Retained negative evidence: the first ordinary Windows archive contained CRLF
despite committed LF blobs and the patch failed before compilation. The corrected
LF archive is explicitly byte-checked. First real Linux tests then recorded
93 PASS / 8 FAIL: Clang normalized backslashes and did not quote tabs, GCC `-H`
omitted its implicit stdc-predef header, and Clang included successful
`__has_include` probes in `-M`. These observations informed narrow rejection and
independent-fixture corrections. Five initial `.incbin` calls were correctly
uncacheable; no hit/performance claim is inferred from them.

Raw logs, JUnit, source hashes, objects, store manifests, cache stats and receipts
are retained in ignored `.cache/pr2444-linux-evidence.tar.gz` (SHA-256
`312b040ffbbe7fb9276a5a74c0327bb7b9bf2653870814037b48741b2aeaf42b`) and the
independent experiment directory
`/inspire/ssd/project/chemicalreaction/czxs25220150/pr2444-clang-20261013-c330f089`.
A one-off old/new GCC differential additionally preserves identical recipe,
closure payload/key and diagnostics for three ordered flag combinations; this
is evidence, not a permanent generated-body/hash CI lock.

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
