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
Require exact equality between Clang's make header set and opened-header trace,
not merely trace containment. Successful but unopened `__has_include` probes are
explicitly non-reusable because their original path cannot be witnessed by `-H`.

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

### Superseding unopened-probe guard

Further audit identified that a successful `__has_include` probe can appear in
Clang `-M` without appearing in `-H`. Trace containment alone would not reject a
tab-split or slash-normalized unopened probe whose decoys exist. Exact set equality
now rejects it; this is a deliberate Clang-only support-domain restriction, not
a claim that the preceding snapshot produced incorrect object mathematics.

Source-matched real Linux requalification of this guard: **149 passed,
22 subtests passed, no SKIP**, including successful real tab/backslash probes
with real split/normalized decoys. Both stored-object Decimal oracles and all
compiler/native API/docstring/Ruff gates pass again. The ordinary-cache probe on
this same snapshot again records one replay hit and one refreshed-salt miss.

The newer snapshot uses committed `a993f85cfff22b5ebf55d9d77d17ba269482a04b`, LF
archive SHA-256 `6188aeb61fcfc9ad949b3253c11000ea12f93054cd11a4de65440dc6afa9dea6`,
plus patch `392b270f6c1dc12b8cb26a928f2d7793e63e9524d6a01bbe6b589f19ac5fe813`.
Collector/test SHA-256 values are
`98f1a3cf564efa7ed7664e7982b8c461d13e26133a38769a2cad0610286ffaa0` /
`0477d39b24c580462558766173758e3a56092beaee9e7609f5da29c76d2ff0a8`.
Cacheable ordinary object SHA-256:
`59986b11920e73472c03c0ee28b3734dcdfcb4ef360b2d2feea2bc17e2d5ed7e`;
recipe/closure/counter receipt SHA-256:
`1582f7b7a6ba8bc597c689dbd7bdfdb0a34d0069bcd5fcb4ab69844c62f2c7ce`.
The expanded retained archive is `.cache/pr2444-probe-evidence.tar.gz`, SHA-256
`ed872e179a18866f9bc68fd0eb43c0c26632ddfa6a7d9f9601ea983fe71a21a5`.
It includes the preceding observations and receipts, not only the final PASS.

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
