# Decision: scope stock weighted-Z packaging without restricting its algorithm

Status: implemented locally
Date: 2026-11-17

## Problem

PR #3105 correctly removed SM90-only algorithm admission, but the portable
manifest then packaged weighted-Z on additional default targets. The production
AOT checker rejected that compiled-mask extension. Runtime-default-off does not
eliminate module-size and first-use costs, and fixture execution is not a fresh
complete-endpoint promotion campaign.

## Decision

Resolve optional per-row `capability_architectures` scopes while loading a
production manifest. Scope the two stock portable weighted-Z companions to
SM90, preserving the original effective compiled masks. The reusable companion
generator and IR retain architecture-independent legality. Custom manifests can
explicitly declare other targets; the unchanged promotion gate still rejects
unqualified changes to stock packaging.

Scopes may only name declared shared-Rys companion capabilities and must contain
non-empty lists of valid architecture strings. These three companions have
independent compiled masks in the unchanged promotion gate. Do not silently
extend scoping to a wrapper whose packaging is not separately inventoried.
Unscoped declared capabilities retain their prior
behavior. Native runtime opt-in, coefficient guards, precision, resource limits,
fallbacks and numerical acceptance thresholds do not change.

Keep the original algorithm declarations separate from resolved packaging
exclusions. Removing a declared capability would also change the sealed
selection inventory even when the original compiler had omitted its companion.
Emitters consume effective capabilities; the promotion gate still observes the
actual companion masks without rewriting its historical inputs.

## Rejected alternatives

- Restoring a hard-coded SM90 test would reintroduce the portability defect.
- Resealing the production baseline or weakening the checker would authorize an
  unmeasured default change rather than resolve it.
- Claiming the existing SM120 fixture as complete endpoint qualification would
  conflate plan/scientific-fixture evidence with packaging promotion.

## Evidence

The executable manifest tests exercise preserved stock availability, explicit
cross-target declaration, malformed-scope refusal and the unchanged fixed thread
mapping. The default-off negative control expands the candidate manifest rather
than mutating its retained baseline. The production checker must pass against
the original sealed baseline; synthetic unit fixtures are not qualification.

The upstream Libxc snapshot is retained as historical evidence when integrating
master. Source drift is not repaired by rewriting its scientific receipts.

Final focused validation: 192 tests pass across manifest scoping, weighted-Z
plans and host dispatch, shared-XY integration, default-promotion controls and
production-profile contracts. The real `tools/check_cuda_aot_policy.py` accepts
the original sealed baseline with no changed targets: SM75/80/86/89/90/100/120.
This is preservation of effective policy, not a new scientific promotion.
The compiler structure check covers 605 modules with zero dependency errors.

The broader 12-module related suite passes 307 tests with three explicit skips
before the final scope-normalization/retargeting hardening; the final 192-test
run includes that hardening. An earlier broader run failed its CMake dependency
probe because this shell did not expose the installed OpenSSL/LibArchive
development prefix. Setting `OPENSSL_ROOT_DIR` and `CMAKE_PREFIX_PATH` to the
existing `/home/jzzeng/miniconda3` prefix fixes that environmental prerequisite;
no project dependency check is bypassed and no system package is modified.

## References

This supersedes the unresolved stock-packaging disposition in
[the original portability note](2026-11-16-weighted-z-capability-not-sm-whitelist.md),
not its historical device evidence or limitations.
