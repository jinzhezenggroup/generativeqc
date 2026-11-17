# Decision: separate AOT packaging, preference, evidence scope and force policy

Status: implemented
Date: 2026-11-16

## Problem

The SM120 profile bundled several independent decisions: three optional Fock
families, exact derivative packaging, Rys-task preference, an exact-tuned class
override and native force scheduling inferred from a portable label. This made
it difficult to test a single optimization on another target without inheriting
unrelated compilation cost or silently losing its preference.

The bounded-force rejection also tested `tuned && compatible`. Normal profile
resolution makes these mutually exclusive: exact tuned matches are not
compatible matches, and compatible matches are not tuned. The old comment
described a strict ownership contract that was not actually enforced by any
stock resolved profile. Enabling it during a structural refactor would change
the accepted endpoint behavior rather than merely clarify it.

## Decision

- `AotPackaging` owns an explicit subset of `rys_value`, `k_block` and
  `rys_task`, plus independent exact derivative packaging. Each family owner
  checks only its own inclusion. Missing packaging retains the old all-family
  default; derivative packaging still requires explicit opt-in.
- Rys-task default preference is explicit and separate from compilation. The
  shared nine-class `bounded_spd` set has one authority, used by the shared
  candidate policy and referenced by the SM120 manifest. An explicit class
  list or class-set reference can declare target scope, rather than depending
  on the meaning of `tuned`. Scope does not confer capability or qualification.
- `ForceExecutionPolicy` owns small-work angular-queue selection and optional
  strict bounded s/p/d force ownership. Generated registries store it per
  device target. The existing `ProfileInfo` layout and public C profile ABI
  are unchanged; no-match and AOT-disabled registries use explicit generic
  policy. Energy-only requests cannot fail a force ownership check.
- Stock policies retain the effective defaults: SM120 includes all optional
  families and derivatives, enables its scoped Rys-task preference and disables
  the angular crossover. Portable includes no optional families/derivatives,
  disables task preference and enables the materialized small-work crossover.
  Strict bounded ownership is explicitly false for every stock profile.
- Live manifests reject the obsolete top-level packaging flags. The build
  sidecar and live capability audit move to schema version 2. Repository
  consumers migrate directly; no live compatibility aliases remain.

## Qualification and integrity boundary

This change neither expands weighted-Z capability nor merges the independent
weighted-Z qualification work. No recurrence, scientific arithmetic, scheduling
row or resource gate is promoted. A portable label is not an algorithmic veto,
but explicit compilation/preference is also not scientific or endpoint evidence.

The sealed historical default snapshot and manifest retain their bytes and
integrity hashes. After verifying those hashes, the default gate decodes their
old all-or-none packaging into the equivalent family set and reconstructs the
historical force policy from that archived manifest. It does not re-resolve old
inputs with the new compiler or rewrite the baseline. Changes to any effective
family, preference or force policy still require a source-matched promotion.

The regenerated live CUDA capability audit correctly marks old compile and
endpoint receipts historical after source identity changes; the retained raw
evidence is not deleted or relabeled current. The Libxc live catalog refresh
changes only current source identities and revalidation input lists, not
scientific statuses, executed identities, receipts or accepted product counts.

## 2026-11-17 upstream integration

The follow-up integrates master through `50482bc35`, including native direct-J/K
topology policy and the offline Libxc acceptance boundary. The conflicting Libxc
snapshot is retained from upstream, rather than refreshing historical scientific
acceptance or resetting its previous-summary baseline. Consequently the original
live-catalog freshness result above belongs to the original source revision, not
this integration. `check_libxc_native_catalog.py --allow-stale-generated-views`
passes with an explicit historical-presentation advisory; it reports zero current
qualified products and does not confer current scientific acceptance. The normal
strict presentation check still rejects the stale generated view. The live CUDA
audit is regenerated against the integrated source; the sealed default baseline
and failed scientific observations remain unchanged.

## Rejected alternatives

- Enabling every optional family on portable would mix a compilation-cost and
  runtime-default promotion into a structural refactor without endpoint evidence.
- Keeping exact-tuned or portable checks alongside explicit preference would
  retain two conflicting authorities for the same decision.
- Copying the nine-class list into each profile would duplicate algorithm policy.
- Enabling strict bounded ownership by default would silently activate a
  previously unreachable rejection. That requires a separate behavioral change
  and endpoint acceptance, including fallback coverage failures.
- Rewriting the frozen baseline would weaken promotion integrity. A narrow
  historical-format decoder preserves its original semantic content instead.

## Evidence

Validation artifacts are local under
`/data/jzzeng/sm120-decoupling-validation-20261116/`, based on master source
`da9b2871a149133e34a7c97246b24833d7c59255`:

- Before/after production bundles compare stock SM80, SM90 and SM120 auto
  selection plus forced-portable SM120. Generated scientific CUDA inputs are
  byte-identical; only registry policy serialization and diagnostic sidecars
  change. This is one-time migration evidence, not a permanent source hash test.
- Executable generated-registry host probes test independent force policies
  under relabeled profile metadata, exact matches, single-device visibility,
  device switches and unmatched targets. Multi-device assertions are confined
  to the mock configuration so real single-GPU CTest jobs remain valid. CUDA
  discovery is mocked; these are not GPU performance tests.
- Host-compiled native force-policy probes exercise energy/force request scope,
  explicit diagnostic exemptions, missing ownership and work/capacity crossover
  boundaries, including large unsigned masks and work counts.
- Python tests cover independent family subsets, derivative opt-in and global
  opt-out, portable task preference, scoped evidence validation, missing-capability
  rejection and promotion-gate negative controls.

No new NVCC, real-GPU, sanitizer or complete-endpoint performance qualification
is claimed by these host and compiler-generation checks.

Recorded validation: the 28-module regression run passed 898 tests with two
optional tests skipped; documentation helpers passed 94 tests and 106 subtests,
and Sphinx HTML built with warnings as errors. The final single/multi-device
registry module passed 42 tests, and final audit-format/evidence modules passed
117 tests. Compiler ownership, default-promotion inventory, CUDA AOT policy,
Libxc live-catalog consistency, Ruff and changed-line C++ formatting passed.
The local CMake probe used the existing Conda OpenSSL/LibArchive installation
via `CMAKE_PREFIX_PATH`, without changing project dependency detection.

Representative reproduction commands:

```bash
PYTHONPATH=python:. pytest -q tests/python/test_aot_profile_selection.py \
  tests/python/test_aot_policy_dimensions.py \
  tests/python/test_force_coverage_request_scope.py \
  tests/python/test_cuda_aot_global_promotion.py
python tools/check_compiler_structure.py
python tools/check_default_promotion_inventory.py
python tools/check_cuda_aot_policy.py
python tools/check_libxc_native_catalog.py
```

## Revisit when

Source-identified endpoint evidence supports changing a stock family set,
default preference or force policy. Strict bounded ownership needs independent
failure/fallback qualification before a default change. Cross-target performance
evidence may widen explicit scopes without copying algorithmic policy.

## References

- Supersedes the label-based portions of
  [exact-profile overrides](2026-10-10-rys-task-profile-overrides.md).
- Current contracts: `docs/developer/shell_codegen.md` and
  `docs/developer/direct_rys_tasks.md`.
- Frozen-default gate: `tools/generativeqc_validation/cuda_aot_policy.py`.
