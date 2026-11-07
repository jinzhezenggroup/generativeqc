# Decision: preserve measured derivative AD while repairing CI integration

Status: implemented
Date: 2026-11-08

## Problem

The derivative retirement qualification was frozen against master
`97c34eeb0eb1a0fdabfb24b5436f667149701f32` and published in commit
`4f555d074700fdb0d340de90f57a9f8d7e4f58cf` (PR #3044). The first CI run found
type-checking errors, bare-build import dependencies, an atomic-grid namespace
collision, stale Libxc identity metadata and an aggregate evidence storage
overflow when combined with newer master. These are integration defects, not
permission to relabel the original performance samples or weaken qualification.

## Decision

- Admit existing numeric payload types explicitly after IR operation admission;
  type generated execution namespaces and fixed-size tuples accurately. These
  changes must preserve emitted native mathematics and compiled Python schedules.
- Make `geometry` a canonical lazy-export package, as already done for other
  compiler domains. Remove the obsolete fake-package bootstrap from the GFN2
  pair and CPU XC generators. Build-time emission does not need NumPy or the
  runtime, and generators cannot replace already imported canonical packages.
- Move the unchanged nuclear Hessian-vector array executor to
  `method.nuclear_repulsion_array`; leave the primal and source generator in
  `method.nuclear_repulsion`. Migrate every repository caller directly, with no
  old-location alias. Array execution still differentiates the shared primal.
- Fully qualify generated atomic-grid calls so a translation unit including
  `dft::generated` recipe metadata cannot shadow the independent generated grid
  namespace. This changes name lookup, not the selected generated functions.
- Integrate master `d52d2ae2a8093023bde15292a108c3cc5f0d2cfc` without rebasing or
  rewriting the already published carrier commit.
- Give the nine new AD test modules supplemental CI slots. Inserting them into
  the ordinary modulo enumeration moved 1272 existing module owners, including
  the Cs ECP Hessian from core-a to core-c. The original core-c run timed out at
  83% even though both Cs cases had already passed (889.0 and 1157.1 worker
  seconds). Restore every one of the 2309 existing selected module owners and
  spread the new suites across the same three core lanes, retaining every test,
  priority rule, worker count, cache boundary and finite time limit.
- Delete the unused `build_s_integrals` compatibility alias and the
  `build_cartesian_integrals` alias, migrating its sole native-test caller to
  `build_integrals`. These internal pre-release wrappers own no mathematics,
  budget admission or ABI; retaining them would only preserve obsolete names.
  Keep the actively used budgeted DF contraction overload and its resource
  checks: it is not a dead compatibility shim.

## Scientific and storage identity

The Libxc catalog was already stale on master: neither the retained nor live
summary admits a current native endpoint. Refresh its exact live snapshot and
acknowledge the 222 identity-only transitions from master's retained summary.
All functional inventories, qualified/ready/blocked stage sets and public method
selection remain identical. Historical campaign identities remain historical;
the refresh does not promote, demote or freshly qualify any native endpoint.

The original 126184-byte `source.patch.xz` remains byte-recoverable from commit
`4f555d074700fdb0d340de90f57a9f8d7e4f58cf`. Replace only its redundant checkout
storage with `source-snapshot.json`, using the established Git snapshot schema
and shared offline restorer. The publication authenticates that capsule; replay
also authenticates both compressed patch bytes and the decompressed measured
source hash. Preserve the measured baseline, original source hash, samples,
first-use observations, negative results and all numerical/work gates.

The carrier revision is not a new measured producer revision. A shallow clone
missing the carrier must explicitly obtain its Git object; replay fails closed
and never fetches or publishes an external archive. No storage limit increases,
release assets, new hosts or scientific sample deletions are required.

## Evidence

- Workflow-pinned `ty==0.0.82` passes the complete compiler check.
- Bare generator, canonical-package, classical derivative, nuclear HVP and recipe
  coexistence tests: 158 passed, 4 opt-in skips.
- Focused scalar/AO/density/nonlocal/quadrature/Becke/COSX and SCF tests:
  716 passed, 13 opt-in skips. Libxc catalog tests: 63 passed.
- Evidence retention, shared publication and authenticated offline recovery
  tests: 187 passed. The full 74-endpoint replay retains a maximum scaled
  numerical error of 0.033858515990914384.
- Executable insertion/routing tests across all five CI events and
  absent/untracked/tracked/missing modules: 100 passed. The exact real-inventory
  before/after census is retained in `pr-shard-ownership-{before,after}.json`.
- The full shard-ownership suite passes all 955 cases. Canonical CPU XC/import
  and semilocal emission tests pass 70 cases. Bare CPU XC emission is identical
  before/after bootstrap removal, SHA-256
  `c5261f78dae8a84f4ac53ddab70c8dba6ea76c7b82026998b4749ccfbcc7dfd4`.
  The migrated native Cartesian test compiles/links through verified sccache
  against the qualified library and validates all eight s/p AOs against its
  independent PySCF/libcint reference values.
- Ownership checks pass for 601 compiler modules, 266 shared SCF modules and
  431 CUDA inventory entries.
- A one-off comparison against the published implementation finds all 62 native
  generated headers and compiled Python response schedules byte-identical.
- The atomic-grid namespace repair produces the same Release host object before
  and after qualification (`-O3 -DNDEBUG`, GCC, verified sccache), SHA-256
  `6ac64eda0e53d7776878bbc5b4c717a959e23521eef0522e8b784aa752f0cf51`.
  These comparisons are migration evidence, not permanent source-fingerprint CI.

Local receipts live under `.artifacts/classical-derivatives/pr-final-*`,
`typing-outputs-{before,after}.json` and `grid-namespace-{before,after}/`.
The frozen performance and scientific campaign remains in
`benchmarks/results/derivative-ad-20261108/`; these CI repairs do not claim a new
timing campaign for the merged master tree.

## Rejected alternatives

Adding NumPy to bare CMake generation, keeping fake packages or compatibility
aliases, moving handwritten slopes into templates, silently marking stale
Libxc receipts current, dropping endpoint samples, increasing the storage cap,
or regenerating the measured source patch from the repaired tree would each
hide a different contract violation rather than repair it.

## References

- [Derivative retirement closure](2026-11-08-derivative-retirement-closure.md).
- `benchmarks/results/derivative-ad-20261108/README.md` documents source recovery
  and replay without an external archive or GPU.
