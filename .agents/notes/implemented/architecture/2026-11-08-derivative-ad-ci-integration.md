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

The intermediate integration kept the original 126184-byte `source.patch.xz`
byte-recoverable from commit
`4f555d074700fdb0d340de90f57a9f8d7e4f58cf` and replaced its redundant checkout
storage with `source-snapshot.json`, using the established Git snapshot schema
and shared offline restorer. That intermediate publication authenticated the
capsule; replay also authenticated compressed patch bytes and the measured
source hash. This storage choice is superseded by the self-contained lossless
compression below. The measured baseline, original source hash, samples,
first-use observations, negative results and all numerical/work gates are kept.

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
- Before the separate erf finite-range repair, a one-off comparison against the
  published implementation finds all 62 native generated headers and compiled
  Python response schedules byte-identical after type/import integration.
- The atomic-grid namespace repair produces the same Release host object before
  and after qualification (`-O3 -DNDEBUG`, GCC, verified sccache), SHA-256
  `6ac64eda0e53d7776878bbc5b4c717a959e23521eef0522e8b784aa752f0cf51`.
  These comparisons are migration evidence, not permanent source-fingerprint CI.

Local receipts live under `.artifacts/classical-derivatives/pr-final-*`,
`typing-outputs-{before,after}.json` and `grid-namespace-{before,after}/`.
The frozen performance and scientific campaign remains in
`benchmarks/results/derivative-ad-20261108/`; these CI repairs do not claim a new
timing campaign for the merged master tree.

## Superseding integration

The concurrent PR repair is merged without discarding either branch's work.
Its [review integration note](2026-11-07-derivative-ad-review-integration.md)
records an explicit numerical change: differentiate the erf primal before
contracting its slope with the direction, avoiding intermediate overflow for
large finite seeds. Its independent high-precision oracle is retained. The
earlier byte-preservation comparison does not cover this deliberate generated
mathematics change, and the frozen endpoint timings are not relabelled as a new
post-repair GPU campaign.

Self-contained lossless XZ compression of the small JSON records and denser
compression of the unchanged sample JSON fit the same aggregate budget while
retaining the original `source.patch.xz` directly. This supersedes the capsule
above: remove `source-snapshot.json` and its replay/restorer dependency. The
decoded input, provenance, qualification, summary and sample records remain
identical; only the evidence attachment's compressed-sample checksum changes.
Replay verifies the publication's compressed files and the decompressed
measured-source hash. No Git-object recovery is required for this bundle.

Keep the canonical nuclear array executor and generator cleanup described above.
Merge the bare isolated-generator and host-generated-header fixture tests, and
consolidate duplicate CI insertion tests while checking all shard owners.

The follow-up CI run also exposed the standalone AO jet oracle's missing
generated-axis include and the bulk Libxc catalog's stale scalar-owner digest.
The AO probe now includes the actual production generated header and compiles
through the shared verified cache fixture; the independent Leibniz reference is
unchanged. Rebuilding the complete pinned bulk catalog reproduces both catalog
and report bytes exactly (544 registrations, 222 imported, 322 blocked), allowing
its scalar-owner provenance to be refreshed without weakening offline freshness
checks or mutation rejection.

The final erf/integral scalar probe was freshly compiled and executed on n1
through Slurm job 16768 with assigned device visibility unchanged: 26 host/CUDA
tests passed without skips, including the high-precision extreme-seed oracle.
The earlier job 16767 skipped because the optional host-probe cache executable
was not selected; it is not counted as GPU coverage. Selecting verified sccache
explicitly fixes that validation environment, without bypassing the scheduler.

Final integration validation passes 975 routing/AO cases, 150 scalar-owner,
source-registry, Libxc and publication cases, and 113 focused host cases with
13 explicitly opt-in CUDA skips. The separately scheduled 26-case scalar run
covers those host/CUDA schedules without skips. Workflow-pinned type checking,
all three ownership inventories and the 74-endpoint historical replay pass.

A separate complete Cs ECP unit-x HVP diagnostic passed on both frozen CPU
libraries with matched Python trees and one-thread BLAS: baseline 847.86 s,
candidate 849.71 s. This single pair (ratio about 1.00218) is diagnostic, not a
statistical performance claim or post-erf qualification. Both jobs completed
before the concurrent repair was integrated.

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
