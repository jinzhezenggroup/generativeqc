# Decision: enable the complete native Hessian package by default

Status: implemented
Date: 2026-10-16

## Problem

Enabling native Hessians previously selected a small development inventory unless
the builder independently generated and selected the full closure. That was a
useful incremental development boundary, but an enabled package could still
reject otherwise supported ordered orbital-f / auxiliary-g shell products.
The source-identity manifest also omitted the new CMake owner, inventory seed
and response-codegen entry points. Numerical sources were covered, but edits to
those packaging/generation inputs did not invalidate the deployed source identity.

## Decision

An empty `GENERATIVEQC_HESSIAN_AOT_INVENTORY` now means complete factorized
orbital-through-f / auxiliary-through-g closure. CMake materializes the existing
completion recipe at configure time and compiles the existing generated leaves;
there is no new numerical algebra or runtime generator. The inventory is written
under the build directory, not into the source checkout. Unchanged regeneration
preserves the inventory timestamp.

Explicit inventories remain authoritative, including cached selections from
older builds. We do not silently expand a deliberately bounded development
package or overwrite its input. Clear the cache value to restore the full
default. A missing explicit file fails closed. Hessian support itself remains
opt-in, and a missing linked leaf never falls back to JIT, Python or CPU numerics.

The source manifest now includes the Hessian CMake owner, seed, completion and
leaf tools, and every response-codegen entry point registered by that owner.
Compiler and runtime Python modules remain covered by their existing recursive
groups. Scientific program identities still describe their graphs/selectors;
the content source identity describes the build inputs, not just those programs.

## Evidence and invariants

The host CMake tests execute real configure-time selection without CUDA. They
check the full default against the independent explicit completion interface,
idempotent regeneration, preservation of an explicit partial inventory, and
missing-input/build-time-Python rejection. Source-manifest gates protect the
new inputs as actual identity members rather than pinning implementation text
or generated-source hashes.

This changes package selection and provenance, not the numerical method or
acceptance thresholds. Earlier source-matched GPU/timing receipts must not be
relabeled as measurements of this build. The full inventory remains reproducible
from its finite recipe rather than duplicated as a large tracked JSON file.

## Clean-package qualification

Clean source `f8ee2dd66299a5704626a662f6dbe05d7725e3ed` was deployed using
`git archive`. Slurm 3237 completed 52 native AOT tests (80.63 seconds) and
all four actual C++ CLI/public-native-frontend cases (472.89 seconds), with no
skips. The independently regenerated content identity matched the compiled
header exactly: `3f39b33212b3452bcd228139241dbc019e37bc18c6a6f562a27db08a6318666c`.
The complete default inventory matches the earlier explicit recipe byte-for-byte;
its SHA256 is `c349bc9f63d543f5a1a33bd9ca2552a568d6846c9d5f4d83b5222d15a04770c6`.

The preceding Slurm 3235 attempt is not an all-green result: 52 AOT and two CLI
cases passed, but the other two tests refused test-only CPU-oracle cache-miss
compilation because the remote compiler-cache directory was absent from PATH.
The actual native calls completed before those CPU-oracle failures. The retry
restored that PATH and verified sccache 0.16.0, without changing production code,
scientific inputs, or numerical gates. Both attempts remain in local artifacts.

Selected raw CLI results, independent reconverged energy/HVP receipts, full
high-angular matrices, semantic work counts, exact source/binary identities and
reproduction prerequisites are retained in
`benchmarks/results/wb97mv-hessian-default-fg-20261016/`. Earlier wider molecular
measurements inside that envelope explicitly retain their distinct Slurm 3233
production/test revisions and library identity. This is numerical qualification,
not a new timing comparison or an acceptance of the failed GPU4PySCF ratios.

## Revisit when

The supported finite angular domain changes, or a new scheduling strategy is
qualified against the same primitive and complete-molecular endpoint gates.
