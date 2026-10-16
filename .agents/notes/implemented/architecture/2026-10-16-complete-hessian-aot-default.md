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

## Revisit when

The supported finite angular domain changes, or a new scheduling strategy is
qualified against the same primitive and complete-molecular endpoint gates.
