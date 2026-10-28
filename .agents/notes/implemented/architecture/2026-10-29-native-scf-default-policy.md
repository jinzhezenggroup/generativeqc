# Decision: share default SCF controls through the native catalog

Status: implemented
Date: 2026-10-29

## Problem

The application facade selected default SCF tolerances in Python. Its native
capability exposed a version number but not the actual policy, and checking the
five qualified KS models resolved compiler MethodIR objects again. SDK consumers
could not ask the same owner for its default controls.

## Decision

Expose one versioned metadata query over the existing backend/method and KS grid
descriptors. The native method layer compares full electronic identities against the
compiler-generated native catalog and compares the grid against the existing
native default-grid owner. It does not copy model hashes or quadrature tables.
The policy belongs in `src/methods/scf_defaults.hpp` because its grid/catalog
dependencies are method-specific; the reusable runtime layer must not depend on
DFT. No provider, GPU context, grid materialization or scientific solve is involved.

Keep the incumbent narrow policy: all-electron CUDA RHF, or the exact five
qualified restricted KS electronic models with the complete default Treutler
level-3 prescription and fused device XC. Other models, grids, schedules, CPU
and ECP contexts keep strict defaults. These are defaults only; explicit user
controls and physical final-state checks remain authoritative and unchanged.
Python copies metadata and validates the returned schema. Only a missing symbol
uses the old policy. Native rejection never retries Python selection.

## Retained boundaries

This does not qualify new methods or move initial guesses, precision execution,
gradient qualification, observable audits or independent seed histories. Custom
MethodIR is still compiler-owned. General frontend/SDK/CLI model resolution and
complete precision lifecycle remain separate #2227 work; R3 is excluded.

The bounded Python grid-descriptor cache retains immutable host metadata only.
It owns the borrowed dense radius array and never retains molecular geometry,
SCF state or device memory. Unknown identities stay strict rather than being
inferred from a familiar method name or ingredient.

Each Calculator retains at most one copied default answer keyed by the selected
query and every consumed model/backend/ECP/grid fact. Precision validation asks
the same question repeatedly; rebuilding ctypes descriptors adds host allocation
work to very short endpoints. Replacing any input invalidates the answer, and
callers receive detached controls. No geometry, density or device lifetime enters
this metadata cache. Complete endpoint qualification remains required.

The transport also shares at most eight immutable query bindings by native
library handle. Each entry retains its library, so a live cache key cannot refer
to an unloaded/reused handle. Direct CFUNCTYPE binding avoids adding a CDLL ↔
function reference cycle for every short-lived audit Calculator. Explicitly
supplied bindings still take precedence; missing symbols alone select compatibility.

## Validation

Public/native contracts and actual linked Python queries cover every incumbent
model, restricted versus unrestricted methods, missing/unknown protocol, exact
grid and radius neighbours, changed schedules, custom identities, malformed
receipts and transactional failures. Explicit caller overrides remain covered by
the existing constructor tests. Final endpoint receipts accompany qualification.
