# Decision: expose the existing post-HF CUDA DF dependency

Status: implemented
Date: 2026-10-12

## Problem

Before the prepared-batch capability extraction, `posthf/bridge.cpp` obtained
`scf/cuda_density_fitting.hpp` through `api/handles.hpp` and
`methods/method.hpp`. The bridge already owned `PostHfRhfJkPlan`, called the
CUDA DF source/plan factories and destructors, executed RHF J/K response, and
read metric diagnostics and source counters. Those calls are present in base
commit `fd01d1fa9748be058341d7794b49337480116eb5`.

Removing the backend diagnostic header from the method-neutral base exposed
that dependency: the non-PCH CPU build lost required function declarations and
the complete metric-diagnostic type. Adding the direct consumer include fixed
compilation but also made the previously hidden edge visible to the boundary
audit.

## Decision and limits

Keep the direct include at the existing bridge and inventory precisely
`posthf/bridge.cpp -> scf/cuda_density_fitting.hpp` as known architectural debt.
The production bridge change is solely that include; no new provider call,
ownership transfer, scientific algorithm, or numerical tolerance is introduced.
This does not approve new post-HF consumers or additional SCF dependencies.

The exact-edge regression checks both limits: another SCF header in this bridge
and the same header in another post-HF consumer still fail. The report continues
to label the retained edge as `known_debt`, and removing it remains allowed.

## Rejected alternatives and revisit condition

Reintroducing the include in `method.hpp`, inserting it through another accepted
header, or forward-declaring implementation APIs would hide the coupling or undo
the capability boundary. No neutral shared header currently owns these plan
factories, execution functions, counters, and diagnostics. Moving those contracts
and their consumers to a shared provider owner is a separate architectural
migration, not a batch-facet compile repair. Remove this debt entry when that
migration eliminates the bridge's direct use of the SCF-owned API.

## Evidence

- The retained provider-reuse rationale is recorded in
  `../performance/2026-09-20-posthf-generated-df-device-handoff.md`; exposing this
  declaration dependency does not introduce a new integral or response owner
- PR #2381: the base bridge implementation and its retained CUDA DF plan calls
- Commit `4aca6538529769691f1f94a5d4d70379b1ac293e`: direct include only;
  complete non-PCH CPU build, prepared lifecycle, and capability ABI tests pass
- `tests/python/test_electronic_structure_boundaries.py`: exact consumer/header
  ceiling, new-edge rejection, and current inventory checks

Agent: dot
