# Decision: prepared native owners bind their own device admission

Status: implemented
Date: 2026-10-17

## Problem

Selected-inventory admission already used the native sparse lifetime sweep and
actual native buffer/provider owners, but Python's observation scope still
controlled ledger binding. An installed C++ SDK or CLI could not independently
hold that same admission contract through preparation and execution. Merely
removing the observer would also lose its serialized host-workspace policy.

## Decision

Share the existing inventory admission factory between the private metadata ABI
and a typed public resource handle. Single calculations and batches retain that
shared ledger before the scientific plan member, so plan/provider/buffer
destruction releases final charges before the ledger owner. Native RAII scopes
bind preparation and execution; metadata handles may close independently.

Same-owner nesting borrows counters; foreign or concurrent bindings refuse.
Optional frontend observation is not authority. Native scope depth enforces the
same serialized workspace even after an optional observer ends. An otherwise
unobserved operation owns an isolated native sampling scope and saves the samples
in its diagnostic. Existing outer observations are borrowed without claiming an
isolated native peak. Missing samples and requested-byte overflow stay unavailable.

The Python adapter selects the public path once. Only a missing preparation
symbol permits old-library compatibility; rejection, partial ABI or malformed
publication cannot trigger private creation or unbudgeted preparation. CLI
`run --owned-device-bytes` exposes only numeric/provider-reservation admission,
not the misleading promise of a global device/RSS cap.

## Rejected alternatives

- Keep binding in Python: does not satisfy installed SDK/CLI runtime ownership.
- Copy inventory arithmetic into another planner: creates competing authority.
- Reuse frontend observers as a mandatory native scope: retains the original
  dependency and can disable serialized workspace when optional observation ends.
- Name the CLI option a total device budget: runtime/library internals, physical
  coverage and global/model validation are not supplied by this ledger.

## Invariants

No scientific equations or capability bits change. Retained numeric charges and
provider reservations remain distinct, survive warm calls and metadata closure,
and follow actual buffer release, including cross-thread destruction. Failed
admission is not an alternative backend or an excuse to omit a budget.

## Evidence

Behavioral coverage lives in `tests/native/test_resource_ledger_plan.cpp`,
`tests/native/test_device_resources.cpp`,
`tests/python/test_native_prepared_resources.py`, and the installed-CLI controls
in `tests/native/test_cli_runtime.py`. Host metadata and synthetic observation
tests do not establish real-GPU numerical or endpoint performance acceptance;
source-matched scheduled CUDA qualification remains a separate gate.

## Consequences and revisit conditions

This removes a frontend lifetime authority without pretending to finish #2227
R2. Global selection/model validation, provider-internal physical coverage and
full allocation/RSS authority remain separate work. Extend the public contract
when those native authorities exist, preserving explicit old-library behavior
and avoiding a second planning engine. R3 response/HVP remains outside this slice.
