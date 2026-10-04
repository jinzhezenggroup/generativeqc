# Decision: restore matrix Lambda defaults

Status: implemented
Date: 2026-10-04

## Problem and decision

A review follow-up temporarily changed the merged #1829 matrix Lambda schedule
to explicit opt-in. The maintainer requested restoring its original default in
a separate change after #1829 landed. This supersedes the temporary decision in
[the matrix-batch note](../performance/2026-10-04-df-lambda-matrix-batches.md#review-follow-up-explicit-matrix-selection-pending-promotion-qualification),
while retaining that record as historical rationale.

`LambdaOptions`, the internal complete-force declaration, and the benchmark CLI
again select matrix Lambda by default. The batch limit remains eight. Explicit
scalar selection and all existing bounded admission/allocation fallbacks remain
available. The shared-state diagnostic again begins with the matrix default and
then explicitly selects its scalar control; the native complete-force probe
again follows the endpoint defaults.

## Scope and invariants

This restores only default selection and its matching tests/documentation.
Both provider-lifetime mutex repairs, complete admission accounting, resource
fallbacks, generated equations, and error propagation remain unchanged. The
retained one-pair cold timing and numerical evidence keeps its original scope
and thresholds. This policy change adds no new device qualification or
complete-force performance claim.

## Validation

Host tests compile the actual options and endpoint declarations plus CLI
selection, checking matrix defaults, explicit scalar selection, and invalid
selectors. Existing provider-release concurrency regressions remain enabled.
The separate PR runs the repository's normal current-head CI before landing.
