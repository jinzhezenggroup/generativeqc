# Decision: Apply CPU KS HVP panels on prepared fleet items

Status: implemented
Date: 2026-11-24

Use the same live token, exact electronic binding and native component byte
floor for sizing and executing a prepared action. An accepted item's normalized
system already exists; do not recreate it to plan or run another SCF to apply an
HVP. Keep caller-supplied expected coordinates rather than mirroring accepted
geometries in Python: after a failed or moved replay only the native state token
can certify which geometry is available. No new resident response cache or
mutable CPKS state is retained between calls.

For a budgeted prepared fleet, freeze its chosen primal candidates and overlap
the action with the entire admitted fleet's peak. There is no proven release
point that would justify subtracting another item's stored numerical data.
The temporary derivative reservation does not alter the prepared replay plan;
short budgets, stale state and a failed solve publish no action. Unchanged
scope exclusions include export scratch, implicit BLAS workspace and RSS.

Tests compare against independently prepared actions, exercise accepted and
failed geometry replays, and check exact/one-byte-short host composition on a
ragged fleet. SDK queries agree with the original pre-SCF floor. Existing
independent moving-gradient regressions protect the unchanged equations.
