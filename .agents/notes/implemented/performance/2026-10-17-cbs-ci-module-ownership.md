# Decision: retain complete CBS fixture ownership in CI

Status: implemented
Date: 2026-10-17

## Problem

Adding the independent take-default-axis test changed sorted modulo ownership of
`test_ccsdt_cbs_native.py` from core-c to core-a. The core-a Python step in
[run 38996287111](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38996287111/job/117289480168)
failed after 1813 seconds against the existing 30-minute limit. The retained
artifact contains a passing 1520.473-second HVP and a late unfinished CBS
finite-difference case, with CBS cases distributed across three workers. It has
no JUnit result. Timeout is suspected, not a proven failure cause; this scheduling
repair does not establish that the failed job was numerically clean.

## Decision and invariants

Transfer the complete CBS module to the existing posthf loadfile lane. Count its
original core enumeration slot before excluding it, so this transfer does not
change any unrelated core owner. Select tracked missing files too, preserving
fail-closed behavior, and do not admit absent or untracked future files.

Insert CBS after complete CC and before ECP-heavy in the existing measured
posthf priority list. Preserve the first four expensive owners and the relative
order of every unrelated file. Keep the worker count, test/job limits, event
matrix, scientific cases, numerical thresholds, reference data, and exclusions.
One loadfile worker owns all module-scoped CBS fixtures. No numerical algorithm,
CPU optimization, or CUDA qualification changes are included.

## Evidence and limitations

The exact-head posthf Python step completed in 1112 seconds:
[run 38996287111](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38996287111/job/117289484146).
Eight recent successful posthf steps measured 723, 974, 984, 1107, 1112, 1341,
1373, and 1403 seconds. The 1403-second run retained 418 cases, 361 passed and
57 existing skips, on four loadfile workers. Its complete-file costs include
1196.958 seconds of UCCSD Lambda, 1034.064 seconds of physical response, and
769.938 seconds of corrected owner work.

Completed CBS module blocks measured 308.579, 311.046, and 328.754 seconds.
The 308-second JUnit receipt contains all 25 cases, 9 passed and 16 existing
CPU-large/CUDA qualification skips; its summed setup/call/teardown time is
308.324 seconds. The recent 311-second comparator matches the current test,
CBS wrapper, and checked native/CC/methods/integrals/SCF sources. The older
329-second comparator matches the test, CBS wrapper/reference tree, native
calculation entry point, and CC tree, but differs in other native sources.

Conservatively adding the rounded 329-second historical CBS cost to the slowest
observed 1403-second posthf step gives 1732 seconds, 68 seconds below the routine
limit. This is a historical extrapolation, not a guaranteed future wall time or
a measurement of the modified lane. Required hosted CI must validate the new
ownership. Scheduled/manual qualification retains its existing 90-minute budget;
this change does not claim a new complete nightly timing measurement.

Replaying the actual xdist 3.8.0 loadfile scheduler with the slowest posthf
JUnit durations and CBS conservatively padded to 329 seconds models a
1377.756-second original queue, versus the observed 1403-second step. Including
that 25.244-second residual models 1646.693 seconds with CBS appended and
1525.600 seconds with CBS inserted after complete CC. The chosen insertion
starts CBS at modeled 769.938 seconds instead of 1292.449 seconds, while
retaining the first four owners. These are scheduling estimates only: they do
not model changed contention, runner variability, or establish modified CI success.

Runtime-heavy was rejected: a recent 1557-second step plus CBS would exceed the
routine limit despite its faster 1098-second exact-head run. Scattering CBS cases
was rejected because it can repeat the expensive module fixture. Removing tests,
relaxing thresholds, and extending limits were not considered acceptable repairs.

## Regression coverage and revisit conditions

Executable selector regressions cover all five event types, tracked/missing/
untracked/absent files, and filename insertions at all three modulo offsets.
They require stable dedicated ownership, unchanged unrelated modulo slots, no
new case exclusions, and posthf loadfile. A one-off full-inventory comparison
checks every one of the 1923 test files and preserves all per-shard arguments.
Existing real-xdist regressions protect complete node and module-fixture reuse.

Revisit if actual posthf CI approaches its limit, CBS fixture cost grows, or the
lane's workload changes. Use completed per-file receipts and full-lane wall time;
do not infer headroom from test counts or move the timeout to another lane.

## Additional references

- [Slowest retained posthf run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38991714295/job/117278939729)
- [Older complete CBS comparator](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38812030470/job/116694927833)
- [Recent complete CBS receipt and source](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38974227696/job/117219698495)
- [CBS JUnit reference receipt](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38987002246/job/117262113429)
