# Decision: rebalance native contracts after precision/defaults integration

Status: implemented
Date: 2026-10-31

## Evidence

PR #2873 run `41183809495` validates the runtime/ECP transfer: runtime-heavy
passes in 1,717 seconds and ECP forces in 1,314 seconds. However, after the
precision/defaults stack entered the branch, core-a reached its unchanged
30-minute ceiling with all four workers still active. Its diagnostics retain
18,780 passed and 227 skipped call reports, with no failures; the suite is
incomplete and must not be reported as passing.

The shared stationary kernel module accounts for at least 623.5 reported
worker-seconds and still has active cases at cancellation. The native CUDA
binding module adds 291.3 worker-seconds. Core-b's complete Python step takes
975 seconds in the same run, leaving capacity for these parallel contracts.
The earlier core-a choice was appropriate to a different integrated inventory;
new test files can change ordinary modulo ownership even when expensive files
have explicit pins.

## Decision and invariants

Move the two complete native modules from the explicit core-a pins to core-b.
Keep their ordinary enumeration slots, all cases and assertions, the existing
worksteal workers, and every routine/nightly time limit. No independent oracle,
failure case or compiled contract is removed. Existing ownership tests cover
tracked, missing and untracked modules, unrelated core slots and five event
classes; actual xdist scheduling tests remain part of that suite.

This supersedes the owner choice in
[the earlier core-c balance](2026-10-31-core-c-native-contract-balance.md).
It does not undo the independently validated ECP Hessian transfer.

## Revisit when

Verify complete hosted CI for the resulting integrated tree. These measured
worker totals justify a transfer but do not predict exact wall time. If core-b
becomes the critical path, use the new run's complete per-module diagnostics;
do not bounce modules between owners using measurements from an older inventory.

Raw evidence: job `124486531211`, artifact
`benchmark-debug-41183809495-core-a`, `native-diagnostics/worker-*.jsonl`.
