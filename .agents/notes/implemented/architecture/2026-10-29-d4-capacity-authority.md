# Decision: native D4 sizing owns global correction capacity

Status: implemented
Date: 2026-10-29

## Problem

After D4 retained allocations joined the shared device ledger, global planning
still rejected D4 corrections. Copying its EEQ workspace formula into Python
would create another authority and confuse the standalone ragged schedule with
KS's retained per-system owners and replicated immutable tables.

## Decision

Extract the incumbent bounded schedule into a native allocation-free capacity
query used by D4Plan preparation. Sum individual plans for the public named-KS
query; use the largest per-item complete capacity times system count for the
existing equal-share local budget. Python presents the native receipt and
combines it with the existing electronic resource inventory. The SDK query
works without a CUDA driver, including with a CPU-only library.

## Boundaries

Only complete named strict-FP64 programs admitted by the existing electronic
planner can select global D4 ownership. CPU plans reserve an explicitly sized
standalone gradient consumer because electronic native force admission remains
per-item. CUDA requires existing native stationary-capacity support and cannot
silently create a second correction owner. Missing additive inventory refuses
global planning while preserving ordinary older-library D4 execution. gCP
composites are excluded; their additional storage is not D4 capacity.

## Rejected alternatives

Do not reuse one standalone ragged capacity for a KS fleet, omit replicated
tables, treat total retained bytes as the equal-share local cap, or accept a
metadata estimate without actual ledger allocation/lifetime accounting. Do not
infer electronic force qualification from a correction-only shape query.

## Invariants and evidence

Keep the same descending worker search and EEQ atom bound. Native receipts are
transactional. Tests compare sizing with actual independent prepared owners,
cover exact/below local and global limits, preserve correction/force identity,
and exercise failed/moved replay, native refusal and missing protocol behavior.
Independent references and complete endpoint qualification remain required;
retained measurements identify their actual source and binaries separately.

## References

This follows the [allocation authority prerequisite](2026-10-29-d4-device-ledger.md)
and advances #2227 R2/R6 only. R3 and parent closure are outside this slice.
