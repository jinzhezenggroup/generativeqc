# Decision: compose resident nonlocal force producers and consumers

Status: implemented in the candidate branch; GPU qualification pending
Date: 2026-09-27

## Problem

The full-grid producer in #1504 and device-seed consumer in #1497 did not yet join in the public stationary-force driver. Python still downloaded density features, compacted an active set, downloaded pair adjoints, and uploaded each seed tile.

## Decision

Join the existing owners through ingredient-driven device feature leases. Keep the full-grid producer alive until the geometry consumer drains. Reuse it only after a completed generation. Preserve the old compacted CUDA route for unavailable producer/consumer ABIs and explicit capacity failures, and report the selected route. CUDA, numerical and identity failures must not silently fall back.

The first slice keeps two bounded AO traversals. It does not pretend that retaining rho/gradient also retains all AO jets. A later memory-accounted lease-retention slice can remove the second collocation where storage permits.

## Invariants

No scientific kernel, threshold, exact pair ordering, force source, or final-state validation is replaced. Zeroed inactive device seed rows permit the original quadrature weights without downloading an active mask. Same-stream order and the final drain protect borrowed inputs. The nonlocal owner is destroyed before the grid.

## Evidence

The source-only sequencing suite has 8 passing tests in this session. It tests the real adapter with explicit fake device owners, not GPU arithmetic. Native CUDA compilation, independent molecular force comparison and complete endpoint timings have not been run: n3 is offline. Pair enqueue time is labeled as enqueue time; its completion belongs to the following geometry/drain boundary. Unknown active/pair counts remain null, with a separate upper bound.

## References

#1423, #1479, #1482, #1494, #1497, #1504. Agent: ChatGPT. Model: GPT-6 Astra Pro.
