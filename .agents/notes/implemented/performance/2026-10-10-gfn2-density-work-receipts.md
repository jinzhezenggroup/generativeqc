# Decision: count executed GFN2 density work inside the SCC graph

Status: implemented
Date: 2026-10-10

## Problem

The device-launched GFN2 SCC graph hid plain and energy-weighted density
contractions from host-node profiling. Whole-graph occupancy and the absence of
per-node NCU captures could not establish a density-specific cost or a rank-k
provider crossover. The shared #2083 weighted-Gram emitter already owns the
scientific scalar loop; duplicating its formula for a profiler would break the
single-owner contract.

## Decision

Keep the ordinary `emit_gfn2_density_contract()` bytes unchanged (SHA256
`19701164b11a0afb1490daa147237f8022ea9d684d58299ac119fa470b3c2538`).
The GFN2 method adapter optionally inserts integer visit counters at unique,
checked anchors in that same selected checked-pair loop and emits a second
include. `GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=ON` compiles that sibling
into a separate CUDA artifact. The default build retains its original density
workspace ABI and kernel path. The internal cache must enable readback before
preparing a topology; otherwise no receipt arena is attached.

One instrumented CTA record identifies system, spin channel, matrix tile,
orbital count and triangular pair count. Counters advance only at executed
pair/orbital/check/publication statements. In particular, `plain_visits` counts
orbital entries reaching two coefficient and one occupation-weight read
expressions; `weighted_visits` counts energy-weight expressions reached after
the plain check. These are semantic visits, not memory-transaction counts.
`plain_completed` and `weighted_completed` count successful checked updates.
Four dense scratch stores occur per published pair, including duplicate stores
on the diagonal; the receipt retains `published_pairs` rather than pretending
there were only unique memory addresses. Zero occupation still incurs a loop
visit. Finite failure leaves partial counts and suppresses public science for
the whole affected system through the existing publication gate.

Each CTA writes one status, including inactive/invalid-active/unused-channel/prior-error
skips. A single atomic slot counter counts every launched CTA; the host resets
it once before a synchronous endpoint and reads it only after settlement.
`attempted_receipts > receipt_capacity` explicitly reports overflow. Admission
reserves at most 64 MiB, using the selected generated tile schedule, two channel
slots per system and `maximum_iterations + 2` density launches. Rejected or
failed replacement topologies can retain a bounded host snapshot after
settlement, while their device arena is released. Call and plan tokens prevent
an earlier prepared graph from masquerading as the failed call's evidence.

`cta_cycles` brackets one CTA's contraction body with block barriers and is a
same-SM instrumented duration. It is not whole-kernel elapsed time, graph time,
an additive cross-SM quantity, or a complete SCC fraction. Plain and weighted
updates are interleaved within each orbital visit, so this fused CTA duration
cannot be split into independent P and W elapsed times. Graph submission is
reported from `submitted_graphs`, separately from call completion. A zero
receipt count with no submitted graph carries no density-time inference.

## Rejected alternatives

- Do not infer executed work from shape alone: active masks, ragged spin
  layouts, finite breaks and graph submission can all change actual visits.
- Do not add a GFN2-local provider selector or a second weighted-Gram formula.
- Do not synchronize or reconstruct density on the host inside SCC iterations.

## Evidence and acceptance boundary

The source-identity and anchor checks are in
`tests/python/test_gfn2_density_receipt_codegen.py`. The native ragged harness
uses an independent long-double matrix oracle, fractional occupations,
restricted/unrestricted members, inactive peers, late plain/weighted failures,
graph replay and receipt-count checks. The complete endpoint bootstrap checks
graph/replay/reset/failure topology ownership against independent small tblite
fixtures. Record actual device/build/source identities and complete E plus host
force off/on/replay timing separately before claiming runtime qualification.
Neither these receipts nor small fixtures resolve the independent 768-AO
tblite E/F disagreement or close #1879/#560.

## Revisit when

A source-matched, independently validated shared rank-k provider executes in
the production SCC graph and has measured complete-endpoint crossover data.

## References

#1879, #560, #1814, #2083, #2157 and
`docs/maintainer/performance_engineering.md`.
