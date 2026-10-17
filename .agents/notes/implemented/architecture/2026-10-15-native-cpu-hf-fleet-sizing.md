# Decision: Native CPU HF fleet capacity aggregation

Status: implemented
Date: 2026-10-15

## Problem

The shared phase owner described in `2026-10-15-shared-cpu-hf-phase-sizing.md`
left Python and CLI with duplicate retained-capacity polynomials and fleet
composition. A prepared fleet retains every item's topology, warm densities
and outputs while sharing the largest serialized phase workspace. Summing
workspace per item or taking only the largest retained item is incorrect.

## Decision

`scf::CpuHfHostFleetResources` validates shell/primitive counts, sums retained
owners, maximizes phase workspace and checks the complete signed-int64 peak.
Append is transactional; homogeneous CLI batches use a repetition count without
building a repeated descriptor array. The additive private fleet query accepts
14 compact counts per item and returns retained/workspace/peak plus per-item
phase diagnostics. Python marshals metadata rather than applying these capacity
polynomials or requesting each phase individually.

The fleet query validates every item before any publication. A second scalar
pass publishes the already validated phase diagnostics. This deliberately uses
two phase evaluations per item with O(1) native scratch and one FFI call per
fleet, rather than allocating an O(items) native result vector or issuing one
FFI call per item. Inputs must remain immutable for the call; overlapping
input/output ranges and byte-length overflow are rejected before reading items.
No scientific tensors, providers, GPU work or heap allocation are needed on
successful queries. This is an ownership/compatibility change, not a measured
endpoint performance claim.

## Compatibility and remaining boundaries

The phase-only ABI retains native phase bounds and portable fleet composition.
No-library and older-library dry runs keep their provider-free portable sizing.
Phase and fleet provenance are separate schedule-identity fields. A discovered
fleet query's failure never falls back to phase-only queries or Python math.

Global candidate/budget composition, admission, leases and allocator receipts
are not migrated here. Numeric/value-metadata bounds still exclude runtime and
allocator overhead; this does not complete #2227 R2 or widen scientific/CUDA
capabilities. Scientific equations and warm-state behavior are unchanged.

## Evidence

Native tests pin H2 retained/serialized fleet capacities, compare homogeneous
repetition with descriptor fleets, cover heterogeneous direct/DF and public/
Cartesian shapes, and test invalid later items, transactional append, malformed
schemas, overlapping buffers, huge byte lengths and exact/overflowing combined
peaks. Python tests compare independent portable ragged bounds across spin,
history and ECP variants, poison phase/portable paths after discovery, reject
silent fallback, and retain phase-only ABI provenance. Native CLI contracts
compare RHF/UHF direct/DF batch results with the same fleet query. Complete CPU
resource tests retain exact/below-budget cold/warm energy+force acceptance.

## Revisit when

Native prepared calculations own global resource admission and leases, or a
qualified metadata publication interface can eliminate the second scalar pass
without O(items) native scratch or weaker transactional output guarantees.

## References

#2227 R2/R5/R6; #933/#934;
`docs/maintainer/resource_planning.md`;
`src/scf/hf_host_resources.hpp`;
`python/generativeqc/resources_hf.py`;
`.agents/notes/implemented/architecture/2026-10-15-shared-cpu-hf-phase-sizing.md`.

## Follow-up: ECP input authority

`2026-10-15-native-ecp-workspace-authority.md` moves the previously Python-sized
ECP input to the existing native scalar query. ECP-bearing CPU HF fleets now
make one additional scalar query per ECP item alongside the single fleet
query; all-electron fleets retain the one-query path. The fleet ABI, two-pass
scalar work and O(1) native scratch described here remain unchanged.
