# Decision: Share modeled CPU DF-KS capacity across prepared owners

Status: implemented
Date: 2026-11-24

## Problem

The native per-owner DF-KS gate authenticates topology and properties, but two
individually feasible handles can exceed a common host allowance. Their contexts
may be independent, so a context's solve mutex cannot serialize their scratch.

## Decision

The optional native host cap leases the existing per-owner retained and
workspace inventory, not a second set of estimates. Under one cap, each live
owner reserves retained capacity and participates in a maximum scratch
reservation. Native preparation and replay take the same workspace lock across
contexts. Ledger state has a separate mutex so snapshot and lease retirement
do not need the workspace or context of another owner.

Preparation reserves before copying modeled system/provider storage, publishes
only after the existing native gates pass, and unwinds the lease on failure.
The plan releases its providers before the lease retires. An optional metadata
handle shares the ledger but does not govern the lifetime of prepared owners.
The original per-owner limit and admitted properties still apply at replay.

## Boundaries

- Retained plus maximum scratch is sound because each shared cap serializes
  its supported preparations and executions. Do not admit another concurrent
  scratch consumer without adding its capacity or sharing the workspace lock.
- The cap covers the same modeled all-electron FP64 LDA/PBE DF-J energy and
  force endpoint as the per-owner gate, not allocator telemetry, checkpoints,
  unrelated owners, response work or process RSS.
- An explicit scope, rather than a process-wide cap, avoids silently counting
  unrelated native contexts. Separate caps are not one resource plan.

## Evidence

The native named KS test combines single-system and fleet reservations, checks
exact/one-byte-short capacity, rolls back a rejected preparation, and replays
after closing metadata. The Python public DF test exercises shared admission
and capacity reuse across independent contexts, including concurrent execution
with unchanged energy, forces, iterations and Fock-build counts.

Related: #2227 (R2), [per-owner CPU DF-KS gate](2026-11-24-native-cpu-df-ks-host-admission.md).
