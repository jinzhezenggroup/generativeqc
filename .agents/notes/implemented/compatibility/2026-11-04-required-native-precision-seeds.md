# Decision: require native prepared precision seed transport

Status: implemented
Date: 2026-11-04

## Problem

Prepared precision already stages warm densities through immutable native
snapshots. Missing capacity/capture symbols still selected duplicate Python
AO/spin sizing and legacy descriptor copying. This retained a second production
transport for pre-release older libraries and obscured the ownership boundary.

## Decision

Require the selected runtime's capacity, capture, restore, read and destroy
protocol before preparing or executing a precision fleet. Bind through the
existing typed selected-library transport, including explicit symbol overrides.
Remove Python topology sizing and legacy descriptor array staging. Snapshot
storage, physical restoration and source-independent lifetime keep their native
owners; no new scientific equations or native ABI are introduced.

An incomplete installation raises RuntimeError, distinct from a supported native
domain refusal. Optional auditor history may still cold-fallback on MemoryError
or NotImplementedError. It must not swallow missing installation support.
Candidate staging refusal remains an audited item failure, not a silent cold
restart. Disabled warm starts require the current protocol but allocate no seeds.

This supersedes only the absent-protocol compatibility in the
[capacity](../architecture/2026-10-29-native-warm-seed-capacity.md) and
[snapshot](../architecture/2026-10-29-native-warm-seed-snapshots.md) decisions.
Those notes' bounded storage, library lifetime and physical validation rationale
remain applicable.

## Invariants and remaining scope

- Preserve independent candidate/auditor histories, frozen source geometry,
  per-item failure isolation and no overlapping device owners.
- Bound all-slot numeric staging before materialization and retain validation
  scratch reservations, combined history limits and optional cold audit.
- Production capture/restore never materializes Python diagnostic arrays.
- Python still owns seed eligibility, model identity, resource reservations and
  initial prepared-fleet orchestration. This is not completion of #2227 R6.
- This does not qualify new method/backend domains or establish full allocator,
  provider/RSS accounting, whole-interface parity or original CBS campaign gates.

## Evidence and revisit conditions

The warm-seed capacity/snapshot suites check independent AO dimensions,
RHF/UHF/RKS/UKS frozen/moved replay, exact/below-limit budgets, source teardown,
invalid receipts and legacy-ABI exclusion. Prepared precision tests exercise
missing selected symbols before allocation/execution, mutated owner bindings,
independent histories and non-swallowed incomplete auditor protocols.
Real CUDA variants still require explicit finite Slurm qualification.

No speedup is claimed from removing an unused older-library branch. Complete
endpoint and semantic work gates remain required for broader #2227 promotion.
Revisit only with a separately authorized supported-version contract, or when a
native controller owns model/history eligibility and resource admission together.
