# Decision: observe CUDA KS value-provider applications

Status: implemented
Date: 2026-10-24

## Problem

The CUDA KS result retained an unavailable J/K value-provider record, while
its precision/Fock ledger described a different contract. Counting SCF
iterations or recurrence work cannot recover actual value-provider calls,
especially for spin/range composition, validation, retries and warm rebases.

## Decision

Record known successful value-provider submissions at their existing call
sites. J applies to the total density; each executed K term applies per spin.
This covers resident, ordinary, occupied, provisional-screening and fused or
fallback range dispatches without changing their operands, matrix bindings,
screening selection or streams. A compatibility refusal before submission
adds no work; the actual fallback applications supply their own records.

The observer checks capture state on the actual dispatch stream, including
an explicit default-stream handle. Capture, failed/unknown queries, failed
submissions and overflow permanently revoke the execution-local record.
Unsupported device chunks and AUTO/mixed conditional routes remain
unavailable. Fixed-density profiling does not append to an accepted solve.

Reset availability before beginning a replacement solve, then initialize an
empty complete record only after the fresh execution state is established.
Both method enqueue and finish catches poison the aggregate and any adapted
diagnostic before retry. A later successful attempt cannot repair unknown
prior work; an independent neighboring item retains its own record.

## Invariants

- These counts are value-provider applications, not ERI work or FLOPs.
- An unavailable count remains unavailable, rather than becoming zero.
- Preserve existing failure classification, last-good warm state and finite
  retry policy. Completeness changes do not change numerical success.
- Existing full precision/operator, source/binary, independent numerical,
  resource and complete endpoint gates remain separate requirements.

## Evidence and qualification

The header-only native unit and source-bound preliminary CUDA scheduler
fixture exercise spin/range composition, large counters, sticky failure,
unknown/overflow attempts, retry isolation, replay and certified zero work.
Actual allocated CUDA tests exercise registered LDA/PBE/PBE0/WB97M-V cases,
replay reset and failed-begin recovery. Host test success alone does not
qualify the current CUDA binary or the full 48/96 endpoint campaign.

References: #2072, #2250; docs/developer/ks_diagnostics.md.
