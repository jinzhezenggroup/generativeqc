# Decision: bind retained qualification to complete measured populations

Status: implemented
Date: 2026-11-08

## Boundary

PR #3048 integrates master `c2bf1061a` without changing the independently
measured GPU populations. The generated Libxc catalog remains stale with zero
current-source admissions. The existing HF force-stationarity correction and
selected-device ownership repair remain intact.

## Retained-evidence checks

Require the recorded GPU, workload, default controls and native image for every
DFT ABBA and Direct48 endpoint. Both historical and current-master cohorts use
the same recorded defaults. Reject interposers, incomplete raw-K/sanitizer
populations, nonfinite error summaries and any nonzero sanitizer summary.
Source-bound HF force replays validate their original positive integer work
counts, rather than imposing those counts on future implementations.

Strict HF48 auxiliary records carry native image metadata. The older
moved/restored HF96 schema does not. Retain that limitation explicitly: its
supporting provenance is the original authenticated observation, producer,
worker, image receipt and same-job chain. Exact hashes here protect immutable
historical evidence inputs, not the spelling of current production code.
Reject substituted geometry, solver settings, image or worker records rather
than inventing missing per-record image metadata.

The synthetic raw-exchange oracle contracts each unique AO ERI orbit once,
including repeated-shell components. Retained executed fixture bytes are not
regenerated or relabelled.

## Invariants

No numerical gate, evidence budget, failed observation, independent reference
or production fallback is relaxed. No new native-GPU qualification or
performance result follows from these reader and integration repairs.
Executable corruption tests cover the new rejection paths; replay retains the
historical default HF48 rejection separately from the later force repair.
