# Decision: bound optional trusted CUDA snapshot restoration

Status: implemented
Date: 2026-10-13

## Problem and measured evidence

Merge-queue job [115238953228](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38380329295/job/115238953228)
spent 10m28s restoring the trusted master cache before its required CUDA build.
Artifact discovery and the blob redirect completed in less than a second. The
combined download, hash, and streaming-extraction interval was
10:09:21.430--10:19:48.089 UTC (626.659s). Artifact 11803181089 contained
3,773,233,465 bytes; its expected and actual SHA256 matched. The subsequent
production compile took 3m57s and the job succeeded at 10:24:16 UTC.

The same immutable artifact restored in 42.916s in overlapping queue job
[115242722322](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38381471126/job/115242722322),
and in 113.628s in job 115237932866. Other nearby successful snapshots took
roughly 117s, 192s, and 324s. These are observations, not a controlled benchmark.
Normal logs cannot distinguish network transfer from streaming extraction or
exclude internal retries, whose messages are debug-only.

The pinned downloader uses five total stream attempts with five-second waits
after errors. Its 30-second inactivity timer refreshes on each data chunk;
there is no total transfer deadline. `continue-on-error` alone therefore did
not prevent an optional restore from consuming the 35-minute job budget.

## Decision

- Limit this merge-queue snapshot download to two minutes. Keep the 35-minute
  job and 30-minute production-compile budgets unchanged.
- Extract into a run/attempt-specific directory under `runner.temp`. Preserve
  successful-master-CI provenance selection and explicitly require
  `digest-mismatch: error` on the existing pinned action.
- Only a successful completed download can activate the directory via
  `CCACHE_DIR`. Selecting the directory avoids another multi-GB copy.
- On failure, timeout, missing snapshot, or failed activation, use the existing
  normal-directory `actions/cache` fallback and still perform every required
  build and probe. Use step `outcome`, not `conclusion`: `continue-on-error`
  changes the latter to success even after a failed step.
- Discard only an attempted, unselected staging directory. The cleanup checks
  the exact runner-temp/run/attempt path, rejects an empty or unexpected path,
  and refuses to delete the selected cache. Skipped/missing downloads do not
  authorize cleanup. The preexisting normal cache is never cleared.

Only the full trusted snapshot consumer in `cuda-compile` needs this change.
The separate master fast-cache delta already has a two-minute limit and
success-only, read-only activation. Publication, compile identities, cache keys,
permissions, and mandatory scientific qualification remain unchanged.

## Tests and tradeoffs

Execute the actual activation and cleanup shell against temporary caches,
including missing, skipped, timed-out, checksum-failed and successful download
outcomes. Verify rejected cleanup paths preserve both caches. Exercise workflow
conditions and negative controls for digest enforcement, deadline, provenance,
isolation, fallback and non-gating behavior. Existing cold-budget and master
handoff qualification remain required.

Two minutes admits the observed 43--117s transfers and bounds long optional
work, but may abandon a slower transfer that would eventually succeed. The
fallback can miss or be slow too; this change does not prove an overall speedup
or guarantee the full compile allowance remains after every possible setup
delay. Revisit the cap using hosted outcomes, without bypassing cache integrity
or relaxing required builds. Download interruption can add runner shutdown
grace beyond the nominal step timeout.

## Primary implementation references

- [Pinned downloader](https://github.com/actions/download-artifact/blob/3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c/src/download-artifact.ts#L201-L251)
- [Matching toolkit stream retry, timeout and extraction](https://github.com/actions/toolkit/blob/21229dc09e551e6da18e72e4e454ee145709c713/packages/artifact/src/internal/download/download-artifact.ts#L46-L173)
- [Runner step timeout versus job cancellation](https://github.com/actions/runner/blob/5f0b3ddcc6ee9fe9441e00d55e85b94ea4aad798/src/Runner.Worker/StepsRunner.cs#L296-L353)
- [Runner continue-on-error outcome handling](https://github.com/actions/runner/blob/5f0b3ddcc6ee9fe9441e00d55e85b94ea4aad798/src/Runner.Worker/ExecutionContext.cs#L1422-L1452)
