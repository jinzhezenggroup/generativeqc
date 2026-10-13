# Decision: Bound new CLI work on an already-red CI attempt

Status: implemented
Date: 2026-10-12

## Problem and measured observation

The native CLI job starts after CUDA compilation, independently of CPU, Python,
and the other required CUDA qualification jobs. This preserves green-run
parallelism, but it can start substantial new compilation after a sibling has
already made the required aggregate fail.

In PR #2383's old run `38317461329`, Python core-c job `115037432962`
failed at 22:57:28 UTC. CLI job `115043994991` had started at 22:57:05;
its fast CUDA seed began at 22:58:48. Thus a check immediately before the seed
could have observed the failure without waiting for other jobs.

The repaired run `38320127161` was still in the CLI at the 23:15 UTC
observation: 11m22s of CLI elapsed time, of which only 9m14s was exclusively
on the workflow tail after the last other job ended at 23:05:49. These are
observations of existing runs, not a measured saving from this guard. The
repaired green run must still perform full CLI qualification. No running job
was cancelled or benchmark/GPU experiment performed for this change.

## Decision

Add a single read-only guard immediately before the fast CUDA seed, only for
selected PR and merge-group CLI builds. It makes one request to the exact run
attempt's jobs endpoint, with a five-second supervised-process deadline, no polling,
no redirects, a fixed GitHub API origin, a bounded response, and the existing
step-scoped read token. No permission or credential is added. A process-level
timeout also bounds native DNS resolution, which a Python signal handler or
socket timeout alone cannot reliably interrupt. A timed-out worker is killed
and reaped without touching another job, build or shared compiler process.

Validate the entire response before deciding: complete single-page count below
100, unique job IDs/names, run ID, attempt, repository/run URL, workflow name,
original event source head and recognized job states. PR job heads use the
immutable event's `pull_request.head.sha`, not synthetic `GITHUB_SHA` or the
current live PR. Merge groups use the event's `merge_group.head_sha`.

Only the required sibling inventory from `Pass CI checks` is authoritative:
two CPU legs, three CUDA jobs, eight Python shards and merge-queue liveness.
Completed `failure`, `timed_out`, `cancelled`, `action_required` or
`startup_failure` results stop new CLI work. Optional performance, coverage,
cache and reporting failures do not. Active jobs do not cause a wait.

A verified failure returns **nonzero**, retaining a failed, unqualified CLI
job. Rerunning only the previously failing Python job cannot qualify the CLI
or make the aggregate green. The CLI itself must subsequently run successfully.
Exact-attempt lookup prevents a stale failure from an earlier attempt from
permanently blocking reruns. Missing siblings, including reused successes
absent from a partial rerun, cause normal full CLI qualification.

Any API error, timeout, redirect, pagination uncertainty, schema/identity
mismatch or unknown state warns and runs the existing qualification. This is
fail-open for doing work, never a shortcut to a passing qualification result.

## Preserved invariants

- The dependency graph, impact classifier, selected build tiers, every existing
  qualification step, and required aggregate are unchanged
- Master, tag, schedule and manual behavior and trusted cache producers are
  unchanged; no partial PR cache is promoted
- There is no `continue-on-error`, allowed failure or passing/skipped shortcut
  for a stopped CLI job
- Required-job display names are a maintained API identity boundary, with a
  workflow-inventory test to detect drift; source/body fingerprints are not used

## Rejected alternatives and tradeoffs

Waiting on all sibling jobs before starting the CLI would serialize the green
path. Success/skipped early exit could incorrectly admit an unqualified CLI
after a partial rerun. Run-wide latest-job queries can mix attempts. Cancellation
would affect already-running work and is not part of this bounded repair.

Green runs pay one bounded lookup. Already-red runs lose additional CLI failure
diagnostics and must rerun CLI after repair. A later sibling failure, unavailable
API, missing current-attempt sibling or unfamiliar schema still incurs the full
CLI build. The guard is intentionally not an optimizer for those cases.

## Evidence and revisit conditions

Mocked transport and executable decision tests cover all required siblings,
success/active/optional cases, wrong run/head/attempt, incomplete responses,
duplicates, malformed states, partial reruns, API errors, redirects and a stalled
native libc call inside the lookup worker. Workflow contracts verify insertion, event/tier scope, unchanged read
permissions, required inventory and aggregate failure/skip policy. Existing CLI,
cache-stage, merge-queue and time-budget tests remain in place.

Revisit if GitHub changes attempt-job identity semantics, CI exceeds the bounded
single page, required matrix names change, or measurements show lookup overhead
or lost diagnostics outweigh the avoided already-red compilation. No quantified
speedup is asserted before source-matched hosted evidence exists.

## References

- [Old PR run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38317461329)
- [Repaired PR run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38320127161)
- [Attempt-specific jobs API](https://docs.github.com/en/rest/actions/workflow-jobs#list-jobs-for-a-workflow-run-attempt)
- [GitHub rerun semantics](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
