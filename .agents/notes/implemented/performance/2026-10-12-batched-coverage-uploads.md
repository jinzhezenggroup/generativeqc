# Decision: batch optional coverage uploads without sharing their failure state

Status: implemented
Date: 2026-10-12

## Problem and evidence

CI run [38261551954](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38261551954)
finished every coverage producer at 15:12:20 UTC and `Pass CI checks` at
15:13:36. The first three coverage jobs did not start until 15:17:05,
15:17:31, and 15:17:36: 285–316 seconds of queue time for 17–21 seconds
of actual job time. Their checkout steps took 7–10 seconds, artifact downloads
one second, and Codecov actions 4–5 seconds. Five more uploads were still
queued at 15:20:45. The final upload completed at 15:25:00, 11 minutes
24 seconds after the correctness gate and 12 minutes 40 seconds after all
producers completed. Across all eight uploads, queue time ranged from 285 to
741 seconds and actual job time from 15 to 21 seconds; checkout was 6–10 seconds, download
0–1 seconds, and Codecov 3–7 seconds. Because running workflows
are intentionally preserved, these optional jobs kept the workflow concurrency
slot occupied. The newly landed UCCSD shard increases eight uploads to nine.

## Decision

Use three fixed batches of three independent reports in the existing upload-only
job. This removes six runner admissions and six repeated repository checkouts
without merging any reports. Keep every artifact name, report basename,
language flag, and Codecov upload name. Download into separate report directories
and upload only the corresponding explicit file with discovery disabled.

The checkout and each of the six transfer actions have a one-minute timeout.
Their total seven-minute allowance fits the existing eight-minute job timeout
with a minute reserved for runner/action cleanup. This gives observed download
and upload times at least eight-fold headroom. A report action timeout is a step
failure; `continue-on-error` lets later independent reports proceed. Each upload
requires its own download's successful **outcome**, not its post-tolerance
conclusion. Missing, failed, timed-out, or skipped downloads never trigger an
upload of an absent/partial report. Actual job cancellation still stops uploads.
Checkout failure remains fatal to that batch, since source metadata is shared.

## Invariants

- Keep OIDC exclusively in the upload-only job; do not run repository scripts,
  tests, builds, or downloaded artifacts in this permission context.
- Preserve action/CLI pins, `contents: read`, `id-token: write`, credential-free
  checkout, event guard, upstream-success dependencies, and nonblocking coverage.
- Keep producer uploads, retained artifacts, all scientific jobs, required checks,
  release behavior, and `cancel-in-progress: false` unchanged.
- Keep explicit named downloads and explicit per-report files. Do not replace
  them with wildcard download/search or consume unrelated artifacts.
- When adding a producer, update report ownership and ensure each batch's bounded
  steps fit below the whole-job timeout; do not silently omit a report.

## Alternatives and tradeoffs

A single job needs at least 19 active minutes to allow nine one-minute download/
upload pairs and checkout. Giving it only eight minutes could let one or more
hung reports prevent later attempts; enlarging the workflow tail to twenty minutes
would weaken the existing bound. Three small batches preserve the bound while
reducing runner demand. Nine matrix jobs retain unnecessary queue admissions.
Moving this to a privileged `workflow_run`, executing a local composite action
with OIDC, or changing cancellation policy would alter the trust/lifecycle
contract and is unnecessary.

This reduces runner work, not the number of Codecov requests. It does not promise
an exact wall-time improvement under changing runner availability. Three serial
reports take longer on an individual runner than one report. Optional service
outages remain visible in step outcomes and do not become correctness failures.

## Validation

- 168 coverage/time-budget/CodSpeed/shard tests, Ruff, actionlint 1.7.12,
  and repository aggregate ownership/integrity/public-contract checks pass.
- Parsed old/new workflow comparison: all nine artifact/basename/flag/upload-name
  tuples preserved exactly; every other job and top-level setting identical.
- Contract tests check producer completeness, directory isolation, scoped OIDC,
  pinned actions, report-local outcome gates, and finite aggregate budgets.
- Local policy-model enumeration covers 12,288 success/failure/skipped/timeout
  combinations across the three batches, retaining all independent download
  attempts and suppressing uploads whose own download did not succeed. This is
  configuration validation, not execution of GitHub's runner or Codecov.
- GitHub runner [step timeout handling](https://github.com/actions/runner/blob/main/src/Runner.Worker/StepsRunner.cs)
  distinguishes step timeout failures from cancellation of the whole job and
  applies `continue-on-error` afterward.

## Revisit when

Actual transfer steps approach one minute, the number of producer reports changes,
or runner queue contention disappears. Re-evaluate batch size and per-step budgets
jointly using measured end-to-end workflow completion, without dropping reports.
