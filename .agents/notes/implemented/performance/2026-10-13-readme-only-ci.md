# Decision: keep heavy CI off verified root README-only changes

Status: implemented
Date: 2026-10-13

## Problem

A root README-only PR still launched eight Python environments/native builds,
two CPU compiler/CTest variants and three CUDA compile/resource jobs. The native
CLI already had a change-aware skip tier. Rebuilding unchanged scientific code
did not test the changed prose or improve its editorial qualification.

## Decision

Classify after each existing checkout and omit expensive steps only for a
positively verified modification of root README.md. Preserve all job/check names,
dependencies and the strict aggregate. Keep the GCC ABI, native-package/scope and
CUDA ownership checks. Documentation and pre-commit remain unchanged. Coverage
artifacts are omitted when their tests did not run; no empty coverage is
manufactured. Existing independent artifact downloads determine which real
reports reach Codecov. Never infer artifact absence from a shared matrix output:
classifier fallbacks can legitimately produce coverage in only some matrix legs.

No new prerequisite runner, checkout, dependency barrier or credential is needed.
Duplicating the small local classifier avoids serial runner-queue latency on
ordinary source PRs. Potential prose queue tails still need a fresh target fetch
in each existing job. Git operations have finite bounds; the workflow caps the
whole selector at 75 seconds and falls back to normal coverage on any failure.

## Invariants

- Admit only an existing nonempty regular UTF-8 README.md modification; reject
  mode changes, binary/control bytes, additions, deletions and renames.
- All other paths, including docs Markdown, assets, evidence, fixtures,
  manifests, workflow/helper/test/config/security changes, retain normal CI.
- Resolve the actual verified PR test-merge base and the complete unmerged
  queue against the current target branch. Never approve a queue skip from
  only its last synthetic predecessor. Reuse the native CLI ancestry guard.
- A local non-README queue tail may immediately select normal coverage without
  a remote fetch. This shortcut may only reject, never approve the prose tier.
- Missing, malformed, ambiguous, divergent, stale or failed comparisons select
  normal CI. Only a successful classifier step with exact `prose` output can
  omit heavy work; missing/failed outputs retain the old conditional behavior.
- Master pushes, tags, schedules, manual runs and unknown events keep their
  existing coverage/cache/publication roles. Native CLI release/fast selection,
  scientific acceptance and test-shard ownership remain unchanged. This change
  does not equate an earlier fast-tier result with full-release qualification.
- Preserve original step conditions for full CI, including failure diagnostics
  and post-test cache handling. Do not invoke absent build/cache tools after an
  intentional omission.

## Rejected alternatives

A blanket docs/** or *.md rule would admit machine-consumed inventories and
benchmark evidence, and miss documentation contracts checked by Python tests.
README-only scope is intentionally narrower. A new prerequisite scope job would
introduce another runner admission/checkout to the source-PR critical path.
Changing allowed aggregate skips would unnecessarily alter required-check
admission; successful cheap versions of the existing jobs preserve it instead.

## Evidence and limits

Synthetic Git tests cover source deletion/rename, modes/binary data, missing
refs, failed fetches, stacked queues, target advances/divergence and ambiguous
ancestry. Executable workflow tests cover all 13 matrix/job instances, actual
classifier shell behavior, uncertain outputs, failure cleanup and coverage
omission. Existing native package/scope/workflow and cache tests remain.

This is avoided redundant work, not new numerical or end-to-end benchmark
qualification. Root README scientific claims and its complete link/anchor set
have no existing standalone executable gate; Sphinx's repository-link helper
does not turn those claims into validated scientific evidence. Editorial review
remains necessary. Benchmark code/data changes always retain normal tests.

## Revisit when

Expand the allowlist only after identifying every machine consumer and retained
contract for the new paths. If repeated README-tail target fetches become costly,
measure that cost before adding shared prerequisite work to all source changes.

## References

- README-only example: https://github.com/jinzhezenggroup/generativeqc/pull/2455
- Native queue range guard: tools/native_cli_ci_scope.py
- Admission: tools/ci_prose_scope.py and tests/python/test_ci_prose_scope.py
- Workflow behavior: tests/python/test_ci_prose_workflow.py
