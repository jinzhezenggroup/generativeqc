# Decision: run CLI qualification alongside fast CUDA compilation

Status: implemented
Date: 2026-10-13

## Problem

PR and merge-group native CLI packaging waited for the required fast CUDA job,
although those events consumed none of its files, caches, or outputs. Observed
merge-group predecessors took 23m17s, 21m39s, and 14m13s before the CLI could start.
The master producer does consume its trusted-cache publication decision, so
removing the dependency for every event would lose a real ordering requirement.

## Decision

Keep `native-cli-artifact` ordered after successful `cuda-compile` on master/tag
pushes and manual runs. Add `native-cli-qualification` on PRs and merge groups,
depending only on successful merge-queue liveness. A YAML anchor supplies the
same complete package recipe to both jobs; this does not duplicate build stages
within one event. Only the ordered job receives the publication output through
its environment; the parallel job receives an explicit false value.

The existing `Pass CI checks` aggregate needs both branches. Its event-specific
skip allowance names only the unused branch, or both branches on the pre-existing
scheduled no-artifact path. It never permits failures or a skipped selected
branch. Master cleanup and versioned-release dependencies remain unchanged.

## Invariants

- Preserve full queued-range tier classification, genuine CLI link/package,
  complete automatic XC catalog, stationary-program checks, smoke tests,
  installed/archive-byte comparison, checksums, and artifact delivery.
- Preserve explicit orphan liveness, default success dependency admission, and
  the bounded same-attempt required-failure guard. An already-failed required
  sibling fails qualification; uncertainty or a running sibling does not wait.
- Both required CUDA and selected CLI qualification must finish successfully.
  Rerunning only a failed sibling cannot convert a failed/skipped CLI to success.
- Preserve master-only cache production, successful ordered output transport,
  cache identities and trust checks, read-only qualification permissions,
  exact checkout identity, and no cancellation of running workflows.
- Distinct job names prevent the failure guard's duplicate-name rejection from
  unnecessarily treating a valid jobs page as uncertain.

## Rejected alternatives

- Removing the dependency globally loses master ordering and its output.
- Adding a condition to the existing dependency does not remove the DAG edge.
- Copying the package recipe into a second job risks divergent qualification.
- Cross-queue compiled-object reuse would expand cache trust; matching source
  inputs alone do not establish identical apt-installed toolchains.

## Evidence and limits

`test_native_cli_workflow.py` executes conditions and dependency admission for
same-repository/fork PRs, merge groups, master/tag/other pushes, manual and
scheduled events, failure/cancellation/skip, orphan liveness, and publication
outputs. The existing guard tests include both branch records and retain
required-sibling failures. Packaging, full-queue scope, cache lifecycle, and
same-attempt guard tests remain required. The candidate was also parsed with
PyYAML and actionlint 1.7.12; the alias resolves to all 25 original steps, with
only the publication output transported through the job environment.

This changes scheduling, not compiler work or a measured end-to-end speedup.
Runner availability may still serialize jobs. No GPU or nvcc validation was
available locally; hosted elapsed-time effects require a subsequent real run.
No live run was cancelled or manually dispatched for this repair.

## References

- [Observed predecessor 1](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38341966041/job/115120479334)
- [Observed predecessor 2](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38342772968/job/115122586186)
- [Observed predecessor 3](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38344084359/job/115124946495)
- [GitHub YAML anchors](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations#yaml-anchors-and-aliases)
