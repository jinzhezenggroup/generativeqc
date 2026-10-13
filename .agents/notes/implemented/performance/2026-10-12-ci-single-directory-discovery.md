# Decision: avoid repeated pytest directory discovery in broad core shards

Status: implemented
Date: 2026-10-12

## Problem

The broad core shards pass hundreds of individual files to pytest. With pytest
9.1.1, a local core-b collection of 531 explicit paths reported 531 collections of
`tests/python` before importing the test modules. This repeats directory discovery
on every xdist worker. Hosted #2371 core-b needed 4m48s from test-step start to its
first reported four-worker test, even after a 227/227-hit native build.

## Decision

Keep the existing shell selector as the sole owner of shard files, event-dependent
deselection, and fresh-frontier ordering. A separate, opt-in CI argument adapter
uses one `tests/python` target plus pytest's standard `--ignore` complement when
core-a/b/c supplies sorted, distinct, immediate test files. Enumerate the immediate
directory entries once; ignore whole unselected directories and symlink directories
before pytest can import their conftests. Fail closed if any selected file is
missing, the selection is empty, or enumeration fails.

Retain the original explicit targets for all other shards, priority-ordered files,
node IDs, duplicate paths, and nested file targets. Ordinary developer pytest
invocations do not load the adapter. No test body, oracle, fixture, assertion,
parameter, tolerance, scheduling mode, coverage option, or CI timeout is removed.

## Evidence

On composed commit d286f2f2a plus this test-only adapter, Python 3.12.14 and pytest
9.1.1, a clean serial core-b collection took 98.4 seconds, collecting 12,825 items.
The per-collector profiler recorded 531 directory traversals. A separate matched
four-worker discovery-only harness (same interpreter/environment/worktree/inventory,
worksteal, no scientific test execution) took 106.56 seconds with explicit files
and 14.97–17.57 seconds in two runs with hardened directory/ignore selection,
an 83.5–86.0% local reduction (the initial file-only variant took 16.91 seconds).
All four workers had exactly the same ordered 12,669 node IDs before and after.
The different serial/parallel item counts use different local Python environments;
only matched-environment comparisons establish parity or speedup.

The diagnostic harness intercepts the runtest protocol after normal collection;
it is not committed and is not evidence of scientific execution. Hosted CI must
establish the effect on end-to-end feedback time; runner queuing, native builds,
and scientific test execution remain separate costs.

Executable regressions cover every shard and all five events, unchanged fallback
ordering, repeat sourcing without broadening, missing/empty inputs, partial
enumeration failure, serial and four-worker
node order, parametrized module fixtures, preserved skips, foreign/untracked file
imports, foreign and symlinked child conftests, fresh-frontier composition, and unchanged
developer discovery.

## Rejected alternatives

Do not replace collection with post-import deselection: it keeps repeated import
and parameter-generation work and can execute foreign import side effects. No
custom pytest collector or private pytest API is needed. Do not change worksteal,
module fixture scope, reference caches, or shard ownership to solve discovery cost.

## References

- https://docs.pytest.org/en/stable/example/pythoncollection.html#ignore-paths-during-test-collection
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38265826506/job/114882282301

Revisit if pytest eliminates repeated explicit-file directory traversal, or the
selector begins producing nested/priority-ordered core targets. The explicit-path
fallback preserves those semantics until they have a separately measured solution.
