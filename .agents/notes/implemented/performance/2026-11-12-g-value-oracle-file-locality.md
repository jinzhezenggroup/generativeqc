# Decision: keep the new emitted-g oracle outside ordinary core modulo routing

Status: implemented; hosted wall-time follow-up pending
Date: 2026-11-12

## Evidence and problem

PR #3057 head `56b2b6bb15b551dd99cd92b6e69f0614876df4c7`, workflow
43221290971, core-b job 131193221547 reached its existing 35-minute test-step
ceiling. Retained XML reports 28,519 cases, zero failures/errors, 5,200 skips,
and 2,118.459 seconds; the deadline still makes the overall attempt failed.
The new `test_one_electron_g_shared_values.py` module has 181 passing cases
and 132.974 recorded worker-seconds. Controller records include all four worker
exits without an OOM event; this is not evidence of a numerical assertion failure.
The comparable master queue run 43157119998 completed core-b in 1,368.163 seconds.
Many common scientific modules were also slower in the new run, so the new file
alone does not explain the complete timing difference.

The new test filename was automatically inserted into the sorted ordinary core
modulo inventory. A one-off exact-head census, comparing the published inventory
with that same inventory absent only this new file, finds **730 existing module
owners displaced**. Merely moving its workload while retaining the inserted slot
would preserve this unintended cache/workload migration.

## Decision and invariants

Give the new module no ordinary core modulo slot, and select it as a complete
tracked file on the existing compiler-heavy loadfile lane. This follows the
existing no-slot treatment for new qualification files, rather than a transfer
of a historical file whose old slot must be retained. The one-off census then
restores all **2,264** pre-insertion core owners exactly. Compared with actual
master, 900 common module owners differ before this repair and 170 after it;
those remaining 170 differences already come from the three earlier #3057
qualification/diagnostic files and are outside this g-only correction.
Compiler-heavy's test step
completed in 622 seconds in this run, leaving measured headroom for this module.

Keep its shared generated-header fixture with one worker. Preserve all 181
independent Libcint/invalid-input cases and unchanged numerical gates; no test,
scientific work, deadline, worker count, marker or cache trust is reduced.
Tracked-but-missing files remain selected so a damaged checkout fails closed.
Untracked/absent future files do not become required targets.

## Validation and limits

Executable synthetic inventory tests cover all five CI event types and absent,
untracked, tracked and tracked-missing states. They assert unchanged prior core
owners, exact once-only selection, and compiler-heavy ownership. The real xdist
scheduling regression exercises a synthetic module fixture once with competing
files under loadfile, preserving every case without setup duplication.

Actual hosted wall time remains to be measured; this is not a CPU/GPU numerical
or endpoint-performance qualification and does not resolve the separate CLI
ptxas memory problem. Do not increase the deadline or label the failed CI attempt
successful simply because its retained XML contains no assertion failures.
