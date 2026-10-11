# Decision: start the complete Libxc force matrix on the force shard

Status: implemented
Date: 2026-10-11

## Problem and evidence

PR #2253's ordinary CI run 38113859405 checked merge tree
`b73edcfafcc4700533f09ecab3d6d03ed9e2bdf7`: PR head `500a875` plus master
`0f78eab`. The complete automatic Libxc force module arrived through master
#2242 and is absent from the PR head by itself.

The core-a test step started at 05:39:28 UTC. Its first completed test was at
05:45:02, so startup/collection consumed about 334 seconds before useful test
progress. The Libxc lifecycle matrix began at 05:52:53. Nine consecutive
functional cases completed on one worksteal worker in 103.4--110.8 seconds each
(970.9 worker-seconds total). GVT4 was running, and TPSS had not started, when
the unchanged 30-minute test limit stopped the job at 99%. No assertion failure
was displayed. Other core workers were still completing tests near the limit;
this was a workload/scheduling problem, not evidence of one hung native call.

Each lifecycle case deliberately executes two spins, three lifecycle phases,
and thirteen central/displaced endpoints per phase: 78 native and 78 independent
PySCF/Libxc solves. All eleven functionals therefore require 858 native and 858
oracle solves. The file has 17 tests, including six auxiliary contracts, and no
expensive module fixture or mutable shared lifecycle state.

The same run's existing ecp-forces shard passed in 495.4 seconds. Its routine
file sizes are 50, 48, 22, 5 and 5 collected cases. Under the existing loadfile
scheduler the 17-case Libxc module is the fourth-largest file and starts on an
initial worker, instead of 13 minutes into broad core. Extrapolating the nine
observed lifecycle cases gives about 19.8 minutes for eleven cases, before the
auxiliary checks. This is an estimate, not a measured repaired endpoint: GVT4
and TPSS were not timed to completion, and TPSS has a larger imported graph.
Complete hosted CI remains the acceptance check for wall time.

## Decision and invariants

Move only the complete `test_libxc_automatic_forces.py` module to the existing
ecp-forces loadfile shard on every CI event. Count it in core parity before
excluding it, preserving every unrelated file's owner. Select the destination
target only when Git tracks it, accommodating the older PR tree without an
unrelated base merge. A missing tracked file remains selected and fails pytest;
an untracked local file does not enter the CI inventory.

Keep all functionals, spins, cold/replay/changed-geometry phases, finite
differences, independent references, and numerical assertions unchanged. Retain
all existing event deselections, coverage arguments, six jobs, worker counts,
cache identities, permissions, cancellation policy and finite time ceilings.

## Rejected alternatives and revisit condition

Do not blindly retry or enlarge the timeout. Do not alter production identity
caching or reduce the finite-difference work: neither is needed for this routing
repair. Splitting the scientific module into family-sized files is a possible
later remedy if complete hosted timing shows that a single initial worker still
approaches the 30-minute ceiling; it would require a larger source change and
additional test-inventory proof.

Regression tests cover all five events with the file absent, untracked,
tracked, or tracked-but-missing, and check exact ownership and core parity.
Before/after full-tree routing comparisons must preserve every other owner,
distribution mode, deselection and coverage argument.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38113859405/job/114398555883
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38113859405/job/114398555931
- `tests/python/test_ci_shard_ownership.py`
- `tests/python/test_libxc_automatic_forces.py` and `tools/qualify_libxc_forces.py`
