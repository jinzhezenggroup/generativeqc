# Decision: preserve complete UCCSD coverage while balancing CI owners

Status: implemented
Date: 2026-10-12

## Problem and evidence

PR #2329 at f3037cba75705acd1169ea952303fe2513aa1d09 failed the
30-minute core-c pytest step in run 38196621228, job 114643437293.
Its checkout was merge tree 16dd2fc85bbcb14c5c97df51abd0ed95a35f3870.
The log completed 9,041 of 9,052 cases (8,193 passed, 848 skipped) with no
reported assertion failure before timeout. Collection/startup took about
282 seconds. Old Lambda, old unrestricted triples-energy, and new physical
response cases were still passing in the late tail.

Filename-modulo routing is sensitive to additions in the merge tree:
test_ks_coulomb_exchange_work.py was absent from the PR head but present in
the tested merge tree. It shifted later unpinned core owners, concentrating
these UCCSD files together. This is aggregate shard imbalance, not evidence
of a new numerical defect or a complete performance measurement.

The same run's pytest summaries were 1,582.41 s for core-a, 907.14 s for
core-b, 726.46 s for posthf, 755.08 s for runtime-heavy, and 1,166.07 s for
compiler-heavy. In the preceding #2323 run 38187785285, core-c completed in
1,298.59 s and core-b in 1,479.81 s. The unchanged Lambda and triples-energy
owners already contributed substantial native/oracle work there.

The dependent #2340 run 38199980410 independently passed all 22 new host
triples-response cases (7 CUDA cases skipped) in core-a, job 114651269374.
The reported slowest response cases were H6 66.85 s, NH2 49.03 s, H4
20.87 s, and the memory-fallback/lifetime case 28.76 s. These are individual
case durations, not a whole-file timing. Its core-b later reached the same
30-minute deadline with older UCCSD cases still passing in the tail.

## Decision and invariants

Move the complete Lambda, triples-energy, and physical-response test files
to the existing posthf loadfile shard. Reserve the dependent
test_uccsd_triples_response.py for runtime-heavy when Git tracks it.
Keep all four paths in rebalance_paths, so the original core inventory
index advances before exclusion and unrelated modulo owners stay stable.

Git tracking, rather than filesystem existence, controls conditional
admission. An absent or merely untracked future file is not selected.
A tracked but missing file remains selected and fails naturally in pytest.
Every tracked test retains exactly one shard; no numerical, failure,
lifetime, nightly, or independent-oracle case is deselected. Existing
30-minute budgets, worker counts, and cache behavior are unchanged.

## Rejected alternatives and verification limits

Do not raise the timeout or drop tests to conceal imbalance. Do not move
the entire gradient file to loadfile: the failing run's other core shard
already recorded about 1,659 summed worker-seconds in that file, including
repeated function-scoped native setup, so serializing it could create a new
long tail. The selected owners have useful module-local oracle reuse.

The real Bash selector and ownership regression are executable without
scientific builds. Coverage/slot tests include five event types and
absent, untracked, tracked, and tracked-but-missing inputs, plus full
head/merge/stacked filename inventories. Passing those tests establishes
selection, not the final runtime of the rerouted scientific workload.
Full rerouted CI remains pending at publication. Revisit allocation using
complete per-test timings if another measured shard approaches its budget.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38196621228
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38187785285
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38199980410
- tests/python/test_ci_shard_ownership.py

## Follow-up: keep the complete gradient tail parallel

Run 38202660731 tested merge tree fb66c86bfa65e1ce0d30a28c363741881aaef6c5
after the complete-owner transfer above. Core-c completed 16,495 of 16,497
cases (12,370 passed, 4,125 skipped), leaving only gradient admission
options0 and native hcore-binding checks at its 30-minute deadline. The
five independent molecular gradient cases, including spherical and
broken-symmetry cases, and the reconverged finite-difference case passed.
No assertion failure was reported.

The passing stacked #2340 run 38202771274 used identical gradient source
(blob 0ff724562688bf6ccf7f5b825ddbc1d8677970e7) and finished core-c in
1,537.41 s. Its slowest-duration table alone records 1,751.22 worker-seconds
in the gradient file: five complete molecular calls, nine fresh fixture
setups, and the finite-difference call. This is a lower bound on summed
test work, not a whole-file wall time or predicted speedup.

Pin the complete gradient file to core-a through the existing hotspot
owner mechanism. Keep worksteal, all 17 cases, fresh function-scoped
fixtures, and the original core inventory index. Same-run core-a/core-b
were 1,126.86/1,269.84 s; the stacked run measured 1,258.70/1,327.50 s.
Core-a therefore has the larger observed margin in both runs. This does
not establish the resulting critical path: final rerouted CI must still
measure it. Serial loadfile routing remains rejected, and the unchanged
30-minute budget remains enforced. Posthf passed in 1,728.94 s on #2329
and 1,736.37 s on #2340; no additional load is assigned there.

Extend the executable all-event ownership regression with the gradient
pin, worksteal mode, absent/untracked/tracked/missing behavior, and
unchanged indices for unrelated files. No scientific code, test oracle,
fixture reuse, acceptance gate, or worker count changes.
