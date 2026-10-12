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
