# Decision: move two bounded native files to the existing compiler lane

Status: implemented
Date: 2026-10-12

## Observed failure and capacity

PR #2350 head c0884754c161bea72f8e980ef6faabe9678028b1, CI run
38219795252, tested merge 29f21502cb6688f3e8a62c9ddd194a43929ae803.
Its unchanged 1800-second core-c test deadline expired at 08:33:54 UTC,
with corrected-Lambda cases still running at 99 percent. No numerical
assertion failure appeared in the log. Core-a and core-b passed.

Same-run pytest elapsed times, in seconds:

| Existing lane | Observed time |
| --- | ---: |
| core-a | 1711.77 |
| core-b | 1386.14 |
| core-c | 1800-second timeout |
| runtime-heavy | 1110.32 |
| compiler-heavy | 688.51 |
| posthf | 949.38 |
| ecp-forces | 904.33 |

The complete native TensorIR CPU file, `test_cc_native_tensor_cpu.py`,
had six cases on one core-a worker: 248.057 summed seconds between visible
test-start/result records, spanning 258.372 seconds from first start to last
result. The complete native UCCSD producer file, `test_uccsd_native.py`,
had three cases on one core-c worker: 394.577 summed observed seconds,
spanning 397.529 seconds. These are log-derived worker intervals, not
complete JUnit phase timings, a speedup measurement, or a predicted shard
wall time.

Runner variability remains material: the repaired #2329 core-a passed in
1754.96 seconds while #2351's same owner inventory passed in 834.48 seconds.
A favorable single run does not establish reliable spare capacity.
The new complete corrected-force module in #2355 has no qualified hosted
runtime in this evidence set and is not reallocated here.

## Bounded transfer

Move only those two complete native files to the existing compiler-heavy
loadfile lane. Retain their original core enumeration slots in
`rebalance_paths`, admit them only when Git tracks them, and preserve every
other file's ordered ownership. Tracked-but-missing files remain selected
and fail closed. Remove the superseded native TensorIR core-a pin.
The gradient, corrected-Lambda and full-Fock files remain parallel and
unsplit on their existing core owners.

Both transferred files construct their owners inside each test; tmp_path
and fresh numerical instances are unchanged. Neither defines a module- or
session-scoped fixture. The compiler lane already installs the same
reference-test dependencies including PySCF, C++ tools, and native CPU
library. Each transferred file's observed serial interval is bounded well
below the lane's unchanged cap; actual combined runtime must be measured
by CI, not inferred by adding these intervals.

The destination keeps its existing loadfile scheduling, four workers,
coverage, thread limits, and time budgets. It does not use the core-only
native fault-diagnostic preload. This is an explicit existing lane
difference; no numerical assertion, scientific negative control, fixture
lifetime, or test case is removed. Diagnostic instrumentation is not
expanded by this repair.

## Verification boundary

The real Bash ownership regressions cover all five event types and
absent, untracked, tracked and tracked-but-missing forms of both new
transfers alongside the existing response owners. Complete static Git
inventories establish exactly-one ownership and unchanged ordered owners
for every other test. Local execution remains blocked before process
launch by ENOSPC; no local Bash/pytest/Ruff pass is claimed. Hosted CI
remains the executable selector, lint, scientific and runtime authority.

This refines the earlier [parallel response split](2026-10-12-uccsd-parallel-tail-balance.md).
That note and its measurements remain intact.

## Evidence

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219795252
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219795252/job/114713251749
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219795252/job/114713251890
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219795252/job/114713251770
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219781445
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38219800549
