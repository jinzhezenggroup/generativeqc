# Decision: stabilize the complete UCCSD response owner after inventory growth

Status: implemented
Date: 2026-10-12

## Observed failure and actual owner inventory

PR #2356 head bc8575a364bfef5e0ece55a084404d24f19918b0, run
38227091725, tested merge 3d9ed5ccbf6220df3d1099836e437a8d45c977ad
into master e72c1167. The tested selector is c390a400, the landed
parallel response split. Core-c hit the unchanged 1800-second test
deadline at 09:46:25 UTC with three corrected-Lambda host cases
unfinished; two were still executing. No numerical assertion failed.

Same-run pytest elapsed seconds:

| Existing lane | Observed time |
| --- | ---: |
| core-a | 1342.99 |
| core-b | 761.55 |
| core-c | 1800-second timeout |
| runtime-heavy | 1126.36 |
| compiler-heavy | 1224.12 |
| posthf | 899.00 |
| ecp-forces | 1201.93 |

The added automatic-restricted CUDA test filename changes later unpinned
modulo owners. On this exact inventory, the complete UCCSD response file
`test_uccsd_response.py` falls on core-c immediately before the
corrected-Lambda tail. Its 14 completed cases account for 301.942 summed
seconds between visible start/result records, spanning 319.697 seconds
from first start to last result. Core-b has 1787.686 summed observed
worker-seconds; core-c has 4541.053 before its timeout. These are
log-derived intervals, not full JUnit timings or wall-time predictions.

Blindly copying the separate #2350 two-native-file transfer would not
relieve this failure: both of those files ran on core-a here, with
175.352 and 232.768 observed worker-seconds. Core-c contains neither.
No such unrelated transfer is included in this repair.

## Single complete-file pin

Pin `test_uccsd_response.py` to the existing core-b owner. This restores
the owner it already had in the measured #2350 and #2355 inventories.
Keep its original enumeration slot, every test, fresh owner lifetime,
parallel worksteal, diagnostics, dependencies and all CI budgets.
Corrected Lambda remains complete and parallel on core-c.

Only this file changes owner on the actual tested inventory. All other
ordered selections are unchanged on each compared inventory. The
existing tracked-file enumeration naturally omits absent/untracked
files; tracked-but-missing files remain selected and fail closed.
The ownership regression covers all five events and the four presence
forms, with a neighboring unpinned filename to detect modulo drift.

## Composition and verification

The repair composes the current PR with already-landed master only.
All reviewed automatic-CUDA scientific source, numerical tests and
documentation are retained byte-for-byte; all unrelated master changes
are retained. No unlanded full-Fock or complete corrected-force science
is imported.

Static checks cover the complete current and prospective routing unions
with #2350/#2355, including the additional oracle test filename, across
all five events. They establish exactly-one ownership and unchanged
unrelated ordering, not runtime. Source/lint execution remains a hosted
CI responsibility because local process launch fails before execution
with ENOSPC. No local Bash/pytest/Ruff pass is claimed.

The lighter measured core-b is a reason to test this bounded placement,
not a promised tail bound. Final CI and the ordinary merge queue must
qualify the actual combined workload.

## Evidence

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38227091725
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38227091725/job/114738146107
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38227091725/job/114738146004
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38227091725/job/114738146040
- tests/python/test_ci_shard_ownership.py


## Landed native-transfer integration after the remaining A timeout

Run 38232996979 on head `86ddf9a7` verified the response-to-B change:
core-c passed 10691 cases/855 skips in 933.54 seconds and core-b passed
8797 cases/4310 skips in 1137.56 seconds. Core-a instead reached the
unchanged 30-minute test-step cap at 10:28:42 UTC in job 114756776752,
with standard-gradient cases still active and no visible assertion failure.
The other pytest times were runtime-heavy 1105.36, compiler-heavy 1232.31,
posthf 1754.70 and ecp-forces 866.39 seconds. These are observations from
this run, not runtime guarantees for another inventory.

Both complete native transfer files ran on failed core-a: the six
`test_cc_native_tensor_cpu.py` cases account for 254.947 observed
start/result worker-seconds and the three `test_uccsd_native.py` cases for
275.880. The same bounded transfer already reviewed and qualified in #2350
has now landed in master `61a0e715fde35566722870389af8e137036232db`.
Compose that actual master with the reviewed restricted-CUDA head, retaining
this response-to-B pin and both original modulo slots. No new allocation
policy or unlanded scientific dependency is introduced.

The existing compiler lane retains its dependencies, loadfile scheduling and
diagnostics, including the previously documented absence of core-only fault
preloading. Every transferred case and fresh owner remains unchanged. The
sum of these worker intervals plus the prior compiler duration is not a
wall-time bound; the final combined hosted run and normal merge queue remain
the deadline gates. All seven shard budgets and worker counts are unchanged.
