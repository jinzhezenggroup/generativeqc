# Decision: reuse landed routing and qualified fresh-worksteal ordering

Status: implemented
Date: 2026-10-12

## Scope

Reuse the unchanged `tools/pytest_ci_fresh_frontier.py` reviewed at #2355
`a03ebbe4d75ebd458565797b9b355dd8ebce1512`, together with its executable
property and actual four-worker subprocess regressions. Only the existing
standard-gradient core-a and corrected-Lambda core-c files opt in. Core-b
emits no priority metadata. First reuse the landed phase-two selector and
ownership contracts from master `61a0e715fde35566722870389af8e137036232db`:
move the complete bounded `test_cc_native_tensor_cpu.py` and
`test_uccsd_native.py` files to compiler-heavy while retaining their core
modulo slots. Relative to that landed routing, the frontier metadata changes
no ordered file ownership or dedicated-shard policy. The existing full-Fock
suite stays in core-b; no corrected-force owner or scientific dependency is
imported.

The plugin permutes existing final collected items, spreading fresh cases
across the existing contiguous worksteal queues. It preserves parametrized
nodes and ordinary relative order, validates the controller scheduler, uses
the actual worker count, and leaves serial execution unchanged. It has no
runtest/error/report hooks, fixture replacement or scientific execution.
Dedicated loadfile owners do not load it. Metadata is emitted only for
selected tracked files; absent/untracked paths are not admitted and tracked
missing paths still fail ordinary collection.

## Evidence and limits

The donor's actual hosted run 38236462737 passed all seven shards and all 135
selector/plugin tests, including six real subprocess checks covering four-
worker startup, filtered node preservation, serial order and unsupported-mode
rejection. Observed pytest seconds were A1699.78/B1094.12/C1763.21,
runtime685.96/posthf1754.00/compiler1529.80/ECP614.41. These observations
validate that donor's complete run; they are not a wall-time promise for a
new inventory. Some margins remain thin.

PR #2360 run 38237354798, core-c job 114771238173, reached the unchanged
30-minute cap in the corrected-Lambda tail. A repeated tail in #2358 run
38235501782, core-c job 114774207564, reached the same cap at 11:20:01 UTC
after the last restricted-limit case started at 11:20:00. The #2360 compiler,
output-retention and extracted-harness failures are repaired separately in
`0275f40c77e6ae2c343fc073d69e41a41047e10f`, with its formatter follow-up
`00a6fb922e02712633fd9d0dd5797227076fba99` preserved unchanged. Startup ordering does not conceal a failed assertion or
replace source repairs. Each adopting head must pass its own full required
CI and normal merge queue.

The two opted-in suites retain their function-scoped independent owners;
all numerical and negative cases, tolerances, workers, deadlines and caches
remain unchanged. Static inventory checks are not local Bash/pytest execution
when the executor is blocked by ENOSPC. Revisit if fixture scopes change or
actual scheduler startup no longer matches the behavior regressions.
