# Decision: separate the complete standard and corrected force CI tails

Status: implemented
Date: 2026-10-12

## Measured failure

PR #2355 run 38223531357 tested merge
866e2e63e (head c4e74932953913c0f3d55ea90de217b30127a038 into
master 160bf3d). The later head
fc0303cc501ee02da8ab41528aec040992d0dc66 has the identical source tree.
Core-a reached its unchanged 1800-second test deadline at 09:07:35 UTC,
at 99 percent, with complete standard-gradient and corrected-force cases
still running. No numerical assertion failed.

Same-run pytest elapsed seconds:

| Existing lane | Observed time |
| --- | ---: |
| core-a | 1800-second timeout |
| core-b | 1168.33 |
| core-c | 1325.23 |
| runtime-heavy | 1120.65 |
| compiler-heavy | 960.94 |
| posthf | 1277.59 |
| ecp-forces | 1189.18 |

The corrected-force file has 16 host and four CUDA cases. Nine host cases
and four host-skipped CUDA cases completed, accounting for 1272.515
observed worker-seconds; two more host numerical cases were still running.
This incomplete sum is a lower bound on its full work, not a serial runtime
qualification. The standard-gradient file also had four cases unfinished.
Do not serialize either complete force family merely to move the timeout.

## Two complete-file ownership changes

Keep standard gradients on core-a. Move the complete corrected-force file
`test_uccsd_t_gradient.py` to core-b with unchanged parallel worksteal.
Move the smaller complete full-Fock file
`test_uccsd_triples_fock_response.py` from core-b to the existing
compiler-heavy loadfile lane to make room.

All 25 full-Fock cases were observed: 19 host passes and six CUDA skips.
The log contains 481.488 summed seconds between individual start/result
records, with a 257.785-second parallel first-start/last-result span.
The largest observed individual interval is 110.640 seconds. These
log-derived values are not complete JUnit phase timings or a guaranteed
serial wall time. Both modules retain fresh function-scoped fixtures;
no molecular fixture, independent oracle, fault injection, replay,
stationarity, admission, lifetime, or CUDA gate is removed or weakened.

Preserve the full-Fock file's original core modulo slot before exclusion.
The new compiler admission uses Git tracking: absent/untracked paths are
not selected, while tracked-but-missing paths remain selected and fail.
All unrelated ordered owners remain unchanged. The corrected-force pin
only changes its explicit core owner, not its enumeration position.

Compiler-heavy already has the C++ tools, native CPU library, PySCF and
other reference dependencies. Its existing loadfile mode, four workers,
thread limits, coverage and time caps stay unchanged. The corrected-force
file keeps core native fault diagnostics; the full-Fock file takes the
destination lane's existing configuration without the core-only preload.
Scientific assertions and negative controls are unchanged.

## Composition and runtime boundary

The separate #2350 repair transfers two other bounded native files into
compiler-heavy. On this #2355 run their observed completed worker
intervals are 215.678 and 246.626 seconds. Together with full-Fock this is
943.792 observed incoming worker-seconds across three complete files.
That is not a combined wall-time forecast. Adding it to the old
960.94-second compiler wall time would exceed 30 minutes if the work all
serialized late. Four-worker file scheduling and actual CI must establish
the composed critical path; no safe bound is claimed from arithmetic or
from a favorable runner. This repair does not silently include those two
independent transfers before their source union requires them.

The selector regressions cover all five events and absent, untracked,
tracked and tracked-missing forms. Static complete-inventory checks cover
the current tree, latest-master composition, and the prospective #2350
routing union, with exactly-one ownership and identical unrelated order.
Local process launch remains blocked by ENOSPC, so no new local
Bash/pytest/Ruff pass is claimed. Running CI is retained; normal checks
and the merge queue remain authoritative.

## Evidence

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38223531357
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38223531357/job/114725766915
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38223531357/job/114725766961
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38223531357/job/114725766911
- tests/python/test_ci_shard_ownership.py
