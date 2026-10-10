# Decision: bounded artifact-only same-runner CodSpeed diagnosis

Status: implemented
Date: 2026-10-10

## Problem

The [identical-selection and tested-base repairs](2026-10-10-codspeed-identical-pr-selection.md)
removed two demonstrated comparison defects without dismissing PR #2232's
5.37% changed-geometry RHF signal. Job 114306797073 authenticated tested merge
`62c11283100411aed0d2e8e8ff2b74d4f2aebeb0`, PR head
`9a30eb35b5a2f3e48a3e82f8fb80591b289b8871` and actual master first parent
`82bdc5c1863ac98e2a18f14568c0e649c3b99079`. Its five-case selector/source matched
the published master receipt, but its CPU fingerprint did not. The ordinary
upload correctly remained unqualified. Hosted runner allocation cannot be
assumed to reproduce a particular master CPU by blindly retrying.

## Decision

Only an authenticated exact-base receipt with identical source/selection and a
CPU-only environment mismatch enables one diagnostic. Missing/invalid receipts,
unproven ancestry, unknown CPU identity, source/selection drift or any other
runtime difference do not enable it. Ordinary qualified-only CodSpeed upload,
master receipt provenance/schema and all numerical/threshold checks remain.

The pinned action bootstraps runner 5.0.1 and its instruments with `run: true`
and `CODSPEED_SKIP_UPLOAD=true`. Real measurements then use four independent
runner/pytest processes in baseline/head/head/baseline order. Each executes the
same unchanged five-case sequence, including all warmup, allocation, convergence,
finite-value and force checks. There is no benchmark filtering or retry loop.

The existing head CPU library is verified and snapshotted. The authenticated
first-parent baseline is checked out and incrementally built through the same
build directory and verified ccache 4.14 launcher. Both library images are
hashed. All arms use one physical checkout, Python import path and native load
path, avoiding an additional path-length/layout difference between worktrees.
Source commits, benchmark bytes, library hashes and CPU/runtime fingerprint are
checked for each arm. The original tested checkout is restored even on failure,
without force/reset/clean when the process can finish its cleanup; an external
job cancellation cannot guarantee that finalization. Compiler/repository cache
contents are never cleared or weakened. Each fresh process uses the same owned,
empty `PYTHONPYCACHEPREFIX` with bytecode writes disabled, preventing stale
checkout-preserved `.pyc` reuse. ELF dynamic entries/build IDs are retained for
both images; actual linkage is checked by the unchanged endpoint execution.

The helper has a 33-minute total execution budget, eight minutes per arm and
five minutes for the incremental control build. Owned process groups are
terminated on timeout; unrelated jobs or cache services are not stopped. PR jobs
have a 40-minute ceiling to include their normal build/setup and artifact
preservation; push/scheduled/manual jobs retain 15 minutes. No GPU work is added.

## Measurement contract and limits

Require exactly the expected five URI-tagged Callgrind parts, in order, in one
pytest process per arm. Read each part's exclusive `totals` by its `events`
header, padding omitted trailing zero counters. Do not sum call-edge rows,
the exact pinned plugin metadata/known termination parts, `summary` plus
`totals`, or adjacent cumulative
differences. Preserve `summary` separately. Reject missing, duplicate, reordered,
split-process, empty or malformed measurements.

Report each of the two paired observations independently, plus repeat spread:

- `Ir`: executed instructions
- `Ct` and `Cl`: reciprocal-throughput and latency instruction costs, in
  centi-cycles, kept separate
- Data accesses, cache misses and syscall counters: retained independently
- Cost change for metric M: `100 * (M_head / M_base - 1)`
- Inverse-cost change: `100 * (M_base / M_head - 1)`

Zero denominators yield an undefined ratio, not an infinite/JSON-invalid value.
No averages, cache weights, Ct/Cl mixture or aggregate five-case score are
invented. Execution duration is labelled housekeeping, not endpoint walltime.

The exact CodSpeed backend cost mapping is not publicly established. These raw
observations can locate reproducible work/cache-cost growth or demonstrate
repeat variation under controlled histories; they cannot themselves reproduce
or clear the remote 5.37% score. The artifact explicitly retains
`backend_score_available=false` and `performance_clearance=false`. A successful
diagnostic job means data collection completed, not that performance passed.

## Storage and trust

Both the environment flag and explicit CLI `--skip-upload` disable CodSpeed
upload/polling. Neither arm is recorded or published as a master baseline. The
artifact has a distinct diagnostic schema/name and explicitly identifies its
origin as the same PR job. No GitHub permission, OIDC grant, release or secret is
added. No CI result is acknowledged or overridden.

Only allowlisted provenance/comparisons, controlled-command stdout logs and
losslessly gzipped numeric Callgrind profiles plus Valgrind logs are uploaded through the existing pinned
GitHub artifact action, with seven-day retention. Arbitrary runner/environment
metadata, libraries, source checkouts and perf maps are excluded. Raw profile
retention is capped at 512 MiB. No evidence is committed under benchmarks/results
and the repository's evidence caps are unchanged.

## Evidence and revisit conditions

Offline tests cover strict eligibility, malformed per-part counters, exact
selection/history, two separate paired ratios, safe zero handling, native build
and SONAME identity, lossless bounded allowlisted storage, owned timeout cleanup,
canonical execution paths, failed arms and checkout restoration. All diagnostic
command execution is stubbed in local tests; a real CI diagnostic remains
necessary before drawing any new performance conclusion.

Revisit when hosted runner matching is reliable or a documented, authenticated
same-runner backend comparison becomes available. Do not broaden eligibility,
add samples or reinterpret a proxy as a backend threshold without review.

## References

- [Observed CPU mismatch](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38084085258/job/114306797073)
- [Pinned CLI upload/profile controls](https://github.com/CodSpeedHQ/codspeed/blob/8b253c6a2d3a435bc404f8a74f86d3c4ed2a2402/src/cli/shared.rs)
- [Pinned upload branch](https://github.com/CodSpeedHQ/codspeed/blob/8b253c6a2d3a435bc404f8a74f86d3c4ed2a2402/src/executor/orchestrator.rs)
- [Per-part exclusive counter dump](https://github.com/CodSpeedHQ/valgrind-codspeed/blob/76a6a648d2bea1ef2c99abaf31a3d9615f841c0b/callgrind/dump.c)
- [Cycle-estimation units](https://github.com/CodSpeedHQ/valgrind-codspeed/blob/76a6a648d2bea1ef2c99abaf31a3d9615f841c0b/callgrind/cycledecode.h)
- [Measurement model and syscall limits](https://codspeed.io/docs/instruments/cpu)

- [Primary variance guidance](https://codspeed.io/docs/instruments/cpu/reducing-variance)

## Follow-up: generated launcher indirection

The first diagnostic job [114312391272](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38085962118/job/114312391272)
authenticated merge `e9b4fce248595fa40558efbd92e62c69d82d7735`, head
`e95f306446493e0f9b1360415c08d8295267ab55` and master `82bdc5c`, then correctly
identified an Intel Xeon Platinum 8370C versus AMD EPYC 9V74 CPU mismatch.
It stopped before any diagnostic build or endpoint because the initial verifier
looked for a literal `ccache` inside `rules.ninja`. The standard build had logged
`/usr/local/bin/ccache` and 219/219 cache hits. Modern CMake stores `${LAUNCHER}` in
that shared rule and its concrete command on each build edge, so that substring
check was insufficient evidence of missing cache use.

The verifier now asks Ninja's read-only `-t compdb` tool to expand only the native
library's Release CXX rules. Every returned object must have the expected target
identity, cache/environment prefix and configured C++ compiler. Empty, malformed,
uncached, misleading or conflicting commands fail closed. Repeated input records
must agree on the command for their output; reported object counts are unique.
The expanded commands and their hash are retained for both builds. A tiny Ninja
graph test reproduces launcher indirection without invoking a compiler, generator
or benchmark. This repair does not change selection, authentication, upload,
CPU/threshold gates or the four-arm budget; the performance question is still open.

Primary implementations: [CMake launcher binding](https://github.com/Kitware/CMake/blob/master/Source/cmNinjaTargetGenerator.cxx)
and [Ninja extra tools](https://ninja-build.org/manual.html#_extra_tools).
