# Decision: preserve host Hessian fixture locality and wheel completion time

Status: implemented
Date: 2026-11-18

## Observed failures

PR #3083 at `b320e86e4a416a82c99836f8d80fdc3675993193` has four
independent failing jobs, not a new failed chemistry qualification:

- CI run `44034488465`: core-a completes with one stale timeout-policy
  assertion, expecting core-c's previous 30-minute allowance instead of 35.
- The same run: core-b exhausts 35 minutes at 99%, with three workers still
  executing `test_host_ks_nuclear_hessian.py`. Core-c exhausts 35 minutes at
  84%, with `test_host_ks_nuclear_response.py` still active. Both modules
  build the same substantial module-scoped native response fixture, which
  worksteal can recreate on several workers.
- Core-c also spends 982.9 completed worker-seconds on
  `test_host_hessian_aot_cmake.py`. Successful same-run Python steps take
  905 seconds on runtime-heavy, 818 on compiler-heavy and 615 on posthf.
- Wheel run `44034488504` exhausts the 330-minute build step at task
  1181/1200 while recording stationary AOT manifests. The native links
  have completed; no compiler failure precedes the timeout. This is an
  actual step deadline, unlike the earlier runner-loss/OOM observations.

Raw logs, original worker ledgers and local validation receipts are retained
outside Git under `/data/jzzeng/pr3083-ci-20261118/`. Completed worker time
does not include unreported in-progress calls and is not an isolated benchmark.

## Decision

Transfer complete nuclear-Hessian, nuclear-response and AOT-CMake test modules
to posthf, runtime-heavy and compiler-heavy respectively. Use their existing
loadfile scheduling; start the first two modules before their ordinary queues.
Count each transferred file in its original core modulo slot before excluding
it, so other files retain their owners. Keep every numerical, independent
reference, resource, fault and Python-free executable case. Do not raise the
Python deadlines or replace independent scientific controls with scheduling
tests. The selector regressions exercise every CI event and retain the exact
whole-repository test inventory, coverage and deselection arguments.

Allocate 345 minutes to the wheel build/repair/test step within its unchanged
360-minute job. The remaining 15 minutes cover setup and bounded cleanup:
10 for saving completed compiler outputs, two for run-owned container
diagnostics/cleanup and three for other overhead. This allocation is not a
cold-build timing guarantee. Keep the 12 GiB cgroup, no-swap policy,
whole-graph concurrency one, full architecture selection, optimization flags,
all structural providers and required installed-wheel/payload gates unchanged.

Update both wheel-policy consumers and the stale core-c allowance assertion
with the workflow. Optional cache publication must never mask build failure.

## Rejected alternatives and limits

Do not merely raise core shard deadlines, split shared fixtures among more
workers, drop tests, disable CUDA providers or restore the concurrency that
previously OOMed. Do not report the timed-out original runs as passed. The
wheel reallocation only provides room for the observed final build tail;
hosted CI remains authoritative for complete wheel acceptance.

This supersedes only the build-step allocation portion of
[the earlier wheel resource envelope](2026-11-15-wheel-cold-build-resource-envelope.md).
Its serialization, memory isolation, architecture, scientific and cache
correctness boundaries remain in force. No production numerical source or
public scientific capability changes here.

## Validation

The completed local CI-policy suite passes 1,241 cases in 82.62 seconds with
the workflow-pinned cibuildwheel 4.2.0. It includes executable xdist tests for
the three transferred modules, unchanged ordinary core ownership, all five
events, absent/untracked/tracked/missing inputs, complete inventory and retained
coverage/deselection arguments. Shell syntax, Ruff and whitespace checks pass.
These configuration tests are not a full manylinux build or GPU execution.

The separate pre-integration three-module native run passes 51 cases and
explicitly skips two in 1,171.57 seconds. Its production inputs are the
unchanged CPU inputs from `b320e86e4`; it starts before the CI-only commit
`16b1f26e5` and is not relabelled as a later upstream qualification. The
actual CMake compiler commands invoke verified ccache, with before/after
statistics retained alongside the log.

## Concurrent upstream integration

After pushing the CI fix, GitHub reports the PR unmergeable and does not start
the required PR workflows. The advertised PR base snapshot still names
`f73be1630`, but the actual remote master ref has advanced to
`2791504cc238fe83d328fc1718080642b288222a` (#3109 and #3110).
Check the actual branch ref and freshly fetched commit rather than inferring
current mergeability from that snapshot alone.

Integrate both upstream commits, preserving the native Hessian CMake additions
and upstream one-electron CUDA port/collector changes. The only content
conflict is the derived AOT audit native identity. Compute it from the combined
input inventory as
`d998ad3b1c469e251fe64b36a497f0eca57e92fddf94e315c8b2ac69b0da801a`,
not either parent's identity. Preserve sealed default decisions and original
compile/endpoint evidence, with no numerical promotion or re-labelled receipt.

The rebuilt combined CPU core matches that inventory; its SHA-256 is
`33294b88948027cc363f2a54b183956040f904569121d4b56e6f4e53d18a62ab`.
All 56 selected public native Hessian/ECP contracts and four rebuilt native
CTest groups pass. The compiler structure check covers 615 modules without
dependency errors; the AOT audit retains defaults and historical failures.

The upstream collector test also exposes a helper-import collision with an
installed top-level `tests` package. Import the existing sibling helper directly,
as other repository tests do, instead of depending on an unowned namespace.
Its 20 collector controls pass after that change, including process-group
cleanup. This does not replace or weaken the archived benchmark controls.
