# Decision: isolate cold wheel compilation from its hosted runner

Status: implemented
Date: 2026-11-15

## Problem and evidence

Master wheel runs 43580937901 and 43697904960 failed with GitHub's annotation
"The hosted runner lost communication with the server." Neither reached wheel
repair, smoke tests, or payload validation. Their compiler-cache, semantic-source,
and cibuildwheel-tool restores all missed; the trusted successful-master snapshot
was unavailable. Loss of the runner also skipped the existing `always()` partial
compiler-cache save, so successive attempts repeated expensive cold work.

Run 43697904960 spent 248.6 minutes in the build step and 293.1 minutes in the
complete job, including lost-runner detection. Run 43580937901 also queued for
about four hours (created 09:31:18 UTC, job started 13:32:26 UTC on November 14)
behind another in-flight qualification; preserving in-flight work amplifies wall
time when previous cold builds lose their runners. Keep that concurrency policy.
Ninja in run 43697904960 started its 1122-task
graph at 09:13:34 UTC and last reported task 536 at 12:54:14 UTC. The 50 native
stationary object completions span 09:45:30–10:58:02, and the 33 automatic Libxc
CUDA object completions span 11:05:52–11:42:52. Those are overlapping completion
windows, not isolated target timings or CPU-time measurements. The final reported
task compiled the host UCCSD force component; CUDA-only job pools would not bound
the whole workload. The disconnect annotation alone does **not** establish OOM
as the cause; no memory, disk, or kernel-OOM telemetry survived those runs.

For comparison, successful run 43163376904 spent 182.1 minutes in build/test,
reported 466 hits and 383 misses across 849 cacheable compiler invocations, and
used only 1.0 GB of the existing 3.0 GB compiler-cache budget. Raising cache size
does not address these observed restore misses. Run 43154394256 completed its
build/test step in 23.9 minutes. These are historical endpoint observations, not
a speedup measurement for this change. The complete job durations for those two
successful runs were 186.1 and 26.6 minutes, respectively.

## Decision

- Set `CMAKE_BUILD_PARALLEL_LEVEL=2` **inside** the cibuildwheel environment, so
  the PEP 517 backend's real `cmake --build` bounds host C++, CUDA, code generation,
  and linking rather than just one CUDA target family. Do not change production
  optimization, numerical flags, method inventory, or the full architecture set.
- Cap the Docker build cgroup at 12 GiB with an equal memory-plus-swap limit.
  The standard public Ubuntu runner has 16 GiB: reserve host capacity for the
  Actions runner, Docker, and cache retention. A container-local allocation
  failure must remain a wheel failure, not permission to skip acceptance gates.
- Keep cibuildwheel containers only until a two-minute, best-effort diagnostic
  step reports Docker state (including `OOMKilled`) and configured memory limits,
  then removes only containers labelled for this run and attempt. `OOMKilled`
  describes container state, not a complete history of every child compiler's
  OOM events. Report host memory/disk state too. Keep the cache bind mount and
  existing failure-path cache save unchanged.
- Print ccache statistics before compilation as well as after successful repair.
  A restored Actions cache is not proof that individual compilations hit.

## Rejected alternatives and invariants

Do not reduce architectures, disable automatic XC/force providers, use fast
compiler numerical modes, cancel in-flight qualifications, or extend timeouts
again. The observed failure is a lost runner, not the configured 330-minute
build-step deadline. PR #3094 independently extends trusted snapshot retention
from one to seven days; leave that change to its existing owner.

## Validation and revisit conditions

104 related tests pass with the workflow-pinned cibuildwheel 4.2.0. Executable
tests parse the actual cibuildwheel engine/environment options, run
a small compiler-free CMake/Ninja graph to observe the resulting parallelism,
and inject failed Docker inspection to check run-owned cleanup. No local GPU
test or full manylinux build is claimed. The PR's unchanged full manylinux
build/repair/installed-wheel/payload gates remain the endpoint acceptance test.
Treat concurrency two as a conservative initial resource policy, not a measured
optimum; revisit it with complete cold/warm runs and surviving resource evidence.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43580937901
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43697904960
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43163376904
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/43154394256
- https://github.com/jinzhezenggroup/generativeqc/pull/3094
- Supersedes the unbounded-parallelism choice in
  `2026-10-10-wheel-cold-build-cache-budget.md`; retain its cache-save contract.
