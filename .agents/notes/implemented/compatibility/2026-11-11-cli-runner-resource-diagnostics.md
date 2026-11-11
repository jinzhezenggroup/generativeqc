# Decision: retain live host evidence around native CLI builds

Status: implemented
Date: 2026-11-11

## Problem

PR #3057's native CLI job lost communication with its GitHub-hosted runner.
Earlier failures included runner-shutdown exit 143 at varying compilation points.
These observations do not establish OOM, disk exhaustion, or compiler failure.
Post-build summaries cannot run after runner loss.

## Decision

Wrap the unchanged native CLI build argv with a standard-library-only sampler.
Flush allowlisted memory/swap, CPU/memory pressure, workspace disk availability,
and aggregate host compiler RSS to the live job log every 30 seconds. Limit the
sampler to 512 records, 4096 bytes per record, and 4096 proc entries per sample.
Compiler RSS is host-wide, selected by fixed compiler names; it is not attribution
to this build, and inaccessible processes or a truncated scan are reported.
Never read process command lines or environments, or emit paths or process names.

The sampler is a daemon thread inside the command wrapper. Stop it after the
owned build child exits, with at most a one-second join. Optional sampling,
output, initialization, and cleanup failures cannot replace the build outcome.
Write records directly to the existing stdout descriptor without changing its
blocking mode; a stalled log pipe must not hold a Python buffered-stream lock
during interpreter shutdown. Setup cancellation must not start a new build.
Forward SIGINT/SIGTERM only to the owned child; retain the runner's existing
process group so normal runner cancellation still owns descendant cleanup. Do
not create an isolated build session or kill a group/shared compiler-cache server.

## Invariants and rejected alternatives

Preserve the existing compile tier, flags, cache policy, parallelism, timeouts,
and qualification gates. A successful FAST_COMPILE=ON run with high cache reuse
is not qualification for the release OFF build. Do not infer that lowering
parallelism or replacing release compilation will fix an unexplained runner loss.

## Evidence and limits

`tests/python/test_ci_ninja_timing.py` executes short commands to verify live
flushes, successful/failed/signaled status, unrelated-process survival, missing
metrics, diagnostic failures, cleanup, and bounded logs/scans. It also executes
the workflow build script with dummy cmake. No full build or GPU run is claimed.
Samples can stop before the actual failure, or be absent if output/metrics fail.
They neither fix runner loss nor prove its cause. Revisit resource limits only
when actual measurements identify pressure and equivalent qualification exists.

Reference: https://github.com/jinzhezenggroup/generativeqc/actions/runs/43009841638/job/130486436715

## Measured-pressure follow-up: 2026-11-11

The initial decision above preserved parallelism while collecting evidence. The
next release-tier attempt at source
`d9504e65b4403812b94c80c3f6136bf596318855` retained `FAST_COMPILE=OFF`, global
parallelism 3, and bounded CUDA/AOT pools of 3/2. It was not an unbounded build.
Job 130530443024 ended with a runner-shutdown message at 13:03:17 UTC after
emitting 31 host-resource samples. Available memory fell from 9,079,416 KiB
(8.66 GiB) at 13:01:33 to 798,992 KiB (0.76 GiB) at 13:02:03. The last sample,
at 13:03:04, recorded:

- 484,148 KiB (473 MiB) available out of 16,366,512 KiB physical memory;
- 14.37 GiB aggregate host compiler RSS, with a maximum of 11.76 GiB for one
  sampled compiler process;
- 707,480 KiB (691 MiB) swap in use;
- memory-pressure `avg10` of 21.81% `some` and 21.33% `full`; and
- 75.48 GiB workspace disk space available.

These measurements establish severe memory pressure before the shutdown, not an
explicit OOM kill or the shutdown's definitive cause. Compiler RSS is host-wide
and can double-count shared pages; it does not identify an offending build edge.
The last completed Ninja edge, 453/767, likewise does not identify which of the
concurrent commands was responsible. There is no observed disk-exhaustion signal.

As a controlled mitigation, reduce only this native CLI build invocation's
global parallelism from 3 to 2. The explicit command-line cap overrides an
inherited `CMAKE_BUILD_PARALLEL_LEVEL`; the existing per-pool limits remain
unchanged beneath it. Keep the full release configuration, flags, scientific
inputs and numerical gates, tests, compiler caching, job timeout, fast seed job,
other jobs, and artifact/publication behavior unchanged. Retain the same bounded
telemetry to evaluate the next equivalent release attempt.

`tests/python/test_ci_cli_ccache.py` executes the actual workflow shell and
diagnostic wrapper with recording cmake/ccache substitutes for both ON/OFF tiers,
with empty and larger inherited parallelism defaults. It checks the explicit cap
of 2, strict cache configuration, and unchanged configure/build failure status.
This is workflow-contract coverage, not a native build or GPU qualification.

Two jobs may still exceed this runner's memory, and lower parallelism may increase
wall time. A passing fast-tier/high-cache-hit build does not qualify the release
workload. Do not claim the mitigation fixes runner loss until the equivalent
release build and required gates complete. Revisit the cap or investigate
individual translation-unit memory only with further measured evidence; preserve
the failed attempts and do not relax release flags or numerical gates.

Reference: https://github.com/jinzhezenggroup/generativeqc/actions/runs/43022943858/job/130530443024
