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
