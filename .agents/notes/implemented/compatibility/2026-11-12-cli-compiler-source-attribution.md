# Decision: attribute live CLI compiler memory to public source units

Status: implemented
Date: 2026-11-12

## Evidence and limits

The release CLI at source `5c02fc1e55424fe6e41b2fd251a77533b269114e`, job
130906981436, used NVCC 12.9.86 with GNU 13.3.0, `FAST_COMPILE=OFF` and the
existing global parallelism cap of 2. The last sample at 01:45:31 UTC recorded
277,116 KiB available out of 16,367,528 KiB physical memory, 1.44 GiB swap used,
14.51 GiB aggregate compiler RSS and 14.38 GiB in one compiler process. Memory
PSI avg10 was 58.82% some / 58.40% full; disk still had 75.90 GiB available.
The runner reported shutdown at 01:45:51. No explicit kernel OOM record appears.

This establishes severe host memory pressure, not a definitive shutdown cause
or offending translation unit. Aggregate RSS can double-count shared pages.
Ninja reports completed commands: the last reported `df_derivatives.cu.o`
(428/767) is not the identity of the remaining large process. The earlier
9299ccf3 attempt lost runner communication while its 22 retained samples had at
least 12.12 GiB available and zero memory PSI; do not conflate the two attempts.

One process nearly exhausts this runner's RAM. Another blind global change from
2 to 1 is not an established solution. The retained logs do not justify changing
a guessed source unit, optimizer settings, release coverage or numerical gates.

## Decision

Extend the [original host-resource diagnostics](2026-11-11-cli-runner-resource-diagnostics.md)
with optional, CI-only source attribution for this native CLI invocation. Use
ccache's documented `CCACHE_PREFIX` and `CCACHE_PREFIX_CPP` hooks; leave the real
compiler, CMake launchers, compiler options, cache identity and storage policy
unchanged. Cache hits retain their existing behavior. The prefix records only
one validated public repository-relative source path and numeric process start
ticks in an atomically published PID file, then execs the unchanged command.
Restore the signals Python ignores at startup before native exec, matching
normal subprocess signal restoration. Diagnostic write failures cannot skip
compilation or change its exit status. The optional writer is a daemon thread
with a one-second join bound; exec ends an interruptibly stalled writer. This is
best-effort bounded setup, not an absolute kernel-I/O guarantee: Linux exec can
still wait for a sibling stuck in uninterruptible I/O.

The live sampler matches numeric parent/start-time identities to these records.
It emits at most four largest compiler processes, each with PID, an existing
fixed-list compiler name, RSS, and a validated source path or null. Only existing
sources below `src/` or `build/cuda-release-cli/generated/` are eligible. Reject
outside-root/symlink escapes, ambiguous inputs, unexpected names and overlong
paths. The prefix may inspect its own compiler arguments to select that source,
but never writes those arguments or environment values. Neither component reads
`/proc/*/cmdline` or `/proc/*/environ`. Raw build commands, definitions, arbitrary
process names and absolute paths remain excluded.

The original 512-sample, 4096-byte record and 4096-process scan bounds remain.
Ancestry traversal stops after 16 entries. Start-time checks reject stale PID
records. Process-tree sampling is best-effort, not atomic; exited, inaccessible,
unrecognized or ambiguous sources remain unattributed. Prefix records are local
temporary diagnostics and are not new uploaded artifacts. The sampler, rather
than compiler stdout, emits attribution because Ninja buffers command output.

## Invariants and validation

Keep the release tier, global cap 2, CUDA/AOT pool limits, generated arithmetic,
source identity, cache configuration, timeouts, tests, numerical gates, fast seed
job, other jobs and publication behavior unchanged. This addition gathers the
missing evidence; it is not a memory-pressure fix or release qualification.

Executable tests cover live descendant attribution after exec, numeric identity
and stale-PID rejection, privacy boundaries, bounded output/ancestry, atomic
publication, optional recording failures, argv/environment/I/O preservation and
success/nonzero/TERM/PIPE/file-limit signal outcomes. A bounded real host compile
through verified ccache checks hits/misses and identical objects with both prefix
hooks enabled or disabled. These are diagnostic/cache contracts, not CUDA or
complete native CLI qualification. Existing wrapper cancellation and blocked
output tests remain required.

## Revisit when

An equivalent release attempt identifies the live compiler stage and translation
unit. Then reproduce that exact unit/flags under explicit resource bounds and a
verified compiler cache before choosing a source partitioning or toolchain
remedy. Do not change runtime numerics or switch to FAST_COMPILE to hide the
failure. A larger or paid runner and materially wider toolchain changes require
separate authorization; no such change is included here.

Reference: https://github.com/jinzhezenggroup/generativeqc/actions/runs/43134646774/job/130906981436
