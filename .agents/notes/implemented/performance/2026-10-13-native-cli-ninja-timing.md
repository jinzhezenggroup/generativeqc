# Decision: bounded completed-command timing for the native CLI

Status: implemented
Date: 2026-10-13

## Problem

The [#2429 queue CLI build](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38389026212/job/115267917680)
took 67m51s with 358 compiler-cache hits and 189 misses; the compared
[2fa9b891 master build](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38385680958/job/115270490016)
took 32m16s despite 512/547 hits. These are different source/build populations, not a controlled
performance comparison. Compiler-cache counters and completion timestamps do
not establish the slow command durations or the dependency critical path.
Successful CLI artifacts retain only the distribution and checksums, not a
Ninja timing log. The successful queue run 38381471126 has the same artifact
shape. More compiler optimizations require better naturally occurring evidence.

## Decision and boundaries

Read the existing CLI build's `.ninja_log` after its required configure/build
step and summarize it in the job log and GitHub step summary. There is no new
native build, workflow dispatch, raw-log upload, or change to compiler flags,
parallelism, cache identity, generated inputs, or qualification gates. Both the
master artifact job and PR/queue qualification job share these steps.

Ninja logs append successive invocations and may compact away older records.
They have no invocation IDs. Do not infer a current invocation by sorting times,
looking for the last timestamp reset, or treating file mtimes as command time.
A separate optional step verifies that the CLI log is absent before the existing
configure/build step. The CLI tree is `build/cuda-release-cli-sm120`, separate
from the master fast seed's `build/cuda-release-sm120`; the workflow invokes
exactly one explicit build of the CLI tree. An existing log, failed freshness
check, detected timestamp reset, repeated output, noncontiguous repeated timing
group, malformed/truncated log, or unsupported version makes timing unavailable.
No diagnostic removes or resets Ninja state. If the build lifecycle starts
reusing a tree or invoking Ninja repeatedly, introduce an explicit stronger
boundary before allowing historical logs; timestamps alone cannot prove it.

Adjacent output rows with equal command hash/start/end are counted once, as
Ninja records a multi-output edge that way. Indistinguishable equal-time/hash
commands coalesce, so report *command timing groups*, not an exact compiler-call
count. This includes cache hits, code generation and link commands. Only logged
completed commands are represented, including after a failed build; uncompleted
work is absent. The cumulative duration includes parallel overlap. The first
start-to-last-completion timestamp span is separately labeled; neither it nor
the cumulative duration is the workflow wall time or dependency critical path.
The GitHub build-step duration remains the configure/build wall-time source.

## Resource and disclosure limits

Both freshness and reporting steps are non-gating and have one-minute ceilings.
The parser accepts v5/v6/v7 logs with the documented five fields, at most 16 MiB,
50,000 records, 4 KiB per row and one day per recorded timestamp. Only regular
files are read; symlink paths and FIFOs are refused. A changed file is refused.
Output is capped at 24 KiB and shows category counts/durations and the longest
20 groups. Names are limited to 180-character allowlisted relative native build
outputs; absolute aliases and other names are counted but redacted. No command
hashes, mtimes, commands, environment, memory dumps or raw exceptions are printed.

## Verification and limits

Focused executable tests cover parallel versus cumulative timing, multi-output
aliases, malformed/truncated/oversized logs, historical logs without a boundary,
reset/repeated-output ambiguity, redaction, I/O failure, actual workflow shell
execution with successful/failed build fixtures, optional step bounds and
shared workflow ownership. Existing build/configure and required-gate steps are
compared unchanged as a one-off review receipt, not a permanent source pin.
Hosted command distributions and any future speedup remain unmeasured until
normal CI reaches this diagnostic. No numerical/GPU qualification is claimed.

References: [Ninja build log implementation](https://github.com/ninja-build/ninja/blob/master/src/build_log.cc)
(`RecordCommand`, `WriteEntry`, `Recompact`) and the
[Ninja manual](https://ninja-build.org/manual.html).
