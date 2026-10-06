# Decision: Observe the CuMetal timeout without changing acceptance

Status: implemented
Date: 2026-10-06

## Evidence and boundary

PR #1997 head `de9eebb88be4d7ded840a35541daff2ce43a7a44` selected
CuMetal `e87f368060cf09a45b148b4f8892470c74093ebd`. The exact retained
7,051-byte PTX compiled to MSL in 0.055 seconds, resolving the older pin's
aggregate-parameter rejection. This did not time offline Metal compilation,
pipeline creation, or quantum-chemistry execution.

[Run 37409275173, job 112093833049](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37409275173/job/112093833049)
passed native CUDA runtime CTest in 0.53 seconds but killed the first RHF pytest
after 90 seconds. Pytest FD capture did not survive that kill, and no per-case
XML finished. The logs therefore cannot distinguish JIT cost from a runtime hang
and do not establish a new compiler or numerical defect.

## Decision

Keep the strict 90-second/group, 300-second/suite gate, its per-case Apple-GPU
provenance and numerical comparisons unchanged. Only the first synchronize event
from that exact head, run attempt one, after an actual RHF gate timeout can start
a separate 180-second diagnostic. It uses the same built native library, provider
pin, `cumetal-ir` backend, `fast48` mode, and partially warmed JIT cache. It does
not rebuild, clear caches, retry another backend, or replace acceptance.

The diagnostic runs the identical RHF test with native output uncaptured and
persisted live with relative timestamps. Optional Python markers bracket the CPU
reference, CUDA endpoint, and numerical assertions. Existing upstream registration
markers identify cache lookup, lowering, and compiled-library resolution; GPU
provenance identifies completed dispatches. Process-group snapshots and one
bounded macOS stack sample provide additional evidence when a phase stalls.
Before/after cache hashes and compiler/native-library hashes identify the exact
partially warmed observation. It is not a cold-start performance measurement.
The existing binary32 libdevice limitation remains; this is not IEEE-FP64.

## Cache persistence and integration

Preserve merged master `162b4d88f83860792205a206c84bf04c3057b070` unchanged
outside the existing PR edits. Its explicit cache saves happen after successful
compiler/toolkit/native builds and before scientific qualification, and never
write from merge-group jobs. Keep #1997's exact source/toolchain cache identities.
The earlier failed job had 0/73 provider and 0/411 application compiler-cache
hits; success-only post-job saves had not run. Compiler artifacts may be cached
without claiming scientific qualification.

## Rejected alternatives and revisit condition

Do not raise the gate timeout to obtain a passing badge, remove FD capture from
its per-case provenance gate, or label the latest compiler broken based only on
the timeout. Revisit execution or compilation behavior only after the live phase
and sampled-stack evidence identifies the remaining bottleneck. A successful
diagnostic remains insufficient to qualify the failed required gate.
