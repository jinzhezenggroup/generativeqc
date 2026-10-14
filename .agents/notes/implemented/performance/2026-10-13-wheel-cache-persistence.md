# Decision: bounded wheel compiler snapshot fallback

Status: implemented
Date: 2026-10-13

## Problem

The ordinary manylinux wheel ccache key/format works, but its availability is
not durable. Master run 38385680933 saved a 1,672,926,294-byte Actions cache;
38390174415 missed all compatible v3/v2 prefixes, and its preceding snapshot
was again absent to 38397514912. These observations do not establish whether
quota eviction, deletion, or a cache-service failure caused the losses.
Run 38407263167 subsequently restored a 412,102,984-byte ordinary snapshot,
reported 502/508 compiler hits and completed build/test in 20m13s. The prior
78m03s cold build was a different source/run population, not a controlled
speed comparison. Another key revision has no supporting diagnosis.

## Decision

Keep the normal Actions cache and add one optional wheel consumer/producer.
Reuse `ci_cpu_ccache_snapshot.py`'s trusted-run discovery, streaming bounded
USTAR verifier, checksums and absent-target atomic activation through a closed
`--profile wheel`. The default CPU profile, identity, limits and workflow/action
remain unchanged. Do not reuse the unrelated Ubuntu CUDA cache.

Wheel snapshots bind the canonical repository, exact wheels workflow,
successful completed master push, source commit, run and attempt, artifact and
payload digests, Ubuntu24/X64 host, pinned manylinux image, CUDA12.9/sm120 and
ccache4.14. A small receipt outside the cache records the container's actual
ccache/NVCC versions and effective CUDA flags; missing, incompatible or
wrong-attempt receipts disable publication without failing the wheel build.
The consumer's expected image is validated against the checked-out pyproject.

Only a successful normal restore with no matched key may enter the optional
two-minute fallback. Discovery has 25 seconds, verification 45 seconds and the
whole composite two minutes. Package/upload each have two minutes. The limits
are 2 GiB artifact transport (archive cap reserves 1 MiB for the manifest/ZIP),
3 GiB decoded compiler data and 100,000 files. CPU's 511/512 MiB encoded,
2 GiB decoded and 100,000-file limits are not enlarged. All unsupported,
missing, oversize, corrupt, interrupted or timed-out snapshots leave the normal
cold path available. Config, locks, stats, logs, symlinks and special files are
never imported. Staging and activation stay next to runner-temp/ccache on the
same filesystem, without reading or changing HOME's cache/config.

Publish only after existing wheel build/smoke/payload and wheel upload steps.
Consumers require the entire producer workflow to have succeeded; an artifact
from an in-progress or failed workflow is not eligible. One-day retention and
zero additional ZIP compression bound duplication. Refresh follows the existing
CUDA snapshot's three-hour/input-aware policy over wheel build/toolchain inputs.
Discovery remains bounded to three eligible candidates from the latest twenty
successful runs. If a skipped publication falls outside that window, refresh
conservatively publishes again; this can produce more than one snapshot within
three hours, but avoids unbounded API searches and false durability claims.
No new deletion/eviction job, shard fanout or workflow dispatch is introduced.

## Evidence and limits

Executable CPU and wheel tests cover discovery provenance, toolchain receipts,
archive/type/digest/size rejection, same-filesystem atomic activation, attempt
ownership, interruption, configuration exclusion, host ownership of restored
files, required workflow gates and conservative refresh decisions. Small
incompressible payloads exercise the actual streaming archive path. Final
focused coverage: 141 passed (77 unchanged CPU tests and 64 wheel cases);
the complete `test_ci_*.py` suite passed 1,458 tests plus 42 subtests, with two
unrelated skips. Independent source/security review found no blocking issues.
A parsed-YAML removal proof restores all previous wheel workflow semantics
when only the new optional integration is removed.

A bounded local fixture used 8 MiB random data plus 1,000 small read-only files:
1,016 files, 8,452,608 decoded bytes, 8,480,583 archive bytes. Packaging took
0.310 s and verification 0.061 s on this executor. Activation preserved the
verified directory inode; restored files belonged to runner UID 1000 and the
original read-only source remained untouched. These are small-fixture costs,
not an extrapolated GiB-scale transfer or hosted speedup.

The historical Actions/cache transport size is not this gzip/USTAR artifact's
size. No production cache was downloaded or synthesized at GiB scale locally.
Actual production size and optional transfer/verification throughput remain
natural hosted qualification evidence, enforced by the runtime limits rather
than assumed from that historical measurement. No full build, GPU campaign,
CI cancellation, cache clearing or numerical-test relaxation is part of this
repair.

## Revisit when

Revisit wheel limits only with natural size/timeout evidence and available
runner/storage headroom. A new image, toolchain or archive format needs explicit
compatibility qualification. Do not expand CPU bounds to accommodate wheel data.

## References

- Shared CPU snapshot implementation: PR #2435
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38385680933
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38390174415
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38397514912
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/38407263167
