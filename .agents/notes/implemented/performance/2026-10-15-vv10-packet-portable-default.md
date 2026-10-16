# Decision: architecture-independent VV10 packet default

Status: implemented
Date: 2026-10-15

## Decision

The user challenged the sm_120 whitelist after the initial default promotion.
The compiler-owned packet kernel uses FP64 arithmetic, double atomics and warp
shuffles; there is no sm_120-specific implementation dependency. Remove the
compute-capability query rather than substituting a different model whitelist.
Availability of the compiled CUDA target remains the build/runtime contract.

This supersedes only the hardware scope of
`2026-10-15-vv10-packet-default.md`. The 65,536-point automatic threshold,
molecular VV10/features/masking admission, numerical bounds, disjoint leases,
ordered fallback and explicit selector overrides remain unchanged. No arithmetic,
precision, pair work, storage or reuse identity changes.

## Evidence and limits

Historical sm_120 campaigns retain their original source/library identities.
They do not qualify later binaries or measure other GPUs. Architecture-independent
selection does not assert equal performance across hardware. The unresolved
96-atom moved-geometry timeout is unaffected. Portable compilation and fresh
source-matched qualification are recorded in the PR follow-up; only scheduled
RTX 5090 execution is available for real-device evidence in this campaign.

The actual production CUDA translation unit compiles for sm_60/70/80/90/100
through verified ccache. A fresh Release rebuild and exclusive finite n4 Slurm
956 allocation qualify native source identity
`9490cd6569f3e403d09f5f832b4c9826f5581172696ab0d4833257db26cf96c1`:
224 host tests (including 42 policy cases), 752 device admission/oracle cases,
47 lifecycle/runtime cases, six memcheck and six synccheck boundary cases with
zero errors, six independent complete molecular E/F/finite-difference cases,
and all 24 original-protocol 3/12-atom calls with the selector absent. The latter
are compared offline against retained exact-protocol independent references;
raw native records are not rewritten. A compact source/binary/log receipt is
`benchmarks/results/vv10-packet-default-20261015/portable-qualification.json`.
Full raw outputs remain in ignored local artifacts, not a new tracked archive;
the unchanged evidence budget and historical campaigns are preserved.

## Revisit when

Representative complete endpoints regress or numerical/resource gates fail on
a supported target. Keep explicit `ordered` and previous `parallel` selection
for diagnosis; do not restore a model whitelist solely because timings were
first collected on that model.
