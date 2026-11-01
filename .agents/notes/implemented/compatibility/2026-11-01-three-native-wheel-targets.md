# Decision: three native CUDA wheel targets with a compute-80 PTX fallback

Status: implemented
Date: 2026-11-01

## Decision

The owner selected native SM 80, 90 and 120 instead of six native targets to
reduce build and artifact cost. CMake `80;90-real;120-real` additionally emits
generic compute-80 PTX for forward JIT compatibility. This supersedes the
[multiarchitecture matrix](2026-10-27-multiarch-wheel-qualification.md), not its
frozen measurements. The active six-target run is preserved, not cancelled.

SM 80 cubins can cover compatible 8.x devices; other newer architectures may
use PTX with a capable driver and compatible libraries. Neither route qualifies
unmeasured scientific endpoints or guarantees tuned performance. SM 70/75 are
excluded. PTX JIT can add first-use latency and requires driver support for the
emitted PTX version, not merely the advertised compute capability.

## Invariants and evidence

Keep portable profiles plus the measured SM 120 profile, per-manifest code-object
inventories, independent source verification, binary integrity, smoke/linkage
checks and the existing finite build/cache budgets. Snapshot identity changes
with the exact matrix; source-cache identity and correctness checks are retained.
No numerical implementation, precision, convergence or acceptance gate changes.
Host policy/manifest tests exercise the exact native/PTX inventory and reject the
superseded matrix as a current snapshot. Actual reduced-matrix wheel results and
cold/warm recurring cost must be qualified separately; no speedup is claimed here.

## References

- PR #2832; owner direction: native 80/90/120.
- https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html

## Master integration and completed historical wheel receipts

The `b5ad34e9` three-target wheel passed run `41295942074`: build/test
253m12s, 107/639 compiler cache hits, approximately 2.1 GB wheel. The preceding
six-target `1ac31a2f` run `41279261469` passed in 306m11s with 358/675 hits
and approximately 3.8 GB wheel. These are different source/cache populations,
not a controlled speedup or a cold-build bound.

Integration with master `19f5ddb3` retains architecture-neutral release presets
and full automatic Libxc CUDA AOT from #2922, alongside this PR's matrix,
compute80 PTX, host-link layout, per-manifest validation and finite budgets.
The historical wheels do not qualify that combined full-catalog build.

The catalog previous-summary ratchet advances to the exact `19f5ddb3` snapshot
summary. Its obsolete identity-transition acknowledgements remain in that
master commit's history, not as current transitions. The new summary, retained
execution/history/endpoint evidence and stale state are unchanged; only live
source identities and baseline-relative transition reporting are refreshed.
No new GPU numerical qualification or capability promotion is claimed.
