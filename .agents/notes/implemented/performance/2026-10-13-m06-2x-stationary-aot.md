# Decision: package the exact M06-2X stationary force plans

Status: implemented; endpoint qualification pending
Date: 2026-10-13

## Problem

Issue #2418's 3-atom M06-2X cold endpoint records 226.157 s in force-provider
preparation. A fresh-process diagnostic with the original library reproduced a
91.50-s compilation interval, including 24 object builds and linking. Subsequent
3- and 12-atom processes reused the same artifact without compiler subprocesses.
The old observation is retained; the diagnostic does not reconstruct its exact
226.157-s internal breakdown.

## Decision

Add RKS/UKS M06-2X to the explicit stationary AOT package catalog. Resolve its
point code from the generated split-hybrid registry, and select by the exact
stationary MethodIR plan identity. Reuse the existing s/p and s/p/d primitive
shards, wrappers, precision contract and integrity-checked loader. MN15 remains
an unrepresented plan with its existing bounded cached JIT fallback.

## Invariants and alternatives

Preserve Libxc-derived 27/50 exact exchange, all stationary derivative sources,
FP64/fmad policy and complete E/F checks. Missing or corrupt declared packages
still fail closed. Do not manufacture cache hits, clear shared compiler caches,
exclude compilation from a measured cold call, or relabel the old cold result.
Packaging removes repeated deployment-time work; compiler-cache hits alone do
not remove first-use source generation, compiler startup and linking.

## Evidence and limits

Host admission covers the exact split-hybrid profiles and corrupt/mismatched
packages without invoking a compiler. Real-device first-process execution must
show the packaged route, no runtime compiler calls, and independent energy/force
agreement before the updated cold result is published.
