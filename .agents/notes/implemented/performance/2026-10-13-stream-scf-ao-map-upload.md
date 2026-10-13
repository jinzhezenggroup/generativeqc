# Decision: upload SCF AO maps one tile at a time

Status: implemented; endpoint qualification pending
Date: 2026-10-13

## Problem

Issue #2416's r2SCAN-3c/96 point does not complete its cold endpoint within
1800 seconds. An n1 Nsight sample attributes 30.185 s to dense density products
and 28.007 s to dense potential products, about 69% of sampled GPU kernel time.
Discovery's conservative all-tile host index vector exceeds the fixed 64-MiB
cap at this shape, excluding a capability-qualified local-AO executor.

## Decision

Keep the worst-case complete device map reservation, but stage only one sorted
index tile on the host. Each subsequent discovery drains the same stream before
reusing that tile's host memory. An exception guard drains outstanding uploads
before releasing the vector. Publish the selected layout only after every tile
and the final upload succeed. The persistent host owner still contains offsets
and density launchers, and the device map retains exactly the same indices.

The layout arithmetic is shared by externally supplied complete maps and the
trusted incremental producer. Complete maps retain their ordering/range checks;
the incremental producer establishes those invariants with its increasing AO loop.

## Invariants

No cutoff, derivative capability, FP64 policy, device bound or 64-MiB host cap
changes. Public resource ledgers retain their existing dense fallback because
they do not yet reserve optional maps. Device allocation failure also keeps its
dense fallback. Semantic AO counts and uploaded bytes remain observable.

## Evidence and limits

The original diagnostic source/library are retained with the #2401 campaign.
Host tests exercise the large-grid admission, genuine host-cap rejection,
precision/capability exclusions, owner replacement and failed discovery.
Source-matched GPU independent energy/force and complete endpoint tests are
required before claiming a speedup.

This refines the host lifetime assumed in
[the original discovery decision](2026-10-03-native-scf-ao-discovery.md) and
[default reconciliation](2026-10-04-generic-scf-local-ao-default-reconciliation.md).
Raising the host cap alone was rejected: it retains a redundant full CPU copy
and merely moves the next size cliff.
