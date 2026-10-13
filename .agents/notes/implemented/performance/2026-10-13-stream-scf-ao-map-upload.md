# Decision: upload SCF AO maps one tile at a time

Status: implemented; capacity fix qualified, full 96-atom latency unresolved
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

## Final capacity qualification and remaining latency

Integrated source `227c55c06` with native library SHA256
`bd6dd7b4e132926ac2839edd19ca96409699906668c58c121df3c0d4f29cdb4a`
passes the complete direct r2SCAN-3c population at 3/12/48 atoms. At 48 atoms,
cold is 86.302 s and the five-call warm median is 26.070 s. The 96-atom cold
call now returns in 500.751 s, with no force-grid capacity fallback, and the
first 11 returned calls all pass the independent 1e-8 Eh / 1e-7 Eh/Bohr gates.
The original 1800-s whole-point limit interrupts the last moved-warm replay;
this fixes the diagnosed storage failure but does not close #2416's complete
population acceptance. Its five warm calls have a 105.539-s median, including
approximately 77.8 s in integral derivatives per call.

At 96 atoms the selected SCF AO work is 164,382,040,064 point-AO-squared visits
versus the dense bound 2,792,802,484,224; observed host map peak is 160,520 bytes.
The force owner reports a 263,387,808-byte numeric host bound within 268,435,456
bytes and a 480,267,904-byte device bound within 536,870,912 bytes. No cap,
quadrature or precision gate was increased. The initial capacity failures and
all incomplete populations remain in the separate README benchmark follow-up.

The newly scoped PR head passes 203 host tests (8 CUDA cases explicitly
skipped), compiler structure and CUDA ownership checks. Earlier 8-case real-GPU
grid qualification used the unchanged native AO provider plus the generated
feature-grid implementation; the complete endpoints above use the final native
library. These source boundaries are retained rather than relabeling GPU
measurements as the rebased PR head.
