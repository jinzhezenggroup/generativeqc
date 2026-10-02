# Proposal: prevalidate molecular VV10 pair and row bounds in linear work

Status: experimental; complete endpoint comparison pending
Date: 2026-10-03

## Decision and bounds

The bounded pair closure checked omega/kappa and distance inside every pair.
The factored row additionally checked partner weights and retained an ordered
retry. For molecular VV10 with features, at most 2^32 points and coefficient
magnitude at most 2^32, first scan the observable rows once on the device.
Each non-screened row must have coordinates of magnitude <=2^14, omega/kappa
in [2^-32,2^32], weighted density magnitude <=2^64, and density/local derivative
magnitudes <=2^128. These sufficient bounds imply every squared separation is
at most 3*2^30 (strictly inside 2^32 even after FP64 rounding), and imply the
existing pair and row arithmetic bounds. Positive-zero active rows remain live.

A device predicate selects between compiled prevalidated and general kernels.
Both launches read that predicate, but exactly one executes pair traversals.
The compiler-owned generated admitted entry point contains the same scalar
program as the guarded closure. The prevalidated specialization removes bounds
and row-retry branches from the pair loop. Any unsupported row retains the
existing guarded fallback and ordered exceptional behavior; there is
no new screen, altered pair order, host scientific work or host synchronization.
Absolute coordinate bounds deliberately reject far translations safely rather
than relying on cancellation-sensitive computed extrema.

## Lifetime and work

The stable partner scatter is enqueued before the preflight. Its block-offset
array is then dead, so the predicate borrows the first integer word of that
array on the same stream. At least one slot exists; reset precedes every scan.
No extra allocation, cross-invocation identity, or workspace-capacity change is
introduced. The active count and partner indices occupy separate slots.

Scheduled extra work is one O(N) admission scan and one quickly rejected row
launch. The selected pair traversal count is unchanged on admitted grids.
Unsupported rows retain the previous at-most-two-traversal fallback; dense grid
squared remains a capacity rather than an observed molecular executed count.

## Evidence and limits

130 host tests pass, including exact output and observed pair-visit equality
with admission enabled/disabled for SCF and geometry demand, signed/zero weights,
screened densities, admitted coordinate edges, far translations, oversized
weights/densities and unsupported coefficients. 74 real-device boundary,
canary, signed-domain and failure tests pass on node1 (2.72 s). Complete molecular
qualification and a same-allocation full24 comparison follow. No endpoint speedup
has yet been established.

The candidate library is
`33211b5f0a26537e97faa471a3c6551217f0238a8c677ae23c345e19237b5c39`,
parent `a440afd27` plus recorded experimental row/unit-reciprocal/full-LR patches.
It includes the separately reviewed both-zero exchange fix, which does not affect
the nonzero WB97M-V coefficients. Source archive, patch, tests, ccache/compiler
receipts and exact library hash live under ignored `.artifacts/wb97m-admission/`.
The first remote launch raced source extraction and failed before collecting
tests; its receipt is retained and the completed copy was verified before rerun.

The earlier row-factor bundle slightly regressed on full24. This experiment is
only a candidate to remove its admission overhead, not permission to promote
a slower isolated optimization. All complete samples still require 1e-8 Eh and
1e-7 Eh/Bohr gates plus cold/warm, changed-geometry and larger qualification.

## Review implementation narrows the arithmetic admission

The review branch deliberately keeps the existing #1729 general one-division energy
root and per-pair feature chain for every grid that fails the linear preflight,
and for unmasked/primitive consumers. Only the admitted molecular specialization
selects unit-reciprocal energy and factored row features. It therefore has one
pair pass on both paths, removing the unsuccessful standalone row-factor retry
policy from the production candidate. The original exploratory binary above
remains a separate experiment and does not qualify this revised dispatch.

The review implementation has 132 passing host tests. Both generated scalar
policies receive independent 90-digit energy/partial finite differences; admitted
row rho/gradient energy differences exercise the selected policy, and unsupported
grids preserve exact original outputs and pair-visit counts. It is being built
as a standalone change above #1729 without #1736's full/LR source experiment.
Its final native hash and real-device/complete evidence will be recorded separately.

The standalone review native library is
`834344864bf03fe90319f58c9730aba9bc35b353113fe9cc0bf40d65db20d338`.
It is built with verified CXX/CUDA ccache launchers and checkout-root CCACHE_BASEDIR.
Unlike the exploratory `33211...` binary, it contains no full/LR derivative
change. Its real-device and complete endpoint qualification is pending.
