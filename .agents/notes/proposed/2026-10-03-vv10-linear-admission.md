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

## Final standalone dispatch: completed device and moved qualification

The exact `834344...` review binary passes all 74 real-GPU boundary/canary/failure
cases (3.31 s), and all 74 under memcheck (3.75 s) and synccheck (3.45 s), each
with zero reported sanitizer errors. Node1 job 5467 passes seven complete
independent molecular/rebuild/stale-state tests in 199.58 s. These results
qualify the final one-pass fallback dispatch, not the earlier replay experiment.

A separate node1 allocation evaluates diagnostic water-12/def2-TZVP on the
24x8x16 grid and checks every cold/priming/warm/moved sample against the retained
independent full CPU oracle. Times are 166.792467 / 23.059597 / 23.080774 /
91.935848 s with 22/1/1/11 SCF iterations. Maximum errors are 1.705303e-12 Eh
and 8.977530e-10 Eh/Bohr; fixed-final-state force replay also passes. This
moved-geometry check is an accuracy gate, not a default-grid performance claim.
Raw receipts are retained in `.artifacts/wb97m-admission-review/results/`.

## Historical exploratory matched endpoint

Node1 job 5464 compared the row-factor `33b396...` binary with exploratory
admission `33211...` sequentially on one RTX5090/eight CPUs. Full24 warm medians
were 47.622777 -> 41.552371 s (1.1461x); cold 329.610090 -> 273.628069 s.
Both cold solves used 18 iterations and all warm solves one iteration. Every
one of five pairs passes, candidate maximum errors 2.694378e-11 Eh and
4.165308e-10 Eh/Bohr. GPU4PySCF warm was 28.319312 s. The unchanged traversal
policy retains the same logical work; actual compacted molecular pair counts
remain unavailable. This bundle includes full/LR derivatives and the exploratory
general replay policy, so these timings do not qualify the final review binary.

Node5 job 1392 completed the earlier row-only larger experiment: default48
cold 1425.892841 s, warm 180.852592 s versus GPU4PySCF 176.586697 s. All five
pairs pass, maximum energy 3.024070e-11 Eh and force 5.888827e-10 Eh/Bohr.
This confirms that row factoring alone has not established an endpoint advantage.

## Final standalone controlled full24 endpoint

Node1 job 5467 ran final `834344...` against #1729's `adb4b086...` sequentially
on the same RTX5090/eight CPUs. Both use master `8ecc0391f` geometry scheduling.
Cold is 330.187087 -> 268.827755 s, both 18 iterations. Three one-iteration warm
samples have medians 49.787769 -> 43.473846 s (1.1452x); candidate samples are
43.473846, 43.468507, 43.641679 s. All five pairs pass: maximum candidate
energy error 2.694378e-11 Eh and force error 4.163176e-10 Eh/Bohr. GPU4PySCF
warm is 27.589376 s; no reference superiority is claimed at this size.

Integral derivatives remain 18.061469 -> 18.073653 s while the measured combined
grid/pair drain drops 15.367909 -> 12.198971 s. Pair order, one selected traversal,
grid visits and SCF iteration counts are unchanged; only the linear admission
scan and rejected empty launch are added. Compacted molecular pair counts remain
unavailable, and dense-grid capacity is not relabeled executed work. The larger
standalone default48 run remains pending on node5 job 1393.
