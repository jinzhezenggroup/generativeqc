# Proposal: retain canonical range sources for s/p/d CUDA Direct owners

Status: proposed; complete endpoint qualification in progress
Date: 2026-10-03

## Problem

The generated SPD owner covers full-range exchange, but not SR/LR. Canonical
source preparation was restricted to molecules containing an f shell, so an
SPD range correction fell back to ordered public-AO output-by-density-pair
work. Def2-SVP therefore missed the source reuse already available to TZVP.

Node1 Slurm 5340 profiles one 24-atom water SCF iteration, spherical def2-SVP,
WB97M-V, grid 48 x 16 x 32. On the old baseline, the generic exchange kernel
takes 551.975 s, 97.1% of GPU kernel time. VV10 takes 12.352 s. This is an
intrusive, unconverged one-iteration diagnosis, not complete endpoint timing.

## Decision

Prepare the existing optional canonical source for SPD owners too. Reserve
the established generated SPD value/force owner first and deduct its charged
storage before considering canonical pairs, matrices, projections and screened
row metadata. The generated full-range source still wins dispatch. Through-f
preparation order is unchanged. The existing exact-budget generic fallback and
the dense canonical fallback remain available when optional storage does not fit.

Use the shared compiler recurrence, normalized Cartesian source, symmetry
scatter, Schwarz admission and sparse public-basis projection. No new integral
algebra, precision mode, screening tolerance, molecular tensor or oracle work
enters production. The prepared provider supports subsequent range corrections,
so an SPD full-range request can retain extra optional O(NAO^2) storage even when
that particular consumer never requests SR/LR. This setup cost requires endpoint
qualification; it cannot displace the generated source under a limited budget.

## Evidence and acceptance

The focused real-device gate on node1 Slurm 5353 passes s/p/d, Cartesian and
spherical AOs, two geometries, both spins, all J/K output selections, Full/SR/LR,
arbitrary nonsymmetric densities, and the exact generated-owner budget fallback.
Independent CPU ERIs are only test oracles. Actual zero-screening canonical LR
work for the two-item fixtures is 12/110/3080 source quartets at 2/4/10 Cartesian
AOs; generated full-range requests execute zero canonical quartets. These are
instrumented fixture counts, not molecular derivative counts.

The candidate additionally passes existing screened-row equality/empty-row,
through-f Full/SR/LR matrix, all-center derivative, one-electron reuse, mixed-J,
DF layout/selection, and generic budget gates under node1 Slurm 5355. All six
independent complete RKS/UKS WB97M-V tests pass, including STO-3G, def2-SVP,
def2-TZVP, local def2-TZVPD and reconverged displaced-energy checks. Full-grid
changed-geometry and larger-case evidence remain in progress. Keep 1e-8 Eh /
1e-7 Eh/Bohr acceptance across all samples.

The first candidate library is
`adca78939e38b296d8bc06ec868470cd7c11db26298b43c72d63254e5878bc7a`.
Its one-iteration profile is 18.207 s, of which VV10 takes 12.235 s. It also
contains the separately qualified Hermite and LR-moment changes; comparing it
with the old baseline does not isolate this scheduling change. The initial
energy agrees within 1.3e-12 Eh, which does not qualify converged forces.

Ignored source patches, source archive, ccache receipts, binary hashes and
remote logs are retained under `.artifacts/wb97m-spd-range/` and
`/home/jzzeng/codes/wb97m-20261002/`. Failed initial transfers/profile launches
remain failed records. Node5 requires preloading the copied build's CUDA 12.9
runtime; its system CUDA preload otherwise lacks `cudaStreamGetDevice`.

## Controlled 24-atom complete endpoint

Node1 Slurm 5359 measures spherical def2-SVP, WB97M-V, grid 48 x 16 x 32
(589824 points), on one RTX 5090. The complete SCF plus host-returned force
endpoint takes 381.379 s cold and 67.887 s median over three warm calls.
GPU4PySCF takes 123.671 s cold and 27.748 s warm. All five measured pairs
(cold, priming and three repeats) pass: maximum energy error 2.2852e-11 Eh
and force error 4.1661e-10 Eh/Bohr. Native iterations are 18 cold and one
warm. This does **not** establish a GPU4PySCF speed advantage.

The reference uses the comparator's original 1e-11 energy / 1e-8 orbital
convergence controls, direct_scf_tol=1e-14 and the same 1e-8 VV10 density
threshold in SCF and force. No physical acceptance gate was relaxed.
The initial pre-composed full-grid baseline was still incomplete after more
than 80 minutes; its cancellation is not a baseline timing or an isolated
speedup. Subsequent rebasing onto merged #1713 does not change the identities
of the binaries measured here.

## Revisit when

If optional setup measurably regresses full-range-only consumers, introduce an
explicit retained-operator preparation contract instead of stealing their
generated-owner budget. More aggressive shell reuse needs independent range
value/derivative and complete endpoint evidence. VV10 becomes the next measured
bottleneck after eliminating the generic SPD range traversal.
