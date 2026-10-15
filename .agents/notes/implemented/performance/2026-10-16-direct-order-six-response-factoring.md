# Decision: factor constant nuclear responses outside the order-six contraction

Status: implementation under qualification; not yet performance-accepted
Date: 2026-10-16

## Problem and derivation

The generated order-six spatial primitive contracts 64 pairs of Wick terms.
Each term projects one Coulomb-root response onto three nuclear centers using
the same product-center Jacobians. The bra coefficient response also enters
two centers with opposite signs. Repeating those projections inside the sum
executes unnecessary products and additions.

Let A be the summed bra-coefficient response, C the summed ket-coefficient
response, and R the summed Coulomb-root response, separately for each Cartesian
coordinate. The three independent center derivatives before Gaussian decay are
`A + alpha/p R`, `-A + beta/p R`, and `C - gamma/q R`. Accumulate A, C and R
first, then apply these constant Jacobians once. Gaussian decay, normalization,
the fourth-center translational response and shell/force accumulation are
unchanged. This changes FP64 association, not the mathematical derivative.

## Boundaries

Only the existing admitted spatial order-six consumer uses this factoring.
Orders four/five and the nonspatial fallback retain their original response
accumulation. Full/LR kernels share the generated primitive; no functional,
molecule, AO-count or device-specific dispatch is introduced. Nine response
sums replace nine center sums without additional storage or synchronization.

## Preliminary evidence

A real-GPU, finite-Slurm, alternating five-sample primitive probe evaluates
32768 Cartesian components and four primitive products per component. With
second-pair-fastest ordering, median pair-4/2 time changes
1.215756 to 0.990448 ms and pair-3/3 time 1.077028 to 0.830554 ms.
Maximum derivative differences are 2.276e-15 and 1.110e-15, respectively.
Full, partial and singleton active-lane cases pass. These are isolated device
times, not endpoint acceptance or an independent molecular oracle.

The pair-4/2 probe uses 172 to 178 registers; pair-3/3 uses 248 to 242.
Their stack bounds remain 1056/1312 bytes, with zero reported spills and no CTA
barriers. These probe resources are not the production kernel's resources.

Existing independent Hermite-value finite differences cover all pair partitions
of orders four through six, all nuclear centers, full/LR limits and exponent
scales. Added common-center geometries check the zero radial-displacement limit.
Complete source-matched endpoint and linked-runtime qualification are pending.

## Rejected reuse alternatives

Warp-local Boys/prefactor preparation reduces instrumented full-mask radial
preparations from 131072 to 4096, but regresses pair-4/2 probe time
1.218603 to 1.239000 ms and pair-3/3 1.064220 to 1.067742 ms. Sharing bra Wick
terms with match/shuffle subgroups changes 1.206765 to 1.218532 ms and
1.065542 to 1.063032 ms: no useful consistent gain. Both trials produce exact
zero derivative differences for full, partial and singleton masks. Neither is
enabled. The two probes use different AO layouts and cannot be compared across
trials. Preparation reduction alone does not establish fewer machine
instructions or endpoint benefit; no hardware-stall cause is claimed.

Ignored raw generators, compile/resource logs and GPU samples are under
`.artifacts/direct-reuse-next/`. Qualification must retain source identities,
independent arrays, actual convergence/build counts, the original 1800-second
population limit, and separate complete/device timing before promotion.
