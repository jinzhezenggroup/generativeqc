# Decision: use FP64 BLAS for large CUDA DF cold seed admission

Status: implemented; scoped H100 endpoint qualification complete
Date: 2026-11-05

## Decision and invariants

CUDA DF Hückel target seed admission prefers the existing locally threaded FP64
CPU BLAS service at 512 AOs and above. All other host validation owners remain
default-off. Explicit selector zero, invalid values and unavailable or
local-thread-ineligible providers retain scalar products. Warm/caller densities
bypass this cold work. The atomic model, eigen operation, construction of the
seed, physical admission gates and preliminary native-numeric capacity remain
unchanged. Provider failures retain the existing typed preparation boundary;
there is no implicit product-backend retry.

This narrows the earlier [proposal](../../proposed/2026-11-05-h100-df-cold-seed-products.md).
The existing tensor service owns thread-local selection/restoration. The
preliminary capacity remains the conservative scalar-temporary envelope;
opaque provider memory and process RSS remain outside that existing native
numeric-capacity contract.

## Changed-default evidence

Measured source `87bbc0b4f20f7696cfd22974bbe5b207e7d69c38`, native identity
`d60aea30bf2891a8d5c6976f59ee7f8a3505cd83af5c123dde48093e4b3ea604`,
core SHA256 `4702ac35b5fada834aa15d0569a4f750d1dc756dbca731090ed5b5c6f49d64a9`.
Three rotating scalar/default/GPU4PySCF HF96 DF rounds on one H100 produce
cold default/scalar ratios 0.851577, 0.926679 and 0.928744: **7.332%** median
paired reduction. Warm ratios are 1.005800, 0.989653 and 0.999193, all within
the 1.05 guard; this does not establish a warm improvement.

Median process cold times are 5.61596 s scalar, 5.20865 s default and 5.68354 s
GPU4PySCF. The median paired reference/default ratio is 1.09324; the first
9.17863 s reference cold observation remains included. Energy outputs are
identical between native variants and the maximum matched force difference is
4.2633e-13 Eh/Bohr. Target work remains 11 iterations / 12 Fock builds.

The HF48 one-round boundary guard passes numerical checks but is slightly slower
(cold ratio 1.007841, warm 1.011686); no timing benefit is claimed for it. The
separate WB96 incremental pilot passes E/F gates but regresses to 1.772399 cold
and 2.608771 warm ratios and is not promoted. Its quartet counters are invalid,
so no quantified quartet saving is inferred from six executed delta builds.

All 15 processes / 60 full E/F calls pass the unchanged independent dense/CPU
gates. Cold includes constructor/preparation/SCF/analytic forces/host export
after context initialization, with retained verified compiled artifacts. It
excludes process startup, installation and AOT compilation. Every unfavorable
sample is preserved in the [public receipt bundle](../../../../benchmarks/results/h100-cold-df-seed-20261105/README.md),
whose offline reader recomputes all gates, source bindings and paired ratios.
The original older-source wider-selector 8.43% result remains distinct.

Six compiled analytic provider/admission tests and 44 DF fault/retry/lifetime
cases pass. The actual incremental Release SM90 build takes 56.4038 s for four
C/C++ cache misses, with zero compile/cache failures and unchanged stationary
AOT artifacts; Ninja's automatic CMake regeneration is explicitly retained.
A fresh read-only same-family review finds no blocking production defect and
records scoped WARN/provisional assurance because broader acceptance is absent.

## Boundaries and revisit conditions

This is HF96 DF acceptance plus a numerical size guard, not a full30 result,
universal performance claim, cross-device qualification or intended-state
certificate. Continue the remaining H100 cold work under
[issue #3004](https://github.com/jinzhezenggroup/generativeqc/issues/3004).
Revisit on complete-endpoint/warm regressions or a changed provider/resource
contract; preserve the independent gates and explicit rollback.
