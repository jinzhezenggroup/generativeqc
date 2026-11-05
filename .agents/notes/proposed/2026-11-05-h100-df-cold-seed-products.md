# Decision: prefer FP64 BLAS for large CUDA DF cold seed admission

Status: proposed; changed-default H100 endpoint qualification pending
Date: 2026-11-05

## Problem and scope

H100 HF96 DF cold Hückel preparation spends about 0.61 s in the existing host
route. CUDA Direct already borrows device validation products; DF passes its
eigen operation without a product operation. The seven preliminary iterations
are the strict native atomic model, not seven molecular target solves.

## Decision

Only CUDA DF Hückel target seed admission at 512 AOs and above prefers the
existing locally threaded FP64 CPU BLAS service. The crossover matches the
conservative large-owner boundary already used by CUDA Direct preparation.
All other validation owners remain default-off. Capture the policy once at
preparation, while preserving `GENERATIVEQC_SCF_VALIDATION_GEMM=0` as an exact
same-source scalar ablation. `=1` retains its existing wider explicit scope;
invalid selectors and unavailable providers retain scalar products.

The atomic model, seed construction, eigen operation, physical equations,
acceptance thresholds, target basis/fitting recipe and preliminary numeric
capacity do not change. Explicit/warm densities return before this operation.
The shared tensor service owns local thread changes and restoration. Provider
errors retain the existing typed preparation boundary; they never trigger an
unrecorded scalar retry.

## Evidence and limits

Frozen source `4370d6c4b1bf4033818f31a9fcc622d3045a33bf`, core SHA256
`22bf8d83ad92c574410d57144108c011d993a45602a4e6c64d03ee7bb594386f`:
three rotating scalar/BLAS/GPU4PySCF rounds on one H100, nine fresh processes,
36 complete energy/analytic-force calls. All independent original dense/CPU
gates pass. All 12 scalar/BLAS comparisons pass total energy 1e-8 Eh, maximum
force 1e-7 and RMS force 3e-8 Eh/Bohr gates; maximum force difference is
3.091e-13 Eh/Bohr and returned energies are identical.

Paired cold BLAS/scalar ratios are 0.859733, 0.915692 and 0.923075 (median
8.43% reduction); warm ratios are 1.007057, 0.999975 and 1.006833. Median cold
times are 5.67624 s scalar, 5.15465 s BLAS and 5.61359 s GPU4PySCF. The first
reference cold outlier, 8.81094 s, remains included. Seed preparation falls to
0.163–0.167 s, with unchanged seven preliminary and 11 target iterations.

Raw/input/producer capsule `completed-gemm-v1.tar.xz` is 158020 bytes, SHA256
`809a6a6ee59a5d980cd37fedb4e8d2437ee55158bb8060d871e8d94cef199a74`.
This qualifies the old-source selector experiment, not the new narrow default
or the original full30 matrix. Changed-default device qualification and raw
publication must precede promotion. Six compiled analytic provider/admission
tests pass, including explicit selector overrides, absent/local-thread-missing
providers and caller thread restoration.
The changed DF owner also passes 44 compiled fault-injection cases for typed
seed/eigen/dispatch/finalization errors, bounded retry and neighboring arena
lifetime. These host probes do not establish changed-default device acceptance.

## Rejected scope and revisit conditions

Do not enable all host validation products globally from one HF96 DF result.
Do not infer a warm win, fewer Fock builds, cross-device latency improvement or
new full30 aggregate. Revisit a CUDA borrowed-product route only with a measured
remaining preparation bottleneck and explicit scratch/lifetime qualification.

The earlier [host validation experiment](2026-10-24-host-validation-products.md)
retains its global default-off conclusion and broader historical qualification.
Track current H100 work in [issue #3004](https://github.com/jinzhezenggroup/generativeqc/issues/3004).

## Follow-up: changed-default acceptance

The narrow default was subsequently measured at source `87bbc0b4f20f7696cfd22974bbe5b207e7d69c38`.
Its median paired HF96 DF cold reduction is **7.332%**, with a -0.081% warm
change; HF48 remains a one-round numerical guard and the separate WB incremental
pilot is rejected. The 8.43% result above remains the earlier, different-source
wider-selector experiment. See the [implemented decision](../implemented/performance/2026-11-05-df-cold-seed-products.md)
and [complete source-bound receipts](../../../benchmarks/results/h100-cold-df-seed-20261105/README.md).
The original proposal's pending status records its pre-qualification state.
