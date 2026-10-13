# Rejection: refined FP64 reciprocal as the large-VV10 fix

Status: rejected for promotion
Date: 2026-10-13

## Candidate

Within the existing admitted normal molecular denominator range only, a private
probe replaces the correctly rounded reciprocal with PTX `rcp.approx.ftz.f64`
and two explicit FP64 FMAs implementing one residual-refinement step. All pair
arithmetic and summation remain FP64. No production source was changed.

## Evidence and decision

Finite n1 Slurm job 8110, RTX 5090, CUDA 12.9.1, sm_120, -O3 and --fmad=false
compares the original generated admitted closure and the candidate in one
binary. Verified ccache performs a separate cached object compile before linking.
Three rotated-order observations follow a warmup. CUDA-event timing includes
the complete row kernel, excluding allocation and upload; these are synthetic
component timings, never complete molecular endpoint evidence.

At 320,000 all-active synthetic points (102,400,000,000 ordered pairs), feature
medians are 4.1131 -> 3.7707 s, and geometry medians 5.8944 -> 5.5521 s: only
8.3% and 5.8% improvements. The unchanged 2e-12 relative/2e-11 absolute serial
comparison passes at 17/80,000/160,000/320,000 points; maximum difference is
3.256e-13. A 17-point independent extended-precision direct-rational energy,
feature and coordinate/weight derivative oracle also passes. This does not
establish general correctly rounded equivalence or complete molecular acceptance.

The gain is insufficient to resolve the observed full-population WB97M-V
limits and does not justify adding reciprocal-rounding policy to production.
Retain the original arithmetic and fallback. A first runner setup failed because
CUDA_PATH was missing; that failure is retained separately. Probe source,
binary/source hashes, all timing samples and scheduler provenance remain in
ignored qualification artifacts and the compact benchmark follow-up receipt.
Revisit only as part of a larger qualified endpoint improvement; do not infer
resolution from an isolated reciprocal microbenchmark.
