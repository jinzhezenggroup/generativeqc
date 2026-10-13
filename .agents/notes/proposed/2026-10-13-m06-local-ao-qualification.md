# M06-2X FP64 local AO qualification

Status: proposed (GPU qualification pending)
Date: 2026-10-13

## Problem

Issue #2414 now has supported CUDA DF energy/force composition, but its 96-atom
12-call population still exceeds the original 1800-second deadline. On source
61b9387f3, warm calls take about 116 seconds; approximately 100 seconds belongs
to dense SCF XC density/potential contractions. The generated split-hybrid
point dispatcher unconditionally reports no local-AO capability, even though
M06-2X consumes the same FP64 MGGA ingredient layout as the indexed executor.

## Candidate and boundaries

Qualify a separate `cuda_fp64_local_ao` manifest capability for M06-2X and
project it into generated native point dispatch. The existing physical FP64
layout checks, derivative coverage, map budget, one-tile host discovery and
1e-16 AO cutoff remain authoritative. Capacity rejection retains dense FP64
execution; explicit disabling remains available. MN15 remains unqualified.
No mixed AO/density, response, DF or graph capability follows from this flag.
No quadrature, energy/force tolerance, iteration or deadline changes are made.

## Acceptance

Host compiled dispatch/admission, registry identity and malformed-manifest
checks pass (160 tests). Before promotion, run independent matched-grid
RKS/UKS decomposition and force oracles, and the unchanged 12-call direct/DF
water populations. Report semantic AO work and complete endpoint times;
a kernel speedup alone does not solve the issue.

## Rejected alternatives

Enabling all split hybrids assumes unmeasured numerical qualification. Changing
a general SCF method whitelist would duplicate compiler capability ownership.
Increasing the point timeout would conceal the unresolved scaling problem.

## References

Issues #2414 and #2418; `manifests/cuda_split_hybrids.json`;
`tests/python/test_split_hybrid_endpoints_cuda.py`;
`tests/python/test_m06_2x_df_cuda.py`.
