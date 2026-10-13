# M06-2X FP64 local AO qualification

Status: implemented and qualified
Date: 2026-10-13

## Problem

Issue #2414 now has supported CUDA DF energy/force composition, but its 96-atom
12-call population still exceeds the original 1800-second deadline. On source
61b9387f3, warm calls take about 116 seconds; approximately 100 seconds belongs
to dense SCF XC density/potential contractions. The generated split-hybrid
point dispatcher unconditionally reports no local-AO capability, even though
M06-2X consumes the same FP64 MGGA ingredient layout as the indexed executor.

## Decision and boundaries

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

## Completed independent qualification

Integrated source `9e3c5e6e6` and native library SHA256
`2695be37dec79b9da1268cb4b836d0dd8a434195e69a633a457b17da2aa8e8cb`
pass every cold, five frozen-seed warm, moved and five moved-warm call for
both direct and DF M06-2X at 3/12/48/96 atoms. All 96 calls use the unchanged
1e-8 Eh / 1e-7 Eh/Bohr gates and original point deadline.

| 96-atom complete endpoint | Direct (s) | DF (s) |
| --- | ---: | ---: |
| cold | 150.704 | 150.082 |
| five-call warm median | 20.014 | 36.357 |
| moved | 112.358 | 111.108 |
| five-call moved-warm median | 20.042 | 36.394 |

The earlier dense DF population on `61b9387f3` took 628.378 s cold and
116.061 s warm and timed out after the first moved replay. This population is
retained; it is not overwritten or pooled with the new result. The maximum
96-atom direct errors are 2.274e-11 Eh / 3.320e-11 Eh/Bohr; DF errors are
2.047e-11 Eh / 1.457e-10 Eh/Bohr. The separate benchmark-data PR includes the
complete per-call iterations, AO work counts, exact source and binary identity.

Independent small RKS/UKS split-hybrid decomposition and AUTO fail-closed cases
pass. Four older public-force fixtures passed numerical comparisons before
failing an obsolete assumption that the compatibility owner was always used.
A test-only overlay (`d8cacdb48`) explicitly selects that owner; all six selected
M06 cases pass, preserving moved/fresh cross-route agreement and reconverged
finite-difference gates. Default-route benchmark runs use unchanged source
`9e3c5e6e6`. The rebased, scoped PR head separately passes 157 host tests,
compiler structure, CUDA ownership and Ruff checks; GPU timings are not
relabeled as measurements of that newer Git head.
