# Decision: reject two r2SCAN-3c cold-start controls without endpoint benefit

Status: rejected
Date: 2026-11-01

## Scope and decision

The independently qualified n1/15095 comparison still has r2SCAN-3c/96 cold
latency of 23.294920 s against GPU4PySCF's 23.150760 s (+0.6227%). Warm medians
are 6.822565 s and 9.876204 s (-30.9192%). Closing that small cold deficit requires
complete endpoint evidence; improving a preparatory subroutine alone is not
sufficient. Neither control below changes the production default.

Both campaigns retain explicit cache conditioning, then forward/reverse fresh
owners with cold plus three frozen warm calls. They use stock grids, FP64,
SCF/force tiles 256/1024, eight Slurm CPU slots and single-threaded math. Every
conditioning and measured population passes the existing independent README
reference gates; paired iteration and Fock counts are unchanged. Positive
changes below mean slower complete candidate time. No A100 result is inferred.

## Host-tile versus device-chunk AO discovery

The existing device control keeps the sampled AO predicate and capacity bounds,
but batches offset publication instead of synchronizing each host tile. Slurm
15114 (5090) and 3873 (PRO 6000) each retain two conditioning populations and
host/device/device/host measurements. All twelve populations qualify.

| Device | Cold change | All warm change | Cold + three warm change |
| --- | ---: | ---: | ---: |
| RTX 5090 | +0.199413% | +0.097655% | +0.152065% |
| RTX PRO 6000 | -0.315793% | +0.039156% | -0.151971% |

These observations do not establish a consistent benefit. Keep the existing
host default and the explicit device control. Native source is
`6527e5625b096f10a8c431ce2829b67d44a98abb656889f486f2b13786dc419c`, library
`327a417cb66c115b7f5d35ca9fde6a89ec217da632e2d0cb0c589bead84b614f`.
Full inputs, controls, outputs, timings and replay script remain in the ignored
`.artifacts/xc-ao-discovery-20261101/` directory of worktree
`/data/jzzeng/xc-ao-discovery-20261101`.

## Exact zero absorption in quadrature preparation

The molecular quadrature partition sums oriented logs of clamped Becke factors.
Every term is nonpositive or negative infinity. Once the accumulated value is
negative infinity, its remaining suffix cannot change the product's exact zero.
The experiment stopped that row's loop at this point, without a tolerance,
reordered finite sum, altered grid, new allocation or GPU-name selector.

Twenty-nine host tests pass, including full ordered log comparisons at exact
centers, finite tails, coincident atoms, all five polynomial iteration counts
and both orientations of radius adjustment. n5/1975 passes independent native
quadrature, pruned/Treutler grid, analytic response and allocation-lifetime
checks; six public RKS/UKS energy/force comparisons also pass. Native quadrature
Compute Sanitizer reports zero errors and zero bytes leaked.

Slurm 15142 (5090) and 3874 (PRO 6000) retain a conditioning predecessor and
parent/candidate/candidate/parent owners. All ten complete populations qualify;
maximum paired energy and force errors are below 1.4e-12 Eh and 3e-12 Eh/bohr.

| Device | Cold change | All warm change | Cold + three warm change |
| --- | ---: | ---: | ---: |
| RTX 5090 | -0.068463% | +0.101073% | +0.004497% |
| RTX PRO 6000 | +0.058471% | +0.084343% | +0.047290% |

A separate diagnostic instruments a copy of the generated kernel, with the same
`--fmad=false` policy. It compares every full/reduced log bitwise on the real
96-atom grid and counts actually executed pair terms. Its timings are excluded
from the complete endpoint cohorts.

| Grid level | Points | Full pair terms | Executed pair terms | Log bit mismatches |
| --- | ---: | ---: | ---: | ---: |
| 0 | 74,432 | 678,819,840 | 551,236,506 | 0 |
| 1 | 323,968 | 2,954,588,160 | 2,469,107,351 | 0 |
| 3 | 1,078,336 | 9,834,424,320 | 8,439,671,222 | 0 |

Less executed arithmetic did not improve the measured complete endpoints.
Reject this production change rather than promoting an isolated work reduction.
Its frozen source is
`cd809ee5edb1aee3dce72fb77ec0331eae7a624d5bb0f801411c84b2c623baef`, library
`5c3a6e7dc32cc0b166e90f8d843bcd16ddff29b585488407a494d5ca0e6bdde8`.
The dirty reconstruction patch, frozen images, complete numerical/timing samples,
failed build/setup logs, work probe and ordinary/optimized replay remain under
`.artifacts/quadrature-exact-zero-20261101/` in worktree
`/data/jzzeng/portable-xc-gemm-20261101`. None is relabelled as the accepted VV10
source or a later publication revision.

## Revisit when

A different workload establishes an endpoint gain large enough to distinguish
from variation, with unchanged independent gates and visible complete warm
populations. A new hypothesis should first address the remaining measured
setup, matrix-contraction or force work instead of repeating these controls.
