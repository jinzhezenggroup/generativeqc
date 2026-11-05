# Proposal: contract order-four/five Direct LR force weights before AD

Status: proposed; measured on A100/RTX5090, final master integration pending
Date: 2026-11-06

## Problem

The omega=0.3 WB97M-V full-Direct stationary owner decomposes its full and
long-range integral responses. The LR bounded consumer contracts angular orders
zero through three at shell level, but its covered s/p/d order-four/five tasks
still repeat geometry and screened-moment derivative work per AO quartet.
Previously delivered pure-J pair projection does not apply to exchange weights.

## Candidate decision

Reuse the existing arbitrary-cotangent order-four/five weighted-ERI graphs and
their relative-coordinate responses with the screened Boys ladder. For fixed
rho and omega its derivative is dF_n(T)/dT = -F_(n+1)(T). Precontract the exact
screened AO exchange weights, then evaluate each primitive geometry once.
Supported classes are PPPP, DSPP, DSDS, DPPS, DDSS, DPPP, DPDS and DDPS.

The scalar shell owner consumes each supported task once; its warp subtile
fallback skips exactly those classes. Other classes, short range, higher orders,
existing LR orders zero through three and noncanonical omega owner admission
remain unchanged. This does not introduce a recurrence or pair factorization.

The prepared-owner selector `GENERATIVEQC_DIRECT_LR_WEIGHTED_FORCES` is included
in resource/restart identity. Candidate code captures unset/1 as enabled and
0/empty/unknown as disabled, but this is not yet authorization to publish a
default-on change. The nuclear response selector remains independent. A
nonfinite relative primitive reduction publishes nothing and retries its
strict nuclear weighted graph once.

## Invariants and qualification

- No CPU production computation, Direct/DF substitution, GPU-name dispatch,
  altered precision, grids, screening, convergence or source coefficients.
- Same bounded queue and AO mask, including equality, pair triangles, shell
  canonicalization, RKS/UKS and physical-atom folding.
- Independent Libcint host and actual-device integral-source gates; previous
  pure-J pair response/rollback tests remain required.
- Same-image selector and previous-qualified-image complete E+force controls on
  A100 and RTX5090; also check the unrelated RTX5090 r²SCAN-3c complete endpoint
  for compile-frame regressions. Finite cold plus three warm samples, not a
  broad parameter sweep. No promotion without positive endpoint evidence.

## Initial evidence

The initial independent host cohort passes 96 science cases across eight
classes, RKS/UKS, three positive omegas and signed exchange coefficients.
Sixteen Inf/NaN late-primitive retry tests and two actual captured-policy gates
also pass (114 total). The first fault fixture mistakenly called a huge finite
relative result nonfinite; the fixture now uses NaN. The pair-projection overflow
test retains its distinct finite-magnitude publication bound.

Source image `c313decb8884e69c3b44b6fe2e72c9491869b64313db3e4fcdd56f8e5f52a2a1`
is based on `b27caeec846b6fd28597c06cbbb55498c1e3d730`, preserving the reviewed
nuclear-rollback fix from PR #3014. Compilation is not runtime qualification.
No GPU timing or speedup is claimed yet; full WB97M-V cold still includes
substantial unrelated source preparation, SCF and VV10 work.

## Measured qualification and master synchronization

The later complete source-matched selector0/1 cohort qualifies the arithmetic
change: A100 LSF225805 warm median19.467532→15.051496s (-22.68%); RTX5090
Slurm16016 warm median16.520271→14.406079s (-12.80%). Both retain cold18/18
iterations/Focks, warm1/1 and identical recorded settings/semantic VV10/J/K work.
The 5090 complete r²SCAN-3c previous-image control has warm ratio0.999953.

Old-image WB controls also contain the upstream force-tile256→1024 change.
Only the same-image comparison isolates LR; do not report the whole historical
image improvement as this optimization. No new GPU4PySCF timing or whole cold-gap
closure is established. The complete retained reference supplies observable
gates only, not a current performance ranking.

Independent gates now include392 actual-native LR host cases and8 scheduled
Libcint device-source cases. Initial huge-finite fault-fixture failures and11
local native-library discovery failures remain recorded; corrected fixture114
cases and actual-library27 resource cases pass. No new sanitizer pass is claimed.

Master advanced to1943a24b7 during measurement; PR3014 head9a72d770a is now
preserved by the follow-up branch. The integrated sourceff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa
is rebuilding for a finite default-only verification. Historical paired records
remain tied to their actual sourcec313decb…/ELFd9b33f54…, not relabelled.
See `benchmarks/results/direct-lr-weighted-20261106/README.md` for full precision,
original producers, settings, work, failures and offline replay.

## References

- Issue #2976 decision comment 6668126027.
- Prior delivered PR #3014; this candidate belongs to a separate stacked PR.
- `tests/python/test_direct_lr_weighted_host.py`
- `tests/python/test_direct_lr_weighted_cuda.py`
- Private raw receipts: `.artifacts/direct-next-20261106/`.
