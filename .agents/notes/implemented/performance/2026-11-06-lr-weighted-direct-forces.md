# Decision: contract order-four/five Direct LR force weights before AD

Status: implemented; default qualified on A100/RTX5090 and synchronized master
Date: 2026-11-06

## Problem

The omega=0.3 WB97M-V full-Direct stationary owner decomposes its full and
long-range integral responses. The LR bounded consumer contracts angular orders
zero through three at shell level, but its covered s/p/d order-four/five tasks
still repeat geometry and screened-moment derivative work per AO quartet.
Previously delivered pure-J pair projection does not apply to exchange weights.

## Decision

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
in resource/restart identity. Unset/1 enables the qualified default and
0/empty/unknown retains the incumbent. The nuclear response selector remains independent. A
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
was rebuilt for a finite default-only verification. Historical paired records
remain tied to their actual sourcec313decb…/ELFd9b33f54…, not relabelled.
See `benchmarks/results/direct-lr-weighted-20261106/README.md` for full precision,
original producers, settings, work, failures and offline replay.

## Integration packaging failure

The first integrated rebuild exported the new source stamp but retained some
old objects: unpacking a later Git archive in-place restored commit-time mtimes
older than the existing objects. CMake therefore missed changed source inputs.
Slurm16017 failed its first independent case with the Python-required policy2
versus native policy1 receipt, before any complete endpoint. Its source stamp
`ff6117e3…` and ELF `d5d4181f…` are explicitly **not** integrated qualification.
A source stamp and successful link alone cannot establish object freshness.

Re-extract with `tar --touch`, retain normal Release flags and the shared content
compiler cache, then require the allocation-free native SCF metadata policy2
before further GPU work. This is host API metadata validation, not a production
CPU scientific solve or reference substitution. Preserve the failed case and
the bad-image policy receipt separately from later execution. The independent
original selector0/1 measurements are unaffected. The first A100 integration
transfer also stopped at expired SSH authentication; no LSF job was submitted
by that attempt. Renew the private control channel, not the GPU cohort.

## Final integrated default

The corrected complete rebuild qualifies source `ff6117e3…` with ELF
`4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a`.
The native metadata policy2 gate passes before both scheduled populations.
LSF225806 finishes DONE/exit0; Slurm16018 finishes exit0. The LR selector is
unset, not explicitly enabled, in both endpoint producers.

WB97M-V full-Direct cold/warm medians are111.973577/14.990246s on A100 and
194.419602/14.365366s on5090. The additional5090 r²SCAN-3c control is
28.854132/10.569315s. Against the original measured candidate-on image, warm
ratios are0.995931/0.997174/0.995831, respectively. All settings and semantic
work remain identical, with paired observable errors below the original gates.
These are later-image integration checks, not a new isolated LR ablation.
Another8 independent device-source and27 resource cases pass;16 additional
independent reference-observable pairings pass. No reference timing is reused.

The default is therefore retained rather than hidden behind opt-in. The final
follow-up is PR #3017. Parent #3014 was subsequently squash-merged as master
`863359f4b`; its full tree is identical to qualified base `9a72d770a`.
`delivery.json` retains and authenticates both raw commit objects and equal
tree identities. Synchronizing this ancestry and retargeting to master changes
no scientific input or native source identity; it does not require another GPU
cohort or relabelling any measured binary. Preserve
the initial mixed-object failure and private-auth transport failure alongside
the passing receipts; do not infer full cold or GPU4PySCF parity from this work.

## References

- Issue #2976 decision comment 6668126027.
- Prior delivered PR #3014; follow-up PR #3017 owns this separate optimization.
- `tests/python/test_direct_lr_weighted_host.py`
- `tests/python/test_direct_lr_weighted_cuda.py`
- Private raw receipts: `.artifacts/direct-next-20261106/`.
