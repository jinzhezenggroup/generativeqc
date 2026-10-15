# Decision: preserve strict nonlocal reuse and bound budget confirmation work

Status: implemented
Date: 2026-10-16

## Problem

The normal mixed-to-strict handoff discarded both the provisional nonlocal
potential and the already-admitted frozen-potential outer solver. Consequently,
every subsequent strict Fock refreshed the quadratic VV10 pair operator. Fewer
SCF iterations did not imply less nonlocal work.

The previous 96-atom/768-AO, direct WB97M-V pilot measured 1062.492 s FP64 and
939.501 s Fast, only 1.131x. The FP64 endpoint executed three pair sweeps with
one exact cache hit; Fast executed five sweeps with no cache hits. This receipt
belongs to source identity `0ba780ead15c9ca55aeb438c5f858aaedfcd64260666a35ad7dda795be203669`.

## Decision

At a normal precision handoff, clear the semilocal initializer and frozen-potential
stage, but retain the outer solver if its existing admission selected it. The
first strict iteration therefore rebuilds the complete current-density FP64
target. Only that fresh strict potential may anchor subsequent provisional
inner steps.

For an admitted budget-stationarity solve, stop proposing density changes and
request a fresh full-target energy confirmation when both physical residual
norms are below `min(1e-5, density_tolerance)` and the density-change gate already
passes. Strict solves retain `min(1e-9, density_tolerance)`. This is a proposal
scheduling threshold, not a convergence certificate: the ordinary energy,
density, physical residual, consecutive full-target and canonical export gates
remain unchanged. The next target evaluation uses the unchanged density, making
an exact-identity pair-cache hit possible without replaying a stale target.

Budget frozen-potential inner steps use that same confirmation tolerance rather
than an adaptive strict tolerance that can approach `1e-12`. They only propose
the density at which a fresh full target is evaluated; their residual and energy
cannot publish a result. Explicit strict solves retain the incumbent adaptive
inner tolerance. A tighter caller density tolerance is never weakened.

Automatic range-source selection does not build a second quartic source during
strict refinement after retiring a mixed lease. That short confirmation tail
uses the incumbent streamed operator instead. Explicit resident-source requests
retain their strict source, capacity-refusal and partial-source paths.

An exact pair cache key still includes arithmetic selection, ingredients,
generation and schedule identity; a retained mixed pair result cannot become a
strict result through relabeling. Any strict resident source requires its own
precision-bound lease. An explicit outer-solver disable remains disabled.
Numerical-failure recovery
still disables the provisional outer solver entirely, retaining the separate
fail-closed behavior introduced by `5f0fc6606`.

This is a shared nonlocal scheduling change, not a functional-name special case.
It does not newly qualify FP32 XC or gradients and does not relax observable
budgets, SCF gates, quadrature, density domains or screening policy.

## Rejected alternative: reuse alone

Retaining the outer solver without changing confirmation scheduling was tested
first, not silently replaced with the later measurements. Slurm job 8818
measured 1069.395 s FP64 and **1128.426 s Fast**, a 0.948x regression. Fast
executed 30 Focks (12 mixed and 18 strict), four pair sweeps and one exact cache
hit. At target iteration 22 its physical residual was already `1.3582e-8` and
density change `1.3002e-8`, but the strict-only `1e-9` confirmation threshold
started another frozen inner/outer cycle. Automatic selection also built a new
strict quartic source that the tail could not amortize.

That candidate passed 14 complete endpoint tests, seven independent source-matrix
tests, 420 host tests with nine resource skips, and memcheck, but correctness and
fewer pair sweeps were insufficient evidence of a performance fix. Its source
identity is `5dc7ab5477e9b9336af69e9e78a04517e72725c5d991aef48710ff2efc53ca01`
and core SHA256 is
`527735deb0bd140295a488940a7a7cb1a95bc169c3d7da6ec5410818d0381f74`.

## Rejected alternative: a tighter confirmation threshold alone

The second candidate avoided the second automatic quartic source and used a
`1e-7` budget confirmation threshold, but retained strict adaptive inner
tolerances. Slurm 8831 measured **1053.608 s FP64 / 1110.022 s Fast (0.949x)**.
Fast still executed 30 Focks (12 mixed and 18 strict), four pair sweeps and one
cache hit. The normalized diagnostic at target 22 was `1.3582e-8`, but its
separate AO maximum residual did not pass `1e-7`; another inner/outer cycle
still ran. Do not use the normalized diagnostic as a maximum-norm certificate.

This candidate independently passed the retained 96-atom observable gates
(Fast energy/max-force/RMS-force errors `7.822e-11 Eh / 2.438e-9 /
9.936e-10 Eh/bohr`), 14 endpoint tests, seven source-matrix tests, 460 host tests
with nine skips, and memcheck. It nevertheless failed the performance objective.
Its source identity is
`e71935977246273eea81c78c27b66c1277e0be67c6b316712aa7475fcce4fd3e`,
core SHA256
`b542912c17eaf89aa96247a85e272e0721428d620dd3f28ffd809336911846f7`.

## Final-candidate qualification

- The new public regression fails on the prior core for both 3- and 12-atom
  geometries: no strict frozen-potential inner step survives the handoff.
- On the new core both cases first refresh a full strict target, then reuse its
  anchor in strict provisional inner steps, and finish with full-target states.
- All 14 complete Fast endpoint cases pass independent GPU4PySCF moving-grid
  energy/force gates, including eight XC compositions and four source selections.
- Seven independent source-matrix cases pass in a separate process. These
  endpoint/matrix results belong to final-candidate Slurm 8840, not either
  rejected candidate. The 3- and 12-atom regressions both refresh a strict anchor,
  retain provisional reuse and end with consecutive full-target states. Each
  executes three pair sweeps and one exact cache hit.
- The 3-atom complete Fast endpoint passes Compute Sanitizer memcheck with zero
  errors (Slurm 8841), using 17 iterations. This instrumented time is not clean
  performance evidence.
- Host policy, precision, source and retry selections pass 673 tests. The broader
  host selection skips 864 resource-gated cases; two native-library energy-only
  cases also pass separately with the deployed library under Slurm 8840. The
  executable host contract retains a caller's tighter density tolerance for both
  strict and budget scheduling. Ruff, clang-format, whitespace and a no-op rebuild
  pass; the source and core digests still match the frozen qualification build.
- Final core SHA256:
  `ec84720356280170da31bcaed757d246eab164031f7434a4c15e2b7b7d424045`.
  Source identity:
  `07fd2c0a5d7c208309b52e1d1fb9a7151d5ab5feff8b70d167adf2fd13e72dd5`.

Raw receipts remain outside the tracked benchmark archive under
`evidence-vv10-range-fp32-20261015/`: retain `outer-reuse-*` and
`native96-pair-20261016-outer*` for the first rejected candidate,
`budget-confirm-*` and `native96-pair-20261016-budget*` for the second, and
`budget-inner-*` for final qualification. Scheduler/deployment and host-library
setup failures are retained as distinct logs, not overwritten by successful runs.

## 96-atom measurement and independent accuracy

The matched pilot uses Slurm job 8842 on one RTX 5090. Each arm is a fresh
process with a fresh calculator, hcore guess, no density fitting and no shared
converged state. The endpoint includes SCF, complete moving-grid analytic
forces, host export and teardown. Both arms use direct RKS, spherical def2-SVP
(768 AO), `GridSpec(48, 16, 32)` (2,359,296 points), four CPU threads and at most
180 iterations. This is a single matched pilot, not a repeated median.

| Mode | Complete endpoint (s) | Mixed / strict Focks | Pair sweeps | Exact cache hits |
| --- | ---: | ---: | ---: | ---: |
| FP64 | 1062.766982 | 0 / 34 | 3 | 1 |
| Fast | 851.927469 | 12 / 7 | 3 | 1 |

The observed FP64/Fast ratio is **1.247485x**, a **19.8387%** endpoint-time
reduction. Fast uses 19 total Focks, four full targets and 15 provisional steps.
It rebuilds the first strict target at iteration 13, refreshes it after the
budget inner solve at iteration 18 and confirms the unchanged density with a
fresh full target at iteration 19. That last energy change is `3.861e-14 Eh`.
The exact cache hit avoids another pair sweep, not the final physical Fock.

This is one genuine `qualified_budget_fock` solve with no execution retry or
census fallback. Executed FP32 work includes 12 J applications, 24 full/range K
applications and one nonlocal pair evaluation, with FP64 accumulation. XC,
final physical validation and analytic forces remain FP64; this is not a claim
that the complete endpoint is FP32. The source/census inventory does not provide
a complete quartet-level J/K work census, so none is inferred from these counts.

The independent accuracy-only reference is the retained 96-atom receipt from
`benchmarks/results/readme-wb97mv-20261014/summary.json.xz`, not a native FP64
self-comparison. The archive digest matches its tracked manifest; the raw
receipt equals the archived receipt; the original oracle driver and all recorded
source hashes match Git commit `99e7c54b2aa52f8cc7861f0f9f95226a152d4824`.
The H/O basis shells, geometry, quadrature, non-DF Hamiltonian and both VV10
`1e-8` density domains match the native physical target. Its independently
converged energy is `-2443.2164193689728 Eh`.

The reference used an atomic guess, eight threads and 100 maximum cycles rather
than the native hcore/four-thread/180-iteration protocol. Reuse is only for the
converged observables, not a new live exact-solver-protocol oracle or a reference
timing. The external `check_historical96.py` authenticates these bindings and
checks each complete native endpoint against energy/max-force/RMS-force gates:
FP64 `1e-7 Eh / 1e-5 / 3e-6 Eh/bohr`; Fast
`1e-5 Eh / 1e-4 / 3e-5 Eh/bohr`.

Both final endpoints pass these independent gates. Their absolute errors are:

| Mode | Energy (Eh) | Maximum force (Eh/bohr) | RMS force (Eh/bohr) |
| --- | ---: | ---: | ---: |
| FP64 | 1.045919e-11 | 1.249382e-9 | 6.894677e-10 |
| Fast | 1.305125e-10 | 1.170886e-6 | 4.926347e-7 |

The authenticated comparison is retained as `historical96-inner-check.json`,
binding `native96-pair-20261016-inner.json` to the historical independent receipt.
Neither the old reference timing nor either rejected candidate enters this new
pilot pair. The result improves on the earlier 1.131x pilot but is not evidence
of the same speedup for every functional, geometry or device.

## PR integration provenance

The measured patch is preserved locally by commit
`7490d29558afe7039982a8e4c8cde9f7849e7c25`, based on `11b0f50bf`.
By the time the follow-up PR was requested, #2528 had merged. The follow-up
therefore applies only these six changed files to master `5348b9e66`, retaining
the intervening upstream fixes and refactors rather than repushing the old PR
branch or reverting concurrent work.

Those upstream changes alter scientific source identity. The 96-atom timings,
independent-accuracy receipts and sanitizer evidence above remain attributed to
their exact measured source/core, not relabeled as a new integrated-head run.
Fresh integrated host selections pass 530 plus 154 tests, with 864 resource skips
and two native-library cases reserved for a configured runtime process.
Integrated native build and GPU qualification are recorded separately from
the earlier pilot. Repeated matched 96-atom pairs are still needed before
promoting the single 1.247x observation to a repeated or general speed claim.

## Revisit when

Measure complete endpoints before changing inner/outer stopping policy. Reuse
can save pair sweeps while adding cheaper inner Focks, so a sweep reduction
alone is not a speedup. Extending FP32 to other regions requires its own shared
capability proof and independent complete-endpoint qualification.

## References

- [Shared FP32 qualification history](../numerics/2026-10-15-shared-vv10-range-k-fp32.md)
