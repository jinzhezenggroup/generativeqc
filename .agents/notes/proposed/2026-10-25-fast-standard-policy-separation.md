# Separate accuracy concessions from reusable execution optimizations

Status: proposed; source and retained-receipt audit complete, runtime defaults unchanged
Date: 2026-10-25
Audited master: `7b3dc64f9` (fetched `origin/master`)

Follow-up: the [implemented CBS preconvergence separation](../implemented/performance/2026-10-25-cbs-standard-preconvergence.md)
supersedes the proposed runtime policy below and records new remote-device
measurements. This file preserves the original retained-receipt audit.

## Problem

Fast should offer a useful accuracy/time tradeoff. The homepage compares both
profiles with GPU4PySCF, which does not directly show Fast's incremental benefit
over Standard. The current profiles also couple arithmetic, stopping policy and
some execution choices. Similar energies or identical qualification statuses do
not establish identical force accuracy; a changed arithmetic request need not
change executed work.

This audit uses the immutable HF/DFT campaign at
`2ed9b394a2889b3a373dc91ebbaaed03931056eb` and the CBS workspace campaign based on
`976f756b391bc65edb79f76df1a8c3fe807518a7` plus its retained candidate patch.
Neither campaign is a new measurement of the audited master. In particular,
the later independent Lambda replay and DF response improvements are not
represented by the old homepage CBS times. Do not relabel these receipts.

## What the HF/DFT measurements actually establish

Recomputed ratios below are medians of the per-case Standard/Fast latency
ratios. Each warm latency is the median of all three frozen-density calls.
All 32 cases in each row remain present, including failed accuracy gates.
The 5% band describes the observations; it is not a significance threshold.
The campaign used separate scheduled cards, not alternating same-card trials.

| Endpoint | Median Standard/Fast | Ratios within 5% of 1 | Equal Fock counts |
| --- | ---: | ---: | ---: |
| Direct cold | 1.0709 | 12/32 | 24/32 |
| Direct warm | 1.0129 | 24/32 | 32/32 |
| DF cold | 1.0042 | 20/32 | 29/32 |
| DF warm | 0.9906 | 17/32 | 32/32 |

- Every one of the 192 Fast warm samples reports zero mixed-stage SCF Focks.
  Warm seeds come from the same engine/profile's cold solve, so their density
  errors can still differ. Some force consumers still narrow arithmetic.
- DF Fast falls back to Standard in 20/32 populations: HF, PBE0, B3LYP,
  M06-2X and WB97M-V at all four sizes. Only PBE, r2SCAN and r2SCAN-3c select
  the budget-Fock route in this cohort. Direct cold selects 28 budget-Fock,
  two strict-density mixed-gradient and two Standard-fallback routes.
- Both modes have 58/64 independently qualified populations and the same six
  failures. This audit recomputes energy/force, reference-grid stability and
  CPU/GPU spot gates from the retained oracle inputs, including the explicit
  cross-revision reuse manifest. It does not turn identical statuses into an
  equality-of-accuracy claim.
- Cold total-energy differences are at most `1.79e-8 Eh` (Direct) and
  `8.21e-9 Eh` (DF). Force differences reach `4.29e-5 Eh/bohr` for Direct
  WB97M-V water-12 and `2.92e-5 Eh/bohr` for DF r2SCAN-3c water-12.
  Reporting energy alone hides a real accuracy concession.
- For example, DF M06-2X water-48 falls back to Standard and both arms have
  `8.2041291e-6 Eh/atom` independent energy error and about `5.6724e-5 Eh/bohr`
  maximum force error. Their equality cannot demonstrate a useful Fast path.
  Dense-reference errors also include quadrature and fitting differences; they
  do not isolate arithmetic or convergence error.

Default Standard is already optimized: qualified CUDA RHF and default-grid
PBE/PBE0/B3LYP/r2SCAN/M06-2X select energy/density controls `1e-9`/`1e-5`.
CPU, unsupported grids/models and WB97M-V retain `1e-10`/`1e-8`.
Admitted Fast Fock routes use `1e-6`/`1e-4`, but that distinction does not apply
to fallback populations. `_scf_convergence.py`, `_budget_fock_controls` and the
executed receipts agree on these choices.

## Why CBS has different convergence and similar energies

The standalone restricted CCSD(T) Python defaults are CC energy `1e-11` and
residual `1e-9`. `CCSDTCBSCalculator` explicitly tightens them to `1e-12` and
`1e-10`; its SCF energy request is `1e-12`. Native reference preparation caps
its `1e-10` density request at `1e-11`. This is intentional policy layering,
not a silent failure to transport Fast settings.

The default CBS coefficient L1 sum is `4.196866214617815`. Fast divides its
composite allowance by that sum before selecting the empirical stopping scale
`0.23827302298009143`. Effective Standard/Fast controls in the retained samples:

| Control | Standard | Fast |
| --- | ---: | ---: |
| SCF energy | `1e-12` | `2.38273e-9` |
| SCF density | `1e-11` | `2.38273e-10` |
| CCSD energy | `1e-12` | `1e-8` |
| CCSD residual | `1e-10` | `2.38273e-7` |
| Lambda residual | `1e-9` | `2.38273e-8` |
| Response absolute solve | `1e-12` | `2.38273e-9` |

Correlated energies depend on the canonical orbitals, and analytic forces need
Lambda and orbital response. The canonical-frame, gap, denominator, physical
replay and stationarity checks survive Fast. Copying a DFT density tolerance
of `1e-4` into this owner would not be a justified harmonization. Equally,
these conservative historical defaults are not demonstrated Pareto optima.
The L1 allowance protects against extrapolation amplification; the subsequent
allowance-to-residual mapping is empirical, not a proved observable bound.
Energy-change, amplitude-residual, density-change and orbital-gradient norms
are different quantities, including in the independent PySCF reference.

CBS C1–C7 complete E+F times fall by 8.8–25.6% in the retained single-call
pairs. C7 changes from 1447.813 to 1093.350 s. Its CC iterations fall 21→13;
Lambda shared actions change from 24 FP64 to 12 mixed plus 8 FP64. TZ MP2
group time barely changes (78.780→75.921 s). Lambda accounts for about 285 s
of the 354 s total reduction; this is a phase attribution, not a separate
additive kernel speedup measurement.

Across that series, composite energy differences are `5.42e-11`–`2.38e-9 Eh`,
and maximum force differences are `2.25e-10`–`1.25e-9 Eh/bohr`. C7 Fast's
observed CC residual is `1.52e-8`, below its `2.38e-7` ceiling, with energy
change `3.29e-9`. Multiple stopping/publication checks and discrete iterative
steps can overshoot a ceiling. Small observed errors do not imply a missing
Fast setting or justify relaxing all consumers at once. Only C2 has the
campaign's independent constituent energies and two-step directional force
check; larger-alkane differences are Standard controls, not independent errors.

## Which optimizations belong in Standard

| Mechanism | Current ownership | Decision for this audit |
| --- | --- | --- |
| Same-basis HF/MP2/CC source reuse | Shared CBS groups | Already benefits Standard |
| Corrected GMRES live-storage bound, larger admitted Lambda tiles | Shared controller | Already benefits Standard; not a precision concession |
| Independent smaller-batch Lambda replay | Shared Lambda owner | Already shared on audited master; homepage CBS curve predates it |
| DF force bitmask crossover, DF response algebra/capacity, exact Fock/DIIS reuse | Shared execution owners | Keep shared; do not credit solely to Fast |
| CCSD and Lambda FP32 preconvergence followed by FP64 solves | Requested only when `convergence.relaxed()` in grouped DF CBS | Leading candidate for Standard with unchanged strict final limits; requires an isolated experiment |
| Fixed FP64 J/K anchors in WB97M-V outer proposals | Fast-only admission | Algebraic reuse is a candidate, but prior strict-FP64 trials had no convincing benefit; do not promote from the Fast result |
| FP32 final force geometry, relaxed stopping, approximate screening | Budget-qualified consumers | Keep behind explicit accuracy allowance |

The fixed-anchor limitation has actual negative evidence in
[the original decision](../implemented/performance/2026-10-18-fast-fixed-anchor-delta.md).
The shared capacity correction is documented in
[the workspace decision](../implemented/performance/2026-10-22-cbs-lambda-workspace.md).
The independent replay change is documented in
[its follow-up](../implemented/performance/2026-10-22-df-lambda-independent-replay.md).

## Proposed next implementation

Decouple preconvergence admission from `CorrelationConvergence::relaxed()`.
Qualify a 2×2 experiment: strict/relaxed final controls crossed with FP64/mixed
preconvergence, using identical shapes, numeric budgets and bounded fallbacks.
Keep independent physical replays, triples, Lambda transpose audits and force
response. Do not simulate the strict/mixed cell by only overriding the four
Python SCF/CC controls: the correlated budget would still relax Lambda and
orbital response. The native owner needs an independent execution-policy input.

Promote strict/mixed to Standard only after independent E+F acceptance and
complete endpoint benefit on cold, replay, moved and resource-refused cases.
Retain failures and phase/work counts; use same-card alternating measurements.
Current `fp64` is a documented alias of Standard, so changing Standard's internal
arithmetic also needs an explicit compatibility decision for that alias.

Then tune Fast's remaining approximation budget against full energy/force
errors on representative systems. An allowed error is a ceiling, not an error
that should deliberately be reached. A larger error without lower endpoint
latency has no useful tradeoff. Do not change basis sets, Hamiltonians or final
audit ownership to manufacture a separation between modes.

## Reproduction and validation

```bash
PYTHONPATH=python:. python -m benchmarks.fast_standard_audit \
  --output .artifacts/fast-standard-audit/summary.json
PYTHONPATH=python:. python benchmarks/results/cbs-fast-memory-20261021/verify.py
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_ccsdt_cbs.py tests/python/test_precision_policy.py
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_scf_convergence_defaults.py -k 'not native'
```

The offline audit retains input hashes, all paired observables' differences,
controls, routes, semantic work and freshly recomputed HF/DFT qualifications.
The existing CBS verifier passes its 14 curve receipts, 12 paired ablation
receipts, 20 independent water endpoints and C2 directional gates. These are
rechecks of archived observables, not new calculations. The two Python policy
suites pass 257 tests; the default-control suite passes 16 host tests with
three native tests deliberately deselected. No compiler or GPU executes.
The four documentation/link/public-contract suites pass another 92 tests and
104 subtests; the complete Sphinx HTML build passes with warnings as errors.
All 128 recomputed HF/DFT qualification records exactly match the retained
records. Local failure injections reject changed geometry and a mismatched
CBS Hamiltonian protocol. Ruff and whitespace checks pass.

At audit time this host's Slurm view exposes only `node3`, state `drain`, in
`main`. No permitted GPU allocation is available here. Therefore this change
documents current semantics and supplies reproducible analysis; it does not
promote mixed Standard, change solver thresholds or claim new-device evidence.

Retained sources:
[HF/DFT campaign](../../../benchmarks/results/df-dft48-work-20261025/README.md),
[CBS timings/work](../../../benchmarks/results/cbs-fast-memory-20261021/timings.md),
[CBS convergence decision](../implemented/numerics/2026-10-20-ccsdt-cbs-fast-convergence.md).
