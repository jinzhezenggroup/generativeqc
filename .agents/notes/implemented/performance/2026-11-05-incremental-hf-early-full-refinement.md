# Decision: overlap incremental HF full-energy refinement with the SCF tail

Status: implemented; incremental J/K remains opt-in
Date: 2026-11-05

Later qualification: [96-atom default-promotion decision](../../rejected/2026-11-05-hf-incremental-default-promotion.md)
retains opt-in status after the cold-start effect fails the performance floor.
The [source-matched final publication](../../../../benchmarks/results/hf-incremental-composition-20261105/README.md)
retains independent E/F requalification of the delivered repair.

## Problem

The [convergence-composition repair](../numerics/2026-11-05-incremental-hf-convergence-composition.md)
unified HF scientific acceptance and retained-state ownership, but started
full-energy confirmation only after a delta evaluation passed density and
physical stationarity. Confirmation then extended the otherwise finished SCF
tail. Standard cold 96/192/384-AO fixtures took eleven target Focks with deltas,
versus ten without: three full and eight delta builds, plus the same seven
preliminary native Huckel Focks. That extra iteration outweighed the modest
savings from delta contraction. Those measurements do not measure process/JIT
cold start: their cold endpoint includes calculator construction, preparation,
and complete energy/forces after a common process bootstrap.

## Decision

Start sticky full-only refinement when both density RMS and physical maximum
are within one decade of their ordinary owner-qualified limits. The factor ten
is only a conservative phase-selection lookahead, not an error bound or a new
convergence tolerance. It allows full-energy baselines to be collected during
the remaining physical SCF tail without adding device storage, allocations,
host polling, or another public tuning switch.

The ordinary energy, density, physical-residual, finite-value and finalization
checks are unchanged. A current delta energy is still discarded; a current full
evaluation can still establish the first fresh baseline. After actual deltas,
at least two consecutive full-density energies must pass ordinary closure.
Full-only mode is sticky even if later progress regresses, and consumes the
original iteration budget. No-delta warm solves keep ordinary acceptance and
do not acquire an extra confirmation requirement.

## Rejected alternatives

- Remove full-energy confirmation or reuse a delta energy baseline: screening
  and FP64 reconstruction do not certify full-density physical energy closure.
- Loosen final acceptance: unrelated to selecting a cheaper schedule and would
  invalidate the previous convergence repair.
- Add a new density-dependent anchor-refresh heuristic simultaneously: early
  fixed-anchor deltas need not be small, but that is a separate tuning question.
  The existing refresh cadence and optional upper-RMS admission are unchanged.
- Assume an identical trajectory: linear exact J/K does not make screened,
  finite-precision reconstructed Focks bitwise identical to full builds.

## Validation

Native RHF/UHF regressions cover Standard/strict controls and near-stationary
density/physical residuals independently. They require full-only transition but
reject convergence throughout the scheduling margin, including after repeated
full-energy observations. Current full, delta and no-delta baselines remain
distinct. Existing nonfinite, inactive, coarse-stage, retained-state, resource
fallback and independent-reference endpoint checks remain required.

Final source-matched qualification is Slurm job `15975`, RTX 5090, Release/AOT
FP64, spherical all-electron def2-SVP. The native source identity is
`8fca79987ab68da5d27d12343a694c3c919e400b550ef836a4598d1a47b7adaa`;
library SHA256 is
`d14859497a4038414aa6c4ebb9e8f5f0a435d485ece13b728f0d6d6cf49b0e8d`.
The incremental build uses verified sccache 0.16.0 with an owned foreground
server, recorded compiler commands and before/after cache statistics.

- Both native GPU suites pass, including the new early-transition tests.
- All four live independent CPU PySCF endpoint-composition cases pass.
- All 144 formal energy/force endpoints pass retained independent CPU PySCF
  gates. Every target has one attempt, zero discarded target work and no warm
  fallback. Each cold arm also performs seven native Huckel preliminary Focks.
- Both native suites pass compute-sanitizer memcheck with zero errors; all 43
  focused fixed-anchor/default-control host tests pass.
- Modified native files pass clang-format; `git diff --check` passes.

Standard gates remain energy error <= `2.5e-6 Eh/atom`, force maximum <= `2.5e-5`
and RMS <= `7.5e-6 Eh/Bohr`. Observed maxima are `3.36514e-11 Eh`,
`6.33211e-7` maximum and `2.66119e-7 Eh/Bohr` RMS. Strict gates remain absolute
energy error <= `1e-8 Eh`, force maximum/RMS <= `1e-7 Eh/Bohr`; observed maxima
are `1.00045e-11 Eh`, `1.76271e-10` maximum and `7.19515e-11 Eh/Bohr` RMS.

### Complete endpoints

Four independent cold/frozen-warm/moved sequences per arm, AB/BA/BA/AB, screening
`1e-12`, refresh interval eight. Cold includes calculator construction,
preparation, and complete host energy/forces; imports and a common tiny-water
bootstrap are excluded. Cleanup is recorded separately. Warm and moved each
replay that run's own frozen cold density, never an oracle density.

Medians in seconds, **OFF / ON**, from the final source only:

| AOs | Standard cold | Standard warm | Standard moved | Strict cold | Strict warm | Strict moved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.395629 / 0.378729 | 0.121606 / 0.121415 | 0.500069 / 0.485274 | 0.478666 / 0.472082 | 0.168672 / 0.167700 | 0.788319 / 0.739268 |
| 192 | 1.594531 / 1.560767 | 0.262807 / 0.262238 | 1.354334 / 1.201590 | 2.426034 / 2.181112 | 0.263570 / 0.262188 | 2.075625 / 1.727600 |
| 384 | 5.258145 / 5.209401 | 0.763885 / 0.761610 | 4.039501 / 3.404368 | 7.731370 / 6.678792 | 0.765874 / 0.762572 | 6.615574 / 4.727516 |

Standard cold now has ten target Focks in both arms: OFF ten full; ON three full
plus seven deltas, with no post-SCF Fock. Standard moved has seven target Focks:
OFF seven full; ON three full plus four deltas; both perform two post-SCF full
builds under the unchanged moved-seed finalization fallback. Standard frozen
warm uses one full target Fock and no delta/post-SCF build in both arms.

Strict cold target work, **OFF full / ON full+delta**, is `18 / 5+13`,
`16 / 5+11`, and `16 / 4+12` for 96/192/384 AOs, respectively; no post-SCF build.
Strict moved target work is `14 / 4+10`, `13 / 4+10`, and `14 / 4+10`, with one
post-SCF full build per arm. Strict warm is two full steps at 96 AOs and one at
192/384, without deltas or post-SCF builds. Equal numerical acceptance does not
imply equal iteration counts: the 192-AO strict moved ON arm still needs one
more target iteration than OFF, while its complete endpoint is faster.

Quartet work remains unavailable for these streaming 192/384-AO runs and the
resident 96-AO cold phase. Do not infer an integral-work ratio from Fock counts
or present absent quartet counters as zero work. Standard cold's measured
benefit is modest, roughly 1–4%; sub-percent warm differences are not evidence
of a meaningful speedup. No default promotion or universal speedup is implied.

### Baseline and retained artifacts

Before building, preserve the prior `be3b91e2...` library (`cc9214f9...`) and its
build receipts under the isolated trial's `cold-baseline/`. A sequential
same-job replay in `15971` independently reproduces its eleven Standard cold
target Focks (three full plus eight deltas). The improved schedule uses ten.
Absolute cold timings also drift in the unchanged OFF control during that
sequential replay; those cross-source medians are not a paired speedup claim.

Local ignored evidence root: `.artifacts/delta-jk-20261105/`:

- `cold-qualified/`: final-source job `15975`, raw samples and complete reports.
- `cold-qualified-sanitizer/`: same-job memcheck, host tests and cache receipts.
- `build-cold-qualified-receipts/`: final build identity, commands and cache use.
- `cold-baseline-comparison/`: job `15971` baseline 384-AO Standard/strict
  reports and its successful pre-documentation sanitizer/host receipts.
- `cold-intermediate/` and `cold-pre-doc-final/`: successful jobs `15968` and
  `15971` before final comment/format cleanup; not substituted for final timing.

The isolated remote trial is `/data/jzzeng/qc888-delta-jk-20261105/` on `n1`.
`benchmark.py`, `run-cold-final-n1.sh`, and all failed/earlier evidence remain
available; the final run uses `--no-baseline` because the baseline replay is
already preserved. It never bypasses Slurm GPU visibility.

## Consequences

Starting full-only earlier may trade useful late delta builds for full builds
on other systems. Complete endpoint timing, not only iteration counts, decides
whether the trade is useful. Preliminary guesses, direct integral screening,
bounded resident selection, moved-seed fallback, and gradient work remain part
of the endpoint and must not be omitted from performance claims.

## Revisit when

Broader systems show material regressions, or a separately qualified estimate
can select refinement more accurately without weakening physical closure or
adding an unbounded state/resource dependency.
