# Observation: where complete range-separated DF removes endpoint cost

Status: implemented (measurement; not release-performance qualification)
Date: 2026-10-15

Superseded numerical diagnosis: the
[whitened-response decision](../numerics/2026-10-15-stable-whitened-df-response.md)
fixes the cancellation exposed here. These original negative observations and
their source-matched receipts remain historical evidence; later measurements
must use a separate final-source evidence set.
The qualified follow-up is recorded in
`2026-10-15-stable-range-df-endpoints.md` and the stable campaign directory.

## Decision

Retain a source-matched, reproducible comparison of the existing mixed
DF-J/K + Direct-LR composition and the new complete RI-J/K composition.
Do not interpret a fixed-geometry SCF replay as representative of geometry
optimization, or a numerical CUDA qualification build as an optimized Direct
baseline. Do not promote practical-auxiliary force performance on timings alone.

The data and runnable benchmark/oracle/report harnesses are in
`benchmarks/results/range-separated-df-20261015/`. That evidence owns the exact
settings, source/library identities, raw receipts, numerical gates and timing
ranges. The implementation rationale remains in
`../numerics/2026-10-15-range-separated-density-fitting.md`.

## Findings

- CPU fixed-geometry energy replay changes little or regresses slightly:
  the old prepared Direct owner already retains its molecular LR values.
  Removing its creation/recreation cost matters at preparation and geometry
  changes, so cold totals must include preparation.
- At water/def2-TZVP with a practical cc-pVDZ-JKFIT auxiliary, complete-DF
  energy cold total is about 5.1 times faster and changed-geometry SCF about
  5.9 times faster. The independent energy oracle passes without changing
  its gate. Fixed-geometry energy replay is approximately 3% slower.
- The small, independently qualified water/def2-SVP auxiliary force endpoint
  improves approximately 1.5–1.7 times on CPU. The larger practical-auxiliary
  CPU force endpoint removes substantially more time, but those ratios are
  exploratory because its independent derivative gate does not pass.
- CPU Direct LR quartet semantic passes change from one to zero; the measured
  canonical domains grow from 45,150 to 447,931 between the water orbital cases.
  VV10's 26,873,856 water grid-pair evaluations remain unchanged. CUDA quartet
  counts are not exported; keep them unknown rather than inventing counters.
- GPU endpoint observations favor complete DF, but three legacy Direct
  translation units use `--Ofast-compile=max` in that library. Their ratios
  cannot establish normally optimized Direct or release performance.

## Qualification limits discovered

The practical cc-pVDZ-JKFIT force extension is not covered by the previous
small-auxiliary acceptance. At the timing convergence tolerances, independently
reconverged displaced libcint energies give a water force-component discrepancy
of approximately `4.41e-7 Eh/Bohr` on CPU and `4.83e-7` on CUDA against the
unchanged `3e-7` gate. CPU fixed-geometry fitted-force replay also varies at the
`1e-6` level. The old mixed baseline and the new small-auxiliary endpoint pass.
A stricter CPU SCF probe improves one component but does not qualify all force
components, replay, geometries or the practical-auxiliary domain. The root cause
is not diagnosed by these measurements; preserve the negative evidence.

The water dimer at the same per-atom grid completes both energy endpoints but
both CPU force models fail the existing grid-pair work limit. Complete fitting
does not make the nonlocal/grid owner work-bounded on a larger admitted domain.

## Revisit when

Repeat the matched GPU measurements with normally optimized Direct objects;
qualify practical-auxiliary LR force stability with the independent gates;
and separately qualify the production grid, larger molecules, batch and
constrained-memory behavior. Do not silently substitute a different metric
cutoff, a smaller grid or a larger admission limit to label these data accepted.
