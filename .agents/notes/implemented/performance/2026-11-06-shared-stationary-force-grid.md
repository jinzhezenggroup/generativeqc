# Decision: share strict stationary XC/VV10 force panels

Status: implemented
Date: 2026-11-06

## Problem

Warm WB97M-V endpoints already execute one Fock and can reuse exact VV10 pair
seeds, but the native force controller evaluated identical AO derivative panels
and density jets twice: once for semilocal geometry and again for nonlocal
geometry. The retained diagnostic in
`benchmarks/results/vv10-packet-fp64-fma-20261106/diagnostic.json.xz` records
2,156,672 evaluated points for a 1,078,336-point physical grid. Another cold SCF
threshold or pair-kernel tuning sweep cannot remove that duplicated warm work.

## Decision

Keep a constructor-owned proof of shared device/coordinate pointer/span,
points-per-atom and equal atom-prefix/radius topology. In a strict-FP64 force
call, evaluate the first panel and obtain its valid stream before preparing
nonlocal force seeds from the authenticated final SCF feature lease. Then
consume semilocal and nonlocal geometry from each same-generation panel, on
that stream, before advancing the grid. Each consumer keeps its own weights,
atomic weights and generated source slots. Their transient geometry scratch is
stream-ordered, not shared concurrently.

The existing event orders the borrowed final feature producer. Exact-key
comparison, error-aware force-seed publication, pair schedule identity and the
incomplete-call RAII invalidation remain unchanged. There is no additional
full-grid allocation, retained density, resource tier or CPU scientific work.

Different point owners, topology or points-per-atom, and mixed grid/geometry
precision retain the original two-pass path. The selector
`GENERATIVEQC_STATIONARY_SHARED_GRID=0` disables sharing for causal qualification
and rollback; unset/empty/1 uses the guarded default. There is no GPU-model,
functional-name or molecular-size performance whitelist.

## Rejected alternatives

- Obtaining a stream from a fresh unevaluated grid: there may be no valid view.
- Sharing only by point count or coordinate values: neither authenticates panel
  identity or the installed stationary moving-grid topology.
- Sharing an FP32 first-pass panel with the original FP64 nonlocal consumer:
  that changes derivative precision rather than just eliminating duplicate work.
- Retaining all AO/density-jet panels: unnecessary full-grid memory and admission
  cliffs; sequential same-generation consumption has bounded existing storage.
- Reusing stale SCF solver history: unrelated to the dominant warm force work.

## Invariants and evidence

Independent original/displaced complete E/F, distinct-grid and minimum-capacity
fallbacks, exact-key replay and failure recovery must remain qualified. Numerical
gates, pair screening, physical SCF convergence and all force sources are unchanged.
The two consumers still execute both geometry contractions; only duplicate panel
production is removed. Semantic projection FMA pairs are not hardware instruction
counts, and halving panel work does not halve the complete endpoint.

Source-matched actual A100/RTX5090 evidence and reproduction live in
`benchmarks/results/shared-force-grid-20261106/README.md`. Complete endpoint timing
includes constructor/preparation for cold, SCF, analytic forces and host export.
After all timed samples, an additional frozen-warm replay of the same owner
records force work with intrusive tracing. It is not a clean timing sample:
DF tracing can disable optional MD-J. The initial collector attempted the
automatic-artifact ABI, but that ABI refuses curated WB97M-V recipes; preserve
that failure rather than relabeling its interrupted endpoints as measurements.

The actual A100 LSF225900 warm median is 15.012093 -> 14.831869 s (-1.20%);
RTX5090 Slurm16589 is 14.915725 -> 13.649965 s (-8.49%). Both GPUs halve the
duplicated panel/projection census while retaining both geometry contractions
and zero warm force pair sweeps. Forty independent observable comparisons pass;
worst force max is below 3.553e-7 Eh/bohr. Eleven production-image numerical/work
cases and 271 host tests pass. The production build lacks the deliberate runtime
failure symbol; a separately identified hook-only image passes that recovery
case and is excluded from performance ranking.

A100 cold is +0.89%, and cold plus three warm calls +0.30%; do not claim a cold
or complete aggregate A100 win. The RTX5090 cold shortening is too large to
attribute to this removed panel work. This is one bounded sequential paired
cohort, not a universal speed guarantee or a fresh GPU4PySCF latency comparison.
The final PR also includes documentation/evidence-only master b681a1fc3, whose
production closure is byte-identical to the measured source. Retain these limits
when summarizing the modest A100 versus stronger RTX5090 warm benefit.

## Revisit when

New producer precision, topology ownership, streams or geometry consumers are
introduced, or complete scientific/resource/two-GPU gates show a regression.
Any extension needs independent observables and complete endpoint qualification,
not a panel microbenchmark alone. Tracking memory: #2976; separate from #3020's
provisional-SCF handoff.
