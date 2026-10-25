# Comparison conditions

Eight methods, Direct/DF and 3/12/48/96-atom water clusters use native
Standard/Fast and one GPU4PySCF default SCF profile. Each independent process
measures one complete cold energy-and-force call and three warm calls from its
own frozen cold density. All complete samples appear as performance curves, including accuracy
failures.
Independent accuracy status remains in the tables and receipts.

| Condition | GenerativeQC | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Geometry/state | Identical Bohr coordinates, charge 0, spin 0 | Same |
| Orbital basis | Spherical def2-SVP; r2SCAN-3c uses def2-mTZVPP | Same |
| r2SCAN-3c corrections | Native D4 + gCP, included in energy, forces and timing | Independent D4 + gCP, included |
| XC grid | Default level-3 Treutler/Lebedev with NWChem pruning | Installed default, same prescription |
| VV10 grid | Default level-3 Treutler/Lebedev with NWChem pruning | Installed default nonlocal grid, same prescription |
| Moving-grid forces | Full coordinate and partition response | Full response, explicitly enabled with `grid_response=True` |
| VV10 density cutoff | 1e-10 for energy and forces | Default 1e-10 for both |
| VV10 weight screening | Nonzero signed weights and their derivatives | Absolute weights greater than 1e-14 |
| Standard convergence | Qualified default-grid HF/PBE/PBE0/B3LYP/r2SCAN/r2SCAN-3c/M06-2X: energy change 1e-9 Eh, density RMS 1e-5; ωB97M-V retains 1e-10 / 1e-8. Final physical/canonical validation remains enabled; exact per-call settings are retained | Default energy change 1e-9 Eh and orbital-gradient norm about 3.16e-5 |
| Iteration cap / DIIS history | 100 / 8 | Defaults 50 / 8 |
| Fast | Public `PrecisionCalculator(policy="fast")`; executed proposals, refinements and fallbacks retained | No GPU4PySCF Fast curve |
| Integral screening | Standard 1e-12; admitted Fast proposals up to 1e-10 | Default direct-SCF tolerance 1e-13 |
| XC AO screening | Qualified AO-jet selection at 1e-16 with bounded dense fallback | Shell screening with `AO_THRESHOLD=1e-10` |
| Cold initial density | Native Hückel | Default MINAO |
| Direct Fock reuse | Native defaults; ordinary incremental J/K off, qualified Fast nonlocal anchors retained | Default incremental Direct Fock retained |
| DF auxiliary basis | Exact frozen default reference tables, including auxiliary g shells | Default selection checked against those tables |
| DF metric | Eigensystem, relative cutoff 1e-10 × largest eigenvalue | Cholesky first; fallback eigenvalue cutoff 1e-7 absolute |
| DF pure-J representation | Resident raw/fitted factors or bounded native Hermite contractions under the default resource policy | Installed defaults |
| ωB97M-V DF exchange | Fitted full-range exchange and exact long-range correction | Fitted full-range and long-range exchange |
| Resources | One Slurm-allocated RTX 5090 on n1; one math thread | Same |

The convergence norms differ. Equal numeric tolerances would not establish
common accuracy. The selected protocol retains each public solver policy and
checks every cold/warm sample against independent direct references: energy
error ≤ 1e-5 Eh/atom, maximum force error ≤ 1e-4 Eh/Bohr and RMS force error
≤ 3e-5 Eh/Bohr. Reference-grid stability and CPU/GPU spot comparisons must pass
quarter gates. DF fitting error is included. Original oracle producer identities
and audited input-reuse contracts remain intact; oracle densities never seed
production calculations. Qualification of the plots is limited to these sampled
static geometries. Separate changed-geometry tests qualify the new implementation.

Density-change stopping is already enabled. Incremental J/K instead uses delta D
to reduce operator work; these are different controls. Standard's final-state
proof removes a conservative extra iteration only when the unchanged physical
and canonical state checks already pass. It does not adopt GPU4PySCF's stopping
norm or silently loosen caller tolerances.

Fast is method dependent. Admitted Direct candidates and default-grid pure-J DF
PBE/r2SCAN (including r2SCAN-3c) can use E/D/screening ceilings of
1e-6/1e-4/1e-10 and must close with strict physical Focks. The DF route narrows
shared XC products while J and metric work remain FP64. Separately qualified
semilocal force arithmetic is available over its bounded model/shape domain.
Other DF models retain strict SCF with qualified force arithmetic or a strict
fallback. Explicit tighter controls and observable-audit targets remain intact.
The [work table](work.md) includes all exposed solves, rather than inferring
work from the Fast label. A strict fallback or tiny calculation need not be
faster; small differences between independently allocated cards are observations,
not significance claims.

Equal quadrature does not mean identical ordering, alignment, active-domain
screening or arithmetic. For the 96-atom semilocal grid, native stores 1,078,336
points; the reference's 256-entry alignment stores 1,078,528, including 192
zero-weight padding entries. That is not a different grid. Likewise, unlike
metric/AO screening thresholds cannot be ranked by their numeric values alone.
Stock Direct uses `direct_scf=True`; the reference DF wrapper sets it to `False`.

Cold includes construction, preparation, SCF, complete forces and host export.
It is an application-object cold measurement with installed/prebuilt code, not
process startup or a clean compiler-cache build. Matching core and stationary
AOT libraries/manifests are built before measurement; compiler caches are
preserved. All production precision validation is timed. Independent publication
qualification runs outside the endpoints. Warm reports the median and full range
of three calls, each starting from the same engine-local cold density.

Complete device-launch Graph synccheck remains instrumented-unqualified on the
measured driver/tool stack. A standalone two-kernel reproduction with an
unconditional barrier demonstrates the same failure in two sanitizer versions;
ordinary launch/Graph controls, all output checks, memcheck and racecheck pass.
The original logs and explicit qualification assessment are retained. Full
provider synchronization checks use supported launches; no full device-graph
synccheck pass is claimed.

All 192 populations use one frozen source and matching binary. Eight finite
Slurm workers preserve scheduler GPU visibility; no grid, convergence, memory-cap
or diagnostic override enters the timed jobs. Explicit spatial bisection
(`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2`) and FP64 validation products
(`GENERATIVEQC_SCF_VALIDATION_GEMM=1`) select the measured execution schedules. Measurements on separate cards are not an interleaved repeated-process study. The accelerator snapshot precedes
the timed calls; its inherited `sampling_point` label says “after benchmark
measurements” and is not evidence of post-run clock or temperature stability.

CBS and moved-geometry timings are unchanged. Historical ranking changes caused
by the earlier comparator's stricter GPU4PySCF convergence and disabled Direct
incremental Fock are documented in the
[settings investigation](../../../.agents/notes/implemented/performance/2026-10-20-readme-default-alignment-and-ranking-reversal.md).
They are not attributed to native speedups in this campaign.
