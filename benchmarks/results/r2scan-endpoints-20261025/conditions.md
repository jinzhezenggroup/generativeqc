# Comparison conditions

The #2786 homepage framework measures eight methods, Direct/DF, and
3/12/48/96-atom water clusters, with default GenerativeQC and stock GPU4PySCF.
Every fresh process measures one cold and three warm complete energy-and-force
calls. Each warm call starts from that engine's own frozen cold density. No
reference density enters production. The original 2×4 layout and cold/warm
encoding are preserved. Every completed point appears, including failed
independent accuracy gates; the tables retain that distinction.

| Condition | GenerativeQC | GPU4PySCF 1.8.1 |
| --- | --- | --- |
| Geometry/state | Identical Bohr coordinates, charge 0, spin 0 | Same |
| Orbital basis | Spherical def2-SVP; r2SCAN-3c uses def2-mTZVPP | Same |
| r2SCAN-3c corrections | Native D4 + gCP, included in energy, forces and timing | Independent D4 + gCP, included |
| XC/VV10 grids | Default level-3 Treutler/Lebedev, NWChem pruning | Installed defaults, same prescription |
| Moving-grid forces | Full coordinate and partition response | Full response, explicitly enabled with `grid_response=True` |
| VV10 density cutoff | 1e-10 for energy and forces | Default 1e-10 for both |
| VV10 weight screening | Nonzero signed weights and derivatives | Absolute weights greater than 1e-14 |
| Convergence | Default-grid HF/PBE/PBE0/B3LYP/r2SCAN/r2SCAN-3c/M06-2X: energy change 1e-9 Eh, density RMS 1e-5; ωB97M-V: 1e-10 / 1e-8. Final physical/canonical validation remains enabled | Default energy change 1e-9 Eh, orbital-gradient norm about 3.16e-5 |
| Iteration cap / DIIS history | 100 / 8 | Defaults 50 / 8 |
| Integral screening | FP64 final work at 1e-12; bounded native incremental proposals retain their existing policy | Default direct-SCF tolerance 1e-13 |
| XC AO screening | Qualified AO-jet selection at 1e-16 with bounded dense fallback | Shell screening with `AO_THRESHOLD=1e-10` |
| Cold initial density | Native Hückel; preliminary work retained in receipts | Default MINAO |
| Direct Fock reuse | Public automatic policy, including bounded cold pure-J proposals at qualified sizes; full-density final closure | Default incremental Direct Fock; DF wrapper sets `direct_scf=False` |
| DF auxiliary basis | Exact frozen default reference tables, including auxiliary g shells | Default selection checked against frozen tables |
| DF metric | Eigensystem, relative cutoff 1e-10 × largest eigenvalue | Cholesky first; fallback eigenvalue cutoff 1e-7 absolute |
| DF pure-J representation | Default resource/work policy, including bounded Hermite streaming and metric-aware primal omissions | Installed defaults |
| ωB97M-V DF exchange | Fitted full-range exchange and exact long-range correction | Fitted full-range and long-range exchange |
| Resources | One Slurm-allocated RTX 5090 on n1, one math thread | Same |

Solver residuals have different meanings. Common observable gates are energy
error ≤1e-5 Eh/atom, maximum force error ≤1e-4 Eh/Bohr and RMS force error
≤3e-5 Eh/Bohr. Reference-grid stability and CPU/GPU spot comparisons use quarter
gates. DF fitting error is included. Default-grid failures remain visible;
no grid or tolerance was changed to improve publication status. Figure
qualification is limited to sampled static geometries. Separate original,
warm, displaced and fresh-displaced checks at 96/120 atoms qualify the changed
implementation against independent GPU4PySCF endpoints.

The original oracle receipts and audited reuse manifest are copied byte for
byte from the preceding retained campaign. The current driver changes only
work metadata. Offline reproduction verifies exact geometry, orbital basis,
representation, charge, spin and frozen auxiliary-input identities against that
campaign; the qualifier additionally checks oracle producer/driver identities,
package versions, grids, controls and every energy/force sample. Old timing
values enter only the explicitly labeled historical comparison.

Cold includes construction, preparation, SCF, complete forces and host export.
It is application-object cold with installed/prebuilt code; process startup
and compilation are excluded. Matching core and stationary AOT libraries and
manifests are built before measurement. Warm is the median and full range of
all three calls. Both arms of a scientific pair run consecutively on the same
allocated card; first-arm order alternates across pairs. Seven workers cover
all 128 populations. This is one fresh-process cohort, not a repeated-process
statistical significance study.

The measured execution schedules retain the existing explicit controls
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` and
`GENERATIVEQC_SCF_VALIDATION_GEMM=1`. These are not unset-source defaults.
No experimental screening, memory-cap or profiler overrides enter the timed
jobs. The scheduler's device visibility is preserved. The accelerator snapshot
precedes the calls; its inherited `sampling_point` label says “after benchmark
measurements” and does not establish post-run clock/temperature stability.

Source, core/AOT hashes, actual compiler commands, verified sccache 0.16.0,
before/after cache statistics and finite Slurm allocations are retained.
Exploratory, instrumented and cancelled trials are separate evidence and never
supply figure timing points. Public Schwarz and the DF metric/gradient cases
pass CUDA memcheck; no new full-device-graph synccheck or racecheck pass is
claimed in this campaign. CCSD(T)/CBS inputs, timings, text and figure are
unchanged.
