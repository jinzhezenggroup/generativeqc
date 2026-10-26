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
| Convergence | Default-grid HF/PBE/PBE0/B3LYP/r2SCAN/r2SCAN-3c/M06-2X: energy change 1e-9 Eh, density RMS 1e-5; ωB97M-V: 1e-10 / 1e-8. Large eligible Direct HF owners use bounded screened FP64 proposals before target refinement; final physical/canonical validation remains enabled | Default energy change 1e-9 Eh, orbital-gradient norm about 3.16e-5 |
| Iteration cap / DIIS history | 100 / 8 | Defaults 50 / 8 |
| Integral screening | FP64 final work at 1e-12; bounded native incremental proposals retain their finite policy, with restricted primitive pure-J omission capped at 1e-9 only during explicit provisional actions | Default direct-SCF tolerance 1e-13 |
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
no grid or tolerance was changed to improve publication status. Figure qualification is limited to the sampled static geometries. This refresh
adds no production optimization or new displaced-geometry qualification. The
implementation evidence retains independent 96/120-atom original/warm/moved
checks and explicit graph/proposal-lifetime tests.

The original oracle receipts and audited reuse manifest are copied byte for
byte from the preceding retained campaign. Offline reproduction compares exact
geometry, orbital basis, representation, charge, spin and frozen auxiliary
inputs; it then checks reference producer identities, controls, package versions,
grid stability, CPU/GPU spots and every energy/force sample. All figure timings
are newly measured using the single frozen `r2-default-v2` source, based on
`42b266917f6676ecf0559fb8965a4e19f60d0fe3` plus the retained `source.patch.xz`.
The dirty source archive SHA-256 is
`e4af728d14fc6279faa146748b65fcc38d87b98dbe5e0ee0233d3df5658b6b71`.
The parent SHA alone does not identify the modified implementation. No earlier
cohort or the separate three-round qualification supplies a figure timing.

Cold includes object construction, preparation, SCF, complete forces and host
export. It is application-object cold with installed/prebuilt code; process
startup and compilation are excluded. Matching core and stationary AOT libraries
are built before measurement. Warm is the median and full range of three calls
starting from the engine's own frozen cold density.

All 64 scientific pairs are assigned to seven finite Slurm workers using the
preceding cohort's estimated cost, without selecting prior timing winners.
Each pair runs consecutively on the same allocated RTX 5090; first-arm order
alternates by case index. Every timing is fresh. This figure uses one process
per engine/case; separate three-round qualification of the six formerly slower
cold endpoints is retained in the implementation evidence. The source, runner
and native library are identical across all 128 populations. `jobs.json.xz`
retains the complete plan, commands, scheduler job IDs, assigned device
visibility and exit codes.

The measured execution schedules retain the existing explicit controls
`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` and
`GENERATIVEQC_SCF_VALIDATION_GEMM=1`. These are not unset-source defaults.
Math libraries use one thread; all seven workers reserve eight CPU cores. No experimental screening, memory-cap or
profiler controls enter the figure measurements. Slurm device visibility is
preserved. The accelerator snapshot precedes measurement; its inherited
`sampling_point` label says “after benchmark measurements” and does not establish
post-run clock/temperature stability.

Separate intrusive MD-J work-counter diagnostics belong to
[`cold-r2-defaults-20261026`](../cold-r2-defaults-20261026/README.md); their
timings never enter these figures. Full compiler output and binaries remain in
ignored artifacts. Compact source audit, core/AOT hashes, build settings and
verified sccache 0.16.0 before/after shared-cache statistics are retained in
`provenance.json.xz`. CCSD(T)/CBS inputs, timings, text and figure remain unchanged.

The DF PBE0-96 population in this figure is slower in cold than its paired
reference despite the separate three-round campaign retaining a faster native
median. Its three warm calls are also slower than the native regression
population with the same iteration count. The cause is not established; the
original complete population is retained rather than replaced by a faster
repeat. Accelerator snapshots do not establish clocks throughout timed calls.

Figures use the original Matplotlib 3.11.2 renderer; version, FreeType and font
identity are retained in provenance. Install `render-requirements.txt` for
byte-identical replay. Layout, labels and rendering code are unchanged.
