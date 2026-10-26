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
implementation publications retain their original endpoint and lifetime gates.
The DF headroom repair additionally retains 96-atom moved-geometry and HF
parity, independent references and constrained-capacity fallback checks.

The original oracle receipts and audited reuse manifest are copied byte for
byte from the preceding retained campaign. Offline reproduction compares exact
geometry, orbital basis, representation, charge, spin and frozen auxiliary
inputs; it then checks reference producer identities, controls, package versions,
grid stability, CPU/GPU spots and every energy/force sample. All figure timings
are newly measured from committed source
`46b916ac6df0082d56fbc48639bc1261aecc7d04`, including the DF shell-screening
headroom repair. The clean Git archive identity is retained in
`provenance.json.xz`. All 2,770 relevant source/header/compiler/runner files
match the previously qualified candidate tree. The exact candidate library is
reused; its inherited embedded source identity is not represented as the new
commit. The linked library SHA-256 identifies execution. No earlier cohort or
separate three-round qualification supplies a figure timing.

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
Math libraries use one thread; all seven workers reserve eight CPU cores, 16 GB and a 45-minute time limit.
Jobs are admitted individually as GPUs become available. No experimental
screening, memory-cap or profiler controls enter the figure measurements. Slurm
device visibility is preserved. Each accelerator snapshot is labeled before
measurement and bound to the executed CUDA PCI address and GPU UUID. Such a
snapshot does not establish clocks throughout timed calls.

Separate intrusive work-counter diagnostics belong to the
[DF headroom investigation](../df-screening-headroom-20261026/README.md) and
[earlier cold qualification](../cold-r2-defaults-20261026/README.md); their timings
never enter these figures. The repair used verified sccache 0.16.0 to rebuild
the changed CUDA bridge and independently relink verified parent objects. Its
source-matched build evidence and limitations remain in that publication. This
refresh performs no new compilation. Core/AOT hashes and source parity are
retained in `provenance.json.xz`. CCSD(T)/CBS inputs, timings, text and figure
remain unchanged.

The entire preceding cohort, including its slower DF PBE0-96 observation, is
recoverable byte for byte from existing Git history using `previous-cohort.json`.
This refresh replaces every timing population together. The previous figure's
memory-budget remainder disabled optional screening; the new source reserves
space for norms during tile planning while retaining strict fallback.

Figures use Matplotlib 3.11.2; version, FreeType and font identity are retained
in provenance. Both axes are linear and the time axis starts at zero. The 2×4
layout, labels, colors and cold/warm encoding are preserved. Install
`render-requirements.txt` for byte-identical replay. Historical publications
retain the renderer's default log-log axes.
