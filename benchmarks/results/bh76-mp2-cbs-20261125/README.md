# MP2-assisted CCSD(T)/CBS on HTBH38/08 and NHTBH38/08

Refs #3168. This is a fixed-geometry, classical electronic barrier-height
comparison at GenerativeQC master
`22f72d0692a760c2892cd7791022584d9ae1c3e5` (2026-11-25). The two recipes share the
double-zeta CCSD(T) correction. Each requests a native CUDA energy-only
single point for the transition structure and each isolated fragment or
bound complex in the corresponding entrance or exit channel.

## Round 2 production results

| Recipe | Set | Round 1 endpoints | New round 2 endpoints | Cumulative endpoints | Round 1 barriers | Cumulative barriers | Cumulative MAE | Max abs. error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DT_D | HTBH38 | 14/40 | 9 | 23/40 | 2/38 | 8/38 | 0.704 | 1.287 |
| DT_D | NHTBH38 | 30/47 | 6 | 36/47 | 18/38 | 21/38 | 0.820 | 2.373 |
| DT_D | BH76 | 44/87 | 15 | 59/87 | 20/76 | 29/76 | 0.788 | 2.373 |
| TQ_D | HTBH38 | 11/40 | 1 | 12/40 | 2/38 | 2/38 | 0.330 | 0.330 |
| TQ_D | NHTBH38 | 24/47 | 5 | 29/47 | 10/38 | 18/38 | 0.290 | 0.706 |
| TQ_D | BH76 | 35/87 | 6 | 41/87 | 12/76 | 20/76 | 0.294 | 0.706 |

All errors in this section use kcal/mol. On the 20 common barriers
(26.3% of BH76), DT_D gives MAE 0.739 and maximum absolute error 2.373.
TQ_D gives MAE 0.294 and maximum absolute error 0.706. All 20 TQ_D
errors and 14 of the 20 DT_D errors are within 1 kcal/mol. The
recipe-specific MAEs in the table use all complete barriers for each
recipe.

The cumulative energies retain successful round 1 endpoints and add
round 2 successes under the same scientific model. Some complete
barriers combine energies from both builds. DT_D had already assembled
each barrier in the new common subset in round 1.

Against the retained conventional larger-basis CCSD(T) anchors, DT_D
covers 11/18 with MAE 0.502 and maximum absolute difference 1.648.
TQ_D covers 4/18 with MAE 0.253 and maximum absolute difference 0.538.

| Recipe | Stage | Protocol limit | Resource limit | SCF convergence | Development required | CCSD convergence | Other or unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DT_D | Round 1 | 13 | 29 | 1 | 0 | 0 | 0 |
| DT_D | After round 2 | 19 | 0 | 6 | 2 | 1 | 0 |
| TQ_D | Round 1 | 7 | 6 | 7 | 32 | 0 | 0 |
| TQ_D | After round 2 | 0 | 0 | 0 | 45 | 1 | 0 |

The protocol column counts host-side per-point timeouts. The resource
column counts scalar-work and numeric-memory refusals. SCF convergence
counts native reference stopping failures. The development column counts
the angular-momentum and eigensolve-input refusals, with one endpoint
count when both apply. CCSD amplitude convergence has its own column.
Both recipes reported CCSD amplitude nonconvergence for C5H8, blocking
HTBH38 directions 37 and 38.

All 95 round 2 records reached a terminal outcome. The production
preflight identified 45 angular-momentum exclusions. Their round 1
diagnoses were 30 angular-momentum refusals, two DF eigensolve input
refusals, six memory refusals and seven SCF failures. The other 50 points
received a complete energy attempt up to the two-hour limit.

| Recipe | Newly completed | Median (s) | Maximum (s) |
| --- | ---: | ---: | ---: |
| DT_D | 15 | 1030.678 | 6918.818 |
| TQ_D | 6 | 16.522 | 23.318 |

Round 2 recovered 11/13 former DT_D timeouts, two of them within the
original 420 s. TQ_D recovered 6/7 former timeouts, all within 420 s.
DT_D also recovered four former scalar-work refusals. The timing
observations reflect the combined build and resource changes across
both rounds. The durations in the table measure completed energy
endpoints on the allocated H100s.

[Round 2 summary](round2/summary.json) · [Endpoint status by round](round2/endpoints.csv) · [Barrier directions and blockers](round2/barriers.csv) · [Raw records](round2/raw/)

## Round 1 CUDA barrier heights

Errors are against the authors’ scalar REF1 values in kcal/mol.
Only complete barriers contribute to MAE and maximum error.

| Recipe | Set | Complete / planned | MAE | Max abs. error | Within 1 kcal/mol |
| --- | --- | ---: | ---: | ---: | ---: |
| DT_D | HTBH38 | 2/38 | 0.588 | 0.588 | 2 |
| DT_D | NHTBH38 | 18/38 | 0.755 | 2.373 | 12 |
| DT_D | BH76 | 20/76 | 0.739 | 2.373 | 14 |
| TQ_D | HTBH38 | 2/38 | 0.330 | 0.330 | 2 |
| TQ_D | NHTBH38 | 10/38 | 0.217 | 0.392 | 10 |
| TQ_D | BH76 | 12/76 | 0.236 | 0.392 | 12 |

- DT_D on the 12 common barriers gives MAE 0.653 and maximum absolute error 1.269 kcal/mol.
- TQ_D on the 12 common barriers gives MAE 0.236 and maximum absolute error 0.392 kcal/mol.

The shared HTBH38 directions are a single symmetric H + H₂ reaction.
The largest DT_D error comes from NHTBH38 index 22, and the largest TQ_D
error comes from NHTBH38 index 28. Coverage and error exceedance rule out a
full-set chemical-accuracy claim for these runs.

The planned full-set coverage and per-barrier 1 kcal/mol gate is recorded in [summary.json](summary.json).
[barriers.csv](barriers.csv) retains every direction, its signed error, and the missing molecular keys for unassembled barriers.

## Round 1 endpoint outcomes and timing

The median and maximum are fresh-calculator wall times of successful energy-only CUDA endpoints in seconds.

| Recipe | Set | Successful / planned | Refused or unconverged | 420 s timeouts | Other unfinished | Median (s) | Max (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DT_D | HTBH38 | 14/40 | 17 | 9 | 0 | 47.617 | 295.682 |
| DT_D | NHTBH38 | 30/47 | 13 | 4 | 0 | 73.084 | 355.561 |
| DT_D | BH76 | 44/87 | 30 | 13 | 0 | 63.517 | 355.561 |
| TQ_D | HTBH38 | 11/40 | 27 | 2 | 0 | 59.718 | 173.250 |
| TQ_D | NHTBH38 | 24/47 | 18 | 5 | 0 | 112.962 | 385.608 |
| TQ_D | BH76 | 35/87 | 45 | 7 | 0 | 84.593 | 385.608 |

- DT_D recorded 24 correlation work-policy refusals, five numeric-memory
  refusals, one SCF nonconvergence and 13 timeouts.
- TQ_D recorded 30 angular-momentum refusals, six numeric-memory
  refusals, seven SCF nonconvergences, seven timeouts and two DF eigensolve
  input refusals (HTBH38 TS7 and TS13).

The raw CUDA journals retain the original diagnostic messages and the fresh-process wall times, including interrupted attempts.

## Development required on the pinned CUDA path

- **Open-shell full-DF QZ basis support.** The unrestricted component
  needs orbital g functions and the matching fitting bases include
  auxiliary h functions. Its current CUDA closure supports orbital f
  and fitting auxiliary g. This requires work across the basis admission,
  one-electron integrals, full-DF metric and three-center operators, and
  unrestricted Fock/MP2 consumers. End-to-end qualification needs
  independent FP64 reference energies for the newly admitted basis and
  spin domain. [Basis capability implementation](../../../python/generativeqc/basis_capabilities.py)
- **Ordinary CUDA DF eigensolve input admission.** HTBH38 TS7 and TS13
  had the recorded input rejection, affecting barrier indices 13, 14,
  25 and 26. The validation checks shape, finiteness and symmetry of
  the AO matrix, and of the overlap and orthogonalizer when supplied.
  The diagnostic does not identify the offending argument. The required
  development starts by isolating the inconsistent input and its
  producer, then correcting the producer and validating convergence
  and the resulting energies in a regression for both structures.
  [Input validation](../../../src/scf/cuda/df_eigensystem.cpp)

The angular-momentum gap affects 54/76 TQ_D barriers by transition-structure or channel dependency. The affected indices are HTBH38 1–8, 11–36 and NHTBH38 1–12, 29–36. This leaves a maximum of 22/76 TQ_D barriers at the pinned capability boundary. The TQ_D eigensolve-affected directions are included in the angular-momentum list.

DF eigensolve input refusals affect `DT_D` HTBH38 13–14, 25–26 and `TQ_D` HTBH38 13–14, 25–26.

## Round 1 CUDA comparisons to the conventional anchors

The comparison measures the net difference between the CUDA composite and conventional PySCF CCSD(T) in the larger basis.

| Recipe | Complete / 18 planned | MAE (kcal/mol) | Max abs. (kcal/mol) |
| --- | ---: | ---: | ---: |
| DT_D | 4/18 | 0.185 | 0.339 |
| TQ_D | 4/18 | 0.253 | 0.538 |

## Dataset and reference

The Minnesota database authors' [HTBH38/08](https://comp.chem.umn.edu/db/dbs/htbh38.html) and
[NHTBH38/08](https://comp.chem.umn.edu/db/dbs/nhtbh38.html) releases each
supply 19 reaction pairs. Each contains a `REF1` column for comparison
without spin–orbit coupling and a `REF2` best-estimate column. The error
statistics use `REF1`. The retained `REF2` values document which reference
was selected. References for the compilation and original reactions are
[Zheng, Zhao and Truhlar (2009)](https://doi.org/10.1021/ct800568m),
[Zhao, Lynch and Truhlar (2005)](https://doi.org/10.1039/B416937A), and
[Zhao, González-García and Truhlar (2005)](https://doi.org/10.1021/jp045141s).

[inputs.json](inputs.json) records the original set indices, both reference
columns, reaction stoichiometry, the authors' Cartesian XYZ points in
Ångström, Gaussian-input charges and multiplicities, and SHA-256 digests
of both original HTML pages and four source archives. The releases describe
these geometries as QCISD/MG3. The driver converts the coordinates with 0.529177210903 Å/Bohr. The channel map checks balance
of nuclei and charge. Four pairs in NHTBH38 share an SN2 transition
structure between the separated-fragment and bound-complex definitions.
Both directions and all symmetric degeneracies remain in the authors'
76-entry denominator. The plan contains 87 geometry/basis-family keys
across the two sets. Energies exclude vibrational, thermal and spin–orbit
corrections, and use neither counterpoise ghost functions nor reoptimized
geometries.

## Model and acceptance criteria

| Recipe | HF and MP2 extrapolation | CCSD(T) minus MP2 correction |
| --- | --- | --- |
| `DT_D` | cc-pVDZ / cc-pVTZ | cc-pVDZ |
| `TQ_D` | cc-pVTZ / cc-pVQZ | cc-pVDZ |

NHTBH38 indices 13–28 use the corresponding aug-cc-pVXZ series for every
species in both channels, including neutral partners. This keeps the
same diffuse basis protocol throughout each anionic SN2 reaction.
The driver imports spherical orbital and matching (aug-)cc-pVXZ-JKFIT
records from the installed PySCF 2.14.0 basis library. The identities of
the records passed to GenerativeQC are retained per successful point.
Every native energy uses full density fitting for J/K and correlation,
all electrons, RHF for singlets and UHF for higher multiplicities. The
`fp64` policy applies strict convergence and FP64 preconvergence.
There are no initial guess, damping, level shift, or iteration overrides.
The round 1 numeric capacity limit is 32 GiB. In both rounds the
pinned calculator supplies the default iteration and stopping policies
and publishes the actual limits per point. No stability search was
requested. HF is extrapolated as an exponential (exponent 1.63), whereas
the separate MP2 correlation energy is extrapolated as X^-3.

A completed barrier requires all its molecular endpoints to return
converged, finite CUDA energies. Replay reconstructs the reported
composite energy from its role energies within 1e-8 Hartree, then evaluates
the signed barrier as the transition energy minus the channel sum.
One Hartree is 627.5094740631 kcal/mol. The error summaries give MAE,
maximum absolute error, the number within 1 kcal/mol, and coverage using
the original denominator. The full-set 1 kcal/mol gate requires both
complete coverage and every absolute error at or below that threshold.
The common-coverage view uses the same successful barriers for the two
recipes.

### Round 2 protocol

The [frozen plan](round2/plan.json) contains the 95 endpoints without a
round 1 energy. The production runner allows 7,200 seconds per fresh
process. It sets the documented
[correlation work override](../../../docs/user/correlated_methods.md#control-cost-memory-and-convergence)
to `50000000000000` estimated scalar-work units per phase. The largest
recorded round 1 work estimate was approximately `1.21e13` units. The
runner raises `correlation_memory_budget_bytes` to 64 GiB on the 80 GB H100. The budget
bounds scientific numeric buffers. It is distinct from measured device
or host memory pressure. The geometry, charge, multiplicity, basis
records, fitting, FP64 arithmetic and convergence thresholds stay fixed.

The failed unrestricted references already use the
[documented Hcore default](../../../docs/user/initial_guesses.md).
Explicit preliminary guesses reject unrestricted or second-order
domains. The CBS API exposes CC amplitude damping. It has no documented
SCF damping control for this fitted unrestricted CUDA component.
The [stability controller](../../../docs/user/spin_roots.md#spin-root-backends)
requires native CPU exact J/K. Retried points therefore use the original
SCF defaults. The runner uses the energy-only QZ UMP2 basis capability
report to check whether higher angular momentum makes a complete result
impossible for an open-shell endpoint. An endpoint skipped on that basis
retains its primary round 1 diagnosis. The two recorded DF eigensolve
input refusals are also carried forward as development-required outcomes.

The conventional CPU anchors use independent PySCF RHF/UHF, MP2 and
CCSD(T) with the same orbital bases, charge, spin and geometry. The
preselected forward indices are HTBH38 3, 9, 17 and 23, and NHTBH38 3,
13, 29, 31 and 37. Both directions are included. They cover H transfers,
associations, an anionic substitution and an isomerization. Runs use
all electrons and conventional integrals, SCF thresholds of 1e-12
Hartree and 1e-9 orbital-gradient norm, and CCSD thresholds of
1e-10 Hartree and 1e-8 amplitude-update norm. The raw records retain
occupations and spin squares. Each anchor enables two comparisons:

- Conventional PySCF composite versus conventional PySCF CCSD(T) in the
  larger basis measures the effect of the composite recipe on that
  conventional branch, including its basis extrapolation.
- CUDA composite versus the same conventional composite measures the
  net effect of different fitting, implementation and possibly SCF
  roots. CUDA composite versus conventional larger-basis CCSD(T)
  combines the two differences.

## Independent conventional comparisons

All statistics in this table use the conventional PySCF composite on the
completed pairs of the 18 planned anchor barriers. Errors are kcal/mol.

| Recipe | Complete / planned | MAE vs larger-basis CCSD(T) | Max vs larger-basis CCSD(T) | MAE vs REF1 | Max vs REF1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DT_D | 13/18 | 0.491 | 1.656 | 0.933 | 1.539 |
| TQ_D | 9/18 | 0.339 | 1.262 | 0.808 | 1.534 |

The two rows have different coverage. The largest conventional composite
deviation for DT_D is HTBH38 index 4, and for TQ_D it is HTBH38 index 23.
The CPU job was successful at the process level and reported 68 of 78
endpoint successes, six SCF or CC convergence failures and four 480 s
timeouts. See the per-endpoint journal for the individual causes.

## Execution and reproducibility

Both H100 builds use the full archive of the same master revision and
Release SM90. In round 1, the [build driver](build_qz.sh) enabled CUDA
fast compile and disabled shell and stationary AOT. Fast compile applies
NVCC `--Ofast-compile=max` to most native CUDA owners, reducing compiler
optimization effort. It can change kernel throughput and resource use.
With shell AOT disabled, covered shell work can instead use runtime
preparation or fallback, depending on the operator. Both effects can
enter the measured endpoint wall time.
Stationary AOT packages serve the qualified DFT force profiles. This
correlated benchmark requests energies only. Round 1 provides neither a
production timing comparison nor an isolated measurement of the effect
of any build flag.

Round 2 uses the [production build driver](build_production_qz.sh) with
CUDA fast compile off, shell AOT on (`portable_cuda`), native CUDA
stationary kernels and stationary CUDA force AOT on, and CPU stationary
force AOT off. It builds and audits the same six H100 cohort stationary
profiles in both s/p and s/p/d domains. Each GPU job checks the archive
source identity against the loaded native library and records its SHA-256.
The repeated and first-round successful endpoint cohorts are disjoint.
The measured intervals begin before the outer calculator is created and
end after the energy is retrieved. They include all component
preparation and execution. The separate process interval includes Python
startup, imports, basis snapshot creation and result serialization.
These are single observations from fresh processes with no cross-point
SCF warm start. They describe endpoint cost for the recorded allocation.
The CPU reference timing is for the complete HF/MP2/CCSD(T) sequence
in a single basis.

The [aug-QZ/T–Q force validation for #3055](../full-df-aug-qz-3055-20261109/README.md)
provides separate restricted CH+ and CN− high-angular correctness evidence.
The present calculations request only energies. At the measured revision,
the full-DF unrestricted CUDA Fock path admits orbital shells through f
and auxiliary shells through g. Heavy-atom cc-pVQZ and aug-cc-pVQZ
contain g orbital shells. Both rounds use the complete orbital and fitting basis records.

To rebuild the inputs, save the six URL basenames listed in
`inputs.json` to one directory and run `python prepare_inputs.py <downloads>
rebuilt-inputs.json`. The raw journals under `raw/` contain the calculated
energies, wall times, convergence data and failures. Run
`python replay.py --output /tmp/bh76-replay` to regenerate the barrier table
and summary. The allocation scripts preserve the environment and launch commands.
For a fresh GPU build, generate `source.tar.gz` with
`COPYFILE_DISABLE=1 git archive --format=tar.gz 22f72d069 -o source.tar.gz`,
extract it under `source/`, verify the archive and write its digest with
`sha256sum source.tar.gz > source-staged.sha256`. In the allocated qz
image, run `bash build_qz.sh`, followed by
`bash run_qz_gpu.sh HTBH38 <new-output-directory>` (or `NHTBH38`).
For round 2, use the same staged archive, include the frozen plan as
`plan.json`, and run `bash build_production_qz.sh` followed by
`bash run_round2_qz.sh <new-output-directory> <allocated-GPU-count>`.
The first production allocation ran until eight long DT_D endpoints
completed their 2 h attempts. At the cutover, 36 points had terminal
records. The two [continuation plans](round2/plans/) partition the 59
remaining points. Four started DT_D attempts were interrupted at the
cutover and retained in the first allocation journal. Each received a
fresh-process attempt in just one continuation plan. To reproduce a
continuation, supply its plan file as the third argument to
`run_round2_qz.sh`.

Run `python replay_round2.py --output /tmp/bh76-round2` to reproduce both
rounds and cumulative coverage from the published diagnostic journals.

Run `srun bash run_qz_cpu.sh <new-output-directory>` for the conventional
reference, using the original CPU acquisition input to match the archived
receipt. The CPU reference uses a separate
environment with PySCF 2.14.0, NumPy 2.3.5, SciPy 1.15.3 and h5py 3.15.1.

The CPU acquisition input differs from the GPU input only in the channel
mapping of NHTBH38 indices 27 and 28. The XYZ geometries show that complex
2 retains the C–F bond (reactants), whereas complex 1 retains the C–O
bond (products). The two labels were corrected before the master GPU
run. The CPU anchors never use either complex. Replay checks equality
of all geometry, charge, spin, basis-family and reference values and
retains the original CPU acquisition input and helper bytes.

The earlier revision `d154546d2` was superseded by the updated issue.
Its startup failures and cancelled CUDA jobs are recorded separately.
The conventional PySCF calculation does not load GenerativeQC at any
revision and is retained as the independent reference.

The initial master CUDA jobs used one GPU per set. Their completed
endpoints were transferred to the retained record before the remaining
points were distributed over four H100s. The residual assignment excludes
every previously terminal endpoint. An interrupted endpoint remains in its
original journal and is visible as a separate attempt in the sharded run.
Each rank uses a distinct physical GPU and runs one point at a time.

## Allocation accounting

All Jobs were submitted to CQ-NexChem. The platform read back NORMAL/20 for the maximum GPU request (4) and HIGH/35 for the CPU HPC request (10).
GPU estimates span scheduling through reported finish or stop, including image pulls and build time. The selected quota rows charge one point per H100 card-hour. These are estimates based on observed allocation events.

| Task | Role | Estimated H100 card-hours | Estimated points |
| --- | --- | ---: | ---: |
| `gqc-3168-22f72-ht` | master build and HTBH38 | 0.628 | 0.628 |
| `gqc-3168-22f72-nht` | master NHTBH38 | 0.142 | 0.142 |
| `gqc-3168-22f72-shard` | master sharded endpoints | 5.914 | 5.914 |
| `gqc-3168-htbh38-1125a` | superseded CUDA | 0.039 | 0.039 |
| `gqc-3168-htbh38-1125b` | superseded CUDA | 0.120 | 0.120 |
| `gqc-3168-nhtbh38-1125a` | superseded CUDA | 0.029 | 0.029 |
| `gqc-3168-nhtbh38-1125b` | superseded CUDA | 0.115 | 0.115 |
| `gqc-3168-ref-1125a` | independent CPU anchors | — | 0.000 |

Total estimated charge is 6.987 points, including the superseded attempts.
The CPU job completed its Slurm step and its quota rate was zero points.
The scheduled CPU allocation was approximately 5.691 core-hours.
See [allocation.json](allocation.json) for timings, `raw/allocation-receipts.json.xz` for observed scheduler data, and `raw/quota-rates.json` for the queried rates.
Private qz storage paths in logs are represented by `$GQC_ASSETS`. The original acquired-byte digests are retained before tokenization.

### Round 2 allocation

| Round 2 task | Project | H100 cards | Estimated card-hours | Estimated qz points |
| --- | --- | ---: | ---: | ---: |
| `gqc-3168-r2-build` | CQ-NexChem | 1 | 0.924 | 0.924 |
| `gqc-3168-r2-campaign` | CQ-NexChem | 4 | 16.390 | 16.390 |
| `gqc-3168-r2-resume-a` | CQ-NexChem | 4 | 16.148 | 16.148 |
| `gqc-3168-r2-resume-b` | CQ-NexChem | 4 | 19.703 | 19.703 |

Round 2 used an estimated 53.165 H100 card-hours and 53.165 qz points, including the production build. Estimates run from the scheduling event to finish or stop, using the earliest recorded image event when the scheduling event aged out. The event source, effective priority, quota rate and cleanup are retained for each job.
[Round 2 allocation](round2/allocation.json) · [Acquired and retained SHA-256 digests](round2/acquisition-sha256.json)
