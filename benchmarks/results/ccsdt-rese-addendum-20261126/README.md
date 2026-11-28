# CCSD(T) equilibrium-geometry addendum

This local addendum extends #3166 and the fixed cohort in #3178. Its
scientific revision is `22f72d0692a760c2892cd7791022584d9ae1c3e5`. The
baseline evidence and literature audit come from `99dfcb803c8c176891bc5da951f89d4307cfe54e`
under `benchmarks/results/ccsdt-rese-20261125/`.

## Earlier time limits and rerun protocol

The original 1,200 s and 2,400 s limits killed the DF process with signal
9 for syn-ethenol and aziridine. Both runs had completed RHF preoptimization.
The DF request and timeout receipt survived. Standard output and standard
error were empty, and there was no optimizer checkpoint. Accepted DF steps,
electronic iteration counts, per-step durations, and oscillation cannot be
reconstructed for those attempts.

The reruns start from the original seed and repeat the original CUDA RHF
preoptimization. The same all-electron correlation-only DF-CCSD(T) method,
cc-pCVTZ orbital and cc-pwCVTZ-RIFIT fitting sets (cc-pVTZ and
cc-pVTZ-RIFIT on H), FP64 arithmetic, 2e-5 Eh/bohr maximum atomic force,
80-step cap, 0.25 bohr maximum step, and electronic convergence settings
apply. Each target stage has a 28,800 s process time limit. The H100 jobs
allow 8.25 h to include the CLI relink and RHF preparation.

[The trace patch](optimizer-trace.patch) attaches a read-only observer to
the original L-BFGS implementation and emits one start and end record per
energy-plus-force evaluation and one record per accepted state. Accepted
records include the coordinates, energy, maximum atomic force and elapsed
time since the preceding accepted state. The initial evaluation is timed
separately. The patched CLI translation unit links to the same checksummed
native library used by the baseline. The raw `reference_scf_iterations`
label, as originally emitted, is misleading for DF-RCCSD(T). At the pinned
revision the generic result count is the CCSD count in that path. The summary
uses the dedicated CCSD diagnostic and leaves the reference SCF count
unreported. The RHF trace's generic count is its SCF iteration count.

## Independent curvature

The addendum reuses the three verified DF minima from #3178 when their
native geometry receipt hashes agree. The conventional HF check remains
available in the baseline. Each new check uses PySCF 2.14.0 with exact
all-electron RHF, density fitting in correlation only, and the orbital
and auxiliary basis from the GPU calculation. The test applies the same
projected finite-difference energy Hessian, 0.004 and 0.002 bohr steps,
Richardson first derivative, center-energy and force gates, curvature
floor, and step-stability gate as `check_df_minima.py`. The
[parallel adapter](parallel-minima.patch) distributes displacement
evaluations over independent single-thread PySCF processes. It keeps the
finite-difference summation order and applies five rigid modes to linear
references, including linear polyatomics, or six to nonlinear references.
HPC allocations require the Xeon CPU Max 9462 model before doing any
calculation. Syn-ethenol has seven atoms and 15 vibrational modes,
which require 450 displacements per step size. Its two allocations
run the step sizes against the same checksummed PySCF
center. The coarse job assembles the independent check after the fine
job publishes its completion record. The assembled runtime for this
case covers the final reduction. The raw receipts give each allocation
interval and displacement runtime.

The aziridine HPC request remained queued under the personal 800-CPU
quota and was stopped and deleted before allocation. Its check runs
both step sizes in the existing `xshengrui` notebook. Aziridine has
eight atoms and 18 vibrational modes, which require 648 displacements
per step size. The notebook runs on a verified Xeon
Gold 6530 using the same PySCF environment, inputs and gates. The
launcher records the detached process and coordinator PIDs, timestamps,
hardware, and resource limits. It uses 32 single-threaded workers
with 6 GiB of address space per Python process. Including the coordinator,
the limit is 198 GiB. PySCF also uses asynchronous I/O helpers. The
running process session was capped at 40 logical CPUs across both CPU
sockets, with the applied affinity recorded in the raw notebook output.
Results apply locally to the accepted stationary
geometry on the projected electronic energy surface.

## Combined results

The fixed cohort has **22/22** converged DF geometries, **22** complete curvature checks and **22** verified DF minima.

| Comparison | Parameters | MAE | Maximum absolute error | Largest deviation |
| --- | ---: | ---: | ---: | --- |
| Semi-experimental bonds | 46 | 0.00255 Å | 0.02118 Å | fno N-F |
| Semi-experimental angles | 28 | 0.3243 ° | 1.4539 ° | nh3 H-N-H |

The primary comparison uses the reference set audited in #3178. Using
Bak et al. Table III's `Expt.` angle of 106.7° for NH₃ in place of
the 107.25° `Emp.` entry gives an angle MAE of 0.3046°
and a maximum of 0.9039° over the same angle parameters.
[Bak et al., Table III, p. 6550](https://doi.org/10.1063/1.1357225).

| Molecule | DF outcome | Target steps / evaluations | DF wall (s) | Curvature | Lowest at either step |
| --- | --- | --- | ---: | --- | ---: |
| hf | converged | 3/4 | 68.75 | minimum_verified | 1.264398 |
| h2o | converged | 3/4 | 62.84 | minimum_verified | 0.1702381 |
| co | converged | 4/6 | 226.59 | minimum_verified | 2.428116 |
| n2 | converged | 4/6 | 204.98 | minimum_verified | 2.938631 |
| f2 | converged | 4/5 | 173.63 | minimum_verified | 0.6160083 |
| hcn | converged | 5/7 | 352.60 | minimum_verified | 0.08063012 |
| hnc | converged | 5/7 | 352.07 | minimum_verified | 0.03353523 |
| ch4 | converged | 2/3 | 51.83 | minimum_verified | 0.09431953 |
| nh3 | converged | 5/6 | 116.01 | minimum_verified | 0.1166218 |
| hno | converged | 7/8 | 443.09 | minimum_verified | 0.2233625 |
| hof | converged | 5/6 | 350.88 | minimum_verified | 0.1555348 |
| co2 | converged | 3/4 | 362.17 | minimum_verified | 0.2201536 |
| c2h2 | converged | 6/8 | 314.11 | minimum_verified | 0.03930485 |
| ch2o | converged | 8/9 | 392.86 | minimum_verified | 0.1372959 |
| trans-n2h2 | converged | 7/8 | 325.06 | minimum_verified | 0.1260146 |
| c2h4 | converged | 7/8 | 535.22 | minimum_verified | 0.03050112 |
| fno | converged | 8/9 | 885.26 | minimum_verified | 0.2006029 |
| trans-hono | converged | 11/13 | 1687.81 | minimum_verified | 0.03740389 |
| ch2f2 | converged | 6/7 | 1134.25 | minimum_verified | 0.1038985 |
| syn-ethenol | converged | 20/23 | 6031.85 | minimum_verified | 0.01076969 |
| trans-formic-acid | converged | 11/13 | 2180.44 | minimum_verified | 0.03867969 |
| aziridine | converged | 17/21 | 6177.64 | minimum_verified | 0.02498262 |

Curvatures are unweighted Cartesian values in Eh/bohr². The original
HF conventional minimum remains in the baseline and is outside the DF count.

## Rerun trace

Elapsed cutoffs below use summed energy-plus-force evaluation times in
the new run. They approximate how much of the new trajectory fits
within the old clock limits.

| Molecule | Evaluations | Accepted steps | Rejected trials | Evaluation time min / median / max (s) | CCSD iterations min / median / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| syn-ethenol | 23 | 20 | 2 | 261.35 / 261.96 / 263.78 | 24.0 / 24.0 / 24.0 |
| aziridine | 21 | 17 | 3 | 292.47 / 294.25 / 295.26 | 19.0 / 22.0 / 22.0 |

- syn-ethenol
  - At 1200 s, 4 evaluations, 3 accepted steps and last force 0.0016725 Eh/bohr.
  - At 2400 s, 9 evaluations, 8 accepted steps and last force 0.00071393 Eh/bohr.
  - 3 accepted steps increased the maximum force. The lowest cosine between consecutive displacements was -0.757. Accepted energies remained nonincreasing.
  - The separate RHF preoptimization took 28–31 SCF iterations per evaluation. The RHF endpoint differed from the old runs by at most 3e-12 bohr per coordinate.

- aziridine
  - At 1200 s, 4 evaluations, 3 accepted steps and last force 0.0018358 Eh/bohr.
  - At 2400 s, 8 evaluations, 6 accepted steps and last force 0.00082734 Eh/bohr.
  - 5 accepted steps increased the maximum force. The lowest cosine between consecutive displacements was -0.703. Accepted energies remained nonincreasing.
  - The separate RHF preoptimization took 23–25 SCF iterations per evaluation. The RHF endpoint differed from the old runs by at most 1.2e-12 bohr per coordinate.

The 1,200 and 2,400 s limits fit only a fraction of the accepted steps
needed in the reruns. Per-evaluation time and CCSD iteration count stayed
in narrow ranges. Rejected line-search trials added further time.


## Compute

GPU card-hours estimate the qz allocated interval. CPU-hours are the
allocated core count multiplied by the observed program interval,
or the interval to the stop request for interrupted allocations.
The existing notebook's per-process runtime and worker-slot hours are
recorded separately in the provenance. The H100 quota charged 1 point
per instance-hour. The CPU-only notebook and
Slurm quotas showed 0 points per hour. Invoice totals were unavailable.

| Job | Project | Platform state | GPU card-hours | Allocated CPU-hours | Estimated points |
| --- | --- | --- | ---: | ---: | ---: |
| `gqc-rese-add-1126-aziridine` | CQ-NexChem | SUCCEEDED | 1.725 | — | 1.725 |
| `gqc-rese-add-1126-cpu-aziridine` | cq-nex-chem | STOPPED | 0.000 | 0.000 | 0.000 |
| `gqc-rese-add-1126-cpu-difluoro-resume` | cq-nex-chem | STOPPED | 0.000 | 0.000 | 0.000 |
| `gqc-rese-add-1126-cpu-difluoro-resume40` | cq-nex-chem | SUCCEEDED | 0.000 | 180.822 | 0.000 |
| `gqc-rese-add-1126-cpu-difluoro` | cq-nex-chem | STOPPED | 0.000 | 3.087 | 0.000 |
| `gqc-rese-add-1126-cpu-formic-resume` | cq-nex-chem | STOPPED | 0.000 | 0.000 | 0.000 |
| `gqc-rese-add-1126-cpu-formic-resume40` | cq-nex-chem | SUCCEEDED | 0.000 | 175.289 | 0.000 |
| `gqc-rese-add-1126-cpu-formic` | cq-nex-chem | STOPPED | 0.000 | 2.818 | 0.000 |
| `gqc-rese-add-1126-cpu-medium-resume` | cq-nex-chem | STOPPED | 0.000 | 0.000 | 0.000 |
| `gqc-rese-add-1126-cpu-medium-resume40` | cq-nex-chem | STOPPED | 0.000 | 2.356 | 0.000 |
| `gqc-rese-add-1126-cpu-medium-resume40b` | cq-nex-chem | SUCCEEDED | 0.000 | 211.400 | 0.000 |
| `gqc-rese-add-1126-cpu-medium` | cq-nex-chem | STOPPED | 0.000 | 2.353 | 0.000 |
| `gqc-rese-add-1126-cpu-nox-resume` | cq-nex-chem | SUCCEEDED | 0.000 | 138.667 | 0.000 |
| `gqc-rese-add-1126-cpu-nox` | cq-nex-chem | STOPPED | 0.000 | 1.978 | 0.000 |
| `gqc-rese-add-1126-cpu-small` | cq-nex-chem | SUCCEEDED | 0.000 | 33.404 | 0.000 |
| `gqc-rese-add-1126-cpu-syn-ethenol-fine` | cq-nex-chem | FAILED | 0.000 | 0.189 | 0.000 |
| `gqc-rese-add-1126-cpu-syn-ethenol-fine2` | cq-nex-chem | SUCCEEDED | 0.000 | 427.800 | 0.000 |
| `gqc-rese-add-1126-cpu-syn-ethenol` | cq-nex-chem | SUCCEEDED | 0.000 | 499.789 | 0.000 |
| `gqc-rese-add-1126-syn-ethenol` | CQ-NexChem | SUCCEEDED | 1.688 | — | 1.688 |
| `xshengrui / aziridine curvature process` | CQ-NexChem | SUCCEEDED | 0.000 | — | 0.000 |

Total documented use was 3.414 GPU card-hours, 1680.0 allocated CPU-hours and an estimated 3.414 points. The aziridine notebook process used 43.975 h of elapsed time (1407.2 maximum worker-slot hours) at zero points.

The initial eight-core allocations that projected beyond their eight-hour
windows were stopped after preserving completed points. The restarted
jobs reused those point receipts. Unscheduled 55-core requests were
withdrawn in favor of 40-core allocations. The first medium 40-core
launch omitted the resume environment value and began a duplicate center
calculation. It was stopped and replaced after validating the corrected
command and copied checkpoint. The scheduler records and superseded
outputs are retained alongside the final receipts.

The baseline has separate accounting of 7.146 GPU card-hours and
estimated points, including 0.093 for the withdrawn d154546d2 jobs,
plus 1.403 allocated CPU-hours at zero points.

## Replay

Use Python 3.10 or newer. Point the replay tool at the immutable baseline benchmark directory. When
it is absent in this checkout, use `git archive` on `99dfcb803` to extract
the original directory to scratch or use the original worktree.

```sh
python3 benchmarks/results/ccsdt-rese-addendum-20261126/replay_addendum.py \
  --baseline /path/to/benchmarks/results/ccsdt-rese-20261125 \
  --output /tmp/rese-addendum-summary.json
```

Raw traces and curvature samples are stored in compressed JSON in this
directory. The byte-exact qz transfers are in the local, Git-ignored
`raw/` directory. Their SHA-256 digests, scheduler receipts and persistent
remote output paths are listed in `provenance.json`.
