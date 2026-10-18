# Bounded larger-than-water WB97M-V Hessian observations

Measured October 16, 2026 on one Slurm-assigned NVIDIA GeForce RTX 5090
(32 GiB class), CUDA 12.9, GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 13.6.0.
The node was shared, not reserved exclusively. These are bounded exploratory
observations, not statistically qualified performance promotion or genuinely
large-molecule scaling results.

## Contract and limits

Both engines use direct RKS WB97M-V, STO-3G spherical AOs, self-consistent VV10,
moving primary/NLC/CPHF **8/4/8** quadrature, fresh processes and fresh SCFs.
The CPHF grid equals the primary grid. DF and ECP are not enabled in this
comparison. The deliberately coarse quadrature is not chemically grid-converged;
reference nonconvergence below is specific to this workload, not a general
GPU4PySCF limitation. No convergence, screening or matrix gate was loosened.

Each engine receives a 180 s complete-endpoint deadline and a conservative
24 GiB total-device sampled-memory watchdog. Device visibility is never
overridden. Timings include startup, SCF, the complete host Hessian, JSON and
teardown, not only Hessian kernels. Persistent caches are reused: no empty-cache
cold-start claim. Memory is sampled at approximately 50 ms intervals and includes
the comparator's CUDA context; it is neither an isolated process allocation
counter nor an exact peak/hard allocation cap.

## Observed endpoints

| Molecule | Atoms / AOs | Native complete endpoint | GPU4PySCF complete endpoint | Qualification |
|---|---:|---:|---:|---|
| H2 control | 2 / 2 | 1.8292 s | 6.6526 s | Full matrices pass |
| Water control | 3 / 7 | 8.9032 s | 9.1827 s | Full matrices pass |
| Formaldehyde initial pilot | 4 / 12 | 33.2998 s | 34.0018 s | Full matrices pass |
| Formaldehyde followup primed median | 4 / 12 | 33.4996 s | 24.4927 s | All nine cross-pairs pass |
| Water dimer | 6 / 14 | 93.1989 s | 11.7443 s | Full matrices pass |
| Methanol | 6 / 14 | 102.6281 s | SCF did not converge | No qualified pair or ratio |
| Ethane | 8 / 16 | Stopped at the 180 s deadline | SCF did not converge | No qualified pair or ratio |

Except the explicitly labeled formaldehyde median, these are single fresh-process
observations, not medians. The followup retains one sample-zero process per
engine plus **two** primed fresh-process repeats, alternating engine order;
sample zero is excluded from that median. The initial pilot is also retained,
not replaced by the faster reference followup. The observed formaldehyde median
native/reference ratio is 1.3677368769; native is slower. Water dimer is about
7.94 times slower in this single observation. Neither implies a hardware-wide
or statistically established speed factor.

All available complete cross-engine matrices pass the unchanged absolute
`3e-6 hartree/bohr^2` gate. The largest error is `3.6513983393e-8`, for water
dimer. Native completed endpoints report no runtime Python, runtime compilation,
numerical host fallback, displaced SCF or intermediate panel download. There
is no fresh energy-finite-difference qualification claim in this followup.

Native sampled total-device usage is about **16.75--16.79 GiB**, versus about
**2.38--2.68 GiB** for the reference, including failed scientific attempts but
excluding initial loader failures. Even H2 already samples 16.75 GiB: the small
Hessian-owner `diagnostics.device_bytes` is not total GPU usage. This campaign
does not profile the origin of that footprint. No sampled usage crosses the
24 GiB watchdog; a missed short-lived spike cannot be excluded.

Semantic work is retained for every completed native endpoint. Formaldehyde
has 12 columns, 31 SCF iterations, 268 induced actions, 7,220,016 exact primitive
components and 144 mixed-grid contractions. Water dimer has 18 columns, 21 SCF
iterations, 406 induced actions, 18,580,752 exact primitive components and 324
mixed-grid contractions. Methanol has 18 columns, 26 SCF iterations and 472
induced actions. The ethane timeout has no completed counters or matrix: do not
infer that all its elapsed time was spent in the Hessian rather than SCF.

## Interpretation and retained failures

Within the tested memory/time limits, water dimer completes accurately but is
already substantially slower than the reference. Do not extrapolate the earlier
tiny-water speed win to larger systems. Stop increasing system size until the
actual complete endpoint is profiled and improved. Native default admission is
also bounded: at this source revision STO-3G ethanol has 15 shells and requires
`2 * 15**4 = 101250` ordered exact groups, above the default 100000 limit. That
is a static code/basis bound, not a timed ethanol calculation. Limits are not
raised or bypassed to present an unsupported larger molecule as working.

Slurm jobs 9072, 9099 and 9101 contain the numerical campaign. Pilot methanol
and followup ethane reference attempts fail at the strict 240-cycle SCF ceiling;
native ethane is killed at its common 180 s deadline. Pilot failure wall times
were not recorded precisely, so none are invented. The updated measurement
driver journals failed-attempt envelope durations; these are not successful
complete-Hessian timings. Three earlier deployment-only attempts failed to
load a missing library SONAME symlink; their raw logs/reports are retained
separately and excluded from numerical observations.

## Provenance and replay

Unchanged native artifacts come from compiled revision
`5aa703721fdce6a220838676af615d574f0fed05`; production code is not rebuilt or
modified for this campaign. CLI/library hashes and compiled-source identity
match the preceding direction-reuse publication. Measurement-driver changes
are retained separately as exact compressed snapshots and reconstruction
patches; the manifest's dirty flag refers only to these driver overlays.
`samples.json.xz` retains every report, successful raw matrix, native work
counter, memory observation, exact child stdout/stderr and geometry, including
failures. `qualification.json.xz` retains recomputable matrix gates and
provenance. The manifest authenticates all selected files.

Set `SOURCE_CHECKOUT` to a repository containing the measured revision,
`NATIVE_ROOT` to its already-built linked CLI/library directory (including the
SONAME symlink), and `REFERENCE_PYTHON` to the pinned GPU4PySCF environment.
Set `REFERENCE_LIBRARY_PATH` to its cutensor library directory if necessary.
`reproduce.sh` creates a private archived-source overlay without modifying the
checkout and validates native binary hashes. On the node1 submission host:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:14:00 bash /path/to/reproduce.sh pilot
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:09:00 bash /path/to/reproduce.sh followup
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:07:00 bash /path/to/reproduce.sh dimer
```

Adapt node and GRES together. Reproduction retains all failed cases and exits
nonzero if any required endpoint fails. This standalone replay script is syntax
checked, not separately replayed; the recorded allocations executed the
authenticated driver snapshots. No new broad CUDA qualification or sanitizer
campaign is claimed for unchanged production binaries.
